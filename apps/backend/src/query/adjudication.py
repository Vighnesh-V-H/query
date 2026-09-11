"""Adjudicate heuristic-flagged closures with the configured labeler role.

Ticket 6's keyword heuristics label the obvious endings and flag the ambiguous
middle for adjudication (ADR-0006): brand replies that move the conversation to
DMs, customer messages that only announce a DM, and customer closings that
neither acknowledge nor continue the issue. This stage resolves every flagged
Interaction with one LLM call to the configured ``labeler`` role, producing the
full labeled sample. ADR-0001 requires the hybrid: keywords where the ending is
clear, an LLM where only the conversation can tell, and a one-line justification
per adjudicated label that doubles as audit data.

Each flagged Interaction is rendered as its visible transcript and classified
as Resolved, Uncertain, or Unresolved. A move to DMs hides the outcome, so the
labeler is told to judge only what the transcript shows: a friendly DM hand-off
with no visible evidence of a fix is not Resolved. Every other Interaction keeps
its heuristic label unchanged.

The output is one record per Interaction with its final ``label``, the
provenance ``source`` (``heuristic`` or ``labeler``), the ``reason`` (for
labeler records, the model's one-line justification), the heuristic
``flag_reason`` that sent the case to adjudication, and the labeler ``model``.
The report counts the full sample's label distribution, the heuristic and
adjudicated shares, and the flagged volume by reason.

The labeler is the pipeline's only non-deterministic stage: reruns recall the
model for every flagged Interaction and can move labels. The labeler's replies
are retried once with a repair prompt when they are not valid JSON. Because the
calls are slow and paid, an optional append-only cache records each verdict as
it completes. The cache pins verdicts, so a resumed run reuses any entry whose
prompt hash still matches instead of calling the labeler again; delete the
cache file when changing the labeler model or prompt and fresh verdicts are
wanted.
"""

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from query import cachefile, closure, llm

DEFAULT_INTERACTIONS_PATH = closure.DEFAULT_INPUT_PATH
DEFAULT_LABELS_PATH = closure.DEFAULT_LABELS_PATH
DEFAULT_ADJUDICATED_LABELS_PATH = closure.REPO_ROOT / "data" / "closure-labels-final.jsonl"

LABEL_SOURCES = ("heuristic", "labeler")

SYSTEM_PROMPT = (
    "You are a precise annotation assistant for customer-support research. "
    "You classify the closure of one interaction and reply with raw JSON only."
)
TAXONOMY = (
    "- resolved: evidence of natural closure — the customer explicitly confirms "
    "the issue is solved, or the brand's final reply says the issue/action was "
    "completed.\n"
    "- uncertain: the visible conversation ends without evidence the issue was "
    "actually solved — typically the customer went silent, but also when their "
    "final message neither confirms resolution nor continues the issue.\n"
    "- unresolved: the customer keeps asking, or the brand indicated it could "
    "not help."
)


class AdjudicationError(Exception):
    """Raised when flagged closures cannot be adjudicated."""


@dataclass(frozen=True)
class AdjudicatedLabel:
    """One Interaction's final closure label after adjudication."""

    interaction_id: int
    label: closure.Label
    source: Literal["heuristic", "labeler"]
    reason: str
    flag_reason: str | None
    model: str | None


@dataclass(frozen=True)
class CachedVerdict:
    """A labeler verdict stored for resuming an interrupted run."""

    interaction_id: int
    prompt_sha256: str
    label: closure.Label
    reason: str
    flag_reason: str
    model: str


@dataclass(frozen=True)
class AdjudicationReport:
    """Counts of the full labeled sample, for honest reporting."""

    total: int
    resolved: int
    uncertain: int
    unresolved: int
    heuristic_resolved: int
    heuristic_uncertain: int
    heuristic_unresolved: int
    adjudicated: int
    adjudicated_resolved: int
    adjudicated_uncertain: int
    adjudicated_unresolved: int
    flagged_by_reason: Mapping[str, int]
    models: tuple[str, ...]


class AdjudicationCache:
    """Append-only JSONL cache of labeler verdicts, safe under worker threads.

    Callers load the verdicts once before a run and let every completed
    adjudication be recorded immediately, so a run killed mid-way resumes
    without paying for calls it already made. The crash-safe append and
    tail-repair mechanics live in :mod:`query.cachefile`.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._cache = cachefile.AppendOnlyCache(
            self.path,
            serialize=cached_verdict_to_json,
            parse=_parse_cache_line,
            key=lambda verdict: verdict.interaction_id,
            error_type=AdjudicationError,
        )

    def verdicts(self) -> dict[int, CachedVerdict]:
        """Read the cached verdicts; a missing cache file is an empty cache."""
        return self._cache.entries()

    def record(self, verdict: CachedVerdict) -> None:
        """Append one verdict, atomically with respect to worker threads."""
        self._cache.record(verdict)


def adjudicate_closures(
    interactions: Sequence[closure.Interaction],
    heuristic_labels: Sequence[closure.ClosureLabel],
    infer: Callable[[str], llm.LLMReply] | None = None,
    workers: int = 1,
    cache: AdjudicationCache | None = None,
) -> tuple[tuple[AdjudicatedLabel, ...], AdjudicationReport]:
    """Resolve every flagged ClosureLabel with the labeler role.

    Returns one final label per Interaction, in input order, and a report of
    the full sample's label distribution. ``infer`` is the labeler call, kept
    injectable for tests; ``workers`` bounds concurrent labeler calls; ``cache``
    reuses and records verdicts so interrupted runs can resume.
    """
    if workers < 1:
        raise AdjudicationError(f"workers must be at least 1, got {workers}")
    by_id = _index_labels(interactions, heuristic_labels)
    infer = infer if infer is not None else call_labeler
    saved = cache.verdicts() if cache is not None else {}
    record = cache.record if cache is not None else None

    flagged = [
        (interaction, by_id[interaction.interaction_id].reason)
        for interaction in interactions
        if by_id[interaction.interaction_id].needs_adjudication
    ]
    verdicts = _run_flagged(flagged, infer, workers, saved, record)
    verdict_iter = iter(verdicts)

    labels: list[AdjudicatedLabel] = []
    distribution = {"resolved": 0, "uncertain": 0, "unresolved": 0}
    heuristic_counts = {"resolved": 0, "uncertain": 0, "unresolved": 0}
    adjudicated_counts = {"resolved": 0, "uncertain": 0, "unresolved": 0}
    flagged_by_reason: dict[str, int] = {}
    models: list[str] = []
    for interaction in interactions:
        heuristic = by_id[interaction.interaction_id]
        if heuristic.needs_adjudication:
            verdict = next(verdict_iter)
            labels.append(verdict)
            distribution[verdict.label] += 1
            adjudicated_counts[verdict.label] += 1
            flagged_by_reason[verdict.flag_reason] = (
                flagged_by_reason.get(verdict.flag_reason, 0) + 1
            )
            if verdict.model and verdict.model not in models:
                models.append(verdict.model)
        else:
            labels.append(
                AdjudicatedLabel(
                    interaction_id=interaction.interaction_id,
                    label=heuristic.label,
                    source="heuristic",
                    reason=heuristic.reason,
                    flag_reason=None,
                    model=None,
                )
            )
            distribution[heuristic.label] += 1
            heuristic_counts[heuristic.label] += 1

    report = AdjudicationReport(
        total=len(interactions),
        resolved=distribution["resolved"],
        uncertain=distribution["uncertain"],
        unresolved=distribution["unresolved"],
        heuristic_resolved=heuristic_counts["resolved"],
        heuristic_uncertain=heuristic_counts["uncertain"],
        heuristic_unresolved=heuristic_counts["unresolved"],
        adjudicated=len(flagged),
        adjudicated_resolved=adjudicated_counts["resolved"],
        adjudicated_uncertain=adjudicated_counts["uncertain"],
        adjudicated_unresolved=adjudicated_counts["unresolved"],
        flagged_by_reason=dict(
            sorted(flagged_by_reason.items(), key=lambda item: (-item[1], item[0]))
        ),
        models=tuple(models),
    )
    return tuple(labels), report


def call_labeler(prompt: str) -> llm.LLMReply:
    """Ask the configured labeler role to adjudicate one flagged closure."""
    return llm.call_llm(prompt, role="labeler", system=SYSTEM_PROMPT, temperature=0.0)


def build_prompt(interaction: closure.Interaction, flag_reason: str) -> str:
    """Render the taxonomy, the heuristic flag, and the visible transcript."""
    transcript = "\n".join(
        f"{turn.side}: {turn.text}" for turn in interaction.turns
    )
    return (
        "Classify the closure of this customer-support interaction.\n\n"
        f"Categories:\n{TAXONOMY}\n\n"
        "Judge only what the transcript shows. Moving to DMs hides the outcome: "
        "when the visible conversation contains no evidence the issue was solved, "
        "do not label it resolved however friendly the ending is.\n\n"
        f"Heuristic flag: {flag_reason}\n\n"
        f"Transcript (interaction {interaction.interaction_id}):\n{transcript}\n\n"
        "Reply with raw JSON only, no markdown and no prose:\n"
        '{"label": "resolved" | "uncertain" | "unresolved", '
        '"justification": "one sentence naming the decisive evidence"}'
    )


def adjudicated_label_to_json(label: AdjudicatedLabel) -> dict:
    """Serialize an AdjudicatedLabel to a JSON-compatible mapping."""
    return {
        "interaction_id": label.interaction_id,
        "label": label.label,
        "source": label.source,
        "reason": label.reason,
        "flag_reason": label.flag_reason,
        "model": label.model,
    }


def cached_verdict_to_json(verdict: CachedVerdict) -> dict:
    """Serialize a CachedVerdict to a JSON-compatible mapping."""
    return {
        "interaction_id": verdict.interaction_id,
        "prompt_sha256": verdict.prompt_sha256,
        "label": verdict.label,
        "reason": verdict.reason,
        "flag_reason": verdict.flag_reason,
        "model": verdict.model,
    }


def read_adjudication_cache(input_path: Path | str) -> dict[int, CachedVerdict]:
    """Read the resumable verdict cache; a missing file is an empty cache.

    The cache is an append-only log, so the last record for an Interaction wins
    (a recomputed verdict supersedes a stale one). A trailing record cut short
    by an interruption is dropped so earlier verdicts still resume the run;
    any other malformed record fails the run.
    """
    return cachefile.read_cache(
        input_path,
        _parse_cache_line,
        key=lambda verdict: verdict.interaction_id,
        error_type=AdjudicationError,
    )


def _parse_cache_line(line: str, location: str) -> CachedVerdict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AdjudicationError(f"malformed JSON in cache on {location}: {exc}") from exc
    if not isinstance(record, dict):
        raise AdjudicationError(f"malformed cache record on {location}: expected an object")
    interaction_id = record.get("interaction_id")
    if type(interaction_id) is not int:
        raise AdjudicationError(
            f"malformed cache record on {location}: invalid interaction_id"
        )
    prompt_sha256 = record.get("prompt_sha256")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise AdjudicationError(
            f"malformed cache record on {location}: invalid prompt_sha256"
        )
    label = record.get("label")
    if label not in closure.CLOSURE_LABELS:
        raise AdjudicationError(
            f"malformed cache record on {location}: invalid label {label!r}"
        )
    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise AdjudicationError(
            f"malformed cache record on {location}: invalid reason"
        )
    flag_reason = record.get("flag_reason")
    if not isinstance(flag_reason, str) or not flag_reason:
        raise AdjudicationError(
            f"malformed cache record on {location}: invalid flag_reason"
        )
    model = record.get("model")
    if not isinstance(model, str) or not model:
        raise AdjudicationError(f"malformed cache record on {location}: invalid model")
    return CachedVerdict(
        interaction_id=interaction_id,
        prompt_sha256=prompt_sha256,
        label=label,
        reason=reason,
        flag_reason=flag_reason,
        model=model,
    )


def stage_adjudicated_labels_jsonl(
    labels: tuple[AdjudicatedLabel, ...], output_path: Path | str
) -> Path:
    """Write final labels to a temporary sibling, ready to be swapped in.

    Callers that must update several files as one transaction stage every
    output first and then swap the returned paths in together, so a write
    failure cannot leave a partial set of new files behind.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for label in labels:
                handle.write(json.dumps(adjudicated_label_to_json(label)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_adjudicated_labels_jsonl(
    input_path: Path | str = DEFAULT_ADJUDICATED_LABELS_PATH,
) -> tuple[AdjudicatedLabel, ...]:
    """Read final closure labels back from the JSON Lines format written above.

    Inverse of :func:`stage_adjudicated_labels_jsonl`: parses every record and
    validates its structure and unique interaction ids, so downstream stages
    never see a malformed label file. Every record must carry a final label;
    ``source`` selects which of ``flag_reason`` and ``model`` are required.
    Returns the labels in file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise AdjudicationError(f"adjudicated labels JSONL does not exist: {input_path}")
    labels = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdjudicationError(
                    f"malformed JSON on {location}: {exc}"
                ) from exc
            label = parse_adjudicated_label_record(record, location)
            if label.interaction_id in seen_ids:
                raise AdjudicationError(
                    f"duplicate interaction_id {label.interaction_id} on {location}"
                )
            seen_ids.add(label.interaction_id)
            labels.append(label)
    return tuple(labels)


def label_provenance_error(
    source: str, flag_reason: object, model: object
) -> str | None:
    """Return why a label's provenance does not match its source, if it does not.

    A ``heuristic`` label must not carry labeler fields; a ``labeler`` label
    must carry both the flag it resolved and the model that produced it.
    Returns the problem without a location prefix, or ``None`` when the
    provenance is valid. ``source`` must be one of :data:`LABEL_SOURCES`;
    callers validate that first.
    """
    if source == "heuristic":
        if flag_reason is not None or model is not None:
            return "heuristic labels carry no flag_reason or model"
        return None
    if not isinstance(flag_reason, str) or not flag_reason:
        return "labeler labels need a flag_reason"
    if not isinstance(model, str) or not model:
        return "labeler labels need a model"
    return None


def adjudicated_label_error(label: AdjudicatedLabel) -> str | None:
    """Return why a final label does not satisfy the label contract, if it does not.

    The single contract shared by the label reader and by stages whose records
    embed a final label: a build boundary that validates through this function
    provably emits labels the reader accepts. Returns the problem without a
    location prefix, or ``None`` when the label is valid. A ``heuristic`` label
    must not carry labeler fields; a ``labeler`` label must carry both the flag
    it resolved and the model that produced it.
    """
    if type(label.interaction_id) is not int:
        return "invalid interaction_id"
    if label.label not in closure.CLOSURE_LABELS:
        return f"invalid label {label.label!r}"
    if label.source not in LABEL_SOURCES:
        return f"invalid source {label.source!r}"
    if not isinstance(label.reason, str) or not label.reason.strip():
        return "invalid reason"
    return label_provenance_error(label.source, label.flag_reason, label.model)


def parse_adjudicated_label_record(
    record: object, location: str
) -> AdjudicatedLabel:
    """Validate one decoded final-label mapping and build the AdjudicatedLabel.

    Shared by the label reader and by stages whose records embed the final
    label alongside their own fields, so the label contract is validated in
    one place: the mapping is decoded into an `AdjudicatedLabel` and checked
    with :func:`adjudicated_label_error`.
    """
    if not isinstance(record, dict):
        raise AdjudicationError(
            f"malformed adjudicated label on {location}: expected an object"
        )
    label = AdjudicatedLabel(
        interaction_id=record.get("interaction_id"),
        label=record.get("label"),
        source=record.get("source"),
        reason=record.get("reason"),
        flag_reason=record.get("flag_reason"),
        model=record.get("model"),
    )
    error = adjudicated_label_error(label)
    if error is not None:
        raise AdjudicationError(f"malformed adjudicated label on {location}: {error}")
    return label


def _index_labels(
    interactions: Sequence[closure.Interaction],
    heuristic_labels: Sequence[closure.ClosureLabel],
) -> dict[int, closure.ClosureLabel]:
    """Index heuristic labels by Interaction id and verify they line up."""
    by_id: dict[int, closure.ClosureLabel] = {}
    for label in heuristic_labels:
        if label.interaction_id in by_id:
            raise AdjudicationError(
                f"duplicate closure label for interaction {label.interaction_id}"
            )
        if label.needs_adjudication != (label.label is None):
            raise AdjudicationError(
                f"closure label for interaction {label.interaction_id} has "
                "inconsistent needs_adjudication"
            )
        by_id[label.interaction_id] = label
    input_ids = {interaction.interaction_id for interaction in interactions}
    missing = sorted(input_ids - set(by_id))
    if missing:
        sample = ", ".join(str(interaction_id) for interaction_id in missing[:5])
        raise AdjudicationError(
            f"closure labels missing for {len(missing)} interactions (e.g. {sample})"
        )
    unknown = sorted(set(by_id) - input_ids)
    if unknown:
        sample = ", ".join(str(interaction_id) for interaction_id in unknown[:5])
        raise AdjudicationError(
            f"closure labels reference {len(unknown)} unknown interactions (e.g. {sample})"
        )
    return by_id


def _run_flagged(
    flagged: Sequence[tuple[closure.Interaction, str]],
    infer: Callable[[str], llm.LLMReply],
    workers: int,
    saved: Mapping[int, CachedVerdict],
    record: Callable[[CachedVerdict], None] | None,
) -> list[AdjudicatedLabel]:
    """Adjudicate flagged Interactions, preserving input order."""
    if workers == 1 or len(flagged) <= 1:
        return [
            _adjudicate_one(interaction, reason, infer, saved, record)
            for interaction, reason in flagged
        ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_adjudicate_one, interaction, reason, infer, saved, record)
            for interaction, reason in flagged
        ]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _adjudicate_one(
    interaction: closure.Interaction,
    flag_reason: str,
    infer: Callable[[str], llm.LLMReply],
    saved: Mapping[int, CachedVerdict],
    record: Callable[[CachedVerdict], None] | None,
) -> AdjudicatedLabel:
    """Classify one flagged Interaction, reusing a matching cached verdict.

    The labeler is retried once with a repair prompt when a reply is not valid
    JSON. Fresh verdicts are recorded immediately so an interrupted run can
    resume.
    """
    prompt = build_prompt(interaction, flag_reason)
    cache_key = llm.prompt_sha256(SYSTEM_PROMPT, prompt)
    cached = saved.get(interaction.interaction_id)
    if cached is not None and cached.prompt_sha256 == cache_key:
        return AdjudicatedLabel(
            interaction_id=cached.interaction_id,
            label=cached.label,
            source="labeler",
            reason=cached.reason,
            flag_reason=cached.flag_reason,
            model=cached.model,
        )
    reply = _call_labeler(infer, prompt, interaction.interaction_id)
    try:
        label, justification = _parse_reply(interaction.interaction_id, reply.content)
    except AdjudicationError:
        repair_prompt = llm.build_repair_prompt(prompt, reply.content)
        reply = _call_labeler(infer, repair_prompt, interaction.interaction_id)
        label, justification = _parse_reply(interaction.interaction_id, reply.content)
    verdict = AdjudicatedLabel(
        interaction_id=interaction.interaction_id,
        label=label,
        source="labeler",
        reason=justification,
        flag_reason=flag_reason,
        model=reply.model,
    )
    if record is not None:
        record(
            CachedVerdict(
                interaction_id=verdict.interaction_id,
                prompt_sha256=cache_key,
                label=verdict.label,
                reason=verdict.reason,
                flag_reason=verdict.flag_reason,
                model=verdict.model,
            )
        )
    return verdict


def _call_labeler(
    infer: Callable[[str], llm.LLMReply], prompt: str, interaction_id: int
) -> llm.LLMReply:
    try:
        return infer(prompt)
    except Exception as exc:  # noqa: BLE001 - provider errors vary
        raise AdjudicationError(
            f"labeler call failed for interaction {interaction_id}: {exc}"
        ) from exc


def _parse_reply(interaction_id: int, content: str) -> tuple[closure.Label, str]:
    """Validate the labeler's reply and normalize the justification."""
    payload = llm.extract_json_object(content)
    if payload is None:
        raise AdjudicationError(
            f"labeler returned no JSON object for interaction {interaction_id}: "
            f"{content.strip()[:llm.MAX_REPLY_EXCERPT]!r}"
        )
    label = payload.get("label")
    if label not in closure.CLOSURE_LABELS:
        raise AdjudicationError(
            f"labeler returned invalid label {label!r} for interaction {interaction_id}"
        )
    justification = payload.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise AdjudicationError(
            f"labeler returned an empty justification for interaction {interaction_id}"
        )
    return label, " ".join(justification.split())

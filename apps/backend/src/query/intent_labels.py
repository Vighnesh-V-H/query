"""Label a deterministic dev slice of the RAG pool with final-taxonomy intents.

Ticket 11's final taxonomy (``docs/intent-taxonomy.md``) is the single source
of truth; ticket 13's TF-IDF baseline needs labeled examples to train on and
ticket 14's LLM classifier needs them to sanity-check against. This stage
produces those examples: it draws a deterministic slice of the RAG pool's
opening Customer Messages and labels each one with the configured ``labeler``
role, constrained to the final taxonomy's intent ids.

The slice is a rank, not a roll of the dice: every Interaction is ranked by a
SHA-256 of the seed and its interaction id (the same scheme as the
sample-and-split stage), and the top ``dev_size`` are labeled. The same seed
labels the same slice on any machine regardless of input-file order. Only the
RAG pool is ever labeled here: the holdout stays reserved for the Golden Set,
so a dev slice drawn from it would leak evaluation data into training.

Each label records its provenance. Fresh labels carry ``source: labeler`` with
the model's one-line justification and model id; labels corrected by hand
during spot-checking carry ``source: human`` with no model. The report counts
the label distribution per intent — including intents that receive zero
labels, so thin intents stay visible instead of silently vanishing — plus the
source split and the models used.

The labeler is non-deterministic, so reruns can move labels. Verdicts are
cached by prompt hash in a crash-safe append-only file, like closure
adjudication (ADR-0007) and discovery (ADR-0009): an interrupted run resumes
without paying for finished messages, and a taxonomy edit invalidates stale
verdicts because the prompt embeds the intent definitions. Delete the cache
file when changing the labeler model and fresh verdicts are wanted.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from query import cachefile, closure, llm, taxonomy
from query.interactions import Interaction

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_PATH = closure.DEFAULT_INPUT_PATH
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "intent-dev-labels.jsonl"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "intent-dev-report.json"
DEFAULT_REVIEW_PATH = REPO_ROOT / "docs" / "intent-dev-review.md"

DEFAULT_DEV_SIZE = 500
DEFAULT_SEED = 42
DEFAULT_REVIEW_EXAMPLES = 5

LABEL_SOURCES = ("labeler", "human")

SYSTEM_PROMPT = (
    "You are a precise annotation assistant for customer-support research. "
    "You classify the intent of one customer message and reply with raw JSON only."
)


class IntentLabelError(Exception):
    """Raised when the dev slice cannot be selected or labeled."""


@dataclass(frozen=True)
class IntentLabel:
    """One dev-slice Interaction's intent label with its provenance."""

    interaction_id: int
    customer_message: str
    intent: str
    source: Literal["labeler", "human"]
    justification: str
    model: str | None
    taxonomy_version: int


@dataclass(frozen=True)
class CachedIntentVerdict:
    """A labeler verdict stored for resuming an interrupted run."""

    interaction_id: int
    prompt_sha256: str
    intent: str
    justification: str
    model: str


@dataclass(frozen=True)
class IntentLabelReport:
    """Counts of the labeling run: the per-intent distribution and provenance."""

    total: int
    input_total: int
    requested: int
    seed: int
    taxonomy_version: int
    per_intent: dict[str, int]
    labeler: int
    human: int
    models: tuple[str, ...]


class IntentLabelCache:
    """Append-only JSONL cache of intent verdicts, safe under worker threads.

    Callers load the verdicts once before a run and let every completed label
    be recorded immediately, so a run killed mid-way resumes without paying
    for calls it already made.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._cache = cachefile.AppendOnlyCache(
            self.path,
            serialize=cached_verdict_to_json,
            parse=_parse_cache_line,
            key=lambda verdict: verdict.interaction_id,
            error_type=IntentLabelError,
        )

    def verdicts(self) -> dict[int, CachedIntentVerdict]:
        """Read the cached verdicts; a missing cache file is an empty cache."""
        return self._cache.entries()

    def record(self, verdict: CachedIntentVerdict) -> None:
        """Append one verdict, atomically with respect to worker threads."""
        self._cache.record(verdict)


def select_dev_slice(
    interactions: Sequence[Interaction],
    dev_size: int = DEFAULT_DEV_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[Interaction, ...]:
    """Pick the deterministic dev slice: the top ``dev_size`` by hash rank.

    Returns the picked Interactions ordered by interaction id, so output order
    is stable across machines. When the input holds fewer Interactions than
    requested, the slice covers everything instead of failing.
    """
    if dev_size < 1:
        raise IntentLabelError(f"dev size must be at least 1, got {dev_size}")
    if not interactions:
        raise IntentLabelError("no interactions to label")
    _require_unique_ids(interactions)
    ranked = sorted(
        interactions,
        key=lambda interaction: (
            _rank_key(seed, interaction.interaction_id),
            interaction.interaction_id,
        ),
    )
    picked = ranked[:dev_size]
    return tuple(sorted(picked, key=lambda interaction: interaction.interaction_id))


def label_dev_slice(
    interactions: Sequence[Interaction],
    final: taxonomy.FinalTaxonomy,
    dev_size: int = DEFAULT_DEV_SIZE,
    seed: int = DEFAULT_SEED,
    workers: int = 1,
    cache: IntentLabelCache | None = None,
    infer: Callable[[str], llm.LLMReply] | None = None,
) -> tuple[tuple[IntentLabel, ...], IntentLabelReport]:
    """Select the dev slice and label every opening Customer Message in it.

    Returns one label per selected Interaction, ordered by interaction id, and
    a report with the per-intent distribution. ``infer`` is the labeler call,
    kept injectable for tests; ``workers`` bounds concurrent labeler calls;
    ``cache`` reuses and records verdicts so interrupted runs can resume.
    """
    if workers < 1:
        raise IntentLabelError(f"workers must be at least 1, got {workers}")
    if not final.intents:
        raise IntentLabelError("final taxonomy has no intents to label against")
    dev = select_dev_slice(interactions, dev_size=dev_size, seed=seed)
    infer = infer if infer is not None else call_labeler
    saved = cache.verdicts() if cache is not None else {}
    record = cache.record if cache is not None else None
    intent_ids = final.intent_ids

    if workers == 1 or len(dev) <= 1:
        labels = tuple(
            _label_one(interaction, final, intent_ids, infer, saved, record)
            for interaction in dev
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _label_one, interaction, final, intent_ids, infer, saved, record
                )
                for interaction in dev
            ]
            try:
                labels = tuple(future.result() for future in futures)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    return labels, _report(
        labels, dev, interactions, final, dev_size, seed, intent_ids
    )


def call_labeler(prompt: str) -> llm.LLMReply:
    """Ask the configured labeler role to label one Customer Message."""
    return llm.call_llm(prompt, role="labeler", system=SYSTEM_PROMPT, temperature=0.0)


def build_prompt(
    customer_message: str, interaction_id: int, final: taxonomy.FinalTaxonomy
) -> str:
    """Render the taxonomy definitions and the message to classify."""
    definitions = "\n".join(
        f"- {intent.intent_id}: {intent.definition}" for intent in final.intents
    )
    return (
        "Classify the intent of one customer-support message.\n\n"
        f"Intents (the only valid labels):\n{definitions}\n\n"
        "Decide by what the message asks about, not by its tone: an angry "
        "message about a charge is still a billing issue, and a polite "
        "message that states no problem is not a support issue.\n"
        "Boundary rules:\n"
        "- any charge, receipt, or payment-method problem is billing_payment; "
        "a completed payment that did not unlock the plan is billing_payment;\n"
        "- choosing or changing a plan (Free vs Premium, trials and promos, "
        "Student and Duo plans, upgrades and downgrades, bundles, ads on the "
        "Free tier) is subscription_plans;\n"
        "- managing Family members after purchase (invites, address "
        "verification, members losing access) is family_plan;\n"
        "- a fault in the music stream or player is playback; the app or site "
        "failing anywhere else (crashes, freezes, loading and error screens, "
        "outages, update bugs) and getting Spotify onto or across devices is "
        "app_technical;\n"
        "- presale_codes is only for artist presale code requests and issues;\n"
        "- other is only for messages that state no issue and no request a "
        "support team would act on: praise, jokes, bare mentions, and "
        "support-channel chatter. A message that names a problem or asks for "
        "something is never other.\n\n"
        f"Message (interaction {interaction_id}):\n{customer_message}\n\n"
        "Reply with raw JSON only, no markdown and no prose:\n"
        '{"intent": "<intent id>", '
        '"justification": "one sentence naming the decisive evidence"}'
    )


def parse_label_reply(
    interaction_id: int, content: str, intent_ids: Collection[str]
) -> tuple[str, str]:
    """Validate the labeler's reply and normalize the justification."""
    payload = llm.extract_json_object(content)
    if payload is None:
        raise IntentLabelError(
            f"labeler returned no JSON object for interaction {interaction_id}: "
            f"{content.strip()[: llm.MAX_REPLY_EXCERPT]!r}"
        )
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent:
        raise IntentLabelError(
            f"labeler returned an invalid intent for interaction "
            f"{interaction_id}: {intent!r}"
        )
    if intent not in intent_ids:
        raise IntentLabelError(
            f"labeler returned an unknown intent for interaction "
            f"{interaction_id}: {intent!r}"
        )
    justification = payload.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise IntentLabelError(
            f"labeler returned an empty justification for interaction "
            f"{interaction_id}"
        )
    return intent, " ".join(justification.split())


def intent_label_error(
    label: IntentLabel, intent_ids: Collection[str] | None = None
) -> str | None:
    """Check a label against the contract the label reader enforces.

    Shared by :func:`label_dev_slice` and :func:`read_intent_labels_jsonl` so
    the stage cannot stage a label its own reader would reject. When
    ``intent_ids`` is given, the intent must be one of them; without the
    taxonomy only the intent's shape can be checked. A ``labeler`` label must
    carry the model that produced it; a ``human`` label must not, so the
    source always says who decided.
    """
    if type(label.interaction_id) is not int:
        return "invalid interaction_id"
    if not isinstance(label.customer_message, str) or not label.customer_message.strip():
        return "invalid customer_message"
    if not isinstance(label.intent, str) or not label.intent:
        return "invalid intent"
    if intent_ids is not None and label.intent not in intent_ids:
        return f"unknown intent {label.intent!r}"
    if label.source not in LABEL_SOURCES:
        return f"invalid source {label.source!r}"
    if not isinstance(label.justification, str) or not label.justification.strip():
        return "invalid justification"
    if type(label.taxonomy_version) is not int or label.taxonomy_version < 1:
        return "invalid taxonomy_version"
    if label.source == "labeler":
        if not isinstance(label.model, str) or not label.model:
            return "labeler labels need a model"
    elif label.model is not None:
        return "human labels carry no model"
    return None


def label_to_json(label: IntentLabel) -> dict:
    """Serialize an IntentLabel to a JSON-compatible mapping."""
    return {
        "interaction_id": label.interaction_id,
        "customer_message": label.customer_message,
        "intent": label.intent,
        "source": label.source,
        "justification": label.justification,
        "model": label.model,
        "taxonomy_version": label.taxonomy_version,
    }


def cached_verdict_to_json(verdict: CachedIntentVerdict) -> dict:
    """Serialize a CachedIntentVerdict to a JSON-compatible mapping."""
    return {
        "interaction_id": verdict.interaction_id,
        "prompt_sha256": verdict.prompt_sha256,
        "intent": verdict.intent,
        "justification": verdict.justification,
        "model": verdict.model,
    }


def stage_intent_labels_jsonl(
    labels: Sequence[IntentLabel], output_path: Path | str
) -> Path:
    """Write the labels to a temporary sibling, ready to be swapped in.

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
                handle.write(json.dumps(label_to_json(label)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_intent_labels_jsonl(
    input_path: Path | str = DEFAULT_LABELS_PATH,
    intent_ids: Collection[str] | None = None,
) -> tuple[IntentLabel, ...]:
    """Read the dev labels back from the JSON Lines format written above.

    Inverse of :func:`stage_intent_labels_jsonl`: parses every record and
    validates the shared label contract, plus unique interaction ids. When
    ``intent_ids`` is given, every intent must be one of them. Returns the
    labels in file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise IntentLabelError(f"intent labels JSONL does not exist: {input_path}")
    labels = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntentLabelError(
                    f"malformed JSON on {location}: {exc}"
                ) from exc
            label = _parse_label_record(decoded, location, intent_ids)
            if label.interaction_id in seen_ids:
                raise IntentLabelError(
                    f"duplicate interaction_id {label.interaction_id} on {location}"
                )
            seen_ids.add(label.interaction_id)
            labels.append(label)
    return tuple(labels)


def render_review_markdown(
    labels: Sequence[IntentLabel],
    report: IntentLabelReport,
    final: taxonomy.FinalTaxonomy,
    examples_per_intent: int = DEFAULT_REVIEW_EXAMPLES,
) -> str:
    """Render a human-reviewable Markdown of the labeled slice per intent."""
    lines = [
        "# Intent dev-label review",
        "",
        (
            "Generated by `query label-intents` (ticket 12) over "
            f"{report.total} RAG-pool Customer Messages against taxonomy v"
            f"{report.taxonomy_version}. Each message carries one labeler "
            "verdict with a one-line justification; the human spot-check in "
            "`docs/intent-dev-spot-check.md` audits a subsample of these "
            "labels."
        ),
        "",
        "## Label distribution",
        "",
        f"- messages: {report.total} "
        f"(labeler {report.labeler}, human {report.human})",
        f"- models: {', '.join(report.models) if report.models else 'none'}",
        "",
        "| Intent | Labels |",
        "|--------|-------:|",
    ]
    for intent in final.intents:
        lines.append(f"| `{intent.intent_id}` | {report.per_intent.get(intent.intent_id, 0)} |")
    lines.extend(["", "## Labels per intent"])
    by_intent: dict[str, list[IntentLabel]] = {
        intent.intent_id: [] for intent in final.intents
    }
    for label in labels:
        by_intent.setdefault(label.intent, []).append(label)
    for intent in final.intents:
        group = sorted(
            by_intent[intent.intent_id], key=lambda item: item.interaction_id
        )
        lines.extend(
            [
                "",
                f"### `{intent.intent_id}` ({len(group)} labels)",
                "",
                f"{intent.definition}",
                "",
            ]
        )
        if not group:
            lines.append("_No labels in this slice._")
            continue
        shown = group[:examples_per_intent]
        for label in shown:
            lines.append(
                f"- `[{label.interaction_id}]` {label.customer_message} "
                f"— {label.justification} (`{label.source}`)"
            )
        if len(group) > len(shown):
            lines.append(f"- _…and {len(group) - len(shown)} more._")
    lines.append("")
    return "\n".join(lines)


def stage_review_markdown(
    labels: Sequence[IntentLabel],
    report: IntentLabelReport,
    final: taxonomy.FinalTaxonomy,
    output_path: Path | str,
    examples_per_intent: int = DEFAULT_REVIEW_EXAMPLES,
) -> Path:
    """Write the review Markdown to a temporary sibling, ready to be swapped in."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                render_review_markdown(
                    labels, report, final, examples_per_intent=examples_per_intent
                )
            )
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def _label_one(
    interaction: Interaction,
    final: taxonomy.FinalTaxonomy,
    intent_ids: Collection[str],
    infer: Callable[[str], llm.LLMReply],
    saved: Mapping[int, CachedIntentVerdict],
    record: Callable[[CachedIntentVerdict], None] | None,
) -> IntentLabel:
    """Label one Interaction's opening message, reusing a matching cache entry.

    The labeler is retried once with a repair prompt when a reply is not valid
    JSON. Fresh verdicts are recorded immediately so an interrupted run can
    resume.
    """
    message = interaction.opening_message.text
    prompt = build_prompt(message, interaction.interaction_id, final)
    cache_key = llm.prompt_sha256(SYSTEM_PROMPT, prompt)
    cached = saved.get(interaction.interaction_id)
    if cached is not None and cached.prompt_sha256 == cache_key:
        label = IntentLabel(
            interaction_id=interaction.interaction_id,
            customer_message=message,
            intent=cached.intent,
            source="labeler",
            justification=cached.justification,
            model=cached.model,
            taxonomy_version=final.version,
        )
    else:
        reply = _call_labeler(infer, prompt, interaction.interaction_id)
        try:
            intent, justification = parse_label_reply(
                interaction.interaction_id, reply.content, intent_ids
            )
        except IntentLabelError:
            repair_prompt = llm.build_repair_prompt(prompt, reply.content)
            reply = _call_labeler(infer, repair_prompt, interaction.interaction_id)
            intent, justification = parse_label_reply(
                interaction.interaction_id, reply.content, intent_ids
            )
        label = IntentLabel(
            interaction_id=interaction.interaction_id,
            customer_message=message,
            intent=intent,
            source="labeler",
            justification=justification,
            model=reply.model,
            taxonomy_version=final.version,
        )
        if record is not None:
            record(
                CachedIntentVerdict(
                    interaction_id=label.interaction_id,
                    prompt_sha256=cache_key,
                    intent=intent,
                    justification=justification,
                    model=reply.model,
                )
            )
    error = intent_label_error(label, intent_ids)
    if error is not None:
        raise IntentLabelError(
            f"label for interaction {interaction.interaction_id} violates the "
            f"contract: {error}"
        )
    return label


def _report(
    labels: Sequence[IntentLabel],
    dev: Sequence[Interaction],
    interactions: Sequence[Interaction],
    final: taxonomy.FinalTaxonomy,
    requested: int,
    seed: int,
    intent_ids: Collection[str],
) -> IntentLabelReport:
    per_intent = {intent_id: 0 for intent_id in final.intent_ids}
    labeler = human = 0
    models: list[str] = []
    for label in labels:
        per_intent[label.intent] += 1
        if label.source == "labeler":
            labeler += 1
        else:
            human += 1
        if label.model and label.model not in models:
            models.append(label.model)
    for intent_id in intent_ids:
        per_intent.setdefault(intent_id, 0)
    return IntentLabelReport(
        total=len(dev),
        input_total=len(interactions),
        requested=requested,
        seed=seed,
        taxonomy_version=final.version,
        per_intent=per_intent,
        labeler=labeler,
        human=human,
        models=tuple(models),
    )


def _require_unique_ids(interactions: Sequence[Interaction]) -> None:
    seen: set[int] = set()
    for interaction in interactions:
        if interaction.interaction_id in seen:
            raise IntentLabelError(
                f"duplicate interaction_id {interaction.interaction_id} in the input"
            )
        seen.add(interaction.interaction_id)


def _rank_key(seed: int, interaction_id: int) -> bytes:
    """Stable ranking key: the same seed and id rank the same everywhere."""
    return hashlib.sha256(f"{seed}:{interaction_id}".encode()).digest()


def _call_labeler(
    infer: Callable[[str], llm.LLMReply], prompt: str, interaction_id: int
) -> llm.LLMReply:
    try:
        return infer(prompt)
    except Exception as exc:
        raise IntentLabelError(
            f"labeler call failed for interaction {interaction_id}: {exc}"
        ) from exc


def _parse_label_record(
    record: object, location: str, intent_ids: Collection[str] | None
) -> IntentLabel:
    if not isinstance(record, dict):
        raise IntentLabelError(
            f"malformed intent label on {location}: expected an object"
        )
    label = IntentLabel(
        interaction_id=record.get("interaction_id"),
        customer_message=record.get("customer_message"),
        intent=record.get("intent"),
        source=record.get("source"),
        justification=record.get("justification"),
        model=record.get("model"),
        taxonomy_version=record.get("taxonomy_version"),
    )
    error = intent_label_error(label, intent_ids)
    if error is not None:
        raise IntentLabelError(f"malformed intent label on {location}: {error}")
    return label


def _parse_cache_line(line: str, location: str) -> CachedIntentVerdict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise IntentLabelError(
            f"malformed JSON in cache on {location}: {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise IntentLabelError(
            f"malformed cache record on {location}: expected an object"
        )
    interaction_id = record.get("interaction_id")
    if type(interaction_id) is not int:
        raise IntentLabelError(
            f"malformed cache record on {location}: invalid interaction_id"
        )
    prompt_sha256 = record.get("prompt_sha256")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise IntentLabelError(
            f"malformed cache record on {location}: invalid prompt_sha256"
        )
    intent = record.get("intent")
    if not isinstance(intent, str) or not intent:
        raise IntentLabelError(
            f"malformed cache record on {location}: invalid intent"
        )
    justification = record.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise IntentLabelError(
            f"malformed cache record on {location}: invalid justification"
        )
    model = record.get("model")
    if not isinstance(model, str) or not model:
        raise IntentLabelError(f"malformed cache record on {location}: invalid model")
    return CachedIntentVerdict(
        interaction_id=interaction_id,
        prompt_sha256=prompt_sha256,
        intent=intent,
        justification=justification,
        model=model,
    )

"""Build the hand-labelled Golden Set from the reserved holdout.

Ticket 18 needs a labelling CLI for the 150-250 example Golden Set of
`docs/specs-v0.md` section 8: stratified holdout Interactions served as a full
transcript, hand-labelled for gold intent and the gold auto/escalate decision,
with per-intent floors and an auto/escalate balance target (decision 13 in
``docs/decisions.md``). The set is drawn only from the holdout (decision 6), so
a golden example can never be retrieved as evidence for itself.

The holdout carries no intent labels, and the Golden Set's floors are per
intent, so sampling needs hints. This module asks the configured ``labeler``
role for one intent and one coarse auto/escalate prediction per holdout
message, cached by prompt hash in a crash-safe append-only file like closure
adjudication and intent discovery (ADR-0007, ADR-0009). Hints only choose
*which* Interactions get served for human labelling: they are never shown to
the labeler and never become gold, and the grading rubric is not the evaluated
classifier of ticket 14 — it is a sampling aid (ADR-0011).

Sampling turns those hints into a queue. Intents are represented
proportionally with a per-intent floor, and each intent's quota is split
between predicted-auto and predicted-escalate candidates so the queue lands as
close to the configured auto/escalate share as the hinted capacity allows. The
report states the requested and predicted balance and whether the target was
reachable, so an infeasible target is visible rather than silently missed,
matching the honest-reporting style of the sample/split stage.

The labeler serves the queue in a terminal session: the whole Interaction is
shown, not just the opening message, and the human enters a gold intent, a
gold decision, and optional notes. Each label is written to the Golden Set
file immediately, in queue order, so a paused or interrupted session resumes
by skipping the ids already labelled. Re-running the session is therefore
safe, and the file is always a complete snapshot of the session so far.
"""

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import openai

from query import cachefile, llm, taxonomy
from query.interactions import Interaction

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HOLDOUT_PATH = REPO_ROOT / "data" / "holdout.jsonl"
DEFAULT_HINTS_PATH = REPO_ROOT / "data" / "holdout-hints.jsonl"
DEFAULT_QUEUE_PATH = REPO_ROOT / "data" / "golden-queue.jsonl"
DEFAULT_GOLDEN_PATH = REPO_ROOT / "data" / "golden-set.jsonl"
DEFAULT_SAMPLING_REPORT_PATH = REPO_ROOT / "data" / "golden-sampling-report.json"
DEFAULT_LABELING_REPORT_PATH = REPO_ROOT / "data" / "golden-report.json"

GOLDEN_SET_VERSION = 1
MIN_GOLDEN_SIZE = 150
MAX_GOLDEN_SIZE = 250
DEFAULT_TARGET_SIZE = 200
DEFAULT_FLOOR = 10
DEFAULT_AUTO_SHARE = 0.6
DEFAULT_SEED = 42
HINT_ATTEMPTS = 3
CALL_ATTEMPTS = 5
CALL_BACKOFF_SECONDS = 2.0
TRANSIENT_STATUS_CODES = (408, 409, 425, 429, 500, 502, 503, 504)

AUTO = "auto"
ESCALATE = "escalate"
DECISIONS = (AUTO, ESCALATE)
Decision = Literal["auto", "escalate"]

SYSTEM_PROMPT = (
    "You are a precise support-triage analyst for customer-support research. "
    "You label one customer message with its support intent and whether a "
    "support agent could handle it safely without account access, and reply "
    "with raw JSON only."
)

# The configured labeler is a reasoning model; a hint needs no chain of
# thought, and disabling it cuts each call from ~1,000 completion tokens to
# ~50. This is a provider-specific knob (NVIDIA NIM / HuggingFace chat
# templates), which is why it only applies to this stage's call.
HINT_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

DECISION_HELP = (
    "Decision definitions:\n"
    "  auto: the agent could answer safely from public help content, general "
    "brand behavior, and historical replies, without account access or "
    "personal data.\n"
    "  escalate: the issue needs account-specific investigation or personal "
    "data, or falls in a high-risk category: security/credentials, payment "
    "disputes, account deletion/data requests, abuse or legal threats."
)

_SKIP = object()
_QUIT = object()


class GoldenError(Exception):
    """Raised when the Golden Set cannot be sampled, labelled, or read."""


@dataclass(frozen=True)
class IntentHint:
    """One labeler prediction that chooses how the holdout is stratified.

    ``prompt_sha256`` is ``None`` for hints produced outside this stage; those
    are trusted as given and never refreshed. Hints written here carry the hash
    so a taxonomy or prompt change invalidates them.
    """

    interaction_id: int
    intent: str
    decision: Decision
    reason: str
    model: str
    prompt_sha256: str | None


@dataclass(frozen=True)
class QueueItem:
    """One Interaction selected for hand-labelling, with its sampling stratum."""

    interaction_id: int
    intent: str
    decision: Decision


@dataclass(frozen=True)
class IntentAllocation:
    """What one intent contributed to the sample and what was available."""

    available: int
    available_auto: int
    available_escalate: int
    selected: int
    selected_auto: int
    selected_escalate: int


@dataclass(frozen=True)
class SamplingReport:
    """Counts and balance of the stratified Golden Set queue."""

    seed: int
    hinted: int
    target_size: int
    floor: int
    auto_share_target: float
    selected: int
    selected_auto: int
    selected_escalate: int
    per_intent: dict[str, IntentAllocation]
    balance_target_met: bool

    @property
    def predicted_auto_share(self) -> float:
        return self.selected_auto / self.selected if self.selected else 0.0


@dataclass(frozen=True)
class GoldenExample:
    """One hand-labelled Golden Set example."""

    interaction_id: int
    taxonomy_version: int
    customer_message: str
    gold_intent: str
    gold_decision: Decision
    notes: str
    hint_intent: str
    hint_decision: Decision


@dataclass(frozen=True)
class IntentLabelCounts:
    """Gold labels of one intent, split by decision."""

    total: int
    auto: int
    escalate: int


@dataclass(frozen=True)
class GoldenLabelReport:
    """Distribution of a labelling session over the Golden Set."""

    golden_set_version: int
    taxonomy_version: int
    queue_total: int
    labeled: int
    skipped: int
    remaining: int
    notes_filled: int
    by_intent: dict[str, IntentLabelCounts]
    by_decision: dict[str, int]
    hint_intent_agreement: int
    hint_decision_agreement: int

    @property
    def complete(self) -> bool:
        return self.remaining == 0

    @property
    def auto_share(self) -> float:
        return self.by_decision.get(AUTO, 0) / self.labeled if self.labeled else 0.0

    @property
    def notes_share(self) -> float:
        return self.notes_filled / self.labeled if self.labeled else 0.0

    @property
    def hint_intent_agreement_share(self) -> float:
        return (
            self.hint_intent_agreement / self.labeled if self.labeled else 0.0
        )

    @property
    def hint_decision_agreement_share(self) -> float:
        return (
            self.hint_decision_agreement / self.labeled if self.labeled else 0.0
        )


class HintCache:
    """Append-only JSONL cache of labeler hints, safe under worker threads."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._cache = cachefile.AppendOnlyCache(
            self.path,
            serialize=hint_to_json,
            parse=_parse_hint_record,
            key=lambda hint: hint.interaction_id,
            error_type=GoldenError,
        )

    def hints(self) -> dict[int, IntentHint]:
        """Read the cached hints; a missing cache file is an empty cache."""
        return self._cache.entries()

    def record(self, hint: IntentHint) -> None:
        """Append one hint, atomically with respect to worker threads."""
        self._cache.record(hint)


def call_labeler(prompt: str) -> llm.LLMReply:
    """Ask the configured labeler role for one holdout hint."""
    return llm.call_llm(
        prompt,
        role="labeler",
        system=SYSTEM_PROMPT,
        temperature=0.0,
        extra_body=HINT_EXTRA_BODY,
    )


def build_hint_prompt(message: str, final: taxonomy.FinalTaxonomy) -> str:
    """Render the taxonomy, its examples, and one Customer Message."""
    entries = []
    for intent in final.intents:
        examples = " / ".join(f'"{example}"' for example in intent.examples)
        entries.append(
            f"- {intent.intent_id}: {intent.definition}\n  examples: {examples}"
        )
    definitions = "\n".join(entries)
    return (
        "Reply with one JSON object and nothing else: no analysis, no "
        "thinking, no markdown, no prose.\n\n"
        "Label one opening customer message from SpotifyCares support. "
        "The customer message is the only thing the support agent sees.\n\n"
        f"Intents (the only valid values):\n{definitions}\n\n"
        f"Message:\n{message}\n\n"
        "Decide:\n"
        '- "intent": the intent above the message best fits. Use "other" only '
        "for praise, jokes, bare mentions, support-channel chatter, and other "
        "messages that state no support issue at all.\n"
        '- "decision": "auto" when a support agent could reply safely from '
        "public help content, general brand behavior, and historical replies "
        "alone, without account access or personal data; \"escalate\" when the "
        "issue needs account-specific investigation, personal data, or falls "
        "in a high-risk category - security or credentials, payment disputes, "
        "account deletion or data requests, and abuse or legal threats always "
        "escalate. A question answerable from public documentation (how a "
        "feature works, a known playback error) is auto even when the customer "
        "is frustrated.\n"
        '- "reason": one sentence naming the decisive evidence.\n\n'
        "The whole reply must be exactly this JSON object:\n"
        '{"intent": "<intent id>", "decision": "auto" | "escalate", '
        '"reason": "one sentence"}'
    )


def build_retry_prompt(prompt: str, previous_reply: str) -> str:
    """Ask again, spelling out what the rejected reply got wrong."""
    excerpt = previous_reply.strip()[: llm.MAX_REPLY_EXCERPT]
    return (
        f"{prompt}\n\nYour previous reply was rejected because it was not a "
        f"single JSON object:\n{excerpt}\n\nRespond with exactly one JSON "
        "object and nothing else. Do not include analysis, thinking, markdown, "
        "or prose before or after the JSON."
    )


def classify_hints(
    interactions: Sequence[Interaction],
    final: taxonomy.FinalTaxonomy,
    workers: int = 1,
    cache: HintCache | None = None,
    infer: Callable[[str], llm.LLMReply] | None = None,
) -> tuple[tuple[IntentHint, ...], tuple[str, ...]]:
    """Hint every Interaction's opening message, reusing cached verdicts.

    Returns the hints in input order plus the distinct models that produced
    them. ``infer`` is the labeler call, kept injectable for tests; ``workers``
    bounds concurrent calls; ``cache`` reuses matching verdicts and records
    fresh ones so an interrupted run resumes without paying twice.
    """
    _require_hint_arguments(interactions, workers)
    infer = infer if infer is not None else call_labeler
    saved = cache.hints() if cache is not None else {}
    intent_ids = final.intent_ids

    def hint_one(interaction: Interaction) -> IntentHint:
        prompt = build_hint_prompt(interaction.opening_message.text, final)
        cache_key = llm.prompt_sha256(SYSTEM_PROMPT, prompt)
        cached = saved.get(interaction.interaction_id)
        if cached is not None and _hint_is_current(cached, cache_key, intent_ids):
            return cached
        reply = _call_labeler(infer, prompt, interaction.interaction_id)
        attempts = 1
        while True:
            try:
                intent, decision, reason = parse_hint_reply(
                    interaction.interaction_id, reply.content, intent_ids
                )
                break
            except GoldenError:
                if attempts >= HINT_ATTEMPTS:
                    raise
                attempts += 1
                reply = _call_labeler(
                    infer,
                    build_retry_prompt(prompt, reply.content),
                    interaction.interaction_id,
                )
        hint = IntentHint(
            interaction_id=interaction.interaction_id,
            intent=intent,
            decision=decision,
            reason=reason,
            model=reply.model,
            prompt_sha256=cache_key,
        )
        if cache is not None:
            cache.record(hint)
        return hint

    if workers == 1 or len(interactions) <= 1:
        hints = tuple(hint_one(interaction) for interaction in interactions)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(hint_one, interaction) for interaction in interactions
            ]
            try:
                hints = tuple(future.result() for future in futures)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    models: list[str] = []
    for hint in hints:
        if hint.model and hint.model not in models:
            models.append(hint.model)
    return hints, tuple(models)


def parse_hint_reply(
    interaction_id: int, content: str, intent_ids: Sequence[str]
) -> tuple[str, Decision, str]:
    """Validate the labeler's hint reply and normalize the reason."""
    payload = llm.extract_json_object(content)
    if payload is None:
        raise GoldenError(
            f"labeler returned no JSON object for interaction {interaction_id}: "
            f"{content.strip()[: llm.MAX_REPLY_EXCERPT]!r}"
        )
    intent = payload.get("intent")
    if not isinstance(intent, str) or intent not in intent_ids:
        raise GoldenError(
            f"labeler returned an invalid intent for interaction {interaction_id}: "
            f"{intent!r}"
        )
    decision = payload.get("decision")
    if decision not in DECISIONS:
        raise GoldenError(
            f"labeler returned an invalid decision for interaction "
            f"{interaction_id}: {decision!r}"
        )
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise GoldenError(
            f"labeler returned an empty reason for interaction {interaction_id}"
        )
    return intent, decision, " ".join(reason.split())


def plan_queue(
    hints: Sequence[IntentHint],
    intent_ids: Sequence[str],
    target_size: int = DEFAULT_TARGET_SIZE,
    floor: int = DEFAULT_FLOOR,
    auto_share: float = DEFAULT_AUTO_SHARE,
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[QueueItem, ...], SamplingReport]:
    """Stratify hints into a labelling queue and report the allocation.

    Every intent is represented proportionally to its hinted volume, never
    below ``floor`` and never above what exists; when the floors alone exceed
    ``target_size`` they win and the queue is larger than requested. Within
    each intent's quota, candidates are assigned to predicted-auto and
    predicted-escalate slots so the queue lands on ``auto_share`` when the
    hinted capacity allows, and as close as it can when it does not.
    """
    _require_plan_arguments(hints, intent_ids, target_size, floor, auto_share)
    ordered_ids = tuple(intent_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise GoldenError("intent ids must be unique")
    by_intent: dict[str, list[IntentHint]] = {intent_id: [] for intent_id in ordered_ids}
    seen: set[int] = set()
    for hint in hints:
        if hint.interaction_id in seen:
            raise GoldenError(f"duplicate hint for interaction {hint.interaction_id}")
        seen.add(hint.interaction_id)
        if hint.intent not in by_intent:
            raise GoldenError(
                f"hint for interaction {hint.interaction_id} has unknown intent "
                f"{hint.intent!r}"
            )
        if hint.decision not in DECISIONS:
            raise GoldenError(
                f"hint for interaction {hint.interaction_id} has invalid decision "
                f"{hint.decision!r}"
            )
        by_intent[hint.intent].append(hint)
    available = {intent_id: len(by_intent[intent_id]) for intent_id in ordered_ids}
    quotas = _quotas(available, target_size, floor, ordered_ids)
    ranked = {
        intent_id: sorted(
            by_intent[intent_id],
            key=lambda hint: (_rank_key(seed, hint.interaction_id), hint.interaction_id),
        )
        for intent_id in ordered_ids
    }
    auto_queues = {
        intent_id: [hint for hint in ranked[intent_id] if hint.decision == AUTO]
        for intent_id in ordered_ids
    }
    escalate_queues = {
        intent_id: [hint for hint in ranked[intent_id] if hint.decision == ESCALATE]
        for intent_id in ordered_ids
    }
    selected_total = sum(quotas.values())
    auto_picks, target_met = _split_decisions(
        quotas, auto_queues, escalate_queues, selected_total, auto_share, ordered_ids
    )
    chosen = {
        intent_id: sorted(
            auto_queues[intent_id][: auto_picks[intent_id]]
            + escalate_queues[intent_id][: quotas[intent_id] - auto_picks[intent_id]],
            key=lambda hint: (
                _rank_key(seed, hint.interaction_id),
                hint.interaction_id,
            ),
        )
        for intent_id in ordered_ids
    }
    items = tuple(
        QueueItem(
            interaction_id=hint.interaction_id,
            intent=hint.intent,
            decision=hint.decision,
        )
        for hint in _interleave(chosen, ordered_ids)
    )
    per_intent = {
        intent_id: IntentAllocation(
            available=available[intent_id],
            available_auto=len(auto_queues[intent_id]),
            available_escalate=len(escalate_queues[intent_id]),
            selected=quotas[intent_id],
            selected_auto=auto_picks[intent_id],
            selected_escalate=quotas[intent_id] - auto_picks[intent_id],
        )
        for intent_id in ordered_ids
    }
    report = SamplingReport(
        seed=seed,
        hinted=len(hints),
        target_size=target_size,
        floor=floor,
        auto_share_target=auto_share,
        selected=selected_total,
        selected_auto=sum(auto_picks.values()),
        selected_escalate=selected_total - sum(auto_picks.values()),
        per_intent=per_intent,
        balance_target_met=target_met,
    )
    return items, report


def hint_to_json(hint: IntentHint) -> dict:
    """Serialize an IntentHint to a JSON-compatible mapping."""
    payload = {
        "interaction_id": hint.interaction_id,
        "intent": hint.intent,
        "decision": hint.decision,
        "reason": hint.reason,
        "model": hint.model,
    }
    if hint.prompt_sha256 is not None:
        payload["prompt_sha256"] = hint.prompt_sha256
    return payload


def queue_to_json(item: QueueItem) -> dict:
    """Serialize a QueueItem to a JSON-compatible mapping."""
    return {
        "interaction_id": item.interaction_id,
        "hint_intent": item.intent,
        "hint_decision": item.decision,
    }


def golden_example_to_json(example: GoldenExample) -> dict:
    """Serialize a GoldenExample to a JSON-compatible mapping."""
    return {
        "interaction_id": example.interaction_id,
        "taxonomy_version": example.taxonomy_version,
        "customer_message": example.customer_message,
        "gold_intent": example.gold_intent,
        "gold_decision": example.gold_decision,
        "notes": example.notes,
        "hint_intent": example.hint_intent,
        "hint_decision": example.hint_decision,
    }


def stage_queue_jsonl(items: Sequence[QueueItem], output_path: Path | str) -> Path:
    """Write the queue to a temporary sibling, ready to be swapped in."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(queue_to_json(item)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def write_golden_jsonl(
    examples: Sequence[GoldenExample], output_path: Path | str
) -> Path:
    """Write the Golden Set, swapping the file in atomically."""
    output_path = Path(output_path)
    temporary_path = stage_golden_jsonl(examples, output_path)
    try:
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
    return output_path


def stage_golden_jsonl(
    examples: Sequence[GoldenExample], output_path: Path | str
) -> Path:
    """Write the Golden Set to a temporary sibling, ready to be swapped in."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(golden_example_to_json(example)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_queue_jsonl(
    input_path: Path | str = DEFAULT_QUEUE_PATH,
    intent_ids: Sequence[str] | None = None,
) -> tuple[QueueItem, ...]:
    """Read the queue artifact back from the JSON Lines format written above.

    Parses every record, validates its contract plus unique interaction ids,
    and, when ``intent_ids`` is given, rejects a hint intent outside them.
    Returns the items in file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise GoldenError(f"golden queue JSONL does not exist: {input_path}")
    items = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            record = _decode_record(line, location)
            item = _parse_queue_record(record, location, intent_ids)
            if item.interaction_id in seen_ids:
                raise GoldenError(
                    f"duplicate interaction_id {item.interaction_id} on {location}"
                )
            seen_ids.add(item.interaction_id)
            items.append(item)
    return tuple(items)


def read_golden_jsonl(
    input_path: Path | str = DEFAULT_GOLDEN_PATH,
    intent_ids: Sequence[str] | None = None,
) -> tuple[GoldenExample, ...]:
    """Read the Golden Set back from the JSON Lines format written above.

    Parses every record, validates its contract plus unique interaction ids,
    and, when ``intent_ids`` is given, rejects a gold or hint intent outside
    them. Returns the examples in file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise GoldenError(f"Golden Set JSONL does not exist: {input_path}")
    examples = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            record = _decode_record(line, location)
            example = _parse_golden_record(record, location, intent_ids)
            if example.interaction_id in seen_ids:
                raise GoldenError(
                    f"duplicate interaction_id {example.interaction_id} on {location}"
                )
            seen_ids.add(example.interaction_id)
            examples.append(example)
    return tuple(examples)


def has_customer_message(interaction: Interaction) -> bool:
    """Whether the opening message is non-blank and therefore labelable."""
    return bool(interaction.opening_message.text.strip())


def label_golden(
    queue: Sequence[QueueItem],
    interactions: Sequence[Interaction],
    final: taxonomy.FinalTaxonomy,
    output_path: Path | str,
    ask: Callable[[str], str] | None = None,
    tell: Callable[[str], None] | None = None,
) -> tuple[tuple[GoldenExample, ...], GoldenLabelReport]:
    """Serve the queue for hand-labelling and write the Golden Set.

    The whole Interaction is shown for every queued item. Existing labels in
    ``output_path`` are resumed: already-labelled ids are skipped, so a paused
    session continues where it stopped. Each label is swapped into the file
    before the next one, so the file is always a complete session snapshot.
    """
    ask = ask if ask is not None else input
    tell = tell if tell is not None else print
    _require_label_arguments(queue, interactions, final.intent_ids)
    interactions_by_id = {
        interaction.interaction_id: interaction for interaction in interactions
    }
    queue_ids = {item.interaction_id for item in queue}
    existing = (
        read_golden_jsonl(output_path, final.intent_ids)
        if Path(output_path).is_file()
        else ()
    )
    by_id: dict[int, GoldenExample] = {}
    for example in existing:
        if example.interaction_id not in queue_ids:
            raise GoldenError(
                f"Golden Set contains interaction {example.interaction_id} that is "
                "not in the queue; the file and queue do not belong together"
            )
        if example.taxonomy_version != final.version:
            raise GoldenError(
                f"Golden Set was labelled against taxonomy v{example.taxonomy_version} "
                f"but the current taxonomy is v{final.version}"
            )
        by_id[example.interaction_id] = example
    tell(
        f"Golden Set labeling: {len(queue)} queued, {len(by_id)} already labeled, "
        f"{len(queue) - len(by_id)} to go."
    )
    tell(DECISION_HELP)
    tell("Commands: ? show the intent menu, s skip, q save and pause.")
    labeled = len(by_id)
    skipped = 0
    for item in queue:
        if item.interaction_id in by_id:
            continue
        interaction = interactions_by_id[item.interaction_id]
        if not has_customer_message(interaction):
            skipped += 1
            tell(
                f"warning: interaction {item.interaction_id} has a blank opening "
                "message; skipped."
            )
            continue
        tell("")
        tell(render_transcript(interaction))
        try:
            answer = _prompt_example(final, ask, tell)
        except (EOFError, KeyboardInterrupt):
            tell("")
            tell("session paused; labels so far are saved.")
            break
        if answer is _QUIT:
            break
        if answer is _SKIP:
            skipped += 1
            tell("skipped.")
            continue
        gold_intent, gold_decision, notes = answer
        by_id[item.interaction_id] = GoldenExample(
            interaction_id=item.interaction_id,
            taxonomy_version=final.version,
            customer_message=interaction.opening_message.text,
            gold_intent=gold_intent,
            gold_decision=gold_decision,
            notes=notes,
            hint_intent=item.intent,
            hint_decision=item.decision,
        )
        labeled += 1
        write_golden_jsonl(_ordered_examples(queue, by_id), output_path)
        tell(f"labeled {labeled}/{len(queue)}")
    examples = _ordered_examples(queue, by_id)
    report = _label_report(queue, examples, final, skipped)
    tell(
        f"session done: {report.labeled}/{report.queue_total} labeled "
        f"({report.by_decision[AUTO]} auto, {report.by_decision[ESCALATE]} "
        f"escalate), {report.remaining} remaining."
    )
    return examples, report


def render_transcript(interaction: Interaction) -> str:
    """Render the whole Interaction, so the labeler sees the full transcript."""
    lines = [
        f"Interaction {interaction.interaction_id} ({len(interaction.turns)} turns):"
    ]
    for turn in interaction.turns:
        stamp = turn.created_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{turn.side:>8} {stamp}] {turn.text}")
    return "\n".join(lines)


def render_taxonomy_menu(final: taxonomy.FinalTaxonomy) -> str:
    """Render the final taxonomy as a numbered menu for the labeler."""
    lines = ["Intents (enter the number or the id):"]
    lines.extend(
        f"{number:2}. {intent.intent_id}: {intent.definition}"
        for number, intent in enumerate(final.intents, start=1)
    )
    return "\n".join(lines)


def _require_hint_arguments(
    interactions: Sequence[Interaction], workers: int
) -> None:
    if not interactions:
        raise GoldenError("no interactions to hint")
    if workers < 1:
        raise GoldenError(f"workers must be at least 1, got {workers}")
    seen: set[int] = set()
    for interaction in interactions:
        if interaction.interaction_id in seen:
            raise GoldenError(
                f"duplicate interaction_id {interaction.interaction_id} in the input"
            )
        seen.add(interaction.interaction_id)


def _require_plan_arguments(
    hints: Sequence[IntentHint],
    intent_ids: Sequence[str],
    target_size: int,
    floor: int,
    auto_share: float,
) -> None:
    if not hints:
        raise GoldenError("no hints to sample from")
    if not intent_ids:
        raise GoldenError("the taxonomy has no intents")
    if target_size < 1:
        raise GoldenError(f"target size must be at least 1, got {target_size}")
    if floor < 0:
        raise GoldenError(f"floor must be at least 0, got {floor}")
    if not 0.0 <= auto_share <= 1.0:
        raise GoldenError(f"auto share must be between 0 and 1, got {auto_share}")


def _require_label_arguments(
    queue: Sequence[QueueItem],
    interactions: Sequence[Interaction],
    intent_ids: Sequence[str],
) -> None:
    if not queue:
        raise GoldenError("no queued interactions to label")
    seen_interactions: set[int] = set()
    for interaction in interactions:
        if interaction.interaction_id in seen_interactions:
            raise GoldenError(
                f"duplicate interaction_id {interaction.interaction_id} in the input"
            )
        seen_interactions.add(interaction.interaction_id)
    seen_queue: set[int] = set()
    for item in queue:
        if item.interaction_id in seen_queue:
            raise GoldenError(
                f"duplicate interaction_id {item.interaction_id} in the queue"
            )
        seen_queue.add(item.interaction_id)
        if item.interaction_id not in seen_interactions:
            raise GoldenError(
                f"queued interaction {item.interaction_id} is not in the input"
            )
        if item.intent not in intent_ids:
            raise GoldenError(
                f"queued interaction {item.interaction_id} has unknown intent "
                f"{item.intent!r}"
            )
        if item.decision not in DECISIONS:
            raise GoldenError(
                f"queued interaction {item.interaction_id} has invalid decision "
                f"{item.decision!r}"
            )


def _hint_is_current(
    hint: IntentHint, cache_key: str, intent_ids: Sequence[str]
) -> bool:
    if hint.intent not in intent_ids or hint.decision not in DECISIONS:
        return False
    return hint.prompt_sha256 is None or hint.prompt_sha256 == cache_key


def _quotas(
    available: dict[str, int],
    target: int,
    floor: int,
    intent_ids: Sequence[str],
) -> dict[str, int]:
    """Apportion ``target`` seats proportionally, bounded by floor and supply."""
    total = sum(available.values())
    if total == 0:
        return {intent_id: 0 for intent_id in intent_ids}
    target = min(target, total)
    quotas = {
        intent_id: min(
            available[intent_id],
            max(floor, target * available[intent_id] // total),
        )
        for intent_id in intent_ids
    }
    while sum(quotas.values()) < target:
        addable = [
            intent_id
            for intent_id in intent_ids
            if quotas[intent_id] < available[intent_id]
        ]
        if not addable:
            break
        chosen = max(
            addable,
            key=lambda intent_id: (
                target * available[intent_id] / total - quotas[intent_id],
                -intent_ids.index(intent_id),
            ),
        )
        quotas[chosen] += 1
    while sum(quotas.values()) > target:
        removable = [
            intent_id
            for intent_id in intent_ids
            if quotas[intent_id] > min(floor, available[intent_id])
        ]
        if not removable:
            break
        chosen = min(
            removable,
            key=lambda intent_id: (
                target * available[intent_id] / total - quotas[intent_id],
                intent_ids.index(intent_id),
            ),
        )
        quotas[chosen] -= 1
    return quotas


def _split_decisions(
    quotas: dict[str, int],
    auto_queues: dict[str, list[IntentHint]],
    escalate_queues: dict[str, list[IntentHint]],
    selected_total: int,
    auto_share: float,
    intent_ids: Sequence[str],
) -> tuple[dict[str, int], bool]:
    """Assign each intent's quota to predicted-auto and predicted-escalate slots.

    Returns the auto picks per intent and whether the requested share was
    reachable given the hinted capacity. Every intent must fill its quota, so
    its auto count is bounded below by ``quota - escalate capacity`` and above
    by its auto capacity; the target is clamped into the feasible range and the
    remaining auto slots are handed out in document order.
    """
    requested = _round_half_up(selected_total * auto_share)
    lower: dict[str, int] = {}
    upper: dict[str, int] = {}
    for intent_id in intent_ids:
        quota = quotas[intent_id]
        auto_capacity = min(len(auto_queues[intent_id]), quota)
        escalate_capacity = min(len(escalate_queues[intent_id]), quota)
        lower[intent_id] = quota - escalate_capacity
        upper[intent_id] = min(auto_capacity, quota)
    feasible_target = min(max(requested, sum(lower.values())), sum(upper.values()))
    picks = dict(lower)
    remaining = feasible_target - sum(lower.values())
    for intent_id in intent_ids:
        room = upper[intent_id] - picks[intent_id]
        added = min(room, remaining)
        picks[intent_id] += added
        remaining -= added
    return picks, feasible_target == requested


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def _interleave(
    chosen: dict[str, list[IntentHint]], intent_ids: Sequence[str]
) -> list[IntentHint]:
    """Round-robin the per-intent selections so the queue mixes strata."""
    ordered: list[IntentHint] = []
    position = 0
    while True:
        added = False
        for intent_id in intent_ids:
            if position < len(chosen[intent_id]):
                ordered.append(chosen[intent_id][position])
                added = True
        if not added:
            return ordered
        position += 1


def _ordered_examples(
    queue: Sequence[QueueItem], by_id: dict[int, GoldenExample]
) -> tuple[GoldenExample, ...]:
    return tuple(
        by_id[item.interaction_id]
        for item in queue
        if item.interaction_id in by_id
    )


def _label_report(
    queue: Sequence[QueueItem],
    examples: Sequence[GoldenExample],
    final: taxonomy.FinalTaxonomy,
    skipped: int,
) -> GoldenLabelReport:
    by_intent = {
        intent_id: IntentLabelCounts(total=0, auto=0, escalate=0)
        for intent_id in final.intent_ids
    }
    by_decision = {AUTO: 0, ESCALATE: 0}
    notes_filled = 0
    hint_intent_agreement = 0
    hint_decision_agreement = 0
    for example in examples:
        counts = by_intent[example.gold_intent]
        by_intent[example.gold_intent] = IntentLabelCounts(
            total=counts.total + 1,
            auto=counts.auto + (1 if example.gold_decision == AUTO else 0),
            escalate=counts.escalate + (1 if example.gold_decision == ESCALATE else 0),
        )
        by_decision[example.gold_decision] += 1
        if example.notes.strip():
            notes_filled += 1
        if example.hint_intent == example.gold_intent:
            hint_intent_agreement += 1
        if example.hint_decision == example.gold_decision:
            hint_decision_agreement += 1
    labeled = len(examples)
    return GoldenLabelReport(
        golden_set_version=GOLDEN_SET_VERSION,
        taxonomy_version=final.version,
        queue_total=len(queue),
        labeled=labeled,
        skipped=skipped,
        remaining=len(queue) - labeled,
        notes_filled=notes_filled,
        by_intent=by_intent,
        by_decision=by_decision,
        hint_intent_agreement=hint_intent_agreement,
        hint_decision_agreement=hint_decision_agreement,
    )


def _prompt_example(
    final: taxonomy.FinalTaxonomy,
    ask: Callable[[str], str],
    tell: Callable[[str], None],
) -> tuple[str, str, str] | object:
    """Ask for one gold intent, decision, and notes, or a skip/quit command."""
    by_number = {
        str(number): intent.intent_id
        for number, intent in enumerate(final.intents, start=1)
    }
    by_id = {intent.intent_id: intent.intent_id for intent in final.intents}
    while True:
        answer = ask("intent> ").strip()
        if answer == "?":
            tell(render_taxonomy_menu(final))
            continue
        if answer in ("s", "skip"):
            return _SKIP
        if answer in ("q", "quit"):
            return _QUIT
        gold_intent = by_number.get(answer.lower()) or by_id.get(answer.lower())
        if gold_intent is not None:
            break
        tell("unknown intent: enter a menu number, an intent id, ? for the menu, or q to pause.")
    while True:
        answer = ask("decision (auto/escalate)> ").strip().lower()
        if answer in ("s", "skip"):
            return _SKIP
        if answer in ("q", "quit"):
            return _QUIT
        if answer in ("a", "auto"):
            gold_decision = AUTO
            break
        if answer in ("e", "escalate"):
            gold_decision = ESCALATE
            break
        tell("unknown decision: enter auto (a) or escalate (e), or q to pause.")
    answer = ask("notes (optional)> ").strip()
    if answer in ("s", "skip"):
        return _SKIP
    if answer in ("q", "quit"):
        return _QUIT
    return gold_intent, gold_decision, answer


def _call_labeler(
    infer: Callable[[str], llm.LLMReply], prompt: str, interaction_id: int
) -> llm.LLMReply:
    """Call the labeler, retrying transient API failures with backoff.

    A 1,000-message hint run meets rate limits and brief upstream failures;
    backing off keeps the run alive instead of failing it on one 429.
    """
    for attempt in range(CALL_ATTEMPTS):
        try:
            return infer(prompt)
        except Exception as exc:
            if attempt == CALL_ATTEMPTS - 1 or not _is_transient(exc):
                raise GoldenError(
                    f"labeler call failed for interaction {interaction_id}: {exc}"
                ) from exc
            delay = CALL_BACKOFF_SECONDS * (2 ** attempt)
            time.sleep(delay * (0.5 + random.random()))


def _is_transient(exc: Exception) -> bool:
    """Tell rate limits and connection failures from real errors."""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return getattr(exc, "status_code", None) in TRANSIENT_STATUS_CODES


def _decode_record(line: str, location: str) -> object:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise GoldenError(f"malformed JSON on {location}: {exc}") from exc


def _parse_hint_record(line: str, location: str) -> IntentHint:
    record = _decode_record(line, location)
    if not isinstance(record, dict):
        raise GoldenError(f"malformed hint on {location}: expected an object")
    interaction_id = record.get("interaction_id")
    if type(interaction_id) is not int:
        raise GoldenError(f"malformed hint on {location}: invalid interaction_id")
    intent = record.get("intent")
    if not isinstance(intent, str) or not intent:
        raise GoldenError(f"malformed hint on {location}: invalid intent")
    decision = record.get("decision")
    if decision not in DECISIONS:
        raise GoldenError(f"malformed hint on {location}: invalid decision")
    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise GoldenError(f"malformed hint on {location}: invalid reason")
    model = record.get("model")
    if not isinstance(model, str) or not model:
        raise GoldenError(f"malformed hint on {location}: invalid model")
    prompt_sha256 = record.get("prompt_sha256")
    if prompt_sha256 is not None and (
        not isinstance(prompt_sha256, str) or not prompt_sha256
    ):
        raise GoldenError(f"malformed hint on {location}: invalid prompt_sha256")
    return IntentHint(
        interaction_id=interaction_id,
        intent=intent,
        decision=decision,
        reason=reason,
        model=model,
        prompt_sha256=prompt_sha256,
    )


def _parse_queue_record(
    record: object, location: str, intent_ids: Sequence[str] | None
) -> QueueItem:
    if not isinstance(record, dict):
        raise GoldenError(f"malformed queue item on {location}: expected an object")
    interaction_id = record.get("interaction_id")
    if type(interaction_id) is not int:
        raise GoldenError(
            f"malformed queue item on {location}: invalid interaction_id"
        )
    intent = record.get("hint_intent")
    if not isinstance(intent, str) or not intent:
        raise GoldenError(f"malformed queue item on {location}: invalid hint_intent")
    if intent_ids is not None and intent not in intent_ids:
        raise GoldenError(
            f"malformed queue item on {location}: unknown hint_intent {intent!r}"
        )
    decision = record.get("hint_decision")
    if decision not in DECISIONS:
        raise GoldenError(
            f"malformed queue item on {location}: invalid hint_decision"
        )
    return QueueItem(interaction_id=interaction_id, intent=intent, decision=decision)


def _parse_golden_record(
    record: object, location: str, intent_ids: Sequence[str] | None
) -> GoldenExample:
    if not isinstance(record, dict):
        raise GoldenError(f"malformed Golden Set example on {location}: expected an object")
    interaction_id = record.get("interaction_id")
    if type(interaction_id) is not int:
        raise GoldenError(
            f"malformed Golden Set example on {location}: invalid interaction_id"
        )
    taxonomy_version = record.get("taxonomy_version")
    if type(taxonomy_version) is not int or taxonomy_version < 1:
        raise GoldenError(
            f"malformed Golden Set example on {location}: invalid taxonomy_version"
        )
    customer_message = record.get("customer_message")
    if not isinstance(customer_message, str) or not customer_message.strip():
        raise GoldenError(
            f"malformed Golden Set example on {location}: invalid customer_message"
        )
    checks = (
        ("gold_intent", intent_ids),
        ("hint_intent", intent_ids),
    )
    for field, allowed in checks:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise GoldenError(
                f"malformed Golden Set example on {location}: invalid {field}"
            )
        if allowed is not None and value not in allowed:
            raise GoldenError(
                f"malformed Golden Set example on {location}: unknown {field} {value!r}"
            )
    gold_decision = record.get("gold_decision")
    if gold_decision not in DECISIONS:
        raise GoldenError(
            f"malformed Golden Set example on {location}: invalid gold_decision"
        )
    hint_decision = record.get("hint_decision")
    if hint_decision not in DECISIONS:
        raise GoldenError(
            f"malformed Golden Set example on {location}: invalid hint_decision"
        )
    notes = record.get("notes")
    if not isinstance(notes, str):
        raise GoldenError(f"malformed Golden Set example on {location}: invalid notes")
    return GoldenExample(
        interaction_id=interaction_id,
        taxonomy_version=taxonomy_version,
        customer_message=customer_message,
        gold_intent=record["gold_intent"],
        gold_decision=gold_decision,
        notes=notes,
        hint_intent=record["hint_intent"],
        hint_decision=hint_decision,
    )


def _rank_key(seed: int, interaction_id: int) -> bytes:
    """Stable ranking key: the same seed and id rank the same everywhere."""
    return hashlib.sha256(f"{seed}:{interaction_id}".encode()).digest()

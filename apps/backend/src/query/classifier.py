"""Classify one Customer Message into the final intent taxonomy with an LLM.

Ticket 14's system classifier (``docs/specs-v0.md`` §4): the configured
``labeler`` role — Nemotron-3.5-Lightning per decision 8, the same cheap role
that adjudicates closures and maps discovery clusters — reads a single opening
Customer Message (decision 2's single-turn scope) and returns the intent plus a
confidence. The TF-IDF baseline (ticket 13) is the comparison point; ticket 23
evaluates both on the Golden Set.

The taxonomy version is handled implicitly through the prompt: the prompt is
rendered from the versioned ``FinalTaxonomy`` document (``docs/intent-taxonomy.md``
via :func:`query.taxonomy.read_final_taxonomy`), so a new document version
changes what the model sees without a code change, and the prediction records
the version it was classified under. Replies are constrained to the taxonomy —
an intent outside the document's ids, or a confidence outside 0-1, is retried
once with a repair prompt and then fails rather than guessing.
"""

from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from query import llm, taxonomy

CLASSIFIER_ROLE = "labeler"

DEFAULT_TAXONOMY_PATH = taxonomy.DEFAULT_FINAL_TAXONOMY_PATH

SYSTEM_PROMPT = (
    "You are a precise intent classifier for customer-support research. "
    "You classify one customer message and reply with raw JSON only."
)


class ClassifierError(Exception):
    """Raised when a message cannot be classified into the taxonomy."""


@dataclass(frozen=True)
class IntentPrediction:
    """One message's classified intent with the model's confidence."""

    intent: str
    confidence: float
    model: str
    taxonomy_version: int


def classify_intent(
    message: str,
    final: taxonomy.FinalTaxonomy,
    infer: Callable[[str], llm.LLMReply] | None = None,
) -> IntentPrediction:
    """Classify one Customer Message into the final taxonomy.

    Returns the predicted intent id with a 0-1 confidence, the model that
    produced it, and the taxonomy version from the prompt. ``infer`` is the
    classifier call, kept injectable for tests; an invalid reply is retried
    once with a repair prompt before the run fails.
    """
    _require_message(message)
    _require_taxonomy(final)
    infer = infer if infer is not None else call_labeler
    prompt = build_prompt(message, final)
    reply = _call_classifier(infer, prompt)
    try:
        intent, confidence = parse_reply(reply.content, final.intent_ids)
    except ClassifierError:
        repair_prompt = llm.build_repair_prompt(prompt, reply.content)
        reply = _call_classifier(infer, repair_prompt)
        intent, confidence = parse_reply(reply.content, final.intent_ids)
    return IntentPrediction(
        intent=intent,
        confidence=confidence,
        model=reply.model,
        taxonomy_version=final.version,
    )


def classify_intents(
    messages: Sequence[str],
    final: taxonomy.FinalTaxonomy,
    infer: Callable[[str], llm.LLMReply] | None = None,
    workers: int = 1,
) -> tuple[IntentPrediction, ...]:
    """Classify several messages, preserving input order.

    ``workers`` bounds concurrent classifier calls; ``infer`` stays injectable
    so evaluation harnesses can stub the model.
    """
    if workers < 1:
        raise ClassifierError(f"workers must be at least 1, got {workers}")
    infer = infer if infer is not None else call_labeler
    if workers == 1 or len(messages) <= 1:
        return tuple(
            _classify_one(index, message, final, infer)
            for index, message in enumerate(messages)
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_classify_one, index, message, final, infer)
            for index, message in enumerate(messages)
        ]
        try:
            return tuple(future.result() for future in futures)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def call_labeler(prompt: str) -> llm.LLMReply:
    """Ask the configured labeler role to classify one message."""
    return llm.call_llm(prompt, role=CLASSIFIER_ROLE, system=SYSTEM_PROMPT, temperature=0.0)


def _classify_one(
    index: int,
    message: str,
    final: taxonomy.FinalTaxonomy,
    infer: Callable[[str], llm.LLMReply],
) -> IntentPrediction:
    """Classify one batch entry, tagging failures with its input position."""
    try:
        return classify_intent(message, final, infer)
    except ClassifierError as exc:
        raise ClassifierError(f"message {index}: {exc}") from exc


def build_prompt(message: str, final: taxonomy.FinalTaxonomy) -> str:
    """Render the versioned taxonomy and the message to classify."""
    intent_lines = "\n".join(
        f"- {intent.intent_id}: {intent.definition}" for intent in final.intents
    )
    example_lines = "\n".join(
        f"- {intent.intent_id}: {example}"
        for intent in final.intents
        for example in intent.examples
    )
    valid_ids = ", ".join(f"`{intent_id}`" for intent_id in final.intent_ids)
    return (
        "Classify one customer-support message into the final intent taxonomy.\n\n"
        f"Taxonomy version {final.version} (this prompt fixes the version; "
        "classify against exactly these intents):\n"
        f"{intent_lines}\n\n"
        "Verbatim example messages per intent:\n"
        f"{example_lines}\n\n"
        "Decide by what the message asks for or reports, not by single words: "
        "a message that states no issue and no request a support team would "
        "act on — praise, jokes, bare mentions, support-channel chatter — is "
        "`other`.\n\n"
        f"Customer Message:\n{message.strip()}\n\n"
        "Reply with raw JSON only, no markdown and no prose:\n"
        '{"intent": '
        f"one of {valid_ids}, "
        '"confidence": 0.0-1.0}\n'
        "Confidence is the estimated probability the intent is correct: 1.0 "
        "when certain, around 0.5 when guessing between two intents, lower "
        "when the message is ambiguous or very short."
    )


def parse_reply(content: str, intent_ids: Collection[str]) -> tuple[str, float]:
    """Validate the classifier's reply and normalize the confidence."""
    payload = llm.extract_json_object(content)
    if payload is None:
        raise ClassifierError(
            "classifier returned no JSON object: "
            f"{content.strip()[: llm.MAX_REPLY_EXCERPT]!r}"
        )
    intent = payload.get("intent")
    if intent not in intent_ids:
        raise ClassifierError(f"classifier returned unknown intent {intent!r}")
    confidence = payload.get("confidence")
    error = confidence_error(confidence)
    if error is not None:
        raise ClassifierError(
            f"classifier returned an invalid confidence ({confidence!r}): {error}"
        )
    return intent, float(confidence)


def confidence_error(confidence: object) -> str | None:
    """Check a confidence value: a finite number inside 0-1, or the problem."""
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return "expected a number between 0 and 1"
    value = float(confidence)
    if value != value or value in (float("inf"), float("-inf")):
        return "expected a finite number between 0 and 1"
    if not 0.0 <= value <= 1.0:
        return "expected a number between 0 and 1"
    return None


def prediction_error(
    prediction: IntentPrediction, intent_ids: Collection[str] | None = None
) -> str | None:
    """Check a prediction against the contract the classifier guarantees.

    Shared by the classifier and by stages whose records embed a prediction,
    so a build boundary that validates through this function provably emits
    predictions the classifier could have produced. When ``intent_ids`` is
    given, an intent outside them is invalid. Returns the problem, or ``None``
    when the prediction is valid.
    """
    if not isinstance(prediction.intent, str) or not prediction.intent:
        return "invalid intent"
    if intent_ids is not None and prediction.intent not in intent_ids:
        return f"unknown intent {prediction.intent!r}"
    error = confidence_error(prediction.confidence)
    if error is not None:
        return f"invalid confidence: {error}"
    if not isinstance(prediction.model, str) or not prediction.model:
        return "invalid model"
    if type(prediction.taxonomy_version) is not int or prediction.taxonomy_version < 1:
        return "invalid taxonomy_version"
    return None


def prediction_to_json(prediction: IntentPrediction) -> dict:
    """Serialize an IntentPrediction to a JSON-compatible mapping."""
    return {
        "intent": prediction.intent,
        "confidence": prediction.confidence,
        "model": prediction.model,
        "taxonomy_version": prediction.taxonomy_version,
    }


def read_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> taxonomy.FinalTaxonomy:
    """Read the final taxonomy the classifier prompts against.

    Thin alias over :func:`query.taxonomy.read_final_taxonomy` so downstream
    stages import one classifier entry point.
    """
    return taxonomy.read_final_taxonomy(path)


def _require_message(message: object) -> None:
    if not isinstance(message, str) or not message.strip():
        raise ClassifierError("cannot classify an empty message")


def _require_taxonomy(final: object) -> None:
    intents = getattr(final, "intents", None)
    version = getattr(final, "version", None)
    intent_ids = (
        tuple(getattr(intent, "intent_id", None) for intent in intents)
        if isinstance(intents, tuple) and intents
        else ()
    )
    if (
        not intent_ids
        or type(version) is not int
        or version < 1
        or taxonomy.OTHER_INTENT_ID not in intent_ids
    ):
        raise ClassifierError("classifier needs a non-empty FinalTaxonomy")


def _call_classifier(
    infer: Callable[[str], llm.LLMReply], prompt: str
) -> llm.LLMReply:
    try:
        return infer(prompt)
    except Exception as exc:  # noqa: BLE001 - provider errors vary
        raise ClassifierError(f"classifier call failed: {exc}") from exc

"""Label obvious Interaction closures with keyword heuristics.

Closure detection follows ADR-0001: an Interaction is Resolved (evidence of
natural closure: an explicit customer acknowledgement, or a brand reply
indicating the issue/action was completed), Uncertain (the brand replied and the
customer went silent, with no evidence the issue was solved), or Unresolved (the
customer keeps asking, or the brand said it could not help). Detection is a
hybrid: this stage labels the obvious cases from the closing turns, and flags
the ambiguous middle — DM deflections and unclear customer closing messages —
for LLM adjudication (ticket 7) instead of guessing.

The rules read the closing turns only: the final turn, the final customer turn,
and the final brand turn. They are applied in priority order so contradictory
signals do not leak into a label:

1. a final customer message that mentions moving to DMs is flagged;
2. a final customer message that continues the issue (a question, "still not
   working", a repeated request) is Unresolved, even when it also thanks the
   brand;
3. a final customer message that reports the issue fixed is Resolved;
4. a final customer message that only acknowledges the brand is Resolved —
   unless the brand's preceding reply moved the conversation to DMs, which is
   flagged;
5. a brand reply that claims the issue was fixed or processed is Resolved, even
   when it points at a DM for the details;
6. a brand reply that asks the customer back into DMs is flagged — whatever
   happened privately is not visible in the recorded Interaction;
7. a brand reply that says it cannot help is Unresolved;
8. a customer acknowledgement followed by a closing courtesy ("you're welcome")
   or a plain sign-off is Resolved — unless the brand asks for something ("can
   you send us...") or promises to investigate further, in which case the
   issue is still open;
9. a brand reply that asks for more information, or promises to investigate
   further, with no customer response is Uncertain;
10. any other brand-last ending is Uncertain: the brand replied and the customer
    went silent, which is the Uncertain definition, not a guess.

Two refinements keep false Resolved labels out of the RAG index. A bare
thank-you only counts as acknowledgement when it is essentially the whole
message — a long message that merely contains a thank-you is usually a new
request with a courtesy attached — and the opening customer message can never
acknowledge help it has not received. A closing courtesy on its own ("you're
welcome", "glad to hear") is not a completion claim, and a negated fix ("it's
not fixed yet") is a continuation, not an acknowledgement.

Every verdict carries a human-readable reason; flagged cases carry the reason
they need adjudication. The report counts each label, the flagged volume by
reason, and the share of the sample the heuristics could label definitively.
"""

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from query.interactions import Interaction, Turn

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "rag-pool.jsonl"

CLOSURE_LABELS = ("resolved", "uncertain", "unresolved")
Label = Literal["resolved", "uncertain", "unresolved"]

REASON_CUSTOMER_CONTINUES = "customer keeps asking after the brand's last reply"
REASON_CUSTOMER_RESOLVED = "customer's final message acknowledges the issue is resolved"
REASON_CUSTOMER_DM = "customer moved the conversation to DMs; the outcome is not visible here"
REASON_CUSTOMER_DM_ACK = (
    "customer acknowledged the brand after it moved the conversation to DMs; "
    "the outcome is not visible here"
)
REASON_CUSTOMER_UNCLEAR = (
    "customer's final message neither acknowledges nor continues the issue"
)
REASON_BRAND_COMPLETION = "brand's final reply confirms the issue was handled"
REASON_BRAND_REFUSAL = "brand's final reply says it cannot help"
REASON_BRAND_ACK_CLOSE = "customer acknowledged the brand's help before its closing reply"
REASON_BRAND_DM = "brand moved the conversation to DMs; the outcome is not visible here"
REASON_BRAND_QUESTION = "brand asked for more information and the customer did not reply"
REASON_BRAND_FOLLOWUP = (
    "brand promised to investigate further and the customer did not reply"
)
REASON_BRAND_SILENCE = "brand replied and the customer went silent"

URL_PATTERN = re.compile(r"https?://\S+|t\.co/\S+")
MENTION_PATTERN = re.compile(r"@\w+")
# The dataset mixes straight and typographic apostrophes ("can't" / "can’t");
# patterns are written with straight ones, so texts are normalized first.
APOSTROPHE_PATTERN = re.compile("[\u2019\u2018`\u00b4]")
# A message longer than this that merely contains a thank-you is treated as a
# new request with a courtesy attached, not as an acknowledgement.
MAX_ACKNOWLEDGEMENT_LENGTH = 72

# The customer reports the issue is gone: "that worked", "it's working now".
FIXED_PATTERN = re.compile(
    r"(\b(?:fixed|solved)\b|works? (?:now|perfectly|fine|again)|"
    r"working (?:now|perfectly|fine|again)|that (?:worked|did it|fixed it)|"
    r"it'?s working|it works|back to normal|all good now|no issues? now)",
    re.IGNORECASE,
)
# A negation before "fixed"/"solved" means the opposite: the issue is open.
NEGATED_FIX_PATTERN = re.compile(
    r"\b(?:not|never|isn'?t|wasn'?t|weren'?t|hasn'?t|hadn'?t|ain'?t)\b"
    r"(?: been| yet| even| fully| properly| completely| really| quite)*"
    r" (?:fixed|solved|resolved)",
    re.IGNORECASE,
)
# The customer thanks or approves: only unambiguous acknowledgements, so a
# short "sure" or "ok" is left for adjudication rather than guessed.
ACK_PATTERN = re.compile(
    r"(\bthanks?\b|\bthank you\b|\bthx\b|\bty\b|\bcheers\b|\bappreciate (?:it|your|the)\b|"
    r"\bgot it\b|\bgotcha\b|\bperfect\b|\bawesome\b|\blovely\b|\bthat'?s great\b|"
    r"\ball good\b|\bsorted\b|\bbutiful\b|worked, thank)",
    re.IGNORECASE,
)
# The customer keeps asking: a question, a repeated complaint, a demand.
CONTINUATION_PATTERN = re.compile(
    r"(\?|still |not work|doesn'?t work|didn'?t work|any update|any news|"
    r"any chance|when will|how long|how many|same (?:issue|problem)|"
    r"nothing (?:happened|changed)|yet to|waiting|waited|no (?:reply|response|answer|one)|"
    r"please (?:help|fix|reply|respond)|help me|"
    r"can'?t (?:log|sign|access|play|use|find|see|get)|"
    r"cannot (?:log|sign|access|play|use|find|see|get)|"
    r"still (?:can'?t|cannot|not|waiting|broken|the same|an issue|having)|"
    r"i (?:have|'?ve got|got) (?:an? )?(?:issue|problem)|"
    r"the (?:issue|problem) (?:is|persists|remains)|"
    r"not (?:sure|solved|resolved)|"
    r"where is|why (?:is|are|doesn'?t|did)|what about|need (?:help|this)|give me)",
    re.IGNORECASE,
)
# The customer answers a DM request ("DMed", "replied via DM", "messaged you").
DM_REPLY_PATTERN = re.compile(
    r"(^\W*(?:dmed|dm'?d|dm'?ed)\b|\bi (?:have |just |already )?(?:dmed|dm'?d|dm'?ed|"
    r"messaged|sent)\b|sent .{0,20}\bdms?\b|replied .{0,20}\bdms?\b|"
    r"\bdms?\b .{0,20}\b(?:sent|done|replied)\b|slid into (?:your|the) dms?\b|"
    r"messaged you\b|dm sent\b)",
    re.IGNORECASE,
)
# The brand claims the issue/action was completed. Closing courtesies ("you're
# welcome") are not completion claims and live in COURTESY_PATTERN instead.
COMPLETION_PATTERN = re.compile(
    r"(we'?ve (?:fixed|resolved|processed|updated|sorted|taken care)|"
    r"has been (?:fixed|resolved|processed|updated)|"
    r"is now (?:fixed|resolved|working|available)|"
    r"should (?:now )?(?:be )?(?:working|work|be fixed|be resolved)|"
    r"you'?re all set|back to normal|back (?:up and running|on track)|it'?s fixed)",
    re.IGNORECASE,
)
# A closing invitation ("let us know if you need us") matches the question
# pattern but asks nothing of the customer; it is a sign-off, not a request.
SOFT_INVITE_PATTERN = re.compile(
    r"(if you (?:ever )?(?:need|have)|anything else|give us a shout|"
    r"you know where to find us|we'?re (?:here|around)|just let us know|shout if|"
    r"reach out (?:if|anytime)|don'?t hesitate)",
    re.IGNORECASE,
)# "Thanks anyway" is resignation, not an acknowledgement of resolution.
THANKS_ANYWAY_PATTERN = re.compile(r"thanks? (?:anyway|anyways)", re.IGNORECASE)
# The brand says it cannot help.
REFUSAL_PATTERN = re.compile(
    r"(unfortunately|we'?re afraid|we (?:can'?t|cannot|won'?t)|i'?m afraid|"
    r"(?:we|we do not|we don'?t) (?:currently )?(?:have|offer|support)(?: any)?|"
    r"no updates?|not available|unable to|not something we|outside of our|"
    r"not possible|we'?re not able|isn'?t (?:something|available)|"
    r"don'?t (?:currently )?(?:have|offer|support)|we don'?t have any)",
    re.IGNORECASE,
)
# The brand moves the conversation to DMs, where the outcome is invisible.
DEFLECTION_PATTERN = re.compile(
    r"(\bdms?\b|direct message|sent you a (?:bit more info over )?dm|sent a dm|"
    r"replied to your dm|check your dm|carry on (?:chatting|helping)? ?(?:there|in dm)|"
    r"carry on there|continue (?:chatting|helping|the conversation)? ?(?:there|via dm|in dm)|"
    r"message us|shoot us a dm|send us a dm|via dm|over dm|slide into our dm|"
    r"dm'?d you|we'?ll dm|dm me)",
    re.IGNORECASE,
)
# The brand asks for information it needs before it can help.
QUESTION_PATTERN = re.compile(
    r"(\?\s*$|\? |can you|could you|would you|let us know|send us|shoot us|fire over|"
    r"share (?:the|your)|confirm|tell us|try (?:again|to)|give it another)",
    re.IGNORECASE,
)
# The brand defers: it will investigate and follow up later.
FOLLOWUP_PATTERN = re.compile(
    r"(we'?ll (?:take a look|check|look into|see what|get back|follow up|investigate|"
    r"pass|share|let|make sure|get .{0,20}passed)|"
    r"pass(?:ed)? (?:this|it|your .{0,20})? ?(?:on |onto )?to the (?:right )?team|"
    r"make sure to (?:pass|share|let)|"
    r"let the right (?:team|folks) know|we'?ll keep (?:an? )?(?:eye|ear)|"
    r"our team is|we are (?:looking|investigating))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClosureLabel:
    """One Interaction's closure verdict and the reason behind it."""

    interaction_id: int
    label: Label | None
    reason: str
    needs_adjudication: bool


@dataclass(frozen=True)
class ClosureReport:
    """Counts of the heuristic labeling stage, for honest reporting."""

    total: int
    resolved: int
    uncertain: int
    unresolved: int
    needs_adjudication: int
    flagged_by_reason: Mapping[str, int]

    @property
    def labeled(self) -> int:
        return self.total - self.needs_adjudication

    @property
    def label_rate(self) -> float:
        return self.labeled / self.total if self.total else 0.0

    @property
    def adjudication_rate(self) -> float:
        return self.needs_adjudication / self.total if self.total else 0.0


def label_closures(
    interactions: Sequence[Interaction],
) -> tuple[tuple[ClosureLabel, ...], ClosureReport]:
    """Label obvious closures and flag ambiguous ones for adjudication.

    Returns one verdict per Interaction, in input order, and a report of the
    label distribution and the flagged volume by reason.
    """
    labels: list[ClosureLabel] = []
    counts = {"resolved": 0, "uncertain": 0, "unresolved": 0}
    flagged_by_reason: dict[str, int] = {}
    for interaction in interactions:
        verdict = _label_interaction(interaction)
        labels.append(verdict)
        if verdict.needs_adjudication:
            flagged_by_reason[verdict.reason] = (
                flagged_by_reason.get(verdict.reason, 0) + 1
            )
        else:
            counts[verdict.label] += 1
    report = ClosureReport(
        total=len(interactions),
        resolved=counts["resolved"],
        uncertain=counts["uncertain"],
        unresolved=counts["unresolved"],
        needs_adjudication=sum(flagged_by_reason.values()),
        flagged_by_reason=dict(
            sorted(flagged_by_reason.items(), key=lambda item: (-item[1], item[0]))
        ),
    )
    return tuple(labels), report


def label_to_json(label: ClosureLabel) -> dict:
    """Serialize a ClosureLabel to a JSON-compatible mapping."""
    return {
        "interaction_id": label.interaction_id,
        "label": label.label,
        "reason": label.reason,
        "needs_adjudication": label.needs_adjudication,
    }


def stage_closure_labels_jsonl(
    labels: tuple[ClosureLabel, ...], output_path: Path | str
) -> Path:
    """Write closure labels to a temporary sibling, ready to be swapped in.

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


def _normalize(text: str) -> str:
    """Normalize typographic apostrophes so patterns written with ' match."""
    return APOSTROPHE_PATTERN.sub("'", text)


def _acknowledges_resolution(text: str) -> bool:
    """True when the customer confirms closure or thanks the brand for help.

    A "fixed/works now" claim always counts. A bare thank-you only counts when
    it is essentially the whole message: a long message that merely contains a
    thank-you is usually a new request with a courtesy attached.
    """
    text = _normalize(text)
    if NEGATED_FIX_PATTERN.search(text):
        return False
    if FIXED_PATTERN.search(text):
        return True
    if THANKS_ANYWAY_PATTERN.search(text):
        return False
    if not ACK_PATTERN.search(text):
        return False
    stripped = MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", text)).strip()
    return len(stripped) <= MAX_ACKNOWLEDGEMENT_LENGTH


def _continues_issue(text: str) -> bool:
    """True when the message keeps the issue open rather than closing it."""
    return bool(CONTINUATION_PATTERN.search(text) or NEGATED_FIX_PATTERN.search(text))


def _label_interaction(interaction: Interaction) -> ClosureLabel:
    turns = interaction.turns
    if turns[-1].side == "customer":
        return _label_customer_closing(interaction.interaction_id, turns)
    return _label_brand_closing(interaction.interaction_id, turns)


def _label_customer_closing(interaction_id: int, turns: tuple[Turn, ...]) -> ClosureLabel:
    text = _normalize(turns[-1].text)
    if DM_REPLY_PATTERN.search(text):
        return _flagged(interaction_id, REASON_CUSTOMER_DM)
    if _continues_issue(text):
        return _labeled(interaction_id, "unresolved", REASON_CUSTOMER_CONTINUES)
    if not _acknowledges_resolution(text):
        return _flagged(interaction_id, REASON_CUSTOMER_UNCLEAR)
    previous_brand = next(
        (turn for turn in reversed(turns[:-1]) if turn.side == "brand"), None
    )
    if (
        previous_brand is not None
        and DEFLECTION_PATTERN.search(_normalize(previous_brand.text))
        and not FIXED_PATTERN.search(text)
    ):
        return _flagged(interaction_id, REASON_CUSTOMER_DM_ACK)
    return _labeled(interaction_id, "resolved", REASON_CUSTOMER_RESOLVED)


def _label_brand_closing(interaction_id: int, turns: tuple[Turn, ...]) -> ClosureLabel:
    text = _normalize(turns[-1].text)
    if COMPLETION_PATTERN.search(text):
        return _labeled(interaction_id, "resolved", REASON_BRAND_COMPLETION)
    if DEFLECTION_PATTERN.search(text):
        return _flagged(interaction_id, REASON_BRAND_DM)
    if REFUSAL_PATTERN.search(text):
        return _labeled(interaction_id, "unresolved", REASON_BRAND_REFUSAL)
    last_customer = next((turn for turn in reversed(turns) if turn.side == "customer"), None)
    # The opening message cannot acknowledge help it has not received yet.
    customer_acked = (
        last_customer is not None
        and last_customer is not turns[0]
        and not _continues_issue(_normalize(last_customer.text))
        and _acknowledges_resolution(last_customer.text)
    )
    followup = FOLLOWUP_PATTERN.search(text)
    question = QUESTION_PATTERN.search(text)
    # A closing invitation is a sign-off, not a request for information; a
    # promise to investigate is work still in flight, so it never closes.
    if customer_acked and not followup and (
        not question or SOFT_INVITE_PATTERN.search(text)
    ):
        return _labeled(interaction_id, "resolved", REASON_BRAND_ACK_CLOSE)
    if followup:
        return _labeled(interaction_id, "uncertain", REASON_BRAND_FOLLOWUP)
    if question:
        return _labeled(interaction_id, "uncertain", REASON_BRAND_QUESTION)
    return _labeled(interaction_id, "uncertain", REASON_BRAND_SILENCE)


def _labeled(interaction_id: int, label: Label, reason: str) -> ClosureLabel:
    return ClosureLabel(
        interaction_id=interaction_id,
        label=label,
        reason=reason,
        needs_adjudication=False,
    )


def _flagged(interaction_id: int, reason: str) -> ClosureLabel:
    return ClosureLabel(
        interaction_id=interaction_id,
        label=None,
        reason=reason,
        needs_adjudication=True,
    )

"""Read the seed and final intent taxonomies from their Markdown artifacts.

Ticket 9 wrote the candidate taxonomy as a human-reviewable document,
``docs/intent-seed-taxonomy.md``: a table of intent ids and one-line
definitions, plus one representative message per intent. Discovery (ticket 10)
reconciles its clusters against that document instead of a second, drifting
copy. Ticket 11 then reconciled the seed with those clusters into the final
versioned taxonomy, ``docs/intent-taxonomy.md``, which later stages import as
the single source of truth through :func:`read_final_taxonomy`.

Both readers are strict about the parts the pipeline depends on. The seed
reader requires every table row to carry a backticked snake_case id and a
non-empty definition, and every intent exactly one representative message. The
final reader additionally requires the 8-15 intent target from the
specification, at least one example per intent, and a recorded reconciliation
decision for every seed intent and every discovered theme. A document that
fails these checks raises instead of silently returning a partial taxonomy; a
renamed or dropped row is a taxonomy change that should be visible, not
absorbed.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEED_TAXONOMY_PATH = REPO_ROOT / "docs" / "intent-seed-taxonomy.md"
DEFAULT_FINAL_TAXONOMY_PATH = REPO_ROOT / "docs" / "intent-taxonomy.md"

# The specification's target for the final taxonomy (docs/specs-v0.md, §4).
MIN_FINAL_INTENTS = 8
MAX_FINAL_INTENTS = 15

# Reconciliation decision values, as recorded in the final taxonomy document.
KEPT = "kept"
MERGED = "merged"
DROPPED = "dropped"
PROMOTED = "promoted"
SEED_DECISIONS = (KEPT, MERGED, DROPPED)
NEW_THEME_DECISIONS = (PROMOTED, DROPPED)

# The seed document's `other` row is a fallback class for messages that fit no
# support intent, not a discovery target. Discovery maps clusters to support
# intents or flags them; this id is still parsed so the document stays the one
# source of truth.
OTHER_INTENT_ID = "other"

_INTENT_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_TABLE_ROW = re.compile(r"^\|\s*\d+\s*\|")
_REPRESENTATIVE = re.compile(r"^-\s+`([^`]+)`\s+\[\d+\]:\s+(.+)$")
_FINAL_VERSION = re.compile(r"^\*\*Version:\*\*\s+(\d+)\s*$")
_FINAL_INTENT_HEADER = "| # | Intent id | Definition | Origin |"
_FINAL_EXAMPLE_HEADER = "## Representative examples"
_SEED_DECISION_HEADER = "| Seed intent | Decision | Final intent | Evidence |"
_NEW_THEME_HEADER = "| Cluster | Decision | Final intent | Evidence |"


class TaxonomyError(Exception):
    """Raised when the taxonomy document cannot be turned into intents."""


@dataclass(frozen=True)
class SeedIntent:
    """One candidate intent from the seed taxonomy."""

    intent_id: str
    definition: str
    representative_message: str


def read_seed_taxonomy(
    path: Path | str = DEFAULT_SEED_TAXONOMY_PATH,
) -> tuple[SeedIntent, ...]:
    """Parse the seed taxonomy table and its representative messages.

    Returns the intents in document order. Raises :class:`TaxonomyError` when
    the file is missing, a row is malformed, an id repeats, a definition is
    empty, or an intent has no (or a duplicated) representative message.
    """
    path = Path(path)
    if not path.is_file():
        raise TaxonomyError(f"seed taxonomy does not exist: {path}")
    definitions: dict[str, str] = {}
    representatives: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        location = f"line {line_number} of {path}"
        _read_table_row(line, location, definitions)
        representative = _REPRESENTATIVE.match(line.strip())
        if representative:
            intent_id, message = representative.groups()
            if intent_id in representatives:
                raise TaxonomyError(f"duplicate representative message on {location}")
            representatives[intent_id] = (message.strip(), location)
    if not definitions:
        raise TaxonomyError(f"no intent rows found in {path}")
    unknown = [
        (intent_id, location)
        for intent_id, (_, location) in representatives.items()
        if intent_id not in definitions
    ]
    if unknown:
        intent_id, location = unknown[0]
        raise TaxonomyError(
            f"representative message on {location} references unknown intent {intent_id!r}"
        )
    missing = [
        intent_id for intent_id in definitions if intent_id not in representatives
    ]
    if missing:
        raise TaxonomyError(
            f"intent {missing[0]!r} has no representative message in {path}"
        )
    return tuple(
        SeedIntent(
            intent_id=intent_id,
            definition=definition,
            representative_message=representatives[intent_id][0],
        )
        for intent_id, definition in definitions.items()
    )


def _read_table_row(line: str, location: str, definitions: dict[str, str]) -> None:
    """Read one intent table row into ``definitions`` in document order."""
    if not _TABLE_ROW.match(line.strip()):
        return
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) != 4:
        raise TaxonomyError(
            f"malformed taxonomy row on {location}: expected 4 columns, got {len(cells)}"
        )
    id_cell = cells[1]
    intent_id = _backticked_id(id_cell, location)
    if intent_id in definitions:
        raise TaxonomyError(f"duplicate intent id {intent_id!r} on {location}")
    definition = cells[2]
    if not definition:
        raise TaxonomyError(
            f"intent {intent_id!r} has an empty definition on {location}"
        )
    definitions[intent_id] = definition


@dataclass(frozen=True)
class FinalIntent:
    """One intent of the final versioned taxonomy."""

    intent_id: str
    definition: str
    origin: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class SeedDecision:
    """What the reconciliation did with one seed intent."""

    seed_intent: str
    decision: str
    final_intent: str | None


@dataclass(frozen=True)
class NewThemeDecision:
    """What the reconciliation did with one discovery cluster flagged ``new``."""

    cluster_id: int
    decision: str
    final_intent: str | None


@dataclass(frozen=True)
class FinalTaxonomy:
    """The versioned final taxonomy and the decisions that produced it."""

    version: int
    intents: tuple[FinalIntent, ...]
    seed_decisions: tuple[SeedDecision, ...]
    new_theme_decisions: tuple[NewThemeDecision, ...]

    @property
    def intent_ids(self) -> tuple[str, ...]:
        """The intent ids, in document order."""
        return tuple(intent.intent_id for intent in self.intents)


def seed_decision_error(
    decision: SeedDecision, intent_ids: Collection[str]
) -> str | None:
    """Check one seed decision against the reconciliation contract.

    Shared by the document reader and the reconciliation checker, like the
    discovery cluster contract, so the two cannot disagree about what a valid
    decision is. Returns ``None`` when the decision is valid, or a short
    description of the problem.
    """
    if decision.decision not in SEED_DECISIONS:
        return f"invalid seed decision {decision.decision!r}"
    if decision.decision == DROPPED:
        if decision.final_intent is not None:
            return "dropped seed intent must have no final intent"
        return None
    if decision.final_intent is None:
        return f"{decision.decision} seed intent needs a final intent"
    if decision.final_intent not in intent_ids:
        return f"unknown intent {decision.final_intent!r}"
    if decision.decision == KEPT and decision.final_intent != decision.seed_intent:
        return "kept seed intent must keep its id"
    if decision.decision == MERGED and decision.final_intent == decision.seed_intent:
        return "merged seed intent must point at another intent"
    return None


def new_theme_decision_error(
    decision: NewThemeDecision, intent_ids: Collection[str]
) -> str | None:
    """Check one discovered-theme decision, shared like :func:`seed_decision_error`."""
    if decision.decision not in NEW_THEME_DECISIONS:
        return f"invalid new-theme decision {decision.decision!r}"
    if decision.decision == DROPPED:
        if decision.final_intent is not None:
            return "dropped theme must have no final intent"
        return None
    if decision.final_intent is None:
        return f"{decision.decision} theme needs a final intent"
    if decision.final_intent not in intent_ids:
        return f"unknown intent {decision.final_intent!r}"
    return None


def read_final_taxonomy(
    path: Path | str = DEFAULT_FINAL_TAXONOMY_PATH,
) -> FinalTaxonomy:
    """Parse the final taxonomy document, its examples, and its decisions.

    Returns the version, the intents in document order with at least one
    example each, and the recorded reconciliation decisions. Raises
    :class:`TaxonomyError` when the file is missing, the version is missing or
    malformed, the intent count leaves the 8-15 target, a row, example, or
    decision is malformed or duplicated, or a decision points at an intent the
    document does not define.
    """
    path = Path(path)
    if not path.is_file():
        raise TaxonomyError(f"final taxonomy does not exist: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    version = _read_final_version(lines, path)
    intents = _read_final_intents(lines, path)
    intents = _attach_final_examples(intents, _read_final_examples(lines, path), path)
    intent_ids = tuple(intent.intent_id for intent in intents)
    return FinalTaxonomy(
        version=version,
        intents=intents,
        seed_decisions=_read_seed_decisions(lines, path, intent_ids),
        new_theme_decisions=_read_new_theme_decisions(lines, path, intent_ids),
    )


def _read_final_version(lines: Sequence[str], path: Path) -> int:
    """Read the document's single ``**Version:**`` line."""
    found = [
        (number, match.group(1))
        for number, line in enumerate(lines, 1)
        if (match := _FINAL_VERSION.match(line.strip()))
    ]
    if len(found) != 1:
        raise TaxonomyError(f"expected exactly one version line in {path}")
    number, raw = found[0]
    version = int(raw)
    if version < 1:
        raise TaxonomyError(
            f"invalid taxonomy version {version} on line {number} of {path}"
        )
    return version


def _read_final_intents(
    lines: Sequence[str], path: Path
) -> tuple[FinalIntent, ...]:
    """Read the intent table into intents without their examples."""
    intents = []
    seen: set[str] = set()
    for number, cells in _read_header_rows(lines, _FINAL_INTENT_HEADER, path):
        location = f"line {number} of {path}"
        if len(cells) != 4:
            raise TaxonomyError(
                f"malformed taxonomy row on {location}: "
                f"expected 4 columns, got {len(cells)}"
            )
        intent_id = _backticked_id(cells[1], location)
        if intent_id in seen:
            raise TaxonomyError(f"duplicate intent id {intent_id!r} on {location}")
        if not cells[2]:
            raise TaxonomyError(
                f"intent {intent_id!r} has an empty definition on {location}"
            )
        if not cells[3]:
            raise TaxonomyError(
                f"intent {intent_id!r} has an empty origin on {location}"
            )
        seen.add(intent_id)
        intents.append(
            FinalIntent(
                intent_id=intent_id,
                definition=cells[2],
                origin=cells[3],
                examples=(),
            )
        )
    if not intents:
        raise TaxonomyError(f"no intent rows found in {path}")
    if not MIN_FINAL_INTENTS <= len(intents) <= MAX_FINAL_INTENTS:
        raise TaxonomyError(
            f"final taxonomy has {len(intents)} intents; "
            f"expected {MIN_FINAL_INTENTS}-{MAX_FINAL_INTENTS}"
        )
    return tuple(intents)


def _read_final_examples(
    lines: Sequence[str], path: Path
) -> dict[str, list[tuple[str, int]]]:
    """Read the example bullets of the representative-messages section."""
    headings = [
        index for index, line in enumerate(lines) if line.strip() == _FINAL_EXAMPLE_HEADER
    ]
    if len(headings) != 1:
        raise TaxonomyError(
            f"expected exactly one {_FINAL_EXAMPLE_HEADER!r} section in {path}"
        )
    examples: dict[str, list[tuple[str, int]]] = {}
    for index in range(headings[0] + 1, len(lines)):
        if lines[index].strip().startswith("## "):
            break
        match = _REPRESENTATIVE.match(lines[index].strip())
        if match:
            intent_id, message = match.groups()
            examples.setdefault(intent_id, []).append((message.strip(), index + 1))
    return examples


def _attach_final_examples(
    intents: tuple[FinalIntent, ...],
    examples: dict[str, list[tuple[str, int]]],
    path: Path,
) -> tuple[FinalIntent, ...]:
    """Give every intent its examples, rejecting gaps and duplicates."""
    attached = []
    seen: dict[str, int] = {}
    for intent in intents:
        entries = examples.get(intent.intent_id, [])
        if not entries:
            raise TaxonomyError(
                f"intent {intent.intent_id!r} has no example in {path}"
            )
        messages = []
        for message, number in entries:
            if message in seen:
                raise TaxonomyError(
                    f"duplicate example message on line {number} of {path} "
                    f"(also on line {seen[message]})"
                )
            seen[message] = number
            messages.append(message)
        attached.append(replace(intent, examples=tuple(messages)))
    unknown = sorted(set(examples) - {intent.intent_id for intent in intents})
    if unknown:
        number = examples[unknown[0]][0][1]
        raise TaxonomyError(
            f"example on line {number} of {path} references unknown "
            f"intent {unknown[0]!r}"
        )
    return tuple(attached)


def _read_seed_decisions(
    lines: Sequence[str], path: Path, intent_ids: tuple[str, ...]
) -> tuple[SeedDecision, ...]:
    """Read and check the seed-intent reconciliation table."""
    decisions = []
    seen: set[str] = set()
    for number, cells in _read_header_rows(lines, _SEED_DECISION_HEADER, path):
        location = f"line {number} of {path}"
        if len(cells) != 4:
            raise TaxonomyError(
                f"malformed seed decision on {location}: "
                f"expected 4 columns, got {len(cells)}"
            )
        seed_intent = _backticked_id(cells[0], location)
        if seed_intent in seen:
            raise TaxonomyError(
                f"duplicate seed decision for {seed_intent!r} on {location}"
            )
        if not cells[3]:
            raise TaxonomyError(
                f"seed decision for {seed_intent!r} has no evidence on {location}"
            )
        decision = SeedDecision(
            seed_intent=seed_intent,
            decision=cells[1],
            final_intent=_decision_target(cells[2], location),
        )
        error = seed_decision_error(decision, intent_ids)
        if error is not None:
            raise TaxonomyError(f"{error} on {location}")
        seen.add(seed_intent)
        decisions.append(decision)
    return tuple(decisions)


def _read_new_theme_decisions(
    lines: Sequence[str], path: Path, intent_ids: tuple[str, ...]
) -> tuple[NewThemeDecision, ...]:
    """Read and check the discovered-theme reconciliation table."""
    decisions = []
    seen: set[int] = set()
    for number, cells in _read_header_rows(
        lines, _NEW_THEME_HEADER, path, allow_empty=True
    ):
        location = f"line {number} of {path}"
        if len(cells) != 4:
            raise TaxonomyError(
                f"malformed new-theme decision on {location}: "
                f"expected 4 columns, got {len(cells)}"
            )
        if not cells[0].isdigit():
            raise TaxonomyError(f"invalid cluster id on {location}: {cells[0]!r}")
        cluster_id = int(cells[0])
        if cluster_id in seen:
            raise TaxonomyError(
                f"duplicate new-theme decision for cluster {cluster_id} on {location}"
            )
        if not cells[3]:
            raise TaxonomyError(
                f"new-theme decision for cluster {cluster_id} has no evidence "
                f"on {location}"
            )
        decision = NewThemeDecision(
            cluster_id=cluster_id,
            decision=cells[1],
            final_intent=_decision_target(cells[2], location),
        )
        error = new_theme_decision_error(decision, intent_ids)
        if error is not None:
            raise TaxonomyError(f"{error} on {location}")
        seen.add(cluster_id)
        decisions.append(decision)
    return tuple(decisions)


def _read_header_rows(
    lines: Sequence[str],
    header: str,
    path: Path,
    allow_empty: bool = False,
) -> list[tuple[int, list[str]]]:
    """Return the ``(line_number, cells)`` rows under one unique table header."""
    locations = [index for index, line in enumerate(lines) if line.strip() == header]
    if not locations:
        raise TaxonomyError(f"missing table {header!r} in {path}")
    if len(locations) > 1:
        raise TaxonomyError(f"duplicate table {header!r} in {path}")
    rows = []
    for index in range(locations[0] + 1, len(lines)):
        line = lines[index].strip()
        if not line.startswith("|"):
            break
        if _is_divider_row(line):
            continue
        rows.append((index + 1, [cell.strip() for cell in line.strip("|").split("|")]))
    if not rows and not allow_empty:
        raise TaxonomyError(f"no rows under table {header!r} in {path}")
    return rows


def _is_divider_row(line: str) -> bool:
    """Tell the ``|---|---|`` separator apart from a data row."""
    return all(
        cell and set(cell) <= {"-", ":"} for cell in line.strip("|").split("|")
    )


def _backticked_id(cell: str, location: str) -> str:
    """Read one backticked snake_case id, or raise."""
    candidate = cell.removeprefix("`").removesuffix("`")
    if not (
        cell.startswith("`")
        and cell.endswith("`")
        and _INTENT_ID.match(candidate)
    ):
        raise TaxonomyError(f"invalid intent id on {location}: {cell!r}")
    return candidate


def _decision_target(cell: str, location: str) -> str | None:
    """Read a decision's final-intent cell: ``-`` or a backticked intent id."""
    if cell == "-":
        return None
    candidate = cell.removeprefix("`").removesuffix("`")
    if not (
        cell.startswith("`")
        and cell.endswith("`")
        and _INTENT_ID.match(candidate)
    ):
        raise TaxonomyError(f"invalid final intent {cell!r} on {location}")
    return candidate

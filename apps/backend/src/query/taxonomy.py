"""Read the seed intent taxonomy from its reviewable Markdown artifact.

Ticket 9 wrote the candidate taxonomy as a human-reviewable document,
``docs/intent-seed-taxonomy.md``: a table of intent ids and one-line
definitions, plus one representative message per intent. This module turns
that document into structured data, so discovery (ticket 10) reconciles its
clusters against the same list a human reviews instead of a second, drifting
copy.

The reader is strict about the parts discovery depends on: every table row
must carry a backticked snake_case id and a non-empty definition, every intent
must have exactly one representative message, and no id may appear twice. A
document that fails these checks raises instead of silently returning a
partial taxonomy; a renamed or dropped row is a taxonomy change that should be
visible, not absorbed.
"""

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEED_TAXONOMY_PATH = REPO_ROOT / "docs" / "intent-seed-taxonomy.md"

# The seed document's `other` row is a fallback class for messages that fit no
# support intent, not a discovery target. Discovery maps clusters to support
# intents or flags them; this id is still parsed so the document stays the one
# source of truth.
OTHER_INTENT_ID = "other"

_INTENT_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_TABLE_ROW = re.compile(r"^\|\s*\d+\s*\|")
_REPRESENTATIVE = re.compile(r"^-\s+`([^`]+)`\s+\[\d+\]:\s+(.+)$")


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
    intent_id = id_cell.removeprefix("`").removesuffix("`")
    if not (
        id_cell.startswith("`")
        and id_cell.endswith("`")
        and _INTENT_ID.match(intent_id)
    ):
        raise TaxonomyError(f"invalid intent id on {location}: {id_cell!r}")
    if intent_id in definitions:
        raise TaxonomyError(f"duplicate intent id {intent_id!r} on {location}")
    definition = cells[2]
    if not definition:
        raise TaxonomyError(
            f"intent {intent_id!r} has an empty definition on {location}"
        )
    definitions[intent_id] = definition

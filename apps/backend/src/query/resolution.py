"""Build the final resolution-labeled dataset every downstream stage draws from.

Ticket 7's adjudication stage produces the full labeled sample: every RAG-pool
Interaction carries a final Resolved/Uncertain/Unresolved label with its
provenance (ADR-0007). This stage joins those labels back onto the Interactions
and emits the resolution dataset: one record per Interaction with its label,
label provenance, retrieval eligibility, and the normalized Interaction content
itself.

Only Resolved Cases are retrieval-eligible Historical Cases (ADR-0001,
decision 4). ``retrieval_eligible`` is derived from the label at build time
rather than stored as an independent input, so an Uncertain or Unresolved case
can never be marked eligible. The dataset keeps the whole labeled sample, not
just the Resolved Cases, because downstream evaluation needs the negatives;
retrieval indexes only the eligible records.

Every record carries ``dataset_version`` — the dataset's schema/contract
version — and the label provenance: ``source`` says whether the heuristics or
the labeler decided the case, ``reason`` is the fixed heuristic reason or the
labeler's one-line justification, ``flag_reason`` records what sent the case to
adjudication, and ``model`` names the labeler for adjudicated cases. The build
fails rather than emitting a partial join: the input Interactions must have
unique ids, the labels must cover exactly them, and every label must carry the
provenance its source requires, with no missing, unknown, duplicate, or
provenance-inconsistent entries. The report counts the sample per category and
per source split, plus the retrieval-eligible volume.
"""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from query import adjudication, closure
from query.interactions import (
    Interaction,
    InteractionsError,
    interaction_to_json,
    parse_interaction_record,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INTERACTIONS_PATH = closure.DEFAULT_INPUT_PATH
DEFAULT_LABELS_PATH = adjudication.DEFAULT_ADJUDICATED_LABELS_PATH
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "resolution-dataset.jsonl"

# The dataset's schema/contract version. Bump when the record shape or the
# meaning of a field changes, so consumers can tell snapshots apart.
DATASET_VERSION = 1
RETRIEVAL_ELIGIBLE_LABEL: closure.Label = "resolved"


class ResolutionDatasetError(Exception):
    """Raised when the resolution dataset cannot be built or read."""


@dataclass(frozen=True)
class ResolutionRecord:
    """One Interaction's final label plus the content retrieval draws from."""

    interaction_id: int
    label: closure.Label
    source: Literal["heuristic", "labeler"]
    reason: str
    flag_reason: str | None
    model: str | None
    retrieval_eligible: bool
    interaction: Interaction
    dataset_version: int = DATASET_VERSION


@dataclass(frozen=True)
class SourceCounts:
    """The label distribution of one provenance source (heuristic/labeler)."""

    total: int
    resolved: int
    uncertain: int
    unresolved: int


@dataclass(frozen=True)
class ResolutionReport:
    """Counts of the labeled sample, per category and per source split."""

    dataset_version: int
    total: int
    resolved: int
    uncertain: int
    unresolved: int
    retrieval_eligible: int
    heuristic: SourceCounts
    labeler: SourceCounts
    models: tuple[str, ...]

    @property
    def retrieval_share(self) -> float:
        return self.retrieval_eligible / self.total if self.total else 0.0


def build_resolution_dataset(
    interactions: Sequence[Interaction],
    labels: Sequence[adjudication.AdjudicatedLabel],
) -> tuple[tuple[ResolutionRecord, ...], ResolutionReport]:
    """Join final labels onto Interactions and mark the retrieval-eligible ones.

    Returns one record per Interaction, in input order, and a report of the
    label distribution per category and per source split. The input
    Interactions must have unique ids, and the labels must cover exactly
    them; any missing, unknown, duplicate, or structurally invalid label
    fails the build instead of emitting a partial dataset.
    """
    by_id = _index_labels(interactions, labels)
    records: list[ResolutionRecord] = []
    counts = {"resolved": 0, "uncertain": 0, "unresolved": 0}
    source_counts = {
        "heuristic": {"resolved": 0, "uncertain": 0, "unresolved": 0},
        "labeler": {"resolved": 0, "uncertain": 0, "unresolved": 0},
    }
    models: list[str] = []
    for interaction in interactions:
        label = by_id[interaction.interaction_id]
        records.append(
            ResolutionRecord(
                interaction_id=interaction.interaction_id,
                label=label.label,
                source=label.source,
                reason=label.reason,
                flag_reason=label.flag_reason,
                model=label.model,
                retrieval_eligible=label.label == RETRIEVAL_ELIGIBLE_LABEL,
                interaction=interaction,
            )
        )
        counts[label.label] += 1
        source_counts[label.source][label.label] += 1
        if label.model and label.model not in models:
            models.append(label.model)
    report = ResolutionReport(
        dataset_version=DATASET_VERSION,
        total=len(interactions),
        resolved=counts["resolved"],
        uncertain=counts["uncertain"],
        unresolved=counts["unresolved"],
        retrieval_eligible=counts[RETRIEVAL_ELIGIBLE_LABEL],
        heuristic=_source_counts(source_counts["heuristic"]),
        labeler=_source_counts(source_counts["labeler"]),
        models=tuple(models),
    )
    return tuple(records), report


def _source_counts(counts: Mapping[str, int]) -> SourceCounts:
    return SourceCounts(
        total=sum(counts.values()),
        resolved=counts["resolved"],
        uncertain=counts["uncertain"],
        unresolved=counts["unresolved"],
    )


def _index_labels(
    interactions: Sequence[Interaction],
    labels: Sequence[adjudication.AdjudicatedLabel],
) -> dict[int, adjudication.AdjudicatedLabel]:
    """Index final labels by Interaction id and verify the join is exact.

    Every label is checked against the complete label contract the dataset
    reader enforces (``adjudication.adjudicated_label_error``), so the build
    cannot stage a dataset its own reader would reject.
    """
    by_id: dict[int, adjudication.AdjudicatedLabel] = {}
    for label in labels:
        contract_error = adjudication.adjudicated_label_error(label)
        if contract_error is not None:
            raise ResolutionDatasetError(
                f"invalid label for interaction {label.interaction_id}: "
                f"{contract_error}"
            )
        if label.interaction_id in by_id:
            raise ResolutionDatasetError(
                f"duplicate label for interaction {label.interaction_id}"
            )
        by_id[label.interaction_id] = label
    input_ids: set[int] = set()
    for interaction in interactions:
        if interaction.interaction_id in input_ids:
            raise ResolutionDatasetError(
                f"duplicate interaction_id {interaction.interaction_id} in the input"
            )
        input_ids.add(interaction.interaction_id)
    missing = sorted(input_ids - set(by_id))
    if missing:
        sample = ", ".join(str(interaction_id) for interaction_id in missing[:5])
        raise ResolutionDatasetError(
            f"resolution labels missing for {len(missing)} interactions (e.g. {sample})"
        )
    unknown = sorted(set(by_id) - input_ids)
    if unknown:
        sample = ", ".join(str(interaction_id) for interaction_id in unknown[:5])
        raise ResolutionDatasetError(
            f"resolution labels reference {len(unknown)} unknown interactions "
            f"(e.g. {sample})"
        )
    return by_id


def record_to_json(record: ResolutionRecord) -> dict:
    """Serialize a ResolutionRecord to a JSON-compatible mapping."""
    content = interaction_to_json(record.interaction)
    return {
        "interaction_id": record.interaction_id,
        "dataset_version": record.dataset_version,
        "label": record.label,
        "retrieval_eligible": record.retrieval_eligible,
        "source": record.source,
        "reason": record.reason,
        "flag_reason": record.flag_reason,
        "model": record.model,
        "customer_id": content["customer_id"],
        "brand_id": content["brand_id"],
        "turns": content["turns"],
    }


def stage_resolution_dataset_jsonl(
    records: tuple[ResolutionRecord, ...], output_path: Path | str
) -> Path:
    """Write the resolution dataset to a temporary sibling, ready to be swapped in.

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
            for record in records:
                handle.write(json.dumps(record_to_json(record)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_resolution_dataset_jsonl(
    input_path: Path | str = DEFAULT_DATASET_PATH,
) -> tuple[ResolutionRecord, ...]:
    """Read the resolution dataset back from the JSON Lines format written above.

    Inverse of :func:`stage_resolution_dataset_jsonl`: parses every record and
    validates its structure and unique interaction ids, so retrieval never sees
    a malformed dataset. Every record must carry a supported ``dataset_version``,
    a final label with matching provenance, and a ``retrieval_eligible`` flag
    that agrees with the label. Returns the records in file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ResolutionDatasetError(
            f"resolution dataset JSONL does not exist: {input_path}"
        )
    records = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResolutionDatasetError(
                    f"malformed JSON on {location}: {exc}"
                ) from exc
            record = _parse_record(decoded, location)
            if record.interaction_id in seen_ids:
                raise ResolutionDatasetError(
                    f"duplicate interaction_id {record.interaction_id} on {location}"
                )
            seen_ids.add(record.interaction_id)
            records.append(record)
    return tuple(records)


def _parse_record(record: object, location: str) -> ResolutionRecord:
    if not isinstance(record, dict):
        raise ResolutionDatasetError(
            f"malformed resolution record on {location}: expected an object"
        )
    dataset_version = record.get("dataset_version")
    if type(dataset_version) is not int or dataset_version != DATASET_VERSION:
        raise ResolutionDatasetError(
            f"malformed resolution record on {location}: unsupported "
            f"dataset_version {dataset_version!r} (expected {DATASET_VERSION})"
        )
    try:
        label = adjudication.parse_adjudicated_label_record(record, location)
        interaction = parse_interaction_record(record, location)
    except (
        adjudication.AdjudicationError,
        InteractionsError,
    ) as exc:
        raise ResolutionDatasetError(str(exc)) from exc
    retrieval_eligible = record.get("retrieval_eligible")
    if not isinstance(retrieval_eligible, bool):
        raise ResolutionDatasetError(
            f"malformed resolution record on {location}: invalid retrieval_eligible"
        )
    if retrieval_eligible != (label.label == RETRIEVAL_ELIGIBLE_LABEL):
        raise ResolutionDatasetError(
            f"malformed resolution record on {location}: retrieval_eligible "
            "must be true exactly for resolved labels"
        )
    return ResolutionRecord(
        interaction_id=label.interaction_id,
        label=label.label,
        source=label.source,
        reason=label.reason,
        flag_reason=label.flag_reason,
        model=label.model,
        retrieval_eligible=retrieval_eligible,
        interaction=interaction,
        dataset_version=dataset_version,
    )

"""Draw a deterministic sample of Interactions and split it into two pools.

The sampling stage follows the English filter (ADR-0004): it reads the
normalized Interactions JSONL and draws the development sample of ~4K
Interactions, split into the RAG pool (~3K) and the holdout (~1K) reserved for
the Golden Set (decisions 5 and 6 in ``docs/decisions.md``).

Selection is a rank, not a roll of the dice: every Interaction is ranked by a
SHA-256 of the seed and its interaction id, and the top ``sample_size`` are
kept. Because the ranking key depends only on the seed and the Interaction's
own id, the same seed selects the same sample on any machine, under any Python
version, and regardless of the order the input JSONL arrives in. The first
``holdout_size`` of the ranked sample become the holdout; the rest form the RAG
pool (ADR-0005).

The report states the requested and actual sizes, so an input smaller than the
requested sample is reported honestly instead of silently changing shape: the
sample then covers everything and the holdout scales down with the same ratio.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from query.interactions import Interaction

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "interactions-en.jsonl"
DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 4000
DEFAULT_HOLDOUT_SIZE = 1000


class SamplingError(Exception):
    """Raised when the sample cannot be drawn as requested."""


@dataclass(frozen=True)
class SampleReport:
    """Counts of the sample/split stage, for honest reporting."""

    seed: int
    input_total: int
    sample_size: int
    sampled: int
    rag_pool: int
    holdout: int

    @property
    def sample_rate(self) -> float:
        return self.sampled / self.input_total if self.input_total else 0.0


def sample_and_split(
    interactions: Sequence[Interaction],
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
) -> tuple[tuple[Interaction, ...], tuple[Interaction, ...], SampleReport]:
    """Draw the sample and split it into a RAG pool and a holdout.

    Returns ``(rag_pool, holdout, report)``: the two pools ordered by
    interaction id and a report of the requested and actual sizes. The pools
    are disjoint and together form the sample.
    """
    if sample_size < 1:
        raise SamplingError(f"sample size must be at least 1, got {sample_size}")
    if not 0 <= holdout_size < sample_size:
        raise SamplingError(
            f"holdout size must be at least 0 and smaller than the sample size "
            f"{sample_size}, got {holdout_size}"
        )

    ranked = sorted(
        interactions,
        key=lambda interaction: (
            _rank_key(seed, interaction.interaction_id),
            interaction.interaction_id,
        ),
    )
    sampled = ranked[:sample_size]
    if len(sampled) == sample_size:
        holdout_count = holdout_size
    else:
        # Fewer Interactions than requested: sample them all and scale the
        # holdout down with the same ratio (floored), so the split still
        # reflects the requested RAG/holdout proportions.
        holdout_count = len(sampled) * holdout_size // sample_size

    holdout = tuple(sorted(sampled[:holdout_count], key=_by_interaction_id))
    rag_pool = tuple(sorted(sampled[holdout_count:], key=_by_interaction_id))
    report = SampleReport(
        seed=seed,
        input_total=len(interactions),
        sample_size=sample_size,
        sampled=len(sampled),
        rag_pool=len(rag_pool),
        holdout=len(holdout),
    )
    return rag_pool, holdout, report


def _rank_key(seed: int, interaction_id: int) -> bytes:
    """Stable ranking key: the same seed and id rank the same everywhere."""
    return hashlib.sha256(f"{seed}:{interaction_id}".encode()).digest()


def _by_interaction_id(interaction: Interaction) -> int:
    return interaction.interaction_id

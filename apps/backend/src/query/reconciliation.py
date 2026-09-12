"""Re-check the final taxonomy's reconciliation against the discovery clusters.

Ticket 11 reconciled the seed taxonomy with discovery and recorded the outcome
in ``docs/intent-taxonomy.md``: which seed intents were kept, merged, or
dropped, and which clusters flagged ``new`` were promoted. This module turns
that record into a checked claim. It routes every cluster of the discovery
artifact through the recorded decisions to a final intent — mapped clusters
through their seed intent's decision, promoted clusters to their new intent,
junk to ``other`` — and fails when the artifact no longer matches the document:
a dropped seed intent may not receive mapped clusters, a ``new`` cluster must
have a decision, and every support intent must still receive cluster evidence.

The report counts clusters and messages per final intent, so the taxonomy's
evidence stays visible next to the decisions that produced it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from query import discovery, taxonomy

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "intent-reconciliation-report.json"


class ReconciliationError(Exception):
    """Raised when the clusters no longer route through the recorded decisions."""


@dataclass(frozen=True)
class IntentCoverage:
    """The clusters and messages one final intent receives."""

    clusters: int
    messages: int


@dataclass(frozen=True)
class ReconciliationReport:
    """Per-intent coverage of the recorded discovery run."""

    taxonomy_version: int
    clusters: int
    messages: int
    per_intent: dict[str, IntentCoverage]

    @property
    def covered_messages(self) -> int:
        """Messages routed to a support intent rather than to ``other``."""
        return sum(
            coverage.messages
            for intent_id, coverage in self.per_intent.items()
            if intent_id != taxonomy.OTHER_INTENT_ID
        )

    @property
    def covered_share(self) -> float:
        return self.covered_messages / self.messages if self.messages else 0.0


def reconcile_intents(
    clusters: Sequence[discovery.IntentCluster],
    seeds: Sequence[taxonomy.SeedIntent],
    final: taxonomy.FinalTaxonomy,
) -> ReconciliationReport:
    """Route every cluster through the recorded decisions and count coverage.

    Returns the clusters and messages each final intent receives, with intents
    in taxonomy document order. Raises :class:`ReconciliationError` when the
    decisions do not cover the seed intents or the ``new`` clusters exactly,
    when a dropped seed intent still receives mapped clusters, when a final
    intent would be left without cluster evidence, or when a decision points
    at an intent the taxonomy does not define.
    """
    intent_ids = set(final.intent_ids)
    if taxonomy.OTHER_INTENT_ID not in intent_ids:
        raise ReconciliationError(
            "final taxonomy has no 'other' fallback for non-support clusters"
        )
    decisions = _seed_decisions_by_id(seeds, final, intent_ids)
    new_decisions = _new_theme_decisions_by_cluster(clusters, final, intent_ids)
    coverage = {
        intent.intent_id: IntentCoverage(clusters=0, messages=0)
        for intent in final.intents
    }
    for cluster in clusters:
        target = _route_cluster(cluster, decisions, new_decisions)
        counts = coverage[target]
        coverage[target] = IntentCoverage(
            clusters=counts.clusters + 1, messages=counts.messages + cluster.size
        )
    _require_cluster_evidence(final, coverage)
    return ReconciliationReport(
        taxonomy_version=final.version,
        clusters=len(clusters),
        messages=sum(cluster.size for cluster in clusters),
        per_intent=coverage,
    )


def _seed_decisions_by_id(
    seeds: Sequence[taxonomy.SeedIntent],
    final: taxonomy.FinalTaxonomy,
    intent_ids: set[str],
) -> dict[str, taxonomy.SeedDecision]:
    """Index the decisions by seed intent, requiring exact coverage of the seed."""
    support = [
        intent.intent_id
        for intent in seeds
        if intent.intent_id != taxonomy.OTHER_INTENT_ID
    ]
    by_id: dict[str, taxonomy.SeedDecision] = {}
    for decision in final.seed_decisions:
        if decision.seed_intent in by_id:
            raise ReconciliationError(
                f"duplicate seed decision for {decision.seed_intent!r}"
            )
        by_id[decision.seed_intent] = decision
    missing = [seed_id for seed_id in support if seed_id not in by_id]
    if missing:
        raise ReconciliationError(f"seed intent {missing[0]!r} has no decision")
    unknown = [seed_id for seed_id in by_id if seed_id not in support]
    if unknown:
        raise ReconciliationError(
            f"decision for unknown seed intent {unknown[0]!r}"
        )
    for decision in by_id.values():
        error = taxonomy.seed_decision_error(decision, intent_ids)
        if error is not None:
            raise ReconciliationError(
                f"seed decision for {decision.seed_intent!r}: {error}"
            )
    return by_id


def _new_theme_decisions_by_cluster(
    clusters: Sequence[discovery.IntentCluster],
    final: taxonomy.FinalTaxonomy,
    intent_ids: set[str],
) -> dict[int, taxonomy.NewThemeDecision]:
    """Index the theme decisions by cluster, requiring one per ``new`` cluster."""
    flagged = {
        cluster.cluster_id
        for cluster in clusters
        if cluster.mapping == discovery.MAPPING_NEW
    }
    by_id: dict[int, taxonomy.NewThemeDecision] = {}
    for decision in final.new_theme_decisions:
        if decision.cluster_id in by_id:
            raise ReconciliationError(
                f"duplicate new-theme decision for cluster {decision.cluster_id}"
            )
        by_id[decision.cluster_id] = decision
    missing = sorted(flagged - set(by_id))
    if missing:
        raise ReconciliationError(
            f"new cluster {missing[0]} has no recorded decision"
        )
    unknown = sorted(set(by_id) - flagged)
    if unknown:
        raise ReconciliationError(
            f"decision recorded for cluster {unknown[0]}, which discovery did "
            f"not flag new"
        )
    for decision in by_id.values():
        error = taxonomy.new_theme_decision_error(decision, intent_ids)
        if error is not None:
            raise ReconciliationError(
                f"new-theme decision for cluster {decision.cluster_id}: {error}"
            )
    return by_id


def _route_cluster(
    cluster: discovery.IntentCluster,
    decisions: dict[str, taxonomy.SeedDecision],
    new_decisions: dict[int, taxonomy.NewThemeDecision],
) -> str:
    """Follow one cluster's recorded path to its final intent."""
    if cluster.mapping == discovery.MAPPING_JUNK:
        return taxonomy.OTHER_INTENT_ID
    if cluster.mapping == discovery.MAPPING_NEW:
        decision = new_decisions[cluster.cluster_id]
        return decision.final_intent or taxonomy.OTHER_INTENT_ID
    decision = decisions.get(cluster.mapping)
    if decision is None:
        raise ReconciliationError(
            f"cluster {cluster.cluster_id} maps to unknown seed intent "
            f"{cluster.mapping!r}"
        )
    if decision.decision == taxonomy.DROPPED:
        raise ReconciliationError(
            f"cluster {cluster.cluster_id} maps to dropped seed intent "
            f"{cluster.mapping!r}"
        )
    return decision.final_intent


def _require_cluster_evidence(
    final: taxonomy.FinalTaxonomy, coverage: dict[str, IntentCoverage]
) -> None:
    """Fail when a support intent ends up with no cluster behind it."""
    for intent in final.intents:
        if intent.intent_id == taxonomy.OTHER_INTENT_ID:
            continue
        if coverage[intent.intent_id].clusters == 0:
            raise ReconciliationError(
                f"final intent {intent.intent_id!r} has no cluster evidence"
            )

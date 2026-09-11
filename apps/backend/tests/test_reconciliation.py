import pytest

from query import discovery, reconciliation, taxonomy


def _seed(intent_id):
    return taxonomy.SeedIntent(
        intent_id=intent_id,
        definition=f"{intent_id} issues.",
        representative_message=f"{intent_id} message",
    )


def _cluster(cluster_id, size, mapping):
    return discovery.IntentCluster(
        cluster_id=cluster_id,
        size=size,
        mapping=mapping,
        justification="test justification",
        candidates=(
            discovery.SeedCandidate(intent_id="account", similarity=0.5),
        ),
        examples=(
            discovery.ClusterExample(
                interaction_id=1, message="example message", similarity=0.5
            ),
        ),
    )


def _final(*intent_ids, version=1, seed_decisions=(), new_theme_decisions=()):
    return taxonomy.FinalTaxonomy(
        version=version,
        intents=tuple(
            taxonomy.FinalIntent(
                intent_id=intent_id,
                definition=f"{intent_id} issues.",
                origin="seed",
                examples=(f"{intent_id} example",),
            )
            for intent_id in intent_ids
        ),
        seed_decisions=tuple(seed_decisions),
        new_theme_decisions=tuple(new_theme_decisions),
    )


def _base_seeds():
    return [
        _seed("playback"),
        _seed("account_access"),
        _seed("account_admin"),
        _seed(taxonomy.OTHER_INTENT_ID),
    ]


def _base_final():
    return _final(
        "playback",
        "account",
        "presale_codes",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("account_access", taxonomy.MERGED, "account"),
            taxonomy.SeedDecision("account_admin", taxonomy.MERGED, "account"),
        ],
        new_theme_decisions=[
            taxonomy.NewThemeDecision(2, taxonomy.PROMOTED, "presale_codes"),
        ],
    )


def _base_clusters():
    return [
        _cluster(0, 10, "playback"),
        _cluster(1, 5, "account_access"),
        _cluster(2, 4, discovery.MAPPING_NEW),
        _cluster(3, 6, discovery.MAPPING_JUNK),
    ]


def test_routes_every_cluster_through_the_recorded_decisions():
    report = reconciliation.reconcile_intents(
        _base_clusters(), _base_seeds(), _base_final()
    )

    assert report.taxonomy_version == 1
    assert report.clusters == 4
    assert report.messages == 25
    assert report.per_intent["playback"] == reconciliation.IntentCoverage(1, 10)
    assert report.per_intent["account"] == reconciliation.IntentCoverage(1, 5)
    assert report.per_intent["presale_codes"] == reconciliation.IntentCoverage(1, 4)
    assert report.per_intent[taxonomy.OTHER_INTENT_ID] == (
        reconciliation.IntentCoverage(1, 6)
    )
    assert report.covered_messages == 19
    assert report.covered_share == pytest.approx(19 / 25)


def test_taxonomy_without_other_fallback_raises():
    seeds = [_seed("playback")]
    final = _final(
        "playback",
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
        ],
    )
    clusters = [_cluster(0, 3, "playback"), _cluster(1, 2, "junk")]

    with pytest.raises(
        reconciliation.ReconciliationError, match="no 'other' fallback"
    ):
        reconciliation.reconcile_intents(clusters, seeds, final)


def test_dropped_new_theme_routes_to_other():
    seeds = [_seed("playback"), _seed(taxonomy.OTHER_INTENT_ID)]
    final = _final(
        "playback",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
        ],
        new_theme_decisions=[
            taxonomy.NewThemeDecision(1, taxonomy.DROPPED, None),
        ],
    )
    clusters = [_cluster(0, 3, "playback"), _cluster(1, 2, "new")]

    report = reconciliation.reconcile_intents(clusters, seeds, final)

    assert report.per_intent[taxonomy.OTHER_INTENT_ID] == (
        reconciliation.IntentCoverage(1, 2)
    )


def test_dropped_seed_intent_with_mapped_clusters_raises():
    seeds = [_seed("playback"), _seed("old_theme"), _seed(taxonomy.OTHER_INTENT_ID)]
    final = _final(
        "playback",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("old_theme", taxonomy.DROPPED, None),
        ],
    )
    clusters = [_cluster(0, 1, "playback"), _cluster(1, 1, "old_theme")]

    with pytest.raises(
        reconciliation.ReconciliationError, match="dropped seed intent"
    ):
        reconciliation.reconcile_intents(clusters, seeds, final)


def test_missing_seed_decision_raises():
    seeds = [_seed("playback"), _seed(taxonomy.OTHER_INTENT_ID)]
    final = _final("playback", taxonomy.OTHER_INTENT_ID)

    with pytest.raises(
        reconciliation.ReconciliationError, match="'playback' has no decision"
    ):
        reconciliation.reconcile_intents(_base_clusters(), seeds, final)


def test_decision_for_unknown_seed_intent_raises():
    seeds = [_seed("playback"), _seed(taxonomy.OTHER_INTENT_ID)]
    final = _final(
        "playback",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("ghost", taxonomy.KEPT, "playback"),
        ],
    )

    with pytest.raises(
        reconciliation.ReconciliationError, match="unknown seed intent 'ghost'"
    ):
        reconciliation.reconcile_intents(_base_clusters(), seeds, final)


def test_new_cluster_without_decision_raises():
    final = _final(
        "playback",
        "account",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("account_access", taxonomy.MERGED, "account"),
            taxonomy.SeedDecision("account_admin", taxonomy.MERGED, "account"),
        ],
    )

    with pytest.raises(
        reconciliation.ReconciliationError, match="new cluster 2 has no recorded decision"
    ):
        reconciliation.reconcile_intents(_base_clusters(), _base_seeds(), final)


def test_decision_for_cluster_not_flagged_new_raises():
    final = _final(
        "playback",
        "account",
        "presale_codes",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("account_access", taxonomy.MERGED, "account"),
            taxonomy.SeedDecision("account_admin", taxonomy.MERGED, "account"),
        ],
        new_theme_decisions=[
            taxonomy.NewThemeDecision(2, taxonomy.PROMOTED, "presale_codes"),
            taxonomy.NewThemeDecision(9, taxonomy.PROMOTED, "presale_codes"),
        ],
    )

    with pytest.raises(
        reconciliation.ReconciliationError, match="cluster 9, which discovery did not flag new"
    ):
        reconciliation.reconcile_intents(_base_clusters(), _base_seeds(), final)


def test_promoted_cluster_with_unknown_target_raises():
    final = _final(
        "playback",
        "account",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("account_access", taxonomy.MERGED, "account"),
            taxonomy.SeedDecision("account_admin", taxonomy.MERGED, "account"),
        ],
        new_theme_decisions=[
            taxonomy.NewThemeDecision(2, taxonomy.PROMOTED, "ghost"),
        ],
    )

    with pytest.raises(
        reconciliation.ReconciliationError, match="unknown intent 'ghost'"
    ):
        reconciliation.reconcile_intents(_base_clusters(), _base_seeds(), final)


def test_support_intent_without_cluster_evidence_raises():
    final = _final(
        "playback",
        "account",
        "presale_codes",
        "unused",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("account_access", taxonomy.MERGED, "account"),
            taxonomy.SeedDecision("account_admin", taxonomy.MERGED, "account"),
        ],
        new_theme_decisions=[
            taxonomy.NewThemeDecision(2, taxonomy.PROMOTED, "presale_codes"),
        ],
    )

    with pytest.raises(
        reconciliation.ReconciliationError, match="'unused' has no cluster evidence"
    ):
        reconciliation.reconcile_intents(_base_clusters(), _base_seeds(), final)


def test_cluster_mapped_to_unknown_seed_intent_raises():
    clusters = [*_base_clusters(), _cluster(4, 1, "mystery")]

    with pytest.raises(
        reconciliation.ReconciliationError, match="unknown seed intent 'mystery'"
    ):
        reconciliation.reconcile_intents(clusters, _base_seeds(), _base_final())


def test_duplicate_seed_decision_raises():
    final = _final(
        "playback",
        "account",
        "presale_codes",
        taxonomy.OTHER_INTENT_ID,
        seed_decisions=[
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("playback", taxonomy.KEPT, "playback"),
            taxonomy.SeedDecision("account_access", taxonomy.MERGED, "account"),
            taxonomy.SeedDecision("account_admin", taxonomy.MERGED, "account"),
        ],
        new_theme_decisions=[
            taxonomy.NewThemeDecision(2, taxonomy.PROMOTED, "presale_codes"),
        ],
    )

    with pytest.raises(
        reconciliation.ReconciliationError, match="duplicate seed decision"
    ):
        reconciliation.reconcile_intents(_base_clusters(), _base_seeds(), final)


def test_committed_documents_reconcile():
    seeds = taxonomy.read_seed_taxonomy()
    final = taxonomy.read_final_taxonomy()
    clusters = []
    next_id = 100
    for decision in final.seed_decisions:
        if decision.decision == taxonomy.DROPPED:
            continue
        clusters.append(_cluster(next_id, 10, decision.seed_intent))
        next_id += 1
    for theme in final.new_theme_decisions:
        clusters.append(_cluster(theme.cluster_id, 5, discovery.MAPPING_NEW))
    clusters.append(_cluster(next_id, 7, discovery.MAPPING_JUNK))

    report = reconciliation.reconcile_intents(clusters, seeds, final)

    assert report.taxonomy_version == final.version
    assert report.clusters == len(clusters)
    assert report.messages == 12 * 10 + 5 + 7
    assert report.covered_messages == 12 * 10 + 5
    assert report.per_intent[taxonomy.OTHER_INTENT_ID] == (
        reconciliation.IntentCoverage(1, 7)
    )
    assert all(
        counts.clusters >= 1
        for intent_id, counts in report.per_intent.items()
        if intent_id != taxonomy.OTHER_INTENT_ID
    )

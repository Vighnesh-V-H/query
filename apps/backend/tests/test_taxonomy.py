import pytest

from query import taxonomy


def _document(*intents, representatives=None, extra=""):
    lines = [
        "# Candidate taxonomy",
        "",
        "| # | Intent id | One-line definition | Rough signal |",
        "|---|-----------|---------------------|--------------|",
    ]
    for number, (intent_id, definition) in enumerate(intents, 1):
        lines.append(f"| {number} | `{intent_id}` | {definition} | ~1% |")
    lines.append("")
    if representatives is not None:
        lines.append("## Representative messages")
        lines.append("")
        for intent_id, message in representatives:
            lines.append(f"- `{intent_id}` [123]: {message}")
    return "\n".join(lines) + "\n" + extra


def _write(tmp_path, text):
    path = tmp_path / "taxonomy.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_the_committed_seed_taxonomy():
    intents = taxonomy.read_seed_taxonomy()

    ids = [intent.intent_id for intent in intents]
    assert len(intents) == 14
    assert ids[0] == "account_access"
    assert taxonomy.OTHER_INTENT_ID in ids
    assert len(ids) == len(set(ids))
    for intent in intents:
        assert intent.definition
        assert intent.representative_message


def test_reads_intents_and_representatives_in_document_order(tmp_path):
    path = _write(
        tmp_path,
        _document(
            ("playback", "Songs that will not play."),
            ("billing", "Charges and refunds."),
            representatives=[
                ("playback", "Still not playing music"),
                ("billing", "I was charged twice"),
            ],
        ),
    )

    intents = taxonomy.read_seed_taxonomy(path)

    assert [intent.intent_id for intent in intents] == ["playback", "billing"]
    assert intents[0].definition == "Songs that will not play."
    assert intents[0].representative_message == "Still not playing music"
    assert intents[1].representative_message == "I was charged twice"


def test_missing_file_raises(tmp_path):
    with pytest.raises(taxonomy.TaxonomyError, match="does not exist"):
        taxonomy.read_seed_taxonomy(tmp_path / "missing.md")


def test_document_without_intent_rows_raises(tmp_path):
    path = _write(tmp_path, "# Taxonomy\n\nNo table here.\n")

    with pytest.raises(taxonomy.TaxonomyError, match="no intent rows"):
        taxonomy.read_seed_taxonomy(path)


def test_duplicate_intent_id_raises(tmp_path):
    path = _write(
        tmp_path,
        _document(
            ("playback", "Songs that will not play."),
            ("playback", "Duplicate row."),
            representatives=[("playback", "message")],
        ),
    )

    with pytest.raises(taxonomy.TaxonomyError, match="duplicate intent id 'playback'"):
        taxonomy.read_seed_taxonomy(path)


def test_empty_definition_raises(tmp_path):
    path = _write(
        tmp_path,
        "| # | Intent id | One-line definition | Rough signal |\n"
        "|---|-----------|---------------------|--------------|\n"
        "| 1 | `playback` |  | ~1% |\n"
        "- `playback` [1]: message\n",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="empty definition"):
        taxonomy.read_seed_taxonomy(path)


def test_non_backticked_id_raises(tmp_path):
    path = _write(
        tmp_path,
        "| # | Intent id | One-line definition | Rough signal |\n"
        "|---|-----------|---------------------|--------------|\n"
        "| 1 | playback | Songs that will not play. | ~1% |\n",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="invalid intent id"):
        taxonomy.read_seed_taxonomy(path)


def test_intent_without_representative_raises(tmp_path):
    path = _write(
        tmp_path,
        _document(
            ("playback", "Songs that will not play."),
            ("billing", "Charges and refunds."),
            representatives=[("playback", "message")],
        ),
    )

    with pytest.raises(
        taxonomy.TaxonomyError, match="'billing' has no representative message"
    ):
        taxonomy.read_seed_taxonomy(path)


def test_unknown_representative_intent_raises(tmp_path):
    path = _write(
        tmp_path,
        _document(
            ("playback", "Songs that will not play."),
            representatives=[("billing", "message")],
        ),
    )

    with pytest.raises(
        taxonomy.TaxonomyError, match="references unknown intent 'billing'"
    ):
        taxonomy.read_seed_taxonomy(path)


def test_duplicate_representative_message_raises(tmp_path):
    path = _write(
        tmp_path,
        _document(
            ("playback", "Songs that will not play."),
            representatives=[
                ("playback", "first message"),
                ("playback", "second message"),
            ],
        ),
    )

    with pytest.raises(taxonomy.TaxonomyError, match="duplicate representative"):
        taxonomy.read_seed_taxonomy(path)


_BASE_FINAL_INTENTS = [
    ("alpha", "Alpha issues."),
    ("beta", "Beta issues."),
    ("gamma", "Gamma issues."),
    ("delta", "Delta issues."),
    ("epsilon", "Epsilon issues."),
    ("zeta", "Zeta issues."),
    ("eta", "Eta issues."),
    ("theta", "Theta issues."),
]


def _final_document(
    *intents,
    version=1,
    examples=None,
    seed_decisions=None,
    new_themes=None,
):
    lines = [
        "# Intent taxonomy",
        "",
        f"**Version:** {version}",
        "",
        "| # | Intent id | Definition | Origin |",
        "|---|-----------|------------|--------|",
    ]
    for number, (intent_id, definition) in enumerate(intents, 1):
        lines.append(f"| {number} | `{intent_id}` | {definition} | seed |")
    lines.extend(["", "## Representative examples", ""])
    for intent_id, message in examples or []:
        lines.append(f"- `{intent_id}` [123]: {message}")
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            "| Seed intent | Decision | Final intent | Evidence |",
            "|-------------|----------|--------------|----------|",
        ]
    )
    for seed_intent, decision, target in seed_decisions or []:
        target_cell = f"`{target}`" if target else "-"
        lines.append(
            f"| `{seed_intent}` | {decision} | {target_cell} | evidence |"
        )
    lines.extend(
        [
            "",
            "| Cluster | Decision | Final intent | Evidence |",
            "|---------|----------|--------------|----------|",
        ]
    )
    for cluster_id, decision, target in new_themes or []:
        target_cell = f"`{target}`" if target else "-"
        lines.append(f"| {cluster_id} | {decision} | {target_cell} | evidence |")
    return "\n".join(lines) + "\n"


def _valid_final_document():
    return _final_document(
        *_BASE_FINAL_INTENTS,
        examples=[
            (intent_id, f"{intent_id} example")
            for intent_id, _ in _BASE_FINAL_INTENTS
        ],
        seed_decisions=[
            ("alpha", "merged", "beta"),
            ("beta", "kept", "beta"),
        ],
        new_themes=[(3, "promoted", "gamma")],
    )


def test_reads_the_committed_final_taxonomy():
    final = taxonomy.read_final_taxonomy()

    assert final.version == 1
    ids = final.intent_ids
    assert len(ids) == 13
    assert len(ids) == len(set(ids))
    assert taxonomy.OTHER_INTENT_ID in ids
    assert "account" in ids
    assert "account_access" not in ids
    assert "devices_connectivity" not in ids
    assert "presale_codes" in ids
    for intent in final.intents:
        assert intent.definition
        assert intent.origin
        assert intent.examples
    assert {decision.seed_intent for decision in final.seed_decisions} == {
        intent_id
        for intent_id in (
            "account_access",
            "account_admin",
            "billing_payment",
            "subscription_plans",
            "family_plan",
            "playback",
            "library_playlists",
            "downloads_offline",
            "devices_connectivity",
            "app_technical",
            "content_availability",
            "market_availability",
            "feature_feedback",
        )
    }
    assert final.new_theme_decisions == (
        taxonomy.NewThemeDecision(
            cluster_id=1,
            decision=taxonomy.PROMOTED,
            final_intent="presale_codes",
        ),
    )


def test_reads_synthetic_final_taxonomy(tmp_path):
    path = _write(tmp_path, _valid_final_document())

    final = taxonomy.read_final_taxonomy(path)

    assert final.version == 1
    assert final.intent_ids == tuple(
        intent_id for intent_id, _ in _BASE_FINAL_INTENTS
    )
    assert final.intents[0].examples == ("alpha example",)
    assert final.intents[0].origin == "seed"
    assert final.seed_decisions[0] == taxonomy.SeedDecision(
        seed_intent="alpha",
        decision=taxonomy.MERGED,
        final_intent="beta",
    )
    assert final.new_theme_decisions == (
        taxonomy.NewThemeDecision(3, taxonomy.PROMOTED, "gamma"),
    )


def test_final_taxonomy_missing_file_raises(tmp_path):
    with pytest.raises(taxonomy.TaxonomyError, match="does not exist"):
        taxonomy.read_final_taxonomy(tmp_path / "missing.md")


def test_final_taxonomy_requires_one_version_line(tmp_path):
    path = _write(
        tmp_path,
        _valid_final_document().replace("**Version:** 1\n", ""),
    )

    with pytest.raises(taxonomy.TaxonomyError, match="version line"):
        taxonomy.read_final_taxonomy(path)


def test_final_taxonomy_rejects_invalid_version(tmp_path):
    document = _valid_final_document().replace("**Version:** 1", "**Version:** 0")

    with pytest.raises(taxonomy.TaxonomyError, match="invalid taxonomy version"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_too_few_intents(tmp_path):
    path = _write(tmp_path, _final_document(*_BASE_FINAL_INTENTS[:7]))

    with pytest.raises(taxonomy.TaxonomyError, match="expected 8-15"):
        taxonomy.read_final_taxonomy(path)


def test_final_taxonomy_rejects_too_many_intents(tmp_path):
    intents = list(_BASE_FINAL_INTENTS) + [
        ("iota", "Iota issues."),
        ("kappa", "Kappa issues."),
        ("lambda", "Lambda issues."),
        ("mu", "Mu issues."),
        ("nu", "Nu issues."),
        ("xi", "Xi issues."),
        ("omicron", "Omicron issues."),
        ("pi", "Pi issues."),
        ("rho", "Rho issues."),
    ]

    with pytest.raises(taxonomy.TaxonomyError, match="expected 8-15"):
        taxonomy.read_final_taxonomy(_write(tmp_path, _final_document(*intents)))


def test_final_taxonomy_rejects_duplicate_intent_id(tmp_path):
    intents = [*_BASE_FINAL_INTENTS, ("alpha", "Duplicate row.")]

    with pytest.raises(taxonomy.TaxonomyError, match="duplicate intent id 'alpha'"):
        taxonomy.read_final_taxonomy(_write(tmp_path, _final_document(*intents)))


def test_final_taxonomy_rejects_empty_definition(tmp_path):
    intents = [
        (intent_id, " " if intent_id == "delta" else definition)
        for intent_id, definition in _BASE_FINAL_INTENTS
    ]

    with pytest.raises(taxonomy.TaxonomyError, match="empty definition"):
        taxonomy.read_final_taxonomy(_write(tmp_path, _final_document(*intents)))


def test_final_taxonomy_rejects_intent_without_example(tmp_path):
    document = _valid_final_document().replace(
        "- `gamma` [123]: gamma example\n", ""
    )

    with pytest.raises(taxonomy.TaxonomyError, match="'gamma' has no example"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_unknown_example_intent(tmp_path):
    document = _valid_final_document().replace(
        "- `alpha` [123]: alpha example\n",
        "- `alpha` [123]: alpha example\n- `omega` [9]: stray example\n",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="unknown intent 'omega'"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_duplicate_example_message(tmp_path):
    document = _valid_final_document().replace(
        "- `alpha` [123]: alpha example\n",
        "- `alpha` [123]: alpha example\n- `delta` [9]: alpha example\n",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="duplicate example message"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_kept_decision_with_new_id(tmp_path):
    document = _valid_final_document().replace(
        "| `beta` | kept | `beta` | evidence |",
        "| `beta` | kept | `gamma` | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="must keep its id"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_merged_decision_with_unknown_target(tmp_path):
    document = _valid_final_document().replace(
        "| `alpha` | merged | `beta` | evidence |",
        "| `alpha` | merged | `omega` | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="unknown intent 'omega'"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_invalid_seed_decision(tmp_path):
    document = _valid_final_document().replace(
        "| `alpha` | merged | `beta` | evidence |",
        "| `alpha` | renamed | `beta` | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="invalid seed decision"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_dropped_decision_with_target(tmp_path):
    document = _valid_final_document().replace(
        "| `alpha` | merged | `beta` | evidence |",
        "| `alpha` | dropped | `beta` | evidence |",
    )

    with pytest.raises(
        taxonomy.TaxonomyError, match="dropped seed intent must have no final intent"
    ):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_duplicate_seed_decision(tmp_path):
    document = _valid_final_document().replace(
        "| `beta` | kept | `beta` | evidence |",
        "| `beta` | kept | `beta` | evidence |\n| `beta` | kept | `beta` | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="duplicate seed decision"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_unknown_new_theme_target(tmp_path):
    document = _valid_final_document().replace(
        "| 3 | promoted | `gamma` | evidence |",
        "| 3 | promoted | `omega` | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="unknown intent 'omega'"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_promoted_theme_without_target(tmp_path):
    document = _valid_final_document().replace(
        "| 3 | promoted | `gamma` | evidence |",
        "| 3 | promoted | - | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="needs a final intent"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_rejects_invalid_new_theme_decision(tmp_path):
    document = _valid_final_document().replace(
        "| 3 | promoted | `gamma` | evidence |",
        "| 3 | merged | `gamma` | evidence |",
    )

    with pytest.raises(taxonomy.TaxonomyError, match="invalid new-theme decision"):
        taxonomy.read_final_taxonomy(_write(tmp_path, document))


def test_final_taxonomy_allows_empty_new_theme_table(tmp_path):
    document = _final_document(
        *_BASE_FINAL_INTENTS,
        examples=[
            (intent_id, f"{intent_id} example")
            for intent_id, _ in _BASE_FINAL_INTENTS
        ],
        seed_decisions=[("alpha", "kept", "alpha")],
        new_themes=[],
    )

    final = taxonomy.read_final_taxonomy(_write(tmp_path, document))

    assert final.new_theme_decisions == ()

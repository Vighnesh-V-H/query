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

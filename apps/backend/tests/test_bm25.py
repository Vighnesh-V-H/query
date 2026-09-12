import json
from datetime import datetime, timezone

import pytest

from query import adjudication
from query import bm25 as bm25_mod
from query import cli
from query import resolution
from query.interactions import Interaction, Turn

BRAND = "SpotifyCares"
HEURISTIC_REASON = "customer's final message acknowledges the issue is resolved"


def _turn(tweet_id, side, text, minute=0):
    return Turn(
        tweet_id=tweet_id,
        author_id="cust" if side == "customer" else BRAND,
        side=side,
        created_at=datetime(2017, 11, 1, 10, minute, tzinfo=timezone.utc),
        text=text,
    )


def _interaction(interaction_id, *turns):
    return Interaction(
        interaction_id=interaction_id,
        customer_id="cust",
        brand_id=BRAND,
        turns=tuple(turns),
    )


def _heuristic(interaction_id, label, reason=HEURISTIC_REASON):
    return adjudication.AdjudicatedLabel(
        interaction_id=interaction_id,
        label=label,
        source="heuristic",
        reason=reason,
        flag_reason=None,
        model=None,
    )


def _records(*specs):
    """Build ResolutionRecords from (interaction_id, opening_text, label) specs."""
    found = tuple(
        _interaction(
            interaction_id,
            _turn(interaction_id * 10, "customer", text),
            _turn(interaction_id * 10 + 1, "brand", "Thanks, we are on it.", minute=1),
        )
        for interaction_id, text, _ in specs
    )
    labels = tuple(_heuristic(interaction_id, label) for interaction_id, _, label in specs)
    records, _ = resolution.build_resolution_dataset(found, labels)
    return records


REFUND = "I was charged twice for premium, please refund the duplicate payment"
PLAYLIST = "all my playlists disappeared overnight, please restore them"
LOGIN = "I cannot log into my account, password reset is not working"


def _write_dataset(tmp_path, specs):
    """Write a resolution dataset JSONL from (interaction_id, text, label) specs."""
    records = _records(*specs)
    dataset_path = tmp_path / "resolution-dataset.jsonl"
    staged = resolution.stage_resolution_dataset_jsonl(records, dataset_path)
    staged.replace(dataset_path)
    return dataset_path


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert bm25_mod.tokenize("Refund My CHARGE please") == (
            "refund",
            "my",
            "charge",
            "please",
        )

    def test_strips_mentions_and_links(self):
        assert bm25_mod.tokenize("@SpotifyCares help https://t.co/abc123 please") == (
            "help",
            "please",
        )

    def test_contentless_text_has_no_tokens(self):
        assert bm25_mod.tokenize("@115888 https://t.co/abc") == ()

    def test_keeps_numbers(self):
        assert "4990" in bm25_mod.tokenize("promo Rp 4990 for 3 months")


class TestBuildIndex:
    def test_only_resolved_cases_are_indexed(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "uncertain"),
            (3, LOGIN, "unresolved"),
            (4, "another refund question", "resolved"),
        )

        index, report = bm25_mod.build_bm25_index(records)

        assert index.doc_ids == (1, 4)
        assert report.total_records == 4
        assert report.indexed == 2
        assert report.skipped == 2

    def test_build_time_is_recorded(self):
        records = _records((1, REFUND, "resolved"))

        _, report = bm25_mod.build_bm25_index(records)

        assert report.build_time_s >= 0.0

    def test_empty_input_builds_empty_index(self):
        index, report = bm25_mod.build_bm25_index(())

        assert index.doc_ids == ()
        assert index.avgdl == 0.0
        assert report.total_records == 0
        assert report.indexed == 0
        assert bm25_mod.retrieve_bm25(index, "refund") == ()

    def test_duplicate_interaction_ids_raise(self):
        records = _records((1, REFUND, "resolved"))
        doubled = records + records

        with pytest.raises(bm25_mod.BM25Error, match="duplicate"):
            bm25_mod.build_bm25_index(doubled)

    def test_invalid_params_raise(self):
        records = _records((1, REFUND, "resolved"))

        with pytest.raises(bm25_mod.BM25Error, match="k1"):
            bm25_mod.build_bm25_index(records, k1=-1.0)
        with pytest.raises(bm25_mod.BM25Error, match="b"):
            bm25_mod.build_bm25_index(records, b=1.5)

    def test_indexes_opening_message_only(self):
        records = _records((1, REFUND, "resolved"))

        index, _ = bm25_mod.build_bm25_index(records)

        # the brand reply ("Thanks, we are on it") must not enter the index
        assert "thanks" not in index.doc_freqs
        assert "refund" in index.doc_freqs


class TestRetrieve:
    def test_matching_case_ranks_first(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "resolved"),
            (3, LOGIN, "resolved"),
        )
        index, _ = bm25_mod.build_bm25_index(records)

        cases = bm25_mod.retrieve_bm25(index, "please refund my duplicate charge")

        assert cases[0].interaction_id == 1
        assert all(cases[i].score >= cases[i + 1].score for i in range(len(cases) - 1))

    def test_top_k_is_respected(self):
        records = _records(
            (1, "refund question one", "resolved"),
            (2, "refund question two", "resolved"),
            (3, "refund question three", "resolved"),
        )
        index, _ = bm25_mod.build_bm25_index(records)

        cases = bm25_mod.retrieve_bm25(index, "refund", top_k=2)

        assert len(cases) == 2

    def test_empty_query_returns_no_cases(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = bm25_mod.build_bm25_index(records)

        assert bm25_mod.retrieve_bm25(index, "   ") == ()
        assert bm25_mod.retrieve_bm25(index, "@me https://t.co/x") == ()

    def test_no_term_overlap_returns_no_cases(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = bm25_mod.build_bm25_index(records)

        assert bm25_mod.retrieve_bm25(index, "xylophone zebra quasar") == ()

    def test_ties_break_by_interaction_id(self):
        records = _records(
            (7, "identical refund message here", "resolved"),
            (3, "identical refund message here", "resolved"),
        )
        index, _ = bm25_mod.build_bm25_index(records)

        cases = bm25_mod.retrieve_bm25(index, "identical refund message")

        assert [case.interaction_id for case in cases] == [3, 7]

    def test_invalid_top_k_raises(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = bm25_mod.build_bm25_index(records)

        with pytest.raises(bm25_mod.BM25Error, match="top_k"):
            bm25_mod.retrieve_bm25(index, "refund", top_k=0)

    def test_deterministic_across_builds(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "resolved"),
            (3, LOGIN, "resolved"),
        )

        first, _ = bm25_mod.build_bm25_index(records)
        second, _ = bm25_mod.build_bm25_index(records)

        assert bm25_mod.retrieve_bm25(first, "refund premium") == (
            bm25_mod.retrieve_bm25(second, "refund premium")
        )


class TestPersistence:
    def test_round_trip_preserves_rankings(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "resolved"),
            (3, LOGIN, "resolved"),
        )
        index, _ = bm25_mod.build_bm25_index(records)

        restored = bm25_mod.index_from_json(bm25_mod.index_to_json(index))

        assert restored.doc_ids == index.doc_ids
        assert restored.avgdl == index.avgdl
        assert bm25_mod.retrieve_bm25(restored, "refund") == (
            bm25_mod.retrieve_bm25(index, "refund")
        )

    def test_unsupported_version_raises(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = bm25_mod.build_bm25_index(records)
        payload = bm25_mod.index_to_json(index)
        payload["index_version"] = 999

        with pytest.raises(bm25_mod.BM25Error, match="index_version"):
            bm25_mod.index_from_json(payload)

    def test_malformed_payloads_raise(self):
        with pytest.raises(bm25_mod.BM25Error):
            bm25_mod.index_from_json([])
        with pytest.raises(bm25_mod.BM25Error, match="duplicate"):
            bm25_mod.index_from_json(
                {
                    "index_version": 1,
                    "k1": 1.5,
                    "b": 0.75,
                    "avgdl": 1.0,
                    "docs": [
                        {"interaction_id": 1, "length": 1, "terms": {"a": 1}},
                        {"interaction_id": 1, "length": 1, "terms": {"a": 1}},
                    ],
                }
            )
        with pytest.raises(bm25_mod.BM25Error, match="length"):
            bm25_mod.index_from_json(
                {
                    "index_version": 1,
                    "k1": 1.5,
                    "b": 0.75,
                    "avgdl": 1.0,
                    "docs": [{"interaction_id": 1, "length": 5, "terms": {"a": 1}}],
                }
            )

    def test_stage_and_read_back(self, tmp_path):
        records = _records((1, REFUND, "resolved"), (2, PLAYLIST, "resolved"))
        index, _ = bm25_mod.build_bm25_index(records)
        out_path = tmp_path / "bm25-index.json"

        staged = bm25_mod.stage_bm25_index_json(index, out_path)
        staged.replace(out_path)

        restored = bm25_mod.read_bm25_index_json(out_path)
        assert restored.doc_ids == (1, 2)
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(bm25_mod.BM25Error, match="does not exist"):
            bm25_mod.read_bm25_index_json(tmp_path / "missing.json")

    def test_read_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json\n", encoding="utf-8")

        with pytest.raises(bm25_mod.BM25Error, match="malformed JSON"):
            bm25_mod.read_bm25_index_json(bad)


class TestBuildIndexCli:
    def test_command_writes_index_and_report(self, tmp_path, capsys):
        dataset_path = _write_dataset(
            tmp_path,
            (
                (1, REFUND, "resolved"),
                (2, PLAYLIST, "uncertain"),
                (3, LOGIN, "resolved"),
            ),
        )
        index_path = tmp_path / "bm25-index.json"
        report_path = tmp_path / "bm25-report.json"

        assert cli.main(
            [
                "build-bm25-index",
                "--in",
                str(dataset_path),
                "--index-out",
                str(index_path),
                "--report",
                str(report_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert f"input: {dataset_path} (3 records)" in output
        assert "indexed: 2 resolved cases (skipped 1)" in output
        assert "build time:" in output
        assert f"index written: {index_path}" in output
        assert f"report written: {report_path}" in output

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["index_version"] == bm25_mod.INDEX_VERSION
        assert payload["total_records"] == 3
        assert payload["indexed"] == 2
        assert payload["skipped"] == 1
        assert payload["build_time_s"] >= 0.0
        assert payload["tokenizer"] == bm25_mod.TOKENIZER

        index = bm25_mod.read_bm25_index_json(index_path)
        assert index.doc_ids == (1, 3)
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(self, tmp_path, capsys):
        dataset_path = _write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(["build-bm25-index", "--in", str(dataset_path)]) == 0

        output = capsys.readouterr().out
        assert "indexed: 1 resolved cases (skipped 0)" in output
        assert "index written:" not in output
        assert "report written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys):
        assert cli.main(
            ["build-bm25-index", "--in", str(tmp_path / "missing.jsonl")]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_colliding_paths_return_1(self, tmp_path, capsys):
        dataset_path = _write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(
            [
                "build-bm25-index",
                "--in",
                str(dataset_path),
                "--index-out",
                str(dataset_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err


class TestRetrieveCli:
    def test_retrieve_from_index_file(self, tmp_path, capsys):
        dataset_path = _write_dataset(
            tmp_path,
            ((1, REFUND, "resolved"), (2, PLAYLIST, "resolved")),
        )
        index_path = tmp_path / "bm25-index.json"
        assert cli.main(
            [
                "build-bm25-index",
                "--in",
                str(dataset_path),
                "--index-out",
                str(index_path),
            ]
        ) == 0
        capsys.readouterr()

        assert cli.main(
            ["retrieve-bm25", "--index", str(index_path), "--query", "refund charge"]
        ) == 0

        output = capsys.readouterr().out
        assert "interaction 1" in output
        assert "score" in output

    def test_retrieve_from_dataset_and_writes_report(self, tmp_path, capsys):
        dataset_path = _write_dataset(
            tmp_path,
            ((1, REFUND, "resolved"), (2, PLAYLIST, "resolved")),
        )
        report_path = tmp_path / "retrieval-report.json"

        assert cli.main(
            [
                "retrieve-bm25",
                "--in",
                str(dataset_path),
                "--query",
                "playlists disappeared",
                "--top-k",
                "1",
                "--report",
                str(report_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert "interaction 2" in output
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["query"] == "playlists disappeared"
        assert payload["top_k"] == 1
        assert [r["interaction_id"] for r in payload["results"]] == [2]

    def test_retrieve_needs_exactly_one_source(self, tmp_path, capsys):
        assert cli.main(["retrieve-bm25", "--query", "refund"]) == 1
        assert "error:" in capsys.readouterr().err

        dataset_path = _write_dataset(tmp_path, ((1, REFUND, "resolved"),))
        assert cli.main(
            [
                "retrieve-bm25",
                "--index",
                str(tmp_path / "index.json"),
                "--in",
                str(dataset_path),
                "--query",
                "refund",
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_retrieve_rejects_bad_top_k(self, tmp_path, capsys):
        dataset_path = _write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(
            [
                "retrieve-bm25",
                "--in",
                str(dataset_path),
                "--query",
                "refund",
                "--top-k",
                "0",
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_retrieve_missing_index_returns_1(self, tmp_path, capsys):
        assert cli.main(
            [
                "retrieve-bm25",
                "--index",
                str(tmp_path / "missing.json"),
                "--query",
                "refund",
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_retrieve_without_overlap_reports_no_cases(self, tmp_path, capsys):
        dataset_path = _write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(
            ["retrieve-bm25", "--in", str(dataset_path), "--query", "xylophone zebra"]
        ) == 0

        assert "no matching cases" in capsys.readouterr().out

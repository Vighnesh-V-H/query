import json
from datetime import datetime, timezone

import pytest

from query import adjudication
from query import cli
from query import embedding as embedding_mod
from query import minilm as minilm_mod
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


class KeywordEmbedder:
    """Deterministic offline embedder: topic keywords map to axis vectors."""

    model_name = "test/keyword-3d"

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "refund" in lowered or "charged" in lowered:
                vectors.append((1.0, 0.0, 0.0))
            elif "playlist" in lowered:
                vectors.append((0.0, 1.0, 0.0))
            elif "login" in lowered or "log in" in lowered:
                vectors.append((0.0, 0.0, 1.0))
            else:
                vectors.append((1.0, 1.0, 1.0))
        return tuple(vectors)


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


class TestBuildIndex:
    def test_only_resolved_cases_are_indexed(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "uncertain"),
            (3, LOGIN, "unresolved"),
            (4, "another refund question", "resolved"),
        )

        index, report = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        assert index.doc_ids == (1, 4)
        assert report.total_records == 4
        assert report.indexed == 2
        assert report.skipped == 2

    def test_build_time_and_model_are_recorded(self):
        records = _records((1, REFUND, "resolved"))

        index, report = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        assert report.build_time_s >= 0.0
        assert index.build_time_s == report.build_time_s
        assert report.embedding_model == "test/keyword-3d"
        assert index.embedding_model == "test/keyword-3d"
        assert report.embedding_dim == 3
        assert index.embedding_dim == 3

    def test_vectors_are_l2_normalized(self):
        records = _records((1, "some unfamiliar message here", "resolved"))

        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        (vector,) = index.vectors.values()
        assert vector == pytest.approx((0.5773502691896258,) * 3)

    def test_embedder_failure_wraps_as_minilm_error(self):
        class BrokenEmbedder:
            model_name = "test/broken"

            def embed(self, texts):
                raise RuntimeError("boom")

        with pytest.raises(minilm_mod.MiniLMError, match="boom"):
            minilm_mod.build_minilm_index(
                _records((1, REFUND, "resolved")), BrokenEmbedder()
            )

    def test_embedding_error_propagates_unwrapped(self):
        class FailingEmbedder:
            model_name = "test/failing"

            def embed(self, texts):
                raise embedding_mod.EmbeddingError("model missing")

        with pytest.raises(embedding_mod.EmbeddingError, match="model missing"):
            minilm_mod.build_minilm_index(
                _records((1, REFUND, "resolved")), FailingEmbedder()
            )

    def test_empty_input_builds_empty_index(self):
        index, report = minilm_mod.build_minilm_index((), KeywordEmbedder())

        assert index.doc_ids == ()
        assert index.embedding_dim == 0
        assert report.total_records == 0
        assert report.indexed == 0
        assert minilm_mod.retrieve_minilm(index, "refund", KeywordEmbedder()) == ()

    def test_duplicate_interaction_ids_raise(self):
        records = _records((1, REFUND, "resolved"))
        doubled = records + records

        with pytest.raises(minilm_mod.MiniLMError, match="duplicate"):
            minilm_mod.build_minilm_index(doubled, KeywordEmbedder())

    def test_wrong_vector_count_raises(self):
        class ShortEmbedder:
            model_name = "test/short"

            def embed(self, texts):
                return ()

        with pytest.raises(minilm_mod.MiniLMError, match="0 vectors for 1 texts"):
            minilm_mod.build_minilm_index(
                _records((1, REFUND, "resolved")), ShortEmbedder()
            )

    def test_ragged_vectors_raise(self):
        class RaggedEmbedder:
            model_name = "test/ragged"

            def embed(self, texts):
                return tuple((1.0,) * (position + 1) for position in range(len(texts)))

        with pytest.raises(minilm_mod.MiniLMError, match="ragged"):
            minilm_mod.build_minilm_index(
                _records((1, REFUND, "resolved"), (2, PLAYLIST, "resolved")),
                RaggedEmbedder(),
            )

    def test_non_finite_vectors_raise(self):
        class NanEmbedder:
            model_name = "test/nan"

            def embed(self, texts):
                return tuple((float("nan"),) for _ in texts)

        with pytest.raises(minilm_mod.MiniLMError, match="non-finite"):
            minilm_mod.build_minilm_index(
                _records((1, REFUND, "resolved")), NanEmbedder()
            )

    def test_indexes_opening_message_only(self):
        records = _records((1, REFUND, "resolved"))
        seen = {}

        class RecordingEmbedder(KeywordEmbedder):
            def embed(self, texts):
                seen["texts"] = list(texts)
                return super().embed(texts)

        minilm_mod.build_minilm_index(records, RecordingEmbedder())

        # the brand reply ("Thanks, we are on it") must not be embedded
        assert seen["texts"] == [REFUND]


class TestRetrieve:
    def test_matching_case_ranks_first(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "resolved"),
            (3, LOGIN, "resolved"),
        )
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        cases = minilm_mod.retrieve_minilm(
            index, "please refund my duplicate charge", KeywordEmbedder()
        )

        assert cases[0].interaction_id == 1
        assert cases[0].score == pytest.approx(1.0)
        assert all(cases[i].score >= cases[i + 1].score for i in range(len(cases) - 1))

    def test_top_k_is_respected(self):
        records = _records(
            (1, "refund question one", "resolved"),
            (2, "refund question two", "resolved"),
            (3, "refund question three", "resolved"),
        )
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        cases = minilm_mod.retrieve_minilm(index, "refund", KeywordEmbedder(), top_k=2)

        assert len(cases) == 2

    def test_low_similarity_cases_are_still_returned_with_scores(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        cases = minilm_mod.retrieve_minilm(
            index, "xylophone zebra quasar", KeywordEmbedder()
        )

        # semantic space always has a nearest neighbor; the low score is the
        # weak-evidence signal the escalation policy consumes downstream
        assert [case.interaction_id for case in cases] == [1]
        assert cases[0].score < 1.0

    def test_empty_query_returns_no_cases(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        assert minilm_mod.retrieve_minilm(index, "   ", KeywordEmbedder()) == ()

    def test_ties_break_by_interaction_id(self):
        records = _records(
            (7, "identical refund message here", "resolved"),
            (3, "identical refund message here", "resolved"),
        )
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        cases = minilm_mod.retrieve_minilm(
            index, "identical refund message", KeywordEmbedder()
        )

        assert [case.interaction_id for case in cases] == [3, 7]

    def test_invalid_top_k_raises(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        with pytest.raises(minilm_mod.MiniLMError, match="top_k"):
            minilm_mod.retrieve_minilm(index, "refund", KeywordEmbedder(), top_k=0)

    def test_dimension_mismatch_raises(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        class OtherDimEmbedder:
            model_name = "test/other-dim"

            def embed(self, texts):
                return tuple((1.0, 0.0) for _ in texts)

        with pytest.raises(minilm_mod.MiniLMError, match="dim"):
            minilm_mod.retrieve_minilm(index, "refund", OtherDimEmbedder())

    def test_deterministic_across_builds(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "resolved"),
            (3, LOGIN, "resolved"),
        )

        first, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())
        second, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        assert minilm_mod.retrieve_minilm(
            first, "refund premium", KeywordEmbedder()
        ) == minilm_mod.retrieve_minilm(second, "refund premium", KeywordEmbedder())


class TestPersistence:
    def test_round_trip_preserves_rankings(self):
        records = _records(
            (1, REFUND, "resolved"),
            (2, PLAYLIST, "resolved"),
            (3, LOGIN, "resolved"),
        )
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())

        restored = minilm_mod.index_from_json(minilm_mod.index_to_json(index))

        assert restored.doc_ids == index.doc_ids
        assert restored.embedding_model == index.embedding_model
        assert restored.embedding_dim == index.embedding_dim
        assert minilm_mod.retrieve_minilm(
            restored, "refund", KeywordEmbedder()
        ) == minilm_mod.retrieve_minilm(index, "refund", KeywordEmbedder())

    def test_unsupported_version_raises(self):
        records = _records((1, REFUND, "resolved"))
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())
        payload = minilm_mod.index_to_json(index)
        payload["index_version"] = 999

        with pytest.raises(minilm_mod.MiniLMError, match="index_version"):
            minilm_mod.index_from_json(payload)

    def test_malformed_payloads_raise(self):
        with pytest.raises(minilm_mod.MiniLMError):
            minilm_mod.index_from_json([])
        with pytest.raises(minilm_mod.MiniLMError, match="duplicate"):
            minilm_mod.index_from_json(
                {
                    "index_version": 1,
                    "embedding_model": "test/keyword-3d",
                    "embedding_dim": 2,
                    "docs": [
                        {"interaction_id": 1, "vector": [1.0, 0.0]},
                        {"interaction_id": 1, "vector": [1.0, 0.0]},
                    ],
                }
            )
        with pytest.raises(minilm_mod.MiniLMError, match="vector"):
            minilm_mod.index_from_json(
                {
                    "index_version": 1,
                    "embedding_model": "test/keyword-3d",
                    "embedding_dim": 3,
                    "docs": [{"interaction_id": 1, "vector": [1.0, 0.0]}],
                }
            )

    def test_stage_and_read_back(self, tmp_path):
        records = _records((1, REFUND, "resolved"), (2, PLAYLIST, "resolved"))
        index, _ = minilm_mod.build_minilm_index(records, KeywordEmbedder())
        out_path = tmp_path / "minilm-index.json"

        staged = minilm_mod.stage_minilm_index_json(index, out_path)
        staged.replace(out_path)

        restored = minilm_mod.read_minilm_index_json(out_path)
        assert restored.doc_ids == (1, 2)
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(minilm_mod.MiniLMError, match="does not exist"):
            minilm_mod.read_minilm_index_json(tmp_path / "missing.json")

    def test_read_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json\n", encoding="utf-8")

        with pytest.raises(minilm_mod.MiniLMError, match="malformed JSON"):
            minilm_mod.read_minilm_index_json(bad)


class TestBuildIndexCli:
    def _write_dataset(self, tmp_path, specs):
        records = _records(*specs)
        dataset_path = tmp_path / "resolution-dataset.jsonl"
        staged = resolution.stage_resolution_dataset_jsonl(records, dataset_path)
        staged.replace(dataset_path)
        return dataset_path

    def _patch_embedder(self, monkeypatch):
        monkeypatch.setattr(
            "query.embedding.MiniLMEmbedder", lambda *args, **kwargs: KeywordEmbedder()
        )

    def test_command_writes_index_and_report(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(
            tmp_path,
            (
                (1, REFUND, "resolved"),
                (2, PLAYLIST, "uncertain"),
                (3, LOGIN, "resolved"),
            ),
        )
        index_path = tmp_path / "minilm-index.json"
        report_path = tmp_path / "minilm-report.json"

        assert cli.main(
            [
                "build-minilm-index",
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
        assert "model: test/keyword-3d (dim 3)" in output
        assert "build time:" in output
        assert f"index written: {index_path}" in output
        assert f"report written: {report_path}" in output

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["index_version"] == minilm_mod.INDEX_VERSION
        assert payload["total_records"] == 3
        assert payload["indexed"] == 2
        assert payload["skipped"] == 1
        assert payload["embedding_model"] == "test/keyword-3d"
        assert payload["embedding_dim"] == 3
        assert payload["build_time_s"] >= 0.0

        index = minilm_mod.read_minilm_index_json(index_path)
        assert index.doc_ids == (1, 3)
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(
        self, tmp_path, capsys, monkeypatch
    ):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(["build-minilm-index", "--in", str(dataset_path)]) == 0

        output = capsys.readouterr().out
        assert "indexed: 1 resolved cases (skipped 0)" in output
        assert "index written:" not in output
        assert "report written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        assert cli.main(
            ["build-minilm-index", "--in", str(tmp_path / "missing.jsonl")]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_colliding_paths_return_1(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(
            [
                "build-minilm-index",
                "--in",
                str(dataset_path),
                "--index-out",
                str(dataset_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err


class TestRetrieveCli:
    def _write_dataset(self, tmp_path, specs):
        records = _records(*specs)
        dataset_path = tmp_path / "resolution-dataset.jsonl"
        staged = resolution.stage_resolution_dataset_jsonl(records, dataset_path)
        staged.replace(dataset_path)
        return dataset_path

    def _patch_embedder(self, monkeypatch):
        monkeypatch.setattr(
            "query.embedding.MiniLMEmbedder", lambda *args, **kwargs: KeywordEmbedder()
        )

    def test_retrieve_from_index_file(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(
            tmp_path,
            ((1, REFUND, "resolved"), (2, PLAYLIST, "resolved")),
        )
        index_path = tmp_path / "minilm-index.json"
        assert cli.main(
            [
                "build-minilm-index",
                "--in",
                str(dataset_path),
                "--index-out",
                str(index_path),
            ]
        ) == 0
        capsys.readouterr()

        assert cli.main(
            ["retrieve-minilm", "--index", str(index_path), "--query", "refund charge"]
        ) == 0

        output = capsys.readouterr().out
        assert "interaction 1" in output
        assert "score" in output

    def test_retrieve_from_dataset_and_writes_report(
        self, tmp_path, capsys, monkeypatch
    ):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(
            tmp_path,
            ((1, REFUND, "resolved"), (2, PLAYLIST, "resolved")),
        )
        report_path = tmp_path / "retrieval-report.json"

        assert cli.main(
            [
                "retrieve-minilm",
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

    def test_retrieve_needs_exactly_one_source(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        assert cli.main(["retrieve-minilm", "--query", "refund"]) == 1
        assert "error:" in capsys.readouterr().err

        dataset_path = self._write_dataset(tmp_path, ((1, REFUND, "resolved"),))
        assert cli.main(
            [
                "retrieve-minilm",
                "--index",
                str(tmp_path / "index.json"),
                "--in",
                str(dataset_path),
                "--query",
                "refund",
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_retrieve_rejects_bad_top_k(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(
            [
                "retrieve-minilm",
                "--in",
                str(dataset_path),
                "--query",
                "refund",
                "--top-k",
                "0",
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_retrieve_missing_index_returns_1(self, tmp_path, capsys, monkeypatch):
        self._patch_embedder(monkeypatch)
        assert cli.main(
            [
                "retrieve-minilm",
                "--index",
                str(tmp_path / "missing.json"),
                "--query",
                "refund",
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_retrieve_empty_query_reports_no_cases(
        self, tmp_path, capsys, monkeypatch
    ):
        self._patch_embedder(monkeypatch)
        dataset_path = self._write_dataset(tmp_path, ((1, REFUND, "resolved"),))

        assert cli.main(
            ["retrieve-minilm", "--in", str(dataset_path), "--query", "   "]
        ) == 0

        assert "no matching cases" in capsys.readouterr().out

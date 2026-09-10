import json
from datetime import datetime, timezone

import pytest

from query import cli
from query import interactions as interactions_mod
from query import sampling as sampling_mod
from query.interactions import Interaction, Turn

BRAND = "SpotifyCares"


def _interaction(interaction_id, text="my app keeps crashing on android"):
    return Interaction(
        interaction_id=interaction_id,
        customer_id="cust",
        brand_id=BRAND,
        turns=(
            Turn(
                tweet_id=interaction_id,
                author_id="cust",
                side="customer",
                created_at=datetime(2017, 11, 1, 10, 0, tzinfo=timezone.utc),
                text=text,
            ),
        ),
    )


def _ids(interactions):
    return [interaction.interaction_id for interaction in interactions]


class TestSampleAndSplit:
    def test_reports_requested_and_actual_sizes(self):
        found = tuple(_interaction(i) for i in range(1, 21))

        rag_pool, holdout, report = sampling_mod.sample_and_split(
            found, seed=7, sample_size=10, holdout_size=4
        )

        assert len(rag_pool) == 6
        assert len(holdout) == 4
        assert report.seed == 7
        assert report.input_total == 20
        assert report.sample_size == 10
        assert report.sampled == 10
        assert report.rag_pool == 6
        assert report.holdout == 4
        assert report.sample_rate == 0.5

    def test_pools_are_disjoint_and_cover_the_sample(self):
        found = tuple(_interaction(i) for i in range(1, 51))

        rag_pool, holdout, _ = sampling_mod.sample_and_split(
            found, seed=3, sample_size=20, holdout_size=5
        )

        rag_ids = set(_ids(rag_pool))
        holdout_ids = set(_ids(holdout))
        assert rag_ids.isdisjoint(holdout_ids)
        assert len(rag_ids | holdout_ids) == 20

    def test_deterministic_across_runs(self):
        found = tuple(_interaction(i) for i in range(1, 101))

        first = sampling_mod.sample_and_split(
            found, seed=11, sample_size=30, holdout_size=10
        )
        second = sampling_mod.sample_and_split(
            found, seed=11, sample_size=30, holdout_size=10
        )

        assert _ids(first[0]) == _ids(second[0])
        assert _ids(first[1]) == _ids(second[1])
        assert first[2] == second[2]

    def test_selection_is_independent_of_input_order(self):
        found = tuple(_interaction(i) for i in range(1, 101))
        reordered = tuple(reversed(found))

        straight = sampling_mod.sample_and_split(
            found, seed=5, sample_size=25, holdout_size=8
        )
        shuffled = sampling_mod.sample_and_split(
            reordered, seed=5, sample_size=25, holdout_size=8
        )

        assert _ids(straight[0]) == _ids(shuffled[0])
        assert _ids(straight[1]) == _ids(shuffled[1])

    def test_different_seeds_select_different_samples(self):
        found = tuple(_interaction(i) for i in range(1, 101))

        first = sampling_mod.sample_and_split(
            found, seed=1, sample_size=30, holdout_size=10
        )
        second = sampling_mod.sample_and_split(
            found, seed=2, sample_size=30, holdout_size=10
        )

        assert set(_ids(first[0]) + _ids(first[1])) != set(
            _ids(second[0]) + _ids(second[1])
        )

    def test_larger_sample_extends_smaller_sample(self):
        found = tuple(_interaction(i) for i in range(1, 101))

        small = sampling_mod.sample_and_split(
            found, seed=9, sample_size=20, holdout_size=5
        )
        large = sampling_mod.sample_and_split(
            found, seed=9, sample_size=40, holdout_size=10
        )

        small_ids = set(_ids(small[0]) + _ids(small[1]))
        large_ids = set(_ids(large[0]) + _ids(large[1]))
        assert small_ids < large_ids

    def test_small_input_scales_holdout_down(self):
        found = tuple(_interaction(i) for i in range(1, 4))

        rag_pool, holdout, report = sampling_mod.sample_and_split(
            found, seed=2, sample_size=4, holdout_size=2
        )

        assert report.sampled == 3
        assert report.holdout == 1
        assert report.rag_pool == 2
        assert len(rag_pool) + len(holdout) == 3

    def test_input_below_holdout_ratio_yields_empty_holdout(self):
        found = (_interaction(1),)

        rag_pool, holdout, report = sampling_mod.sample_and_split(
            found, seed=2, sample_size=4, holdout_size=2
        )

        assert rag_pool == (found[0],)
        assert holdout == ()
        assert report.holdout == 0
        assert report.rag_pool == 1

    def test_empty_input_reports_zero_rate(self):
        rag_pool, holdout, report = sampling_mod.sample_and_split(())

        assert rag_pool == ()
        assert holdout == ()
        assert report.input_total == 0
        assert report.sampled == 0
        assert report.sample_rate == 0.0

    def test_pools_are_ordered_by_interaction_id(self):
        found = tuple(_interaction(i) for i in range(1, 201))

        rag_pool, holdout, _ = sampling_mod.sample_and_split(
            found, seed=4, sample_size=50, holdout_size=20
        )

        assert _ids(rag_pool) == sorted(_ids(rag_pool))
        assert _ids(holdout) == sorted(_ids(holdout))

    def test_invalid_sizes_raise(self):
        found = tuple(_interaction(i) for i in range(1, 5))

        with pytest.raises(sampling_mod.SamplingError):
            sampling_mod.sample_and_split(found, sample_size=0)
        with pytest.raises(sampling_mod.SamplingError):
            sampling_mod.sample_and_split(found, sample_size=5, holdout_size=-1)
        with pytest.raises(sampling_mod.SamplingError):
            sampling_mod.sample_and_split(found, sample_size=5, holdout_size=5)


class TestSampleSplitCli:
    def _write_input(self, tmp_path, interactions):
        return interactions_mod.write_interactions_jsonl(
            interactions, tmp_path / "interactions-en.jsonl"
        )

    def test_command_writes_pools_and_report(self, tmp_path, capsys):
        found = tuple(_interaction(i) for i in range(1, 21))
        input_path = self._write_input(tmp_path, found)
        rag_path = tmp_path / "rag-pool.jsonl"
        holdout_path = tmp_path / "holdout.jsonl"
        report_path = tmp_path / "sample-report.json"

        assert cli.main(
            [
                "sample-split",
                "--in",
                str(input_path),
                "--seed",
                "7",
                "--sample-size",
                "10",
                "--holdout-size",
                "4",
                "--rag-out",
                str(rag_path),
                "--holdout-out",
                str(holdout_path),
                "--report",
                str(report_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert f"input: {input_path} (20 interactions)" in output
        assert "seed: 7" in output
        assert "sampled: 10 of 20 (50.00%)" in output
        assert "rag pool: 6 interactions" in output
        assert "holdout: 4 interactions" in output
        assert f"rag pool written: {rag_path}" in output
        assert f"holdout written: {holdout_path}" in output
        assert f"report written: {report_path}" in output

        rag_pool = interactions_mod.read_interactions_jsonl(rag_path)
        holdout = interactions_mod.read_interactions_jsonl(holdout_path)
        rag_ids = set(_ids(rag_pool))
        holdout_ids = set(_ids(holdout))
        assert len(rag_ids) == 6
        assert len(holdout_ids) == 4
        assert rag_ids.isdisjoint(holdout_ids)

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload == {
            "seed": 7,
            "input_total": 20,
            "sample_size": 10,
            "sampled": 10,
            "sample_rate": 0.5,
            "rag_pool": 6,
            "holdout": 4,
        }
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(self, tmp_path, capsys):
        found = tuple(_interaction(i) for i in range(1, 11))
        input_path = self._write_input(tmp_path, found)

        assert cli.main(
            [
                "sample-split",
                "--in",
                str(input_path),
                "--sample-size",
                "5",
                "--holdout-size",
                "2",
            ]
        ) == 0

        output = capsys.readouterr().out
        assert "sampled: 5 of 10 (50.00%)" in output
        assert "written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys):
        assert cli.main(
            ["sample-split", "--in", str(tmp_path / "missing.jsonl")]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_invalid_sizes_return_1(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, (_interaction(1),))

        assert cli.main(
            ["sample-split", "--in", str(input_path), "--sample-size", "0"]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_single_pool_output_returns_1(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, tuple(_interaction(i) for i in range(1, 6)))

        assert cli.main(
            [
                "sample-split",
                "--in",
                str(input_path),
                "--sample-size",
                "4",
                "--holdout-size",
                "1",
                "--rag-out",
                str(tmp_path / "rag-pool.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_colliding_output_paths_return_1(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, tuple(_interaction(i) for i in range(1, 6)))
        shared_path = tmp_path / "shared.jsonl"

        assert cli.main(
            [
                "sample-split",
                "--in",
                str(input_path),
                "--sample-size",
                "4",
                "--holdout-size",
                "1",
                "--rag-out",
                str(shared_path),
                "--holdout-out",
                str(shared_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_output_cannot_overwrite_input_returns_1(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, tuple(_interaction(i) for i in range(1, 6)))

        assert cli.main(
            [
                "sample-split",
                "--in",
                str(input_path),
                "--sample-size",
                "4",
                "--holdout-size",
                "1",
                "--rag-out",
                str(input_path),
                "--holdout-out",
                str(tmp_path / "holdout.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_write_error_returns_1(self, tmp_path, capsys):
        found = tuple(_interaction(i) for i in range(1, 6))
        input_path = self._write_input(tmp_path, found)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        assert cli.main(
            [
                "sample-split",
                "--in",
                str(input_path),
                "--sample-size",
                "4",
                "--holdout-size",
                "1",
                "--rag-out",
                str(blocker / "rag-pool.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

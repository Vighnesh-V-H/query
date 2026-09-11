import json
from datetime import datetime, timezone

import pytest

from query import adjudication
from query import cli
from query import closure as closure_mod
from query import interactions as interactions_mod
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


def _labeler(interaction_id, label, reason="No visible confirmation of a fix."):
    return adjudication.AdjudicatedLabel(
        interaction_id=interaction_id,
        label=label,
        source="labeler",
        reason=reason,
        flag_reason=closure_mod.REASON_BRAND_DM,
        model="test/labeler",
    )


class TestBuildResolutionDataset:
    def test_emits_one_record_per_interaction_in_input_order(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
            _interaction(
                2,
                _turn(4, "customer", "login is broken"),
                _turn(5, "brand", "Can you send us a DM?"),
            ),
        )
        labels = (_heuristic(1, "resolved"), _labeler(2, "uncertain"))

        records, _ = resolution.build_resolution_dataset(found, labels)

        assert [record.interaction_id for record in records] == [1, 2]
        assert records[0].interaction == found[0]
        assert records[1].interaction == found[1]

    def test_only_resolved_cases_are_retrieval_eligible(self):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
            _interaction(3, _turn(5, "customer", "e"), _turn(6, "brand", "f")),
            _interaction(4, _turn(7, "customer", "g"), _turn(8, "brand", "h")),
        )
        labels = (
            _heuristic(1, "resolved"),
            _heuristic(2, "uncertain"),
            _heuristic(3, "unresolved"),
            _labeler(4, "resolved"),
        )

        records, report = resolution.build_resolution_dataset(found, labels)

        assert [record.retrieval_eligible for record in records] == [
            True,
            False,
            False,
            True,
        ]
        assert report.retrieval_eligible == 2

    def test_provenance_is_preserved(self):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
        )
        labels = (
            _heuristic(1, "resolved"),
            _labeler(2, "unresolved", reason="The brand said it could not help."),
        )

        records, _ = resolution.build_resolution_dataset(found, labels)

        assert records[0].source == "heuristic"
        assert records[0].reason == HEURISTIC_REASON
        assert records[0].flag_reason is None
        assert records[0].model is None
        assert records[1].source == "labeler"
        assert records[1].reason == "The brand said it could not help."
        assert records[1].flag_reason == closure_mod.REASON_BRAND_DM
        assert records[1].model == "test/labeler"

    def test_every_record_carries_the_dataset_version(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (_heuristic(1, "resolved"),)

        records, report = resolution.build_resolution_dataset(found, labels)

        assert records[0].dataset_version == resolution.DATASET_VERSION
        assert report.dataset_version == resolution.DATASET_VERSION

    def test_report_counts_each_category(self):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
            _interaction(3, _turn(5, "customer", "e"), _turn(6, "brand", "f")),
            _interaction(4, _turn(7, "customer", "g"), _turn(8, "brand", "h")),
        )
        labels = (
            _heuristic(1, "resolved"),
            _heuristic(2, "uncertain"),
            _heuristic(3, "uncertain"),
            _heuristic(4, "unresolved"),
        )

        _, report = resolution.build_resolution_dataset(found, labels)

        assert report.total == 4
        assert report.resolved == 1
        assert report.uncertain == 2
        assert report.unresolved == 1
        assert report.retrieval_eligible == 1
        assert report.retrieval_share == pytest.approx(0.25)

    def test_report_splits_counts_by_source(self):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
            _interaction(3, _turn(5, "customer", "e"), _turn(6, "brand", "f")),
        )
        labels = (
            _heuristic(1, "resolved"),
            _labeler(2, "resolved"),
            _labeler(3, "uncertain"),
        )

        _, report = resolution.build_resolution_dataset(found, labels)

        assert report.heuristic == resolution.SourceCounts(
            total=1, resolved=1, uncertain=0, unresolved=0
        )
        assert report.labeler == resolution.SourceCounts(
            total=2, resolved=1, uncertain=1, unresolved=0
        )

    def test_report_lists_labeler_models(self):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
        )
        labels = (
            _heuristic(1, "resolved"),
            _labeler(2, "resolved"),
        )

        _, report = resolution.build_resolution_dataset(found, labels)

        assert report.models == ("test/labeler",)

    def test_empty_input_reports_zeroes(self):
        records, report = resolution.build_resolution_dataset((), ())

        assert records == ()
        assert report.total == 0
        assert report.retrieval_eligible == 0
        assert report.retrieval_share == 0.0

    def test_missing_labels_for_an_interaction_raise(self):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
        )
        labels = (_heuristic(1, "resolved"),)

        with pytest.raises(resolution.ResolutionDatasetError) as error:
            resolution.build_resolution_dataset(found, labels)

        assert "missing" in str(error.value)
        assert "2" in str(error.value)

    def test_labels_for_unknown_interaction_raise(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            _heuristic(1, "resolved"),
            _heuristic(2, "resolved"),
        )

        with pytest.raises(resolution.ResolutionDatasetError) as error:
            resolution.build_resolution_dataset(found, labels)

        assert "unknown" in str(error.value)
        assert "2" in str(error.value)

    def test_duplicate_labels_raise(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            _heuristic(1, "resolved"),
            _labeler(1, "uncertain"),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="duplicate"):
            resolution.build_resolution_dataset(found, labels)

    def test_duplicate_input_interactions_raise(self):
        interaction = _interaction(
            1, _turn(1, "customer", "a"), _turn(2, "brand", "b")
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="duplicate"):
            resolution.build_resolution_dataset(
                (interaction, interaction), (_heuristic(1, "resolved"),)
            )

    def test_heuristic_label_carrying_labeler_fields_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="heuristic",
                reason=HEURISTIC_REASON,
                flag_reason=None,
                model="test/labeler",
            ),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="heuristic"):
            resolution.build_resolution_dataset(found, labels)

    def test_labeler_label_without_flag_reason_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="labeler",
                reason="No visible confirmation of a fix.",
                flag_reason=None,
                model="test/labeler",
            ),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="flag_reason"):
            resolution.build_resolution_dataset(found, labels)

    def test_labeler_label_without_model_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="labeler",
                reason="No visible confirmation of a fix.",
                flag_reason=closure_mod.REASON_BRAND_DM,
                model=None,
            ),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="model"):
            resolution.build_resolution_dataset(found, labels)

    def test_invalid_label_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (_heuristic(1, "closed"),)

        with pytest.raises(resolution.ResolutionDatasetError, match="invalid label"):
            resolution.build_resolution_dataset(found, labels)

    def test_invalid_source_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="model",
                reason="x",
                flag_reason=None,
                model=None,
            ),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="invalid source"):
            resolution.build_resolution_dataset(found, labels)

    def test_non_int_interaction_id_types_raise(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)

        for bad_id in ("1", True, 1.0):
            labels = (
                adjudication.AdjudicatedLabel(
                    interaction_id=bad_id,
                    label="resolved",
                    source="heuristic",
                    reason=HEURISTIC_REASON,
                    flag_reason=None,
                    model=None,
                ),
            )

            with pytest.raises(
                resolution.ResolutionDatasetError, match="interaction_id"
            ):
                resolution.build_resolution_dataset(found, labels)

    def test_blank_reason_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="heuristic",
                reason="   ",
                flag_reason=None,
                model=None,
            ),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="reason"):
            resolution.build_resolution_dataset(found, labels)

    def test_non_string_reason_raises(self):
        found = (_interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),)
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="heuristic",
                reason=None,
                flag_reason=None,
                model=None,
            ),
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="reason"):
            resolution.build_resolution_dataset(found, labels)


class TestResolutionDatasetJsonl:
    def _write_records(self, tmp_path, records):
        path = tmp_path / "resolution-dataset.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def _record(self, **overrides):
        record = {
            "interaction_id": 1,
            "dataset_version": resolution.DATASET_VERSION,
            "label": "resolved",
            "retrieval_eligible": True,
            "source": "heuristic",
            "reason": HEURISTIC_REASON,
            "flag_reason": None,
            "model": None,
            "customer_id": "cust",
            "brand_id": BRAND,
            "turns": [
                {
                    "tweet_id": 1,
                    "author_id": "cust",
                    "side": "customer",
                    "created_at": "2017-11-01T10:00:00+00:00",
                    "text": "my app keeps crashing",
                },
                {
                    "tweet_id": 2,
                    "author_id": BRAND,
                    "side": "brand",
                    "created_at": "2017-11-01T10:01:00+00:00",
                    "text": "Try a clean reinstall.",
                },
            ],
        }
        record.update(overrides)
        return record

    def test_stage_writes_the_full_record_shape(self, tmp_path):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall.", minute=1),
                _turn(3, "customer", "That worked, thank you!", minute=2),
            ),
        )
        labels = (_heuristic(1, "resolved"),)
        records, _ = resolution.build_resolution_dataset(found, labels)

        staged = resolution.stage_resolution_dataset_jsonl(
            records, tmp_path / "resolution-dataset.jsonl"
        )

        written = [json.loads(line) for line in staged.read_text(encoding="utf-8").splitlines()]
        assert written == [
            {
                "interaction_id": 1,
                "dataset_version": resolution.DATASET_VERSION,
                "label": "resolved",
                "retrieval_eligible": True,
                "source": "heuristic",
                "reason": HEURISTIC_REASON,
                "flag_reason": None,
                "model": None,
                "customer_id": "cust",
                "brand_id": BRAND,
                "turns": [
                    {
                        "tweet_id": 1,
                        "author_id": "cust",
                        "side": "customer",
                        "created_at": "2017-11-01T10:00:00+00:00",
                        "text": "my app keeps crashing",
                    },
                    {
                        "tweet_id": 2,
                        "author_id": BRAND,
                        "side": "brand",
                        "created_at": "2017-11-01T10:01:00+00:00",
                        "text": "Try a clean reinstall.",
                    },
                    {
                        "tweet_id": 3,
                        "author_id": "cust",
                        "side": "customer",
                        "created_at": "2017-11-01T10:02:00+00:00",
                        "text": "That worked, thank you!",
                    },
                ],
            }
        ]

    def test_round_trips_staged_records(self, tmp_path):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
            _interaction(
                2,
                _turn(4, "customer", "login is broken"),
                _turn(5, "brand", "Can you send us a DM?"),
            ),
        )
        labels = (_heuristic(1, "resolved"), _labeler(2, "uncertain"))
        records, _ = resolution.build_resolution_dataset(found, labels)
        path = tmp_path / "resolution-dataset.jsonl"
        temporary = resolution.stage_resolution_dataset_jsonl(records, path)
        temporary.replace(path)

        read_back = resolution.read_resolution_dataset_jsonl(path)

        assert read_back == records

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(resolution.ResolutionDatasetError, match="does not exist"):
            resolution.read_resolution_dataset_jsonl(tmp_path / "missing.jsonl")

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "resolution-dataset.jsonl"
        path.write_text("not json\n", encoding="utf-8")

        with pytest.raises(resolution.ResolutionDatasetError, match="malformed JSON"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_non_object_line_raises(self, tmp_path):
        path = self._write_records(tmp_path, [[1, 2]])

        with pytest.raises(resolution.ResolutionDatasetError, match="expected an object"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_wrong_dataset_version_raises(self, tmp_path):
        path = self._write_records(
            tmp_path, [self._record(dataset_version=resolution.DATASET_VERSION + 1)]
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="dataset_version"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_non_integer_dataset_version_raises(self, tmp_path):
        path = self._write_records(
            tmp_path, [self._record(dataset_version=float(resolution.DATASET_VERSION))]
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="dataset_version"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_missing_dataset_version_raises(self, tmp_path):
        record = self._record()
        del record["dataset_version"]
        path = self._write_records(tmp_path, [record])

        with pytest.raises(resolution.ResolutionDatasetError, match="dataset_version"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_invalid_label_raises(self, tmp_path):
        path = self._write_records(tmp_path, [self._record(label="closed")])

        with pytest.raises(resolution.ResolutionDatasetError, match="invalid label"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_retrieval_eligibility_must_match_label(self, tmp_path):
        path = self._write_records(
            tmp_path, [self._record(label="uncertain", retrieval_eligible=True)]
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="retrieval_eligible"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_missing_retrieval_eligible_raises(self, tmp_path):
        record = self._record()
        del record["retrieval_eligible"]
        path = self._write_records(tmp_path, [record])

        with pytest.raises(resolution.ResolutionDatasetError, match="retrieval_eligible"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_invalid_source_raises(self, tmp_path):
        path = self._write_records(tmp_path, [self._record(source="model")])

        with pytest.raises(resolution.ResolutionDatasetError, match="invalid source"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_labeler_provenance_is_validated(self, tmp_path):
        path = self._write_records(
            tmp_path,
            [
                self._record(
                    label="resolved",
                    source="labeler",
                    flag_reason=closure_mod.REASON_BRAND_DM,
                )
            ],
        )

        with pytest.raises(resolution.ResolutionDatasetError, match="model"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_invalid_interaction_content_raises(self, tmp_path):
        path = self._write_records(tmp_path, [self._record(turns=[])])

        with pytest.raises(resolution.ResolutionDatasetError, match="turns"):
            resolution.read_resolution_dataset_jsonl(path)

    def test_duplicate_interaction_ids_raise(self, tmp_path):
        path = self._write_records(tmp_path, [self._record(), self._record()])

        with pytest.raises(resolution.ResolutionDatasetError, match="duplicate"):
            resolution.read_resolution_dataset_jsonl(path)


class TestResolutionCli:
    def _write_pool(self, tmp_path, found):
        return interactions_mod.write_interactions_jsonl(
            found, tmp_path / "rag-pool.jsonl"
        )

    def _write_final_labels(self, tmp_path, labels):
        path = tmp_path / "closure-labels-final.jsonl"
        temporary = adjudication.stage_adjudicated_labels_jsonl(labels, path)
        temporary.replace(path)
        return path

    def test_command_writes_dataset_and_report(self, tmp_path, capsys):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
            _interaction(
                2,
                _turn(4, "customer", "login is broken"),
                _turn(5, "brand", "Can you send us a DM?"),
            ),
            _interaction(
                3,
                _turn(6, "customer", "billing question"),
                _turn(7, "brand", "We've fixed the charge."),
            ),
        )
        labels = (
            _heuristic(1, "resolved"),
            _labeler(2, "uncertain"),
            _labeler(3, "resolved"),
        )
        input_path = self._write_pool(tmp_path, found)
        labels_path = self._write_final_labels(tmp_path, labels)
        out_path = tmp_path / "resolution-dataset.jsonl"
        report_path = tmp_path / "resolution-report.json"

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--out",
                str(out_path),
                "--report",
                str(report_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert f"input: {input_path} (3 interactions)" in output
        assert f"labels: {labels_path} (3 labeled)" in output
        assert "resolved: 2" in output
        assert "uncertain: 1" in output
        assert "unresolved: 0" in output
        assert "retrieval-eligible: 2 (66.67%)" in output
        assert "heuristic: 1 (resolved 1, uncertain 0, unresolved 0)" in output
        assert "labeler: 2 (resolved 1, uncertain 1, unresolved 0)" in output
        assert f"dataset written: {out_path}" in output
        assert f"report written: {report_path}" in output

        records = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["interaction_id"] for record in records] == [1, 2, 3]
        assert [record["retrieval_eligible"] for record in records] == [
            True,
            False,
            True,
        ]
        assert records[0]["source"] == "heuristic"
        assert records[1]["source"] == "labeler"
        assert records[1]["flag_reason"] == closure_mod.REASON_BRAND_DM
        assert records[1]["model"] == "test/labeler"
        assert all(record["dataset_version"] == resolution.DATASET_VERSION for record in records)

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["dataset_version"] == resolution.DATASET_VERSION
        assert payload["total"] == 3
        assert payload["resolved"] == 2
        assert payload["uncertain"] == 1
        assert payload["unresolved"] == 0
        assert payload["retrieval_eligible"] == 2
        assert payload["retrieval_share"] == pytest.approx(2 / 3)
        assert payload["by_source"] == {
            "heuristic": {"total": 1, "resolved": 1, "uncertain": 0, "unresolved": 0},
            "labeler": {"total": 2, "resolved": 1, "uncertain": 1, "unresolved": 0},
        }
        assert payload["models"] == ["test/labeler"]
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(self, tmp_path, capsys):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
        )
        input_path = self._write_pool(tmp_path, found)
        labels_path = self._write_final_labels(tmp_path, (_heuristic(1, "resolved"),))

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert "retrieval-eligible: 1 (100.00%)" in output
        assert "heuristic: 1 (resolved 1, uncertain 0, unresolved 0)" in output
        assert "labeler: 0 (resolved 0, uncertain 0, unresolved 0)" in output
        assert "dataset written:" not in output
        assert "report written:" not in output

    def test_missing_labels_file_returns_1(self, tmp_path, capsys):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
            ),
        )
        input_path = self._write_pool(tmp_path, found)

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(tmp_path / "missing.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_labels_do_not_cover_input_returns_1(self, tmp_path, capsys):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
            _interaction(2, _turn(3, "customer", "c"), _turn(4, "brand", "d")),
        )
        input_path = self._write_pool(tmp_path, found)
        labels_path = self._write_final_labels(tmp_path, (_heuristic(1, "resolved"),))

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_output_cannot_overwrite_input_returns_1(self, tmp_path, capsys):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
        )
        input_path = self._write_pool(tmp_path, found)
        labels_path = self._write_final_labels(tmp_path, (_heuristic(1, "resolved"),))

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--out",
                str(input_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_report_cannot_overwrite_labels_returns_1(self, tmp_path, capsys):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
        )
        input_path = self._write_pool(tmp_path, found)
        labels_path = self._write_final_labels(tmp_path, (_heuristic(1, "resolved"),))

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--report",
                str(labels_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_write_failure_leaves_outputs_untouched(self, tmp_path, capsys):
        found = (
            _interaction(1, _turn(1, "customer", "a"), _turn(2, "brand", "b")),
        )
        input_path = self._write_pool(tmp_path, found)
        labels_path = self._write_final_labels(tmp_path, (_heuristic(1, "resolved"),))
        out_path = tmp_path / "resolution-dataset.jsonl"
        report_path = tmp_path / "resolution-report.json"
        out_path.write_text('{"previous": "dataset"}\n', encoding="utf-8")
        report_path.write_text('{"previous": "report"}\n', encoding="utf-8")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        assert cli.main(
            [
                "build-resolution-dataset",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--out",
                str(blocker / "resolution-dataset.jsonl"),
                "--report",
                str(report_path),
            ]
        ) == 1

        assert "error:" in capsys.readouterr().err
        assert out_path.read_text(encoding="utf-8") == '{"previous": "dataset"}\n'
        assert report_path.read_text(encoding="utf-8") == '{"previous": "report"}\n'
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

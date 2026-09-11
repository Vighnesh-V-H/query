import json
from datetime import datetime, timezone

import pytest

from query import adjudication
from query import cli
from query import closure as closure_mod
from query import interactions as interactions_mod
from query import llm
from query.interactions import Interaction, Turn

BRAND = "SpotifyCares"


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


def _dm_interaction(interaction_id, marker):
    return _interaction(
        interaction_id,
        _turn(1, "customer", f"login is broken {marker}"),
        _turn(2, "brand", "Can you send us a DM?"),
    )


class _FakeLabeler:
    """Records prompts and returns canned labeler replies by call order."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        content = self._replies.pop(0)
        return llm.LLMReply(content=content, model="test/labeler")


def _reply(label, justification="The visible thread decides this."):
    return json.dumps({"label": label, "justification": justification})


class TestAdjudicateClosures:
    def test_heuristic_labels_pass_through_without_labeler_calls(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
            _interaction(
                2,
                _turn(1, "customer", "where is my order"),
                _turn(2, "brand", "We'll pass this to the right team."),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler([])

        labels, report = adjudication.adjudicate_closures(
            found, heuristic, infer=fake
        )

        assert fake.prompts == []
        assert [record.label for record in labels] == ["resolved", "uncertain"]
        assert all(record.source == "heuristic" for record in labels)
        assert all(record.flag_reason is None for record in labels)
        assert all(record.model is None for record in labels)
        assert labels[0].reason == heuristic[0].reason
        assert report.total == 2
        assert report.resolved == 1
        assert report.uncertain == 1
        assert report.adjudicated == 0
        assert report.models == ()

    def test_flagged_case_gets_labeler_category_and_justification(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        assert heuristic[0].needs_adjudication is True
        fake = _FakeLabeler([_reply("uncertain", "The brand moved to DMs with no visible outcome.")])

        labels, report = adjudication.adjudicate_closures(
            found, heuristic, infer=fake
        )

        assert len(fake.prompts) == 1
        prompt = fake.prompts[0]
        assert "customer: login is broken" in prompt
        assert "brand: Can you send us a DM?" in prompt
        assert heuristic[0].reason in prompt
        assert "resolved" in prompt and "uncertain" in prompt and "unresolved" in prompt
        assert labels[0].interaction_id == 1
        assert labels[0].label == "uncertain"
        assert labels[0].source == "labeler"
        assert labels[0].reason == "The brand moved to DMs with no visible outcome."
        assert labels[0].flag_reason == heuristic[0].reason
        assert labels[0].model == "test/labeler"
        assert report.adjudicated == 1
        assert report.adjudicated_uncertain == 1
        assert report.flagged_by_reason == {heuristic[0].reason: 1}
        assert report.models == ("test/labeler",)

    def test_labeler_can_recover_resolved_from_flagged_case(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Please DM us."),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler([_reply("resolved", "The customer later confirmed the fix in the thread.")])

        labels, report = adjudication.adjudicate_closures(
            found, heuristic, infer=fake
        )

        assert labels[0].label == "resolved"
        assert labels[0].source == "labeler"
        assert report.adjudicated_resolved == 1
        assert report.resolved == 1
        assert report.uncertain == 0

    def test_code_fenced_reply_is_parsed(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fenced = "```json\n" + _reply("unresolved") + "\n```"
        fake = _FakeLabeler([fenced])

        labels, _ = adjudication.adjudicate_closures(found, heuristic, infer=fake)

        assert labels[0].label == "unresolved"

    def test_prose_wrapped_reply_is_parsed(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler([f"Here is my answer: {_reply('unresolved')}"])

        labels, _ = adjudication.adjudicate_closures(found, heuristic, infer=fake)

        assert labels[0].label == "unresolved"

    def test_malformed_reply_is_retried_with_repair_prompt(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler(["not json at all", _reply("unresolved")])

        labels, _ = adjudication.adjudicate_closures(found, heuristic, infer=fake)

        assert labels[0].label == "unresolved"
        assert len(fake.prompts) == 2
        assert "not json at all" in fake.prompts[1]
        assert "JSON" in fake.prompts[1]

    def test_invalid_reply_twice_raises_adjudication_error(self):
        found = (
            _interaction(
                77,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler(["not json", "still not json"])

        with pytest.raises(adjudication.AdjudicationError, match="77"):
            adjudication.adjudicate_closures(found, heuristic, infer=fake)

    def test_invalid_label_twice_raises_adjudication_error(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler([_reply("closed"), _reply("closed")])

        with pytest.raises(adjudication.AdjudicationError, match="closed"):
            adjudication.adjudicate_closures(found, heuristic, infer=fake)

    def test_blank_justification_twice_raises_adjudication_error(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)
        fake = _FakeLabeler([_reply("uncertain", "  "), _reply("uncertain", "  ")])

        with pytest.raises(adjudication.AdjudicationError, match="justification"):
            adjudication.adjudicate_closures(found, heuristic, infer=fake)

    def test_labeler_call_failure_raises_adjudication_error(self):
        class _BrokenLabeler:
            def __call__(self, prompt):
                raise RuntimeError("connection reset")

        found = (
            _interaction(
                42,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Can you send us a DM?"),
            ),
        )
        heuristic, _ = closure_mod.label_closures(found)

        with pytest.raises(adjudication.AdjudicationError) as error:
            adjudication.adjudicate_closures(found, heuristic, infer=_BrokenLabeler())

        assert "42" in str(error.value)
        assert "connection reset" in str(error.value)

    def test_labels_missing_for_an_interaction_raises(self):
        found = (
            _dm_interaction(1, "one"),
            _dm_interaction(2, "two"),
        )
        heuristic = (
            closure_mod.ClosureLabel(1, None, closure_mod.REASON_BRAND_DM, True),
        )

        with pytest.raises(adjudication.AdjudicationError) as error:
            adjudication.adjudicate_closures(found, heuristic, infer=_FakeLabeler([]))

        assert "missing" in str(error.value)
        assert "2" in str(error.value)

    def test_labels_for_unknown_interaction_raise(self):
        found = (_dm_interaction(1, "one"),)
        heuristic = (
            closure_mod.ClosureLabel(1, None, closure_mod.REASON_BRAND_DM, True),
            closure_mod.ClosureLabel(2, None, closure_mod.REASON_BRAND_DM, True),
        )

        with pytest.raises(adjudication.AdjudicationError) as error:
            adjudication.adjudicate_closures(found, heuristic, infer=_FakeLabeler([]))

        assert "unknown" in str(error.value)
        assert "2" in str(error.value)

    def test_workers_must_be_at_least_one(self):
        found = (_dm_interaction(1, "one"),)
        heuristic = (
            closure_mod.ClosureLabel(1, None, closure_mod.REASON_BRAND_DM, True),
        )

        with pytest.raises(adjudication.AdjudicationError, match="workers"):
            adjudication.adjudicate_closures(
                found, heuristic, infer=_FakeLabeler([]), workers=0
            )

    def test_parallel_workers_preserve_input_order(self):
        found = (
            _dm_interaction(1, "alpha"),
            _dm_interaction(2, "bravo"),
            _dm_interaction(3, "charlie"),
            _dm_interaction(4, "delta"),
        )
        heuristic, _ = closure_mod.label_closures(found)

        class _RoutingLabeler:
            def __init__(self):
                self.prompts = []

            def __call__(self, prompt):
                self.prompts.append(prompt)
                if "alpha" in prompt:
                    label = "resolved"
                elif "bravo" in prompt:
                    label = "uncertain"
                elif "charlie" in prompt:
                    label = "unresolved"
                else:
                    label = "resolved"
                return llm.LLMReply(content=_reply(label), model="test/labeler")

        router = _RoutingLabeler()
        labels, report = adjudication.adjudicate_closures(
            found, heuristic, infer=router, workers=4
        )

        assert [record.interaction_id for record in labels] == [1, 2, 3, 4]
        assert [record.label for record in labels] == [
            "resolved",
            "uncertain",
            "unresolved",
            "resolved",
        ]
        assert all(record.source == "labeler" for record in labels)
        assert report.adjudicated == 4
        assert report.adjudicated_resolved == 2
        assert report.adjudicated_uncertain == 1
        assert report.adjudicated_unresolved == 1

    def test_report_counts_full_sample_distribution(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
            _dm_interaction(2, "two"),
            _dm_interaction(3, "three"),
        )
        heuristic, _ = closure_mod.label_closures(found)

        class _RoutingLabeler:
            def __call__(self, prompt):
                label = "resolved" if "two" in prompt else "unresolved"
                return llm.LLMReply(content=_reply(label), model="test/labeler")

        labels, report = adjudication.adjudicate_closures(
            found, heuristic, infer=_RoutingLabeler()
        )

        assert report.total == 3
        assert report.resolved == 2
        assert report.uncertain == 0
        assert report.unresolved == 1
        assert report.heuristic_resolved == 1
        assert report.heuristic_uncertain == 0
        assert report.heuristic_unresolved == 0
        assert report.adjudicated == 2
        assert report.adjudicated_resolved == 1
        assert report.adjudicated_unresolved == 1
        assert report.flagged_by_reason == {
            closure_mod.REASON_BRAND_DM: 2,
        }
        assert report.models == ("test/labeler",)
        assert [record.interaction_id for record in labels] == [1, 2, 3]

    def test_stage_writes_one_record_per_line(self, tmp_path):
        labels = (
            adjudication.AdjudicatedLabel(
                interaction_id=1,
                label="resolved",
                source="heuristic",
                reason="customer's final message acknowledges the issue is resolved",
                flag_reason=None,
                model=None,
            ),
            adjudication.AdjudicatedLabel(
                interaction_id=2,
                label="uncertain",
                source="labeler",
                reason="The visible thread never confirms a fix.",
                flag_reason=closure_mod.REASON_BRAND_DM,
                model="test/labeler",
            ),
        )

        staged = adjudication.stage_adjudicated_labels_jsonl(labels, tmp_path / "final.jsonl")

        records = [
            json.loads(line)
            for line in staged.read_text(encoding="utf-8").splitlines()
        ]
        assert records[0] == {
            "interaction_id": 1,
            "label": "resolved",
            "source": "heuristic",
            "reason": "customer's final message acknowledges the issue is resolved",
            "flag_reason": None,
            "model": None,
        }
        assert records[1]["source"] == "labeler"
        assert records[1]["flag_reason"] == closure_mod.REASON_BRAND_DM
        assert records[1]["model"] == "test/labeler"

    def test_matching_cached_verdict_is_reused_without_labeler_call(self, tmp_path):
        found = (_dm_interaction(1, "flagged"),)
        heuristic, _ = closure_mod.label_closures(found)
        cache = adjudication.AdjudicationCache(tmp_path / "cache.jsonl")
        fake = _FakeLabeler([_reply("unresolved", "No visible fix.")])

        first, _ = adjudication.adjudicate_closures(
            found, heuristic, infer=fake, cache=cache
        )

        def boom(prompt):
            raise AssertionError("labeler should not be called on a cache hit")

        second, report = adjudication.adjudicate_closures(
            found, heuristic, infer=boom, cache=cache
        )

        assert second == first
        assert second[0].reason == "No visible fix."
        assert second[0].model == "test/labeler"
        assert report.adjudicated_unresolved == 1
        verdicts = cache.verdicts()
        assert list(verdicts) == [1]
        assert verdicts[1].label == "unresolved"
        assert verdicts[1].prompt_sha256

    def test_stale_cached_verdict_is_recomputed(self, tmp_path):
        found = (_dm_interaction(1, "flagged"),)
        heuristic, _ = closure_mod.label_closures(found)
        cache = adjudication.AdjudicationCache(tmp_path / "cache.jsonl")
        cache.record(
            adjudication.CachedVerdict(
                interaction_id=1,
                prompt_sha256="0" * 64,
                label="resolved",
                reason="stale verdict from an older prompt",
                flag_reason=closure_mod.REASON_BRAND_DM,
                model="older/model",
            )
        )
        fake = _FakeLabeler([_reply("uncertain", "Fresh verdict.")])

        labels, _ = adjudication.adjudicate_closures(
            found, heuristic, infer=fake, cache=cache
        )

        assert labels[0].label == "uncertain"
        assert labels[0].reason == "Fresh verdict."
        assert len(fake.prompts) == 1
        verdicts = cache.verdicts()
        assert list(verdicts) == [1]
        assert verdicts[1].reason == "Fresh verdict."

    def test_adjudication_cache_round_trips_verdicts(self, tmp_path):
        verdict = adjudication.CachedVerdict(
            interaction_id=7,
            prompt_sha256="a" * 64,
            label="resolved",
            reason="The customer confirmed the fix.",
            flag_reason=closure_mod.REASON_BRAND_DM,
            model="test/labeler",
        )
        cache = adjudication.AdjudicationCache(tmp_path / "nested" / "cache.jsonl")

        cache.record(verdict)

        assert cache.verdicts() == {7: verdict}

    def test_cache_reader_rejects_malformed_lines(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        path.write_text("not json\n", encoding="utf-8")

        with pytest.raises(adjudication.AdjudicationError, match="malformed JSON"):
            adjudication.read_adjudication_cache(path)

    def test_cache_reader_rejects_invalid_label(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        path.write_text(
            json.dumps(
                {
                    "interaction_id": 1,
                    "prompt_sha256": "a" * 64,
                    "label": "closed",
                    "reason": "x",
                    "flag_reason": "y",
                    "model": "m",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(adjudication.AdjudicationError, match="invalid label"):
            adjudication.read_adjudication_cache(path)

    def test_cache_reader_uses_latest_record_for_an_interaction(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        records = [
            {
                "interaction_id": 1,
                "prompt_sha256": "a" * 64,
                "label": "resolved",
                "reason": "first",
                "flag_reason": "y",
                "model": "m",
            },
            {
                "interaction_id": 1,
                "prompt_sha256": "b" * 64,
                "label": "uncertain",
                "reason": "second",
                "flag_reason": "y",
                "model": "m",
            },
        ]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

        verdicts = adjudication.read_adjudication_cache(path)

        assert verdicts[1].reason == "second"

    def test_call_labeler_uses_the_labeler_role(self, monkeypatch):
        seen = {}

        def fake_call_llm(prompt, role, system, temperature):
            seen["role"] = role
            seen["temperature"] = temperature
            return llm.LLMReply(content="{}", model="test/labeler")

        monkeypatch.setattr(adjudication.llm, "call_llm", fake_call_llm)

        adjudication.call_labeler("prompt")

        assert seen["role"] == "labeler"
        assert seen["temperature"] == 0.0


class TestAdjudicateClosuresCli:
    def _write_pool(self, tmp_path, interactions):
        return interactions_mod.write_interactions_jsonl(
            interactions, tmp_path / "rag-pool.jsonl"
        )

    def _write_heuristic_labels(self, tmp_path, pool_path):
        found = interactions_mod.read_interactions_jsonl(pool_path)
        labels, _ = closure_mod.label_closures(found)
        labels_path = tmp_path / "closure-labels.jsonl"
        temporary = closure_mod.stage_closure_labels_jsonl(labels, labels_path)
        temporary.replace(labels_path)
        return labels_path

    def test_command_writes_final_labels_and_report(self, tmp_path, capsys, monkeypatch):
        pool = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you!"),
            ),
            _dm_interaction(2, "flagged"),
        )
        input_path = self._write_pool(tmp_path, pool)
        labels_path = self._write_heuristic_labels(tmp_path, input_path)
        out_path = tmp_path / "closure-labels-final.jsonl"
        report_path = tmp_path / "closure-adjudication-report.json"
        fake = _FakeLabeler([_reply("unresolved", "No visible confirmation of a fix.")])
        monkeypatch.setattr(adjudication, "call_labeler", fake)

        assert cli.main(
            [
                "adjudicate-closures",
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
        assert f"input: {input_path} (2 interactions)" in output
        assert "adjudicated: 1 flagged" in output
        assert "resolved: 1" in output
        assert "unresolved: 1" in output
        assert f"labels written: {out_path}" in output
        assert f"report written: {report_path}" in output

        records = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["interaction_id"] for record in records] == [1, 2]
        assert records[0]["source"] == "heuristic"
        assert records[1]["source"] == "labeler"
        assert records[1]["reason"] == "No visible confirmation of a fix."
        assert records[1]["flag_reason"] == closure_mod.REASON_BRAND_DM
        assert records[1]["model"] == "test/labeler"

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["total"] == 2
        assert payload["resolved"] == 1
        assert payload["uncertain"] == 0
        assert payload["unresolved"] == 1
        assert payload["adjudicated"] == 1
        assert payload["heuristic"] == {"resolved": 1, "uncertain": 0, "unresolved": 0}
        assert payload["adjudicated_labels"] == {
            "resolved": 0,
            "uncertain": 0,
            "unresolved": 1,
        }
        assert payload["flagged_by_reason"] == {closure_mod.REASON_BRAND_DM: 1}
        assert payload["models"] == ["test/labeler"]
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(self, tmp_path, capsys, monkeypatch):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))
        labels_path = self._write_heuristic_labels(tmp_path, input_path)
        monkeypatch.setattr(
            adjudication, "call_labeler", _FakeLabeler([_reply("uncertain")])
        )

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert "adjudicated: 1 flagged" in output
        assert "labels written:" not in output
        assert "report written:" not in output

    def test_missing_labels_file_returns_1(self, tmp_path, capsys):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(tmp_path / "missing.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_labels_do_not_cover_input_returns_1(self, tmp_path, capsys, monkeypatch):
        input_path = self._write_pool(
            tmp_path, (_dm_interaction(1, "one"), _dm_interaction(2, "two"))
        )
        labels_path = tmp_path / "closure-labels.jsonl"
        temporary = closure_mod.stage_closure_labels_jsonl(
            (closure_mod.ClosureLabel(1, None, closure_mod.REASON_BRAND_DM, True),),
            labels_path,
        )
        temporary.replace(labels_path)
        monkeypatch.setattr(adjudication, "call_labeler", _FakeLabeler([]))

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_output_cannot_overwrite_labels_returns_1(self, tmp_path, capsys):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))
        labels_path = self._write_heuristic_labels(tmp_path, input_path)

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--out",
                str(labels_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_labeler_failure_returns_1(self, tmp_path, capsys, monkeypatch):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))
        labels_path = self._write_heuristic_labels(tmp_path, input_path)

        def broken(prompt):
            raise RuntimeError("upstream down")

        monkeypatch.setattr(adjudication, "call_labeler", broken)

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
            ]
        ) == 1
        error = capsys.readouterr().err
        assert "error:" in error
        assert "upstream down" in error

    def test_command_reuses_cache_on_rerun(self, tmp_path, capsys, monkeypatch):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))
        labels_path = self._write_heuristic_labels(tmp_path, input_path)
        out_path = tmp_path / "closure-labels-final.jsonl"
        report_path = tmp_path / "closure-adjudication-report.json"
        cache_path = tmp_path / "closure-adjudication-cache.jsonl"
        args = [
            "adjudicate-closures",
            "--in",
            str(input_path),
            "--labels",
            str(labels_path),
            "--out",
            str(out_path),
            "--report",
            str(report_path),
            "--cache",
            str(cache_path),
        ]
        monkeypatch.setattr(
            adjudication,
            "call_labeler",
            _FakeLabeler([_reply("uncertain", "Cached verdict.")]),
        )
        assert cli.main(args) == 0
        first = out_path.read_text(encoding="utf-8")
        capsys.readouterr()

        def boom(prompt):
            raise AssertionError("cache should have been used")

        monkeypatch.setattr(adjudication, "call_labeler", boom)

        assert cli.main(args) == 0

        assert out_path.read_text(encoding="utf-8") == first
        assert cache_path.is_file()
        assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_cache_path_cannot_collide_with_outputs_returns_1(self, tmp_path, capsys):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))
        labels_path = self._write_heuristic_labels(tmp_path, input_path)
        out_path = tmp_path / "closure-labels-final.jsonl"

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--out",
                str(out_path),
                "--cache",
                str(out_path),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_write_failure_leaves_outputs_untouched(self, tmp_path, capsys, monkeypatch):
        input_path = self._write_pool(tmp_path, (_dm_interaction(1, "flagged"),))
        labels_path = self._write_heuristic_labels(tmp_path, input_path)
        out_path = tmp_path / "closure-labels-final.jsonl"
        report_path = tmp_path / "closure-adjudication-report.json"
        out_path.write_text('{"previous": "labels"}\n', encoding="utf-8")
        report_path.write_text('{"previous": "report"}\n', encoding="utf-8")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(
            adjudication, "call_labeler", _FakeLabeler([_reply("uncertain")])
        )

        assert cli.main(
            [
                "adjudicate-closures",
                "--in",
                str(input_path),
                "--labels",
                str(labels_path),
                "--out",
                str(blocker / "closure-labels-final.jsonl"),
                "--report",
                str(report_path),
            ]
        ) == 1

        assert "error:" in capsys.readouterr().err
        assert out_path.read_text(encoding="utf-8") == '{"previous": "labels"}\n'
        assert report_path.read_text(encoding="utf-8") == '{"previous": "report"}\n'
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

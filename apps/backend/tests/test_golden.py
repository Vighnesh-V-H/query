import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from query import cli, golden, llm, taxonomy
from query import interactions as interactions_mod
from query.interactions import Interaction, Turn

BRAND = "SpotifyCares"


def _interaction(interaction_id, text="my app keeps crashing", reply="Sorry to hear that!"):
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
            Turn(
                tweet_id=interaction_id + 1000000,
                author_id=BRAND,
                side="brand",
                created_at=datetime(2017, 11, 1, 10, 5, tzinfo=timezone.utc),
                text=reply,
            ),
        ),
    )


def _intent(intent_id):
    return taxonomy.FinalIntent(
        intent_id=intent_id,
        definition=f"definition of {intent_id}",
        origin="kept",
        examples=(f"example {intent_id} one", f"example {intent_id} two"),
    )


def _final(intent_ids=("account", "billing_payment", "other")):
    return taxonomy.FinalTaxonomy(
        version=1,
        intents=tuple(_intent(intent_id) for intent_id in intent_ids),
        seed_decisions=(),
        new_theme_decisions=(),
    )


def _hint(
    interaction_id,
    intent="account",
    decision=golden.AUTO,
    reason="Decisive evidence.",
    model="test/labeler",
    prompt_sha256=None,
):
    return golden.IntentHint(
        interaction_id=interaction_id,
        intent=intent,
        decision=decision,
        reason=reason,
        model=model,
        prompt_sha256=prompt_sha256,
    )


def _hints(*groups):
    """Build hints from ``(count, intent, decision)`` groups, ids from 1."""
    hints = []
    next_id = 1
    for count, intent, decision in groups:
        for _ in range(count):
            hints.append(_hint(next_id, intent, decision))
            next_id += 1
    return tuple(hints)


class _TextLabeler:
    """Answers each hint prompt from a per-message table."""

    def __init__(
        self,
        hints_by_text,
        default=("account", "auto", "Decisive evidence."),
        model="test/labeler",
    ):
        self.hints_by_text = dict(hints_by_text)
        self.default = default
        self.model = model
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        text = prompt.split("Message:\n", 1)[1].split("\n\nDecide:", 1)[0]
        intent, decision, reason = self.hints_by_text.get(text, self.default)
        return llm.LLMReply(
            content=json.dumps(
                {"intent": intent, "decision": decision, "reason": reason}
            ),
            model=self.model,
        )


class _SequenceLabeler:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return llm.LLMReply(content=self.replies.pop(0), model="test/labeler")


class _BoomLabeler:
    def __call__(self, prompt):
        raise AssertionError("labeler must not be called")


class _TransientError(Exception):
    status_code = 429


class _FlakyLabeler:
    def __init__(self, failures, reply):
        self.failures = failures
        self.reply = reply
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise _TransientError("too many requests")
        return llm.LLMReply(content=self.reply, model="test/labeler")


class _ScriptedAsk:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _Teller:
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)

    @property
    def text(self):
        return "\n".join(self.lines)


def _queue_item(interaction_id, intent="account", decision=golden.AUTO):
    return golden.QueueItem(
        interaction_id=interaction_id, intent=intent, decision=decision
    )


def _golden_example(
    interaction_id,
    gold_intent="account",
    gold_decision=golden.AUTO,
    notes="",
    hint_intent=None,
    hint_decision=None,
    taxonomy_version=1,
):
    return golden.GoldenExample(
        interaction_id=interaction_id,
        taxonomy_version=taxonomy_version,
        customer_message=f"message {interaction_id}",
        gold_intent=gold_intent,
        gold_decision=gold_decision,
        notes=notes,
        hint_intent=hint_intent or gold_intent,
        hint_decision=hint_decision or gold_decision,
    )


class TestBuildHintPrompt:
    def test_lists_every_intent_definition_and_example(self):
        prompt = golden.build_hint_prompt("my app crashed", _final())

        assert "my app crashed" in prompt
        for intent_id in ("account", "billing_payment", "other"):
            assert f"- {intent_id}: definition of {intent_id}" in prompt
            assert f"example {intent_id} one" in prompt
            assert f"example {intent_id} two" in prompt
        assert '"decision": "auto" | "escalate"' in prompt
        assert "high-risk" in prompt
        assert "no analysis, no thinking" in prompt


class TestParseHintReply:
    def test_returns_intent_decision_and_normalized_reason(self):
        reply = json.dumps(
            {
                "intent": "account",
                "decision": "escalate",
                "reason": "  Hacked   account.\nNeeds identity checks. ",
            }
        )

        intent, decision, reason = golden.parse_hint_reply(
            7, reply, ("account", "other")
        )

        assert intent == "account"
        assert decision == "escalate"
        assert reason == "Hacked account. Needs identity checks."

    def test_accepts_fenced_json(self):
        reply = '```json\n{"intent": "other", "decision": "auto", "reason": "Praise."}\n```'

        assert golden.parse_hint_reply(1, reply, ("other",)) == (
            "other",
            "auto",
            "Praise.",
        )

    def test_rejects_missing_json(self):
        with pytest.raises(golden.GoldenError, match="no JSON object"):
            golden.parse_hint_reply(1, "account auto", ("account",))

    def test_rejects_unknown_intent(self):
        reply = json.dumps({"intent": "retired", "decision": "auto", "reason": "x"})
        with pytest.raises(golden.GoldenError, match="invalid intent"):
            golden.parse_hint_reply(1, reply, ("account",))

    def test_rejects_invalid_decision(self):
        reply = json.dumps({"intent": "account", "decision": "maybe", "reason": "x"})
        with pytest.raises(golden.GoldenError, match="invalid decision"):
            golden.parse_hint_reply(1, reply, ("account",))

    def test_rejects_empty_reason(self):
        reply = json.dumps({"intent": "account", "decision": "auto", "reason": " "})
        with pytest.raises(golden.GoldenError, match="empty reason"):
            golden.parse_hint_reply(1, reply, ("account",))


class TestClassifyHints:
    def test_hints_every_message_in_input_order(self):
        interactions = tuple(
            _interaction(i, text=f"message {i}") for i in range(1, 4)
        )
        labeler = _TextLabeler(
            {
                "message 1": ("account", "auto", "Login help."),
                "message 2": ("other", "escalate", "No issue."),
                "message 3": ("billing_payment", "escalate", "Charge dispute."),
            }
        )

        hints, models = golden.classify_hints(
            interactions, _final(), infer=labeler
        )

        assert [hint.interaction_id for hint in hints] == [1, 2, 3]
        assert [hint.intent for hint in hints] == [
            "account",
            "other",
            "billing_payment",
        ]
        assert hints[0].decision == "auto"
        assert hints[0].reason == "Login help."
        assert hints[0].model == "test/labeler"
        assert hints[0].prompt_sha256 is not None
        assert models == ("test/labeler",)
        assert len(labeler.prompts) == 3
        assert "message 2" in labeler.prompts[1]

    def test_reuses_matching_cache_without_calling_the_labeler(self, tmp_path):
        interactions = tuple(_interaction(i, text=f"message {i}") for i in range(1, 3))
        path = tmp_path / "hints.jsonl"
        first, _ = golden.classify_hints(
            interactions, _final(), cache=golden.HintCache(path), infer=_TextLabeler({})
        )

        second, _ = golden.classify_hints(
            interactions, _final(), cache=golden.HintCache(path), infer=_BoomLabeler()
        )

        assert first == second
        assert path.read_text(encoding="utf-8").count("\n") == 2

    def test_recomputes_when_the_prompt_hash_is_stale(self, tmp_path):
        path = tmp_path / "hints.jsonl"
        cache = golden.HintCache(path)
        cache.record(_hint(1, intent="other", prompt_sha256="0" * 64))

        hints, _ = golden.classify_hints(
            (_interaction(1, text="message 1"),),
            _final(),
            cache=golden.HintCache(path),
            infer=_TextLabeler({"message 1": ("account", "auto", "Fresh.")}),
        )

        assert hints[0].intent == "account"
        assert hints[0].reason == "Fresh."

    def test_reuses_external_hints_without_a_prompt_hash(self, tmp_path):
        path = tmp_path / "hints.jsonl"
        cache = golden.HintCache(path)
        cache.record(_hint(1, intent="other", prompt_sha256=None))

        hints, _ = golden.classify_hints(
            (_interaction(1),),
            _final(),
            cache=golden.HintCache(path),
            infer=_BoomLabeler(),
        )

        assert hints[0].intent == "other"

    def test_recomputes_a_cached_intent_outside_the_taxonomy(self, tmp_path):
        path = tmp_path / "hints.jsonl"
        cache = golden.HintCache(path)
        cache.record(_hint(1, intent="retired", prompt_sha256=None))

        hints, _ = golden.classify_hints(
            (_interaction(1, text="message 1"),),
            _final(),
            cache=golden.HintCache(path),
            infer=_TextLabeler({"message 1": ("account", "auto", "Fresh.")}),
        )

        assert hints[0].intent == "account"

    def test_repairs_an_invalid_reply_once(self):
        interactions = (_interaction(1, text="message 1"),)
        labeler = _SequenceLabeler(
            [
                "not json",
                json.dumps(
                    {"intent": "account", "decision": "auto", "reason": "Fixed."}
                ),
            ]
        )

        hints, _ = golden.classify_hints(interactions, _final(), infer=labeler)

        assert hints[0].reason == "Fixed."
        assert len(labeler.prompts) == 2
        assert "rejected" in labeler.prompts[1]

    def test_retries_a_flaky_reply_up_to_the_attempt_cap(self):
        labeler = _SequenceLabeler(
            [
                "Here's a thinking process: ...",
                "Still thinking about it ...",
                json.dumps(
                    {"intent": "account", "decision": "auto", "reason": "Third try."}
                ),
            ]
        )

        hints, _ = golden.classify_hints((_interaction(1),), _final(), infer=labeler)

        assert hints[0].reason == "Third try."
        assert len(labeler.prompts) == golden.HINT_ATTEMPTS

    def test_fails_after_the_attempt_cap(self):
        labeler = _SequenceLabeler(["not json", "still not json", "nope again"])

        with pytest.raises(golden.GoldenError, match="no JSON object"):
            golden.classify_hints((_interaction(1),), _final(), infer=labeler)
        assert len(labeler.prompts) == golden.HINT_ATTEMPTS

    def test_workers_bound_parallel_calls(self):
        interactions = tuple(_interaction(i, text=f"message {i}") for i in range(1, 7))
        labeler = _TextLabeler(
            {f"message {i}": ("account", "auto", "Same.") for i in range(1, 7)}
        )

        hints, _ = golden.classify_hints(
            interactions, _final(), workers=4, infer=labeler
        )

        assert [hint.interaction_id for hint in hints] == [1, 2, 3, 4, 5, 6]
        assert len(labeler.prompts) == 6

    def test_labeler_failure_is_wrapped(self):
        def broken(prompt):
            raise RuntimeError("upstream down")

        with pytest.raises(golden.GoldenError, match="labeler call failed"):
            golden.classify_hints((_interaction(1),), _final(), infer=broken)

    def test_retries_transient_labeler_errors_with_backoff(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(golden.time, "sleep", sleeps.append)
        reply = json.dumps(
            {"intent": "account", "decision": "auto", "reason": "Ok."}
        )
        labeler = _FlakyLabeler(2, reply)

        hints, _ = golden.classify_hints((_interaction(1),), _final(), infer=labeler)

        assert hints[0].intent == "account"
        assert labeler.calls == 3
        assert len(sleeps) == 2

    def test_transient_errors_fail_after_the_call_cap(self, monkeypatch):
        monkeypatch.setattr(golden.time, "sleep", lambda seconds: None)
        labeler = _FlakyLabeler(99, "{}")

        with pytest.raises(golden.GoldenError, match="labeler call failed"):
            golden.classify_hints((_interaction(1),), _final(), infer=labeler)
        assert labeler.calls == golden.CALL_ATTEMPTS

    def test_rejects_duplicate_interaction_ids(self):
        with pytest.raises(golden.GoldenError, match="duplicate interaction_id"):
            golden.classify_hints((_interaction(1), _interaction(1)), _final())

    def test_rejects_cache_records_without_reason_or_model(self, tmp_path):
        path = tmp_path / "hints.jsonl"
        path.write_text(
            json.dumps(
                {"interaction_id": 1, "intent": "account", "decision": "auto"}
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(golden.GoldenError, match="invalid reason"):
            golden.HintCache(path).hints()

    def test_rejects_empty_input(self):
        with pytest.raises(golden.GoldenError, match="no interactions"):
            golden.classify_hints((), _final())

    def test_rejects_invalid_worker_count(self):
        with pytest.raises(golden.GoldenError, match="workers"):
            golden.classify_hints((_interaction(1),), _final(), workers=0)


class TestPlanQueue:
    def test_apportions_proportionally_with_a_floor(self):
        hints = _hints((100, "account", "auto"), (100, "billing_payment", "escalate"))

        items, report = golden.plan_queue(
            hints,
            ("account", "billing_payment"),
            target_size=30,
            floor=5,
            auto_share=0.5,
        )

        assert report.selected == 30
        assert report.per_intent["account"].selected == 15
        assert report.per_intent["billing_payment"].selected == 15
        assert len(items) == 30

    def test_floor_caps_at_available_and_keeps_tiny_intents_visible(self):
        hints = _hints(
            (1, "account", "auto"),
            (100, "billing_payment", "auto"),
        )

        _, report = golden.plan_queue(
            hints, ("account", "billing_payment"), target_size=10, floor=3
        )

        assert report.per_intent["account"].selected == 1
        assert report.per_intent["billing_payment"].selected == 9
        assert report.selected == 10

    def test_floors_win_when_they_exceed_the_target(self):
        hints = _hints(
            (5, "account", "auto"),
            (5, "billing_payment", "auto"),
            (5, "other", "auto"),
        )

        _, report = golden.plan_queue(
            hints,
            ("account", "billing_payment", "other"),
            target_size=6,
            floor=3,
        )

        assert report.selected == 9
        assert all(
            allocation.selected == 3 for allocation in report.per_intent.values()
        )

    def test_zero_available_intents_report_zero(self):
        hints = _hints((5, "account", "auto"))

        _, report = golden.plan_queue(
            hints, ("account", "billing_payment", "other"), target_size=3, floor=1
        )

        assert report.per_intent["billing_payment"].selected == 0
        assert report.per_intent["other"].selected == 0

    def test_balance_target_is_met_when_capacity_allows(self):
        hints = _hints(
            (20, "account", "auto"),
            (20, "account", "escalate"),
            (20, "billing_payment", "auto"),
            (20, "billing_payment", "escalate"),
        )

        _, report = golden.plan_queue(
            hints,
            ("account", "billing_payment"),
            target_size=10,
            floor=0,
            auto_share=0.6,
        )

        assert report.balance_target_met is True
        assert report.selected == 10
        assert report.selected_auto == 6
        assert report.selected_escalate == 4
        assert report.predicted_auto_share == 0.6

    def test_balance_target_is_reported_when_infeasible(self):
        hints = _hints(
            (20, "account", "escalate"),
            (20, "billing_payment", "escalate"),
        )

        _, report = golden.plan_queue(
            hints, ("account", "billing_payment"), target_size=10, floor=0
        )

        assert report.balance_target_met is False
        assert report.selected_auto == 0
        assert report.predicted_auto_share == 0.0

    def test_balance_is_clamped_to_the_hinted_capacity(self):
        hints = _hints(
            (100, "account", "auto"),
            (100, "billing_payment", "escalate"),
            (2, "other", "auto"),
        )

        _, report = golden.plan_queue(
            hints,
            ("account", "billing_payment", "other"),
            target_size=30,
            floor=5,
            auto_share=0.5,
        )

        assert report.selected == 30
        assert report.selected_auto == 16
        assert report.selected_escalate == 14
        assert report.balance_target_met is False

    def test_selection_is_deterministic_across_input_order(self):
        hints = _hints(
            (30, "account", "auto"),
            (30, "account", "escalate"),
            (30, "other", "auto"),
        )
        kwargs = dict(
            intent_ids=("account", "other"),
            target_size=12,
            floor=2,
            auto_share=0.5,
            seed=7,
        )

        first, _ = golden.plan_queue(hints, **kwargs)
        second, _ = golden.plan_queue(tuple(reversed(hints)), **kwargs)

        assert [item.interaction_id for item in first] == [
            item.interaction_id for item in second
        ]

    def test_different_seeds_select_different_candidates(self):
        hints = _hints((40, "account", "auto"))
        common = dict(
            intent_ids=("account",),
            target_size=8,
            floor=0,
            auto_share=1.0,
        )

        first, _ = golden.plan_queue(hints, seed=1, **common)
        second, _ = golden.plan_queue(hints, seed=2, **common)

        assert [item.interaction_id for item in first] != [
            item.interaction_id for item in second
        ]

    def test_queue_interleaves_intents(self):
        hints = _hints(
            (10, "account", "auto"),
            (10, "other", "auto"),
        )

        items, _ = golden.plan_queue(
            hints, ("account", "other"), target_size=6, floor=3, auto_share=1.0
        )

        assert [item.intent for item in items[:2]] == ["account", "other"]
        assert [item.intent for item in items[:6]] == [
            "account",
            "other",
            "account",
            "other",
            "account",
            "other",
        ]

    def test_selected_items_are_unique_and_from_the_hints(self):
        hints = _hints(
            (30, "account", "auto"),
            (30, "account", "escalate"),
            (30, "other", "escalate"),
        )

        items, _ = golden.plan_queue(
            hints, ("account", "other"), target_size=20, floor=5, auto_share=0.4
        )

        ids = [item.interaction_id for item in items]
        assert len(set(ids)) == len(ids) == 20
        assert set(ids) <= {hint.interaction_id for hint in hints}

    def test_rejects_invalid_arguments(self):
        hints = _hints((3, "account", "auto"))
        with pytest.raises(golden.GoldenError, match="no hints"):
            golden.plan_queue((), ("account",))
        with pytest.raises(golden.GoldenError, match="target size"):
            golden.plan_queue(hints, ("account",), target_size=0)
        with pytest.raises(golden.GoldenError, match="floor"):
            golden.plan_queue(hints, ("account",), floor=-1)
        with pytest.raises(golden.GoldenError, match="auto share"):
            golden.plan_queue(hints, ("account",), auto_share=1.5)
        with pytest.raises(golden.GoldenError, match="no intents"):
            golden.plan_queue(hints, ())

    def test_rejects_duplicate_and_unknown_hints(self):
        with pytest.raises(golden.GoldenError, match="duplicate hint"):
            golden.plan_queue(
                (_hint(1), _hint(1)), ("account",), target_size=1, floor=0
            )
        with pytest.raises(golden.GoldenError, match="unknown intent"):
            golden.plan_queue(
                (_hint(1, intent="retired"),), ("account",), target_size=1, floor=0
            )
        with pytest.raises(golden.GoldenError, match="invalid decision"):
            golden.plan_queue(
                (_hint(1, decision="maybe"),), ("account",), target_size=1, floor=0
            )
        with pytest.raises(golden.GoldenError, match="unique"):
            golden.plan_queue(
                _hints((2, "account", "auto")),
                ("account", "account"),
                target_size=1,
                floor=0,
            )


class TestQueueJsonl:
    def test_round_trips_items(self, tmp_path):
        items = (
            _queue_item(1, "account", "auto"),
            _queue_item(2, "other", "escalate"),
        )
        path = tmp_path / "queue.jsonl"

        temporary = golden.stage_queue_jsonl(items, path)
        path.write_bytes(Path(temporary).read_bytes())

        assert golden.read_queue_jsonl(path, ("account", "other")) == items
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert payload == {
            "interaction_id": 1,
            "hint_intent": "account",
            "hint_decision": "auto",
        }

    def test_rejects_duplicate_ids_and_bad_records(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        path.write_text(
            '{"interaction_id": 1, "hint_intent": "account", "hint_decision": "auto"}\n'
            '{"interaction_id": 1, "hint_intent": "other", "hint_decision": "auto"}\n',
            encoding="utf-8",
        )
        with pytest.raises(golden.GoldenError, match="duplicate interaction_id"):
            golden.read_queue_jsonl(path)

        path.write_text(
            '{"interaction_id": 1, "hint_intent": "account", "hint_decision": "maybe"}\n',
            encoding="utf-8",
        )
        with pytest.raises(golden.GoldenError, match="invalid hint_decision"):
            golden.read_queue_jsonl(path)

        path.write_text(
            '{"interaction_id": 1, "hint_intent": "retired", "hint_decision": "auto"}\n',
            encoding="utf-8",
        )
        with pytest.raises(golden.GoldenError, match="unknown hint_intent"):
            golden.read_queue_jsonl(path, ("account",))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(golden.GoldenError, match="does not exist"):
            golden.read_queue_jsonl(tmp_path / "missing.jsonl")


class TestGoldenJsonl:
    def test_round_trips_examples(self, tmp_path):
        examples = (
            _golden_example(1, "account", "escalate", notes="Hacked."),
            _golden_example(2, "other", "auto"),
        )
        path = tmp_path / "golden-set.jsonl"

        golden.write_golden_jsonl(examples, path)

        assert golden.read_golden_jsonl(path, ("account", "other")) == examples
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert payload == {
            "interaction_id": 1,
            "taxonomy_version": 1,
            "customer_message": "message 1",
            "gold_intent": "account",
            "gold_decision": "escalate",
            "notes": "Hacked.",
            "hint_intent": "account",
            "hint_decision": "escalate",
        }

    def test_rejects_bad_records(self, tmp_path):
        path = tmp_path / "golden-set.jsonl"
        base = {
            "interaction_id": 1,
            "taxonomy_version": 1,
            "customer_message": "message 1",
            "gold_intent": "account",
            "gold_decision": "auto",
            "notes": "",
            "hint_intent": "account",
            "hint_decision": "auto",
        }

        for field, value, match in (
            ("taxonomy_version", 0, "taxonomy_version"),
            ("gold_intent", "retired", "unknown gold_intent"),
            ("gold_decision", "maybe", "gold_decision"),
            ("hint_decision", "maybe", "hint_decision"),
            ("customer_message", "", "customer_message"),
            ("notes", 3, "notes"),
        ):
            record = dict(base)
            record[field] = value
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with pytest.raises(golden.GoldenError, match=match):
                golden.read_golden_jsonl(path, ("account",))

        path.write_text(
            json.dumps(base) + "\n" + json.dumps(base) + "\n", encoding="utf-8"
        )
        with pytest.raises(golden.GoldenError, match="duplicate interaction_id"):
            golden.read_golden_jsonl(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(golden.GoldenError, match="does not exist"):
            golden.read_golden_jsonl(tmp_path / "missing.jsonl")


class TestLabelGolden:
    def _labeled(self, tmp_path, queue, answers):
        interactions = tuple(
            _interaction(item.interaction_id, text=f"message {item.interaction_id}")
            for item in queue
        )
        output = tmp_path / "golden-set.jsonl"
        ask = _ScriptedAsk(answers)
        tell = _Teller()
        examples, report = golden.label_golden(
            queue, interactions, _final(), output, ask=ask, tell=tell
        )
        return output, examples, report, ask, tell

    def test_labels_the_whole_queue_with_the_full_transcript(self, tmp_path):
        queue = (
            _queue_item(1, "account", "auto"),
            _queue_item(2, "billing_payment", "escalate"),
            _queue_item(3, "other", "auto"),
        )
        output, examples, report, _, tell = self._labeled(
            tmp_path,
            queue,
            ["account", "a", "note 1", "2", "e", "", "other", "e", "note 3"],
        )

        records = golden.read_golden_jsonl(output, _final().intent_ids)
        assert records == examples
        assert [record.gold_intent for record in records] == [
            "account",
            "billing_payment",
            "other",
        ]
        assert [record.gold_decision for record in records] == [
            "auto",
            "escalate",
            "escalate",
        ]
        assert [record.notes for record in records] == ["note 1", "", "note 3"]
        assert records[0].hint_intent == "account"
        assert records[0].hint_decision == "auto"
        assert records[0].taxonomy_version == 1
        assert "message 1" in tell.text
        assert "Sorry to hear that!" in tell.text
        assert "[customer" in tell.text
        assert "[   brand" in tell.text
        assert report.labeled == 3
        assert report.remaining == 0
        assert report.complete is True
        assert report.by_decision == {"auto": 1, "escalate": 2}
        assert report.by_intent["account"].total == 1
        assert report.by_intent["billing_payment"] == golden.IntentLabelCounts(
            total=1, auto=0, escalate=1
        )
        assert report.notes_filled == 2
        assert report.hint_intent_agreement == 3
        assert report.hint_decision_agreement == 2

    def test_writes_each_label_before_the_next_one(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2))
        output = tmp_path / "golden-set.jsonl"
        snapshots = []

        class _WatchingAsk(_ScriptedAsk):
            def __call__(self, prompt):
                snapshots.append(
                    len(golden.read_golden_jsonl(output)) if output.is_file() else 0
                )
                return super().__call__(prompt)

        golden.label_golden(
            queue,
            tuple(_interaction(item.interaction_id) for item in queue),
            _final(),
            output,
            ask=_WatchingAsk(["account", "a", "", "account", "a", ""]),
            tell=_Teller(),
        )

        assert snapshots == [0, 0, 0, 1, 1, 1]
        assert len(golden.read_golden_jsonl(output)) == 2

    def test_resume_skips_already_labelled_ids(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2), _queue_item(3))
        output = tmp_path / "golden-set.jsonl"
        golden.write_golden_jsonl(
            (_golden_example(1, "account", "auto"),), output
        )
        interactions = tuple(_interaction(item.interaction_id) for item in queue)

        _, report = golden.label_golden(
            queue,
            interactions,
            _final(),
            output,
            ask=_ScriptedAsk(["other", "e", "", "account", "a", "done"]),
            tell=_Teller(),
        )

        records = golden.read_golden_jsonl(output)
        assert [record.interaction_id for record in records] == [1, 2, 3]
        assert report.labeled == 3
        assert report.remaining == 0

    def test_pause_keeps_completed_labels_and_reports_remaining(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2))
        output, examples, report, _, _ = self._labeled(
            tmp_path, queue, ["account", "a", "n1", "q"]
        )

        assert [example.interaction_id for example in examples] == [1]
        assert [record.interaction_id for record in golden.read_golden_jsonl(output)] == [1]
        assert report.labeled == 1
        assert report.remaining == 1
        assert report.complete is False
        assert report.skipped == 0

    def test_skip_advances_without_labelling(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2), _queue_item(3))
        _, examples, report, _, _ = self._labeled(
            tmp_path, queue, ["s", "2", "a", "", "3", "e", ""]
        )

        assert [example.interaction_id for example in examples] == [2, 3]
        assert report.labeled == 2
        assert report.skipped == 1
        assert report.remaining == 1

    def test_blank_opening_message_is_skipped_without_writing(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2))
        interactions = (
            _interaction(1, text="   "),
            _interaction(2, text="message 2"),
        )
        output = tmp_path / "golden-set.jsonl"
        ask = _ScriptedAsk(["account", "a", ""])
        tell = _Teller()

        examples, report = golden.label_golden(
            queue, interactions, _final(), output, ask=ask, tell=tell
        )

        assert [example.interaction_id for example in examples] == [2]
        assert report.labeled == 1
        assert report.skipped == 1
        assert report.remaining == 1
        assert "blank opening message" in tell.text
        records = golden.read_golden_jsonl(output, _final().intent_ids)
        assert [record.interaction_id for record in records] == [2]
        assert len(ask.prompts) == 3

    def test_invalid_answers_are_reprompted(self, tmp_path):
        queue = (_queue_item(1),)
        _, examples, report, ask, tell = self._labeled(
            tmp_path, queue, ["99", "1", "x", "a", "  "]
        )

        assert examples[0].gold_intent == "account"
        assert examples[0].gold_decision == "auto"
        assert examples[0].notes == ""
        assert report.notes_filled == 0
        assert "unknown intent" in tell.text
        assert "unknown decision" in tell.text
        assert len(ask.prompts) == 5

    def test_skip_and_quit_commands_work_at_the_notes_prompt(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2))
        _, examples, report, _, _ = self._labeled(
            tmp_path, queue, ["account", "a", "s", "1", "a", "q"]
        )

        assert examples == ()
        assert report.skipped == 1
        assert report.remaining == 2

    def test_question_mark_prints_the_taxonomy_menu(self, tmp_path):
        queue = (_queue_item(1),)
        _, examples, _, _, tell = self._labeled(
            tmp_path, queue, ["?", "1", "a", ""]
        )

        assert examples[0].gold_intent == "account"
        assert "Intents (enter the number or the id):" in tell.text
        assert "definition of billing_payment" in tell.text

    def test_eof_pauses_the_session(self, tmp_path):
        queue = (_queue_item(1), _queue_item(2))
        _, examples, report, _, _ = self._labeled(
            tmp_path, queue, ["account", "a", "", EOFError()]
        )

        assert [example.interaction_id for example in examples] == [1]
        assert report.remaining == 1

    def test_rejects_a_golden_file_outside_the_queue(self, tmp_path):
        queue = (_queue_item(1),)
        output = tmp_path / "golden-set.jsonl"
        golden.write_golden_jsonl((_golden_example(99),), output)

        with pytest.raises(golden.GoldenError, match="not in the queue"):
            golden.label_golden(
                queue,
                (_interaction(1), _interaction(99)),
                _final(),
                output,
                ask=_ScriptedAsk([]),
                tell=_Teller(),
            )

    def test_rejects_a_taxonomy_version_mismatch(self, tmp_path):
        queue = (_queue_item(1),)
        output = tmp_path / "golden-set.jsonl"
        golden.write_golden_jsonl((_golden_example(1, taxonomy_version=2),), output)

        with pytest.raises(golden.GoldenError, match="taxonomy v2"):
            golden.label_golden(
                queue,
                (_interaction(1),),
                _final(),
                output,
                ask=_ScriptedAsk([]),
                tell=_Teller(),
            )

    def test_rejects_a_queued_intent_outside_the_taxonomy(self, tmp_path):
        queue = (_queue_item(1, "retired"),)

        with pytest.raises(golden.GoldenError, match="unknown intent"):
            golden.label_golden(
                queue,
                (_interaction(1),),
                _final(),
                tmp_path / "golden-set.jsonl",
                ask=_ScriptedAsk([]),
                tell=_Teller(),
            )

    def test_rejects_a_queued_interaction_missing_from_the_input(self, tmp_path):
        queue = (_queue_item(5),)

        with pytest.raises(golden.GoldenError, match="not in the input"):
            golden.label_golden(
                queue,
                (_interaction(1),),
                _final(),
                tmp_path / "golden-set.jsonl",
                ask=_ScriptedAsk([]),
                tell=_Teller(),
            )

    def test_rejects_empty_and_duplicate_queues(self):
        with pytest.raises(golden.GoldenError, match="no queued"):
            golden.label_golden((), (), _final(), "unused.jsonl")
        with pytest.raises(golden.GoldenError, match="duplicate interaction_id"):
            golden.label_golden(
                (_queue_item(1), _queue_item(1)),
                (_interaction(1),),
                _final(),
                "unused.jsonl",
            )


class TestSampleGoldenCli:
    def _write_holdout(self, tmp_path, interactions):
        return interactions_mod.write_interactions_jsonl(
            interactions, tmp_path / "holdout.jsonl"
        )

    def _holdout(self):
        return tuple(
            _interaction(i, text=f"message {i}") for i in range(1, 9)
        )

    def _labeler(self):
        return _TextLabeler(
            {
                f"message {i}": (
                    "account",
                    "auto" if i % 2 else "escalate",
                    f"Evidence {i}.",
                )
                for i in range(1, 9)
            }
        )

    def test_command_writes_hints_queue_and_report(
        self, tmp_path, capsys, monkeypatch
    ):
        holdout = self._write_holdout(tmp_path, self._holdout())
        monkeypatch.setattr(golden, "call_labeler", self._labeler())
        hints_path = tmp_path / "hints.jsonl"
        queue_path = tmp_path / "queue.jsonl"
        report_path = tmp_path / "report.json"

        assert cli.main(
            [
                "sample-golden",
                "--in",
                str(holdout),
                "--hints",
                str(hints_path),
                "--out",
                str(queue_path),
                "--report",
                str(report_path),
                "--target",
                "4",
                "--floor",
                "1",
                "--auto-share",
                "0.5",
                "--seed",
                "3",
            ]
        ) == 0

        output = capsys.readouterr().out
        assert f"input: {holdout} (8 interactions)" in output
        assert "hints:" in output and "test/labeler" in output
        assert "target: 4 (floor 1, auto share 50.00%)" in output
        assert "selected: 4" in output
        assert "balance target met" in output
        assert "outside the spec's 150-250" in output
        assert f"queue written: {queue_path}" in output
        assert f"report written: {report_path}" in output

        queue = golden.read_queue_jsonl(queue_path, _final().intent_ids)
        assert len(queue) == 4
        assert len(set(item.interaction_id for item in queue)) == 4
        assert hints_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["seed"] == 3
        assert payload["hinted"] == 8
        assert payload["selected"] == 4
        assert payload["selected_auto"] == 2
        assert payload["selected_escalate"] == 2
        assert payload["predicted_auto_share"] == 0.5
        assert payload["balance_target_met"] is True
        assert payload["models"] == ["test/labeler"]
        assert sum(
            allocation["selected"] for allocation in payload["per_intent"].values()
        ) == 4
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_rerun_reuses_cached_hints(self, tmp_path, capsys, monkeypatch):
        holdout = self._write_holdout(tmp_path, self._holdout())
        monkeypatch.setattr(golden, "call_labeler", self._labeler())
        hints_path = tmp_path / "hints.jsonl"
        args = [
            "sample-golden",
            "--in",
            str(holdout),
            "--hints",
            str(hints_path),
            "--target",
            "4",
            "--floor",
            "1",
            "--auto-share",
            "0.5",
        ]
        assert cli.main(args) == 0

        monkeypatch.setattr(golden, "call_labeler", _BoomLabeler())
        assert cli.main(args) == 0
        capsys.readouterr()

    def test_labeler_failure_returns_1(self, tmp_path, capsys, monkeypatch):
        holdout = self._write_holdout(tmp_path, self._holdout())

        def broken(prompt):
            raise RuntimeError("upstream down")

        monkeypatch.setattr(golden, "call_labeler", broken)
        assert cli.main(
            [
                "sample-golden",
                "--in",
                str(holdout),
                "--hints",
                str(tmp_path / "hints.jsonl"),
                "--target",
                "4",
                "--floor",
                "1",
            ]
        ) == 1
        assert "labeler call failed" in capsys.readouterr().err

    def test_missing_input_returns_1(self, tmp_path, capsys):
        assert cli.main(
            [
                "sample-golden",
                "--in",
                str(tmp_path / "missing.jsonl"),
                "--hints",
                str(tmp_path / "hints.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_invalid_target_returns_1(self, tmp_path, capsys, monkeypatch):
        holdout = self._write_holdout(tmp_path, self._holdout())
        monkeypatch.setattr(golden, "call_labeler", self._labeler())

        assert cli.main(
            [
                "sample-golden",
                "--in",
                str(holdout),
                "--hints",
                str(tmp_path / "hints.jsonl"),
                "--target",
                "0",
            ]
        ) == 1
        assert "target size" in capsys.readouterr().err

    def test_colliding_paths_return_1(self, tmp_path, capsys):
        shared = tmp_path / "shared.jsonl"
        assert cli.main(
            [
                "sample-golden",
                "--hints",
                str(shared),
                "--out",
                str(shared),
            ]
        ) == 1
        assert "distinct" in capsys.readouterr().err

    def test_staging_failure_leaves_outputs_untouched(
        self, tmp_path, capsys, monkeypatch
    ):
        holdout = self._write_holdout(tmp_path, self._holdout())
        monkeypatch.setattr(golden, "call_labeler", self._labeler())
        hints_path = tmp_path / "hints.jsonl"
        queue_path = tmp_path / "queue.jsonl"
        report_path = tmp_path / "report.json"
        queue_path.write_text('{"previous": "queue"}\n', encoding="utf-8")
        report_path.write_text('{"previous": "report"}\n', encoding="utf-8")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        assert cli.main(
            [
                "sample-golden",
                "--in",
                str(holdout),
                "--hints",
                str(hints_path),
                "--out",
                str(queue_path),
                "--report",
                str(blocker / "report.json"),
                "--target",
                "4",
                "--floor",
                "1",
            ]
        ) == 1

        assert "error:" in capsys.readouterr().err
        assert queue_path.read_text(encoding="utf-8") == '{"previous": "queue"}\n'
        assert report_path.read_text(encoding="utf-8") == '{"previous": "report"}\n'

    def test_blank_opening_messages_are_excluded_before_hinting(
        self, tmp_path, capsys, monkeypatch
    ):
        holdout = self._write_holdout(
            tmp_path,
            (
                _interaction(1, text="message 1"),
                _interaction(2, text="   "),
                _interaction(3, text="message 3"),
            ),
        )
        labeler = _TextLabeler(
            {
                "message 1": ("account", "auto", "Evidence 1."),
                "message 3": ("other", "escalate", "Evidence 3."),
            }
        )
        monkeypatch.setattr(golden, "call_labeler", labeler)
        queue_path = tmp_path / "queue.jsonl"
        report_path = tmp_path / "report.json"

        assert cli.main(
            [
                "sample-golden",
                "--in",
                str(holdout),
                "--hints",
                str(tmp_path / "hints.jsonl"),
                "--out",
                str(queue_path),
                "--report",
                str(report_path),
                "--target",
                "2",
                "--floor",
                "1",
            ]
        ) == 0

        output = capsys.readouterr().out
        assert "excluded: 1 interactions with a blank opening message" in output
        assert len(labeler.prompts) == 2
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["hinted"] == 2
        queue = golden.read_queue_jsonl(queue_path, _final().intent_ids)
        assert {item.interaction_id for item in queue} == {1, 3}


class TestLabelGoldenCli:
    def _write_queue(self, tmp_path, queue):
        path = tmp_path / "queue.jsonl"
        path.write_text(
            "\n".join(json.dumps(golden.queue_to_json(item)) for item in queue)
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_command_writes_golden_set_and_report(
        self, tmp_path, capsys, monkeypatch
    ):
        queue = (_queue_item(1, "account", "auto"), _queue_item(2, "other", "escalate"))
        holdout = interactions_mod.write_interactions_jsonl(
            (_interaction(1), _interaction(2)), tmp_path / "holdout.jsonl"
        )
        queue_path = self._write_queue(tmp_path, queue)
        output = tmp_path / "golden-set.jsonl"
        report_path = tmp_path / "report.json"
        answers = iter(["account", "a", "note", "other", "e", ""])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

        assert cli.main(
            [
                "label-golden",
                "--queue",
                str(queue_path),
                "--holdout",
                str(holdout),
                "--out",
                str(output),
                "--report",
                str(report_path),
            ]
        ) == 0

        printed = capsys.readouterr().out
        assert f"queue: {queue_path} (2 queued)" in printed
        assert f"holdout: {holdout} (2 interactions)" in printed
        assert "labeled: 2 (auto 1, escalate 1)" in printed
        assert "remaining: 0" in printed
        assert "notes filled: 1" in printed
        assert f"golden set written: {output}" in printed
        assert f"report written: {report_path}" in printed

        records = golden.read_golden_jsonl(output)
        assert [record.gold_intent for record in records] == ["account", "other"]
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["golden_set_version"] == golden.GOLDEN_SET_VERSION
        assert payload["taxonomy_version"] == 1
        assert payload["queue_total"] == 2
        assert payload["labeled"] == 2
        assert payload["remaining"] == 0
        assert payload["complete"] is True
        assert payload["auto_share"] == 0.5
        assert payload["by_decision"] == {"auto": 1, "escalate": 1}
        assert payload["hint_agreement"]["intent"]["agree"] == 2
        assert payload["hint_agreement"]["decision"]["agree"] == 2

    def test_pause_without_labels_prints_no_golden_set_line(
        self, tmp_path, capsys, monkeypatch
    ):
        queue = (_queue_item(1),)
        holdout = interactions_mod.write_interactions_jsonl(
            (_interaction(1),), tmp_path / "holdout.jsonl"
        )
        queue_path = self._write_queue(tmp_path, queue)
        output = tmp_path / "golden-set.jsonl"
        monkeypatch.setattr("builtins.input", lambda prompt="": "q")

        assert cli.main(
            [
                "label-golden",
                "--queue",
                str(queue_path),
                "--holdout",
                str(holdout),
                "--out",
                str(output),
            ]
        ) == 0

        printed = capsys.readouterr().out
        assert "labeled: 0" in printed
        assert "golden set written" not in printed
        assert not output.is_file()

    def test_missing_queue_returns_1(self, tmp_path, capsys):
        holdout = interactions_mod.write_interactions_jsonl(
            (_interaction(1),), tmp_path / "holdout.jsonl"
        )

        assert cli.main(
            [
                "label-golden",
                "--queue",
                str(tmp_path / "missing.jsonl"),
                "--holdout",
                str(holdout),
                "--out",
                str(tmp_path / "golden-set.jsonl"),
            ]
        ) == 1
        assert "does not exist" in capsys.readouterr().err

    def test_unknown_queue_intent_returns_1(self, tmp_path, capsys):
        queue_path = self._write_queue(tmp_path, (_queue_item(1, "retired"),))
        holdout = interactions_mod.write_interactions_jsonl(
            (_interaction(1),), tmp_path / "holdout.jsonl"
        )

        assert cli.main(
            [
                "label-golden",
                "--queue",
                str(queue_path),
                "--holdout",
                str(holdout),
                "--out",
                str(tmp_path / "golden-set.jsonl"),
            ]
        ) == 1
        assert "unknown hint_intent" in capsys.readouterr().err

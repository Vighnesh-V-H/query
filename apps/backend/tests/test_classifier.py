import json

import pytest

from query import classifier, cli, llm, taxonomy


def _final(version=1, intents=None):
    intents = intents if intents is not None else (
        ("billing_payment", "Money matters: charges and refunds."),
        ("playback", "Getting music to play."),
        ("other", "Fallback class for messages that fit no support intent."),
    )
    return taxonomy.FinalTaxonomy(
        version=version,
        intents=tuple(
            taxonomy.FinalIntent(
                intent_id=intent_id,
                definition=definition,
                origin="seed",
                examples=(f"{intent_id} example one", f"{intent_id} example two"),
            )
            for intent_id, definition in intents
        ),
        seed_decisions=(),
        new_theme_decisions=(),
    )


def _reply(intent, confidence=0.9):
    return json.dumps({"intent": intent, "confidence": confidence})


class _StubClassifier:
    """Returns canned classifier replies by call order, recording prompts."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return llm.LLMReply(content=self._replies.pop(0), model="test/classifier")


class TestBuildPrompt:
    def test_prompt_carries_version_ids_definitions_examples_and_message(self):
        final = _final(version=3)

        prompt = classifier.build_prompt("I was charged twice", final)

        assert "version 3" in prompt
        assert "`billing_payment`" in prompt
        assert "`playback`" in prompt
        assert "`other`" in prompt
        assert "Money matters" in prompt
        assert "billing_payment example one" in prompt
        assert "I was charged twice" in prompt
        assert "confidence" in prompt.lower()

    def test_prompt_changes_with_taxonomy_version(self):
        old = classifier.build_prompt("hello", _final(version=1))
        new = classifier.build_prompt("hello", _final(version=2))

        assert old != new
        assert "version 1" in old
        assert "version 2" in new

    def test_prompt_lists_every_taxonomy_intent(self):
        final = _final()

        prompt = classifier.build_prompt("hello", final)

        for intent in final.intents:
            assert f"`{intent.intent_id}`" in prompt

    def test_prompt_builds_over_the_committed_taxonomy(self):
        final = taxonomy.read_final_taxonomy()

        prompt = classifier.build_prompt("my music is not playing", final)

        assert f"version {final.version}" in prompt
        assert len(final.intent_ids) == 13
        for intent_id in final.intent_ids:
            assert f"`{intent_id}`" in prompt


class TestClassifyIntent:
    def test_returns_intent_confidence_model_and_version(self):
        final = _final(version=2)
        stub = _StubClassifier([_reply("playback", 0.91)])

        prediction = classifier.classify_intent("my music is not playing", final, infer=stub)

        assert prediction.intent == "playback"
        assert prediction.confidence == pytest.approx(0.91)
        assert prediction.model == "test/classifier"
        assert prediction.taxonomy_version == 2
        assert len(stub.prompts) == 1
        assert "my music is not playing" in stub.prompts[0]

    def test_integer_confidence_is_normalized_to_float(self):
        prediction = classifier.classify_intent(
            "hello", _final(), infer=_StubClassifier([_reply("other", 1)])
        )

        assert prediction.confidence == 1.0
        assert isinstance(prediction.confidence, float)

    def test_boundary_confidences_are_accepted(self):
        for confidence in (0.0, 1.0):
            prediction = classifier.classify_intent(
                "hello", _final(), infer=_StubClassifier([_reply("other", confidence)])
            )
            assert prediction.confidence == confidence

    def test_code_fenced_reply_is_parsed(self):
        stub = _StubClassifier(["```json\n" + _reply("playback", 0.8) + "\n```"])

        prediction = classifier.classify_intent("not playing", _final(), infer=stub)

        assert prediction.intent == "playback"
        assert prediction.confidence == pytest.approx(0.8)

    def test_prose_wrapped_reply_is_parsed(self):
        stub = _StubClassifier([f"Here is my answer: {_reply('playback', 0.7)}"])

        prediction = classifier.classify_intent("not playing", _final(), infer=stub)

        assert prediction.intent == "playback"

    def test_unknown_intent_is_retried_then_raises(self):
        stub = _StubClassifier([_reply("refund_status", 0.9), _reply("refund_status", 0.9)])
        final = _final()

        with pytest.raises(classifier.ClassifierError, match="refund_status"):
            classifier.classify_intent("charged twice", final, infer=stub)

        assert len(stub.prompts) == 2
        assert "JSON" in stub.prompts[1]

    def test_recovers_when_retry_returns_a_valid_intent(self):
        stub = _StubClassifier(["not json at all", _reply("billing_payment", 0.85)])

        prediction = classifier.classify_intent("charged twice", _final(), infer=stub)

        assert prediction.intent == "billing_payment"
        assert len(stub.prompts) == 2
        assert "not json at all" in stub.prompts[1]

    @pytest.mark.parametrize(
        "confidence",
        [1.5, -0.1, "high", None, True, float("nan"), float("inf"), [0.9]],
    )
    def test_invalid_confidence_twice_raises(self, confidence):
        payload = json.dumps({"intent": "playback", "confidence": confidence})
        stub = _StubClassifier([payload, payload])

        with pytest.raises(classifier.ClassifierError, match="confidence"):
            classifier.classify_intent("not playing", _final(), infer=stub)

    def test_missing_intent_twice_raises(self):
        payload = json.dumps({"confidence": 0.9})
        stub = _StubClassifier([payload, payload])

        with pytest.raises(classifier.ClassifierError, match="unknown intent"):
            classifier.classify_intent("hello", _final(), infer=stub)

    def test_malformed_reply_twice_raises(self):
        stub = _StubClassifier(["not json", "still not json"])

        with pytest.raises(classifier.ClassifierError, match="no JSON object"):
            classifier.classify_intent("hello", _final(), infer=stub)

    @pytest.mark.parametrize("message", ["", "   ", "\n\t ", None, 123])
    def test_blank_or_non_string_message_raises_without_calling(self, message):
        def boom(prompt):
            raise AssertionError("classifier must not be called")

        with pytest.raises(classifier.ClassifierError, match="empty message"):
            classifier.classify_intent(message, _final(), infer=boom)

    def test_provider_failure_is_wrapped(self):
        def broken(prompt):
            raise RuntimeError("upstream down")

        with pytest.raises(classifier.ClassifierError, match="upstream down"):
            classifier.classify_intent("hello", _final(), infer=broken)

    def test_empty_taxonomy_raises(self):
        final = taxonomy.FinalTaxonomy(
            version=1, intents=(), seed_decisions=(), new_theme_decisions=()
        )

        with pytest.raises(classifier.ClassifierError, match="FinalTaxonomy"):
            classifier.classify_intent(
                "hello", final, infer=_StubClassifier([_reply("other")])
            )

    def test_call_labeler_uses_the_labeler_role(self, monkeypatch):
        seen = {}

        def fake_call_llm(prompt, role, system, temperature):
            seen["role"] = role
            seen["temperature"] = temperature
            return llm.LLMReply(content="{}", model="test/classifier")

        monkeypatch.setattr(classifier.llm, "call_llm", fake_call_llm)

        classifier.call_labeler("prompt")

        assert seen["role"] == "labeler"
        assert seen["temperature"] == 0.0

    def test_default_infer_is_the_configured_labeler(self, monkeypatch):
        monkeypatch.setattr(
            classifier,
            "call_labeler",
            _StubClassifier([_reply("playback", 0.6)]),
        )

        prediction = classifier.classify_intent("not playing", _final())

        assert prediction.intent == "playback"


class TestClassifyIntents:
    def test_batch_preserves_input_order_with_workers(self):
        final = _final()

        def route(prompt):
            if "charged" in prompt:
                intent = "billing_payment"
            elif "playing" in prompt:
                intent = "playback"
            else:
                intent = "other"
            return llm.LLMReply(content=_reply(intent, 0.8), model="test/classifier")

        predictions = classifier.classify_intents(
            ("charged twice", "not playing", "thanks!"),
            final,
            infer=route,
            workers=3,
        )

        assert [p.intent for p in predictions] == [
            "billing_payment",
            "playback",
            "other",
        ]
        assert all(p.taxonomy_version == 1 for p in predictions)

    def test_workers_must_be_at_least_one(self):
        with pytest.raises(classifier.ClassifierError, match="workers"):
            classifier.classify_intents(("hello",), _final(), workers=0)

    def test_batch_failure_names_the_input_position(self):
        def flaky(prompt):
            if "bad" in prompt.split("Customer Message:")[1].split("Reply with")[0]:
                return llm.LLMReply(content="not json", model="test/classifier")
            return llm.LLMReply(content=_reply("playback", 0.9), model="test/classifier")

        with pytest.raises(classifier.ClassifierError, match="message 1"):
            classifier.classify_intents(
                ("music will not play", "bad", "also bad"),
                _final(),
                infer=flaky,
                workers=1,
            )


class TestClassifyIntentCli:
    def test_command_prints_intent_confidence_model_and_version(
        self, capsys, monkeypatch
    ):
        monkeypatch.setattr(
            classifier, "call_labeler", _StubClassifier([_reply("playback", 0.91)])
        )

        assert cli.main(["classify-intent", "--message", "my music is not playing"]) == 0

        output = capsys.readouterr().out
        assert "intent: playback" in output
        assert "confidence: 0.91" in output
        assert "model: test/classifier" in output
        assert "(v1)" in output

    def test_command_with_explicit_taxonomy(self, tmp_path, capsys, monkeypatch):
        path = tmp_path / "taxonomy.md"
        path.write_text(
            classifier.DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            classifier, "call_labeler", _StubClassifier([_reply("other", 0.5)])
        )

        assert cli.main(
            ["classify-intent", "--message", "thanks!", "--taxonomy", str(path)]
        ) == 0

        assert "intent: other" in capsys.readouterr().out

    def test_missing_taxonomy_file_returns_1(self, tmp_path, capsys):
        assert cli.main(
            [
                "classify-intent",
                "--message",
                "hello",
                "--taxonomy",
                str(tmp_path / "missing.md"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_blank_message_returns_1(self, capsys):
        assert cli.main(["classify-intent", "--message", "   "]) == 1
        assert "error:" in capsys.readouterr().err

    def test_invalid_reply_twice_returns_1(self, capsys, monkeypatch):
        monkeypatch.setattr(
            classifier, "call_labeler", _StubClassifier(["nope", "still nope"])
        )

        assert cli.main(["classify-intent", "--message", "hello"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_classifier_failure_returns_1(self, capsys, monkeypatch):
        def broken(prompt):
            raise RuntimeError("upstream down")

        monkeypatch.setattr(classifier, "call_labeler", broken)

        assert cli.main(["classify-intent", "--message", "hello"]) == 1
        error = capsys.readouterr().err
        assert "error:" in error
        assert "upstream down" in error

import json
import threading
from datetime import datetime, timezone

import pytest
from query import cli, intent_labels, llm, taxonomy
from query import interactions as interactions_mod
from query.interactions import Interaction, Turn

BRAND = "SpotifyCares"


def _interaction(interaction_id, text):
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


def _final(*intent_ids, version=1):
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
        seed_decisions=(),
        new_theme_decisions=(),
    )


def _reply(intent, justification="Decisive evidence."):
    return json.dumps({"intent": intent, "justification": justification})


class _SequenceLabeler:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return llm.LLMReply(content=self.replies.pop(0), model="test/labeler")


class _RoutingLabeler:
    """Returns an intent based on a marker in the message, thread-safe.

    Markers (BILLMARK/PLAYMARK/ACCTMARK) appear only in the message text, so
    routing keys off the message rather than the taxonomy definitions and
    boundary rules that the prompt also carries.
    """

    MARKERS = (
        ("BILLMARK", "billing_payment"),
        ("PLAYMARK", "playback"),
        ("ACCTMARK", "account"),
    )

    def __init__(self, default="other"):
        self.default = default
        self.prompts = []
        self._lock = threading.Lock()

    def __call__(self, prompt):
        with self._lock:
            self.prompts.append(prompt)
        for marker, intent in self.MARKERS:
            if marker in prompt:
                chosen = intent
                break
        else:
            chosen = self.default
        return llm.LLMReply(
            content=_reply(chosen, f"The message shows {chosen}."),
            model="test/labeler",
        )


class _BoomLabeler:
    def __call__(self, prompt):
        raise AssertionError("labeler must not be called")


def _pool(*texts):
    return tuple(
        _interaction(number, text) for number, text in enumerate(texts, start=1)
    )


class TestSelectDevSlice:
    def test_same_seed_selects_same_slice_regardless_of_input_order(self):
        forward = _pool("one", "two", "three", "four", "five", "six")
        backward = tuple(reversed(forward))

        first = intent_labels.select_dev_slice(forward, dev_size=3, seed=7)
        second = intent_labels.select_dev_slice(backward, dev_size=3, seed=7)

        assert [item.interaction_id for item in first] == [
            item.interaction_id for item in second
        ]
        assert [item.interaction_id for item in first] == sorted(
            item.interaction_id for item in first
        )

    def test_slice_covers_everything_when_dev_size_exceeds_input(self):
        pool = _pool("one", "two")

        picked = intent_labels.select_dev_slice(pool, dev_size=50, seed=42)

        assert len(picked) == 2

    def test_zero_dev_size_raises(self):
        with pytest.raises(intent_labels.IntentLabelError, match="dev size"):
            intent_labels.select_dev_slice(_pool("one"), dev_size=0)

    def test_empty_input_raises(self):
        with pytest.raises(intent_labels.IntentLabelError, match="no interactions"):
            intent_labels.select_dev_slice((), dev_size=5)

    def test_duplicate_interaction_ids_raise(self):
        pool = (_interaction(1, "one"), _interaction(1, "one again"))

        with pytest.raises(intent_labels.IntentLabelError, match="duplicate"):
            intent_labels.select_dev_slice(pool, dev_size=1)


class TestLabelDevSlice:
    def test_labels_the_ranked_slice_with_provenance(self):
        pool = _pool(
            "my bill charge BILLMARK",
            "music will not play PLAYMARK",
            "cannot log in ACCTMARK",
        )
        final = _final("billing_payment", "playback", "account", "other")

        labels, report = intent_labels.label_dev_slice(
            pool, final, dev_size=3, infer=_RoutingLabeler()
        )

        assert [label.interaction_id for label in labels] == [1, 2, 3]
        assert [label.intent for label in labels] == [
            "billing_payment",
            "playback",
            "account",
        ]
        assert all(label.source == "labeler" for label in labels)
        assert all(label.model == "test/labeler" for label in labels)
        assert all(label.taxonomy_version == 1 for label in labels)
        assert labels[0].customer_message == "my bill charge BILLMARK"
        assert report.total == 3
        assert report.input_total == 3
        assert report.requested == 3
        assert report.seed == intent_labels.DEFAULT_SEED
        assert report.taxonomy_version == 1
        assert report.models == ("test/labeler",)
        assert report.labeler == 3
        assert report.human == 0

    def test_report_distribution_includes_zero_count_intents(self):
        pool = _pool("music will not play PLAYMARK")
        final = _final("playback", "billing_payment", "other")

        _, report = intent_labels.label_dev_slice(
            pool, final, dev_size=1, infer=_RoutingLabeler()
        )

        assert report.per_intent == {
            "playback": 1,
            "billing_payment": 0,
            "other": 0,
        }

    def test_dev_size_selects_a_stable_subset(self):
        pool = _pool("a one", "b two", "c three", "d four", "e five", "f six")
        final = _final("playback", "other")

        labels, report = intent_labels.label_dev_slice(
            pool, final, dev_size=2, seed=9, infer=_RoutingLabeler()
        )
        expected = intent_labels.select_dev_slice(pool, dev_size=2, seed=9)

        assert [label.interaction_id for label in labels] == [
            item.interaction_id for item in expected
        ]
        assert report.requested == 2
        assert report.total == 2
        assert report.input_total == 6

    def test_prompt_carries_definitions_message_and_boundaries(self):
        pool = _pool("charged twice on my card")
        final = _final("billing_payment", "playback", "other")
        labeler = _SequenceLabeler([_reply("billing_payment", "A charge problem.")])

        labels, _ = intent_labels.label_dev_slice(pool, final, infer=labeler)

        prompt = labeler.prompts[0]
        assert "billing_payment: billing_payment issues." in prompt
        assert "charged twice on my card" in prompt
        assert "other is only for messages that state no issue" in prompt
        assert labels[0].justification == "A charge problem."

    def test_justification_whitespace_is_normalized(self):
        pool = _pool("hello")
        final = _final("other")
        labeler = _SequenceLabeler([_reply("other", "  Praise   and chatter.  ")])

        labels, _ = intent_labels.label_dev_slice(pool, final, infer=labeler)

        assert labels[0].justification == "Praise and chatter."

    def test_unknown_intent_is_repaired_once(self):
        pool = _pool("hello")
        final = _final("other")
        labeler = _SequenceLabeler(
            [_reply("mystery"), _reply("other", "No support issue.")]
        )

        labels, _ = intent_labels.label_dev_slice(pool, final, infer=labeler)

        assert labels[0].intent == "other"
        assert len(labeler.prompts) == 2
        assert "previous reply was not valid JSON" in labeler.prompts[1]

    def test_unknown_intent_twice_raises(self):
        pool = _pool("hello")
        final = _final("other")
        labeler = _SequenceLabeler([_reply("mystery"), _reply("mystery")])

        with pytest.raises(intent_labels.IntentLabelError, match="unknown intent"):
            intent_labels.label_dev_slice(pool, final, infer=labeler)

    def test_empty_justification_twice_raises(self):
        pool = _pool("hello")
        final = _final("other")
        labeler = _SequenceLabeler([_reply("other", "   "), _reply("other", "")])

        with pytest.raises(intent_labels.IntentLabelError, match="empty justification"):
            intent_labels.label_dev_slice(pool, final, infer=labeler)

    def test_labeler_failure_is_wrapped_with_interaction_id(self):
        pool = (_interaction(42, "hello"),)
        final = _final("other")

        def broken(prompt):
            raise RuntimeError("upstream down")

        with pytest.raises(intent_labels.IntentLabelError, match="42"):
            intent_labels.label_dev_slice(pool, final, infer=broken)

    def test_workers_must_be_at_least_one(self):
        with pytest.raises(intent_labels.IntentLabelError, match="workers"):
            intent_labels.label_dev_slice(
                _pool("hello"), _final("other"), workers=0, infer=_BoomLabeler()
            )

    def test_parallel_workers_preserve_interaction_order(self):
        pool = (
            _interaction(1, "bill charge BILLMARK"),
            _interaction(2, "music will not play PLAYMARK"),
            _interaction(3, "just saying hello"),
            _interaction(4, "cannot log in ACCTMARK"),
        )
        final = _final("billing_payment", "playback", "account", "other")

        labels, _ = intent_labels.label_dev_slice(
            pool, final, workers=4, infer=_RoutingLabeler()
        )

        assert [label.interaction_id for label in labels] == [1, 2, 3, 4]
        assert [label.intent for label in labels] == [
            "billing_payment",
            "playback",
            "other",
            "account",
        ]

    def test_empty_taxonomy_raises(self):
        with pytest.raises(intent_labels.IntentLabelError, match="no intents"):
            intent_labels.label_dev_slice(
                _pool("hello"), _final(), infer=_BoomLabeler()
            )

    def test_call_labeler_uses_the_labeler_role(self, monkeypatch):
        seen = {}

        def fake_call_llm(prompt, role, system, temperature):
            seen["role"] = role
            seen["temperature"] = temperature
            return llm.LLMReply(content="{}", model="test/labeler")

        monkeypatch.setattr(intent_labels.llm, "call_llm", fake_call_llm)

        intent_labels.call_labeler("prompt")

        assert seen["role"] == "labeler"
        assert seen["temperature"] == 0.0

    def test_committed_taxonomy_is_labelable(self):
        final = taxonomy.read_final_taxonomy()
        pool = _pool("I was charged twice, please refund")
        labeler = _SequenceLabeler([_reply("billing_payment", "A charge problem.")])

        labels, report = intent_labels.label_dev_slice(pool, final, infer=labeler)

        assert labels[0].intent == "billing_payment"
        assert labels[0].taxonomy_version == final.version
        assert set(report.per_intent) == set(final.intent_ids)
        assert "billing_payment" in labeler.prompts[0]


class TestIntentLabelCache:
    def test_reuses_verdicts_on_rerun(self, tmp_path):
        pool = _pool("hello one", "hello two")
        final = _final("other")
        cache = intent_labels.IntentLabelCache(tmp_path / "cache.jsonl")
        labeler = _SequenceLabeler([_reply("other"), _reply("other")])

        first, _ = intent_labels.label_dev_slice(
            pool, final, cache=cache, infer=labeler
        )
        second, _ = intent_labels.label_dev_slice(
            pool, final, cache=cache, infer=_BoomLabeler()
        )

        assert second == first
        assert cache.verdicts().keys() == {1, 2}

    def test_changed_taxonomy_recomputes_stale_entries(self, tmp_path):
        pool = _pool("hello")
        path = tmp_path / "cache.jsonl"
        intent_labels.label_dev_slice(
            pool,
            _final("other"),
            cache=intent_labels.IntentLabelCache(path),
            infer=_SequenceLabeler([_reply("other")]),
        )

        labeler = _SequenceLabeler([_reply("playback", "Now a fault.")])
        labels, _ = intent_labels.label_dev_slice(
            pool,
            _final("playback", "other", version=2),
            cache=intent_labels.IntentLabelCache(path),
            infer=labeler,
        )

        assert labels[0].intent == "playback"
        assert len(labeler.prompts) == 1

    def test_cached_verdict_with_unknown_intent_raises(self, tmp_path):
        pool = _pool("hello")
        path = tmp_path / "cache.jsonl"
        intent_labels.label_dev_slice(
            pool,
            _final("other"),
            cache=intent_labels.IntentLabelCache(path),
            infer=_SequenceLabeler([_reply("other")]),
        )
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        record["intent"] = "mystery"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(intent_labels.IntentLabelError, match="contract"):
            intent_labels.label_dev_slice(
                pool,
                _final("other"),
                cache=intent_labels.IntentLabelCache(path),
                infer=_BoomLabeler(),
            )


def _valid_label_json(**overrides):
    record = {
        "interaction_id": 1,
        "customer_message": "charged twice",
        "intent": "billing_payment",
        "source": "labeler",
        "justification": "A charge problem.",
        "model": "test/labeler",
        "taxonomy_version": 1,
    }
    record.update(overrides)
    return record


def _write_labels(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


class TestIntentLabelArtifacts:
    def test_stage_and_read_round_trip(self, tmp_path):
        labels = (
            intent_labels.IntentLabel(
                interaction_id=1,
                customer_message="charged twice",
                intent="billing_payment",
                source="labeler",
                justification="A charge problem.",
                model="test/labeler",
                taxonomy_version=1,
            ),
            intent_labels.IntentLabel(
                interaction_id=2,
                customer_message="love spotify!",
                intent="other",
                source="human",
                justification="Praise with no request.",
                model=None,
                taxonomy_version=1,
            ),
        )
        path = tmp_path / "labels.jsonl"

        temporary = intent_labels.stage_intent_labels_jsonl(labels, path)
        temporary.replace(path)
        loaded = intent_labels.read_intent_labels_jsonl(
            path, intent_ids=("billing_payment", "other")
        )

        assert loaded == labels

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(intent_labels.IntentLabelError, match="does not exist"):
            intent_labels.read_intent_labels_jsonl(tmp_path / "missing.jsonl")

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text("{not json\n", encoding="utf-8")

        with pytest.raises(intent_labels.IntentLabelError, match="malformed JSON"):
            intent_labels.read_intent_labels_jsonl(path)

    def test_duplicate_interaction_id_raises(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl",
            [_valid_label_json(), _valid_label_json()],
        )

        with pytest.raises(intent_labels.IntentLabelError, match="duplicate"):
            intent_labels.read_intent_labels_jsonl(path)

    def test_unknown_intent_raises_with_taxonomy(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl", [_valid_label_json(intent="mystery")]
        )

        with pytest.raises(intent_labels.IntentLabelError, match="unknown intent"):
            intent_labels.read_intent_labels_jsonl(
                path, intent_ids=("billing_payment",)
            )

    def test_unknown_intent_passes_shape_check_without_taxonomy(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl", [_valid_label_json(intent="mystery")]
        )

        loaded = intent_labels.read_intent_labels_jsonl(path)

        assert loaded[0].intent == "mystery"

    def test_unknown_source_raises(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl", [_valid_label_json(source="heuristic")]
        )

        with pytest.raises(intent_labels.IntentLabelError, match="invalid source"):
            intent_labels.read_intent_labels_jsonl(path)

    def test_empty_justification_raises(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl", [_valid_label_json(justification="  ")]
        )

        with pytest.raises(intent_labels.IntentLabelError, match="justification"):
            intent_labels.read_intent_labels_jsonl(path)

    def test_labeler_record_without_model_raises(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl", [_valid_label_json(model=None)]
        )

        with pytest.raises(intent_labels.IntentLabelError, match="need a model"):
            intent_labels.read_intent_labels_jsonl(path)

    def test_human_record_with_model_raises(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl",
            [_valid_label_json(source="human", model="test/labeler")],
        )

        with pytest.raises(intent_labels.IntentLabelError, match="carry no model"):
            intent_labels.read_intent_labels_jsonl(path)

    def test_bad_taxonomy_version_raises(self, tmp_path):
        path = _write_labels(
            tmp_path / "labels.jsonl", [_valid_label_json(taxonomy_version=0)]
        )

        with pytest.raises(intent_labels.IntentLabelError, match="taxonomy_version"):
            intent_labels.read_intent_labels_jsonl(path)


class TestReviewMarkdown:
    def test_contains_distribution_and_capped_examples(self):
        final = _final("playback", "billing_payment", "other")
        labels = (
            intent_labels.IntentLabel(
                interaction_id=1,
                customer_message="music will not play",
                intent="playback",
                source="labeler",
                justification="A playback fault.",
                model="test/labeler",
                taxonomy_version=1,
            ),
            intent_labels.IntentLabel(
                interaction_id=2,
                customer_message="charged twice",
                intent="billing_payment",
                source="human",
                justification="A charge problem.",
                model=None,
                taxonomy_version=1,
            ),
        )
        report = intent_labels.IntentLabelReport(
            total=2,
            input_total=10,
            requested=2,
            seed=42,
            taxonomy_version=1,
            per_intent={"playback": 1, "billing_payment": 1, "other": 0},
            labeler=1,
            human=1,
            models=("test/labeler",),
        )

        markdown = intent_labels.render_review_markdown(
            labels, report, final, examples_per_intent=1
        )

        assert "# Intent dev-label review" in markdown
        assert "| Intent | Labels |" in markdown
        assert "| `playback` | 1 |" in markdown
        assert "| `other` | 0 |" in markdown
        assert "music will not play" in markdown
        assert "_No labels in this slice._" in markdown


class TestLabelIntentsCli:
    def _write_pool(self, tmp_path):
        pool = _pool(
            "my bill charge BILLMARK",
            "music will not play PLAYMARK",
        )
        return interactions_mod.write_interactions_jsonl(
            pool, tmp_path / "rag-pool.jsonl"
        )

    def test_command_writes_labels_report_and_review(
        self, tmp_path, capsys, monkeypatch
    ):
        input_path = self._write_pool(tmp_path)
        out_path = tmp_path / "intent-dev-labels.jsonl"
        report_path = tmp_path / "intent-dev-report.json"
        review_path = tmp_path / "review.md"
        monkeypatch.setattr(
            intent_labels, "call_labeler", _RoutingLabeler(default="playback")
        )

        assert (
            cli.main(
                [
                    "label-intents",
                    "--in",
                    str(input_path),
                    "--dev-size",
                    "2",
                    "--seed",
                    "42",
                    "--out",
                    str(out_path),
                    "--report",
                    str(report_path),
                    "--review",
                    str(review_path),
                ]
            )
            == 0
        )

        output = capsys.readouterr().out
        assert f"input: {input_path} (2 interactions)" in output
        assert "taxonomy:" in output and "(v1)" in output
        assert "dev slice: 2 of 2 (requested 2, seed 42)" in output
        assert "billing_payment: 1" in output
        assert "sources: labeler 2, human 0" in output
        assert f"labels written: {out_path}" in output
        assert f"report written: {report_path}" in output
        assert f"review written: {review_path}" in output

        records = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 2
        assert all(record["source"] == "labeler" for record in records)
        assert all(record["taxonomy_version"] == 1 for record in records)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["total"] == 2
        assert payload["taxonomy_version"] == 1
        assert payload["sources"] == {"labeler": 2, "human": 0}
        assert payload["per_intent"]["billing_payment"] == 1
        assert "# Intent dev-label review" in review_path.read_text(encoding="utf-8")
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(
        self, tmp_path, capsys, monkeypatch
    ):
        input_path = self._write_pool(tmp_path)
        monkeypatch.setattr(
            intent_labels, "call_labeler", _RoutingLabeler(default="playback")
        )

        assert (
            cli.main(["label-intents", "--in", str(input_path), "--dev-size", "2"])
            == 0
        )

        output = capsys.readouterr().out
        assert "dev slice:" in output
        assert "labels written:" not in output
        assert "report written:" not in output
        assert "review written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys):
        assert (
            cli.main(
                [
                    "label-intents",
                    "--in",
                    str(tmp_path / "missing.jsonl"),
                    "--dev-size",
                    "2",
                ]
            )
            == 1
        )
        assert "error:" in capsys.readouterr().err

    def test_missing_taxonomy_returns_1(self, tmp_path, capsys):
        input_path = self._write_pool(tmp_path)

        assert (
            cli.main(
                [
                    "label-intents",
                    "--in",
                    str(input_path),
                    "--taxonomy",
                    str(tmp_path / "missing.md"),
                    "--dev-size",
                    "2",
                ]
            )
            == 1
        )
        assert "error:" in capsys.readouterr().err

    def test_output_cannot_overwrite_input_returns_1(self, tmp_path, capsys):
        input_path = self._write_pool(tmp_path)

        assert (
            cli.main(
                [
                    "label-intents",
                    "--in",
                    str(input_path),
                    "--dev-size",
                    "2",
                    "--out",
                    str(input_path),
                ]
            )
            == 1
        )
        assert "error:" in capsys.readouterr().err

    def test_command_reuses_cache_on_rerun(self, tmp_path, capsys, monkeypatch):
        input_path = self._write_pool(tmp_path)
        out_path = tmp_path / "intent-dev-labels.jsonl"
        cache_path = tmp_path / "intent-cache.jsonl"
        args = [
            "label-intents",
            "--in",
            str(input_path),
            "--dev-size",
            "2",
            "--out",
            str(out_path),
            "--cache",
            str(cache_path),
        ]
        monkeypatch.setattr(
            intent_labels, "call_labeler", _RoutingLabeler(default="playback")
        )
        assert cli.main(args) == 0
        first = out_path.read_text(encoding="utf-8")
        capsys.readouterr()

        def boom(prompt):
            raise AssertionError("cache should have been used")

        monkeypatch.setattr(intent_labels, "call_labeler", boom)

        assert cli.main(args) == 0

        assert out_path.read_text(encoding="utf-8") == first
        assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2

    def test_write_failure_leaves_outputs_untouched(
        self, tmp_path, capsys, monkeypatch
    ):
        input_path = self._write_pool(tmp_path)
        out_path = tmp_path / "intent-dev-labels.jsonl"
        report_path = tmp_path / "report.json"
        out_path.write_text("previous labels\n", encoding="utf-8")
        report_path.write_text("previous report\n", encoding="utf-8")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(
            intent_labels, "call_labeler", _RoutingLabeler(default="playback")
        )

        assert (
            cli.main(
                [
                    "label-intents",
                    "--in",
                    str(input_path),
                    "--dev-size",
                    "2",
                    "--out",
                    str(blocker / "intent-dev-labels.jsonl"),
                    "--report",
                    str(report_path),
                ]
            )
            == 1
        )

        assert "error:" in capsys.readouterr().err
        assert out_path.read_text(encoding="utf-8") == "previous labels\n"
        assert report_path.read_text(encoding="utf-8") == "previous report\n"
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

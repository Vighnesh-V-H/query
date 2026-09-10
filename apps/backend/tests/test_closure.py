import json
from datetime import datetime, timezone

from query import cli
from query import closure as closure_mod
from query import interactions as interactions_mod
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


def _verdicts(interactions):
    labels, _ = closure_mod.label_closures(interactions)
    return labels


class TestLabelClosures:
    def test_customer_acknowledgement_is_resolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my app keeps crashing"),
                _turn(2, "brand", "Try a clean reinstall."),
                _turn(3, "customer", "That worked, thank you so much!"),
            ),
        )

        labels, report = closure_mod.label_closures(found)

        assert labels[0].label == "resolved"
        assert labels[0].needs_adjudication is False
        assert labels[0].reason
        assert report.resolved == 1
        assert report.needs_adjudication == 0

    def test_customer_fixed_message_is_resolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Try resetting your password."),
                _turn(3, "customer", "It's working now, thanks!"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "resolved"

    def test_customer_continuation_is_unresolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Try resetting your password."),
                _turn(3, "customer", "Still not working, can you help?"),
            ),
        )

        labels, report = closure_mod.label_closures(found)

        assert labels[0].label == "unresolved"
        assert labels[0].needs_adjudication is False
        assert report.unresolved == 1

    def test_customer_thanks_but_continues_is_unresolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Try resetting your password."),
                _turn(3, "customer", "Thanks but it still doesn't work"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "unresolved"

    def test_customer_fixed_but_continues_is_unresolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "login is broken"),
                _turn(2, "brand", "Try resetting your password."),
                _turn(3, "customer", "It works now, but my playlists are still gone"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "unresolved"

    def test_brand_completion_is_resolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I was charged twice"),
                _turn(2, "brand", "We've processed the refund, you're all set."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "resolved"
        assert labels[0].needs_adjudication is False

    def test_brand_closing_after_acknowledgement_is_resolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I was charged twice"),
                _turn(2, "brand", "We've processed the refund."),
                _turn(3, "customer", "Thank you!"),
                _turn(4, "brand", "No problem! We're here if you need us."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "resolved"
        assert labels[0].needs_adjudication is False

    def test_brand_refusal_is_unresolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "can I get a discount?"),
                _turn(2, "brand", "Unfortunately we can't offer discounts on Premium."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "unresolved"
        assert labels[0].needs_adjudication is False

    def test_brand_dm_deflection_is_flagged(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I can't log in"),
                _turn(2, "brand", "Can you DM us your account's email address?"),
            ),
        )

        labels, report = closure_mod.label_closures(found)

        assert labels[0].label is None
        assert labels[0].needs_adjudication is True
        assert "DM" in labels[0].reason
        assert report.needs_adjudication == 1

    def test_brand_promise_to_investigate_is_not_resolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "my playlist disappeared"),
                _turn(2, "brand", "We'll take a look and get back to you."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label != "resolved"
        assert labels[0].needs_adjudication is False

    def test_brand_question_with_silence_is_uncertain(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "the app won't open"),
                _turn(2, "brand", "Can you let us know which device you're using?"),
            ),
        )

        labels, report = closure_mod.label_closures(found)

        assert labels[0].label == "uncertain"
        assert labels[0].needs_adjudication is False
        assert report.uncertain == 1

    def test_brand_plain_answer_with_silence_is_uncertain(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "how much is the offer?"),
                _turn(2, "brand", "It's 129 pesos for mobile payments."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "uncertain"
        assert labels[0].needs_adjudication is False

    def test_completion_wins_over_dm_mention(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I can't log in"),
                _turn(2, "brand", "We've fixed it, check your DMs for the details."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "resolved"

    def test_incidental_thanks_do_not_mask_dm_deflection(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I lost my playlists, can somebody call me? Thanks"),
                _turn(2, "brand", "Can you send us your email address via DM? We'll check."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label is None
        assert labels[0].needs_adjudication is True

    def test_deferred_followup_after_acknowledgement_is_uncertain(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "the app keeps skipping songs"),
                _turn(2, "brand", "We're on it."),
                _turn(3, "customer", "Thanks!"),
                _turn(4, "brand", "No problem, we'll pass your feedback onto the right team."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "uncertain"

    def test_opening_message_cannot_acknowledge_help(self):
        # The opening asks for a feature; "awesome" is not a thank-you.
        found = (
            _interaction(
                1,
                _turn(1, "customer", "it would be awesome if you added lyrics to every song"),
                _turn(2, "brand", "Here is how to find lyrics."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "uncertain"

    def test_customer_ack_after_dm_deflection_is_flagged(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I can't log in"),
                _turn(
                    2,
                    "brand",
                    "We've just replied to your DM. Let's carry on chatting there.",
                ),
                _turn(3, "customer", "Thanks!"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label is None
        assert labels[0].needs_adjudication is True

    def test_long_message_with_incidental_thanks_is_not_resolved(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "why is the account page so useless?"),
                _turn(2, "brand", "We appreciate the feedback."),
                _turn(
                    3,
                    "customer",
                    "Thanks... I suggest you eliminate the account page from the app. "
                    "It is completely useless and confusing.",
                ),
                _turn(4, "brand", "We'll make sure to pass your feedback onto the team."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label != "resolved"

    def test_customer_dmed_reply_is_flagged(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "I can't log in"),
                _turn(2, "brand", "Can you DM us your account's email?"),
                _turn(3, "customer", "DMed"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label is None
        assert labels[0].needs_adjudication is True
        assert labels[0].reason

    def test_customer_neutral_reply_is_flagged(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "the new update is weird"),
                _turn(2, "brand", "We'd love to hear more about it."),
                _turn(3, "customer", "Omg 💀😂"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label is None
        assert labels[0].needs_adjudication is True

    def test_short_agreement_is_not_guessed(self):
        # "Sure" could accept an offer or simply acknowledge a message; it is
        # not a clear acknowledgement of resolution.
        found = (
            _interaction(
                1,
                _turn(1, "customer", "are you there?"),
                _turn(2, "brand", "We can help with that."),
                _turn(3, "customer", "Sure 😊"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label is None
        assert labels[0].needs_adjudication is True

    def test_typographic_apostrophes_still_match(self):
        found = (
            _interaction(
                1,
                _turn(1, "customer", "can I get a discount?"),
                _turn(2, "brand", "We’re afraid we can’t offer discounts on Premium."),
            ),
            _interaction(
                2,
                _turn(1, "customer", "I was charged twice"),
                _turn(2, "brand", "We’ve processed the refund — you’re all set."),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        assert labels[0].label == "unresolved"
        assert labels[1].label == "resolved"

    def test_every_verdict_has_a_reason(self):
        found = (
            _interaction(
                1, _turn(1, "customer", "hi"), _turn(2, "brand", "Can you DM us your email?")
            ),
            _interaction(
                2, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it, you're all set.")
            ),
            _interaction(3, _turn(1, "customer", "hi"), _turn(2, "brand", "Unfortunately we can't help.")),
            _interaction(4, _turn(1, "customer", "hi"), _turn(2, "brand", "Here's how it works.")),
            _interaction(
                5,
                _turn(1, "customer", "hi"),
                _turn(2, "brand", "Here's how it works."),
                _turn(3, "customer", "Thanks!"),
            ),
            _interaction(
                6,
                _turn(1, "customer", "hi"),
                _turn(2, "brand", "Here's how it works."),
                _turn(3, "customer", "Still broken!"),
            ),
            _interaction(
                7,
                _turn(1, "customer", "hi"),
                _turn(2, "brand", "Here's how it works."),
                _turn(3, "customer", "ok"),
            ),
        )

        labels, _ = closure_mod.label_closures(found)

        for label in labels:
            assert label.reason
            if label.needs_adjudication:
                assert label.label is None
            else:
                assert label.label in closure_mod.CLOSURE_LABELS

    def test_report_counts_and_label_rate(self):
        found = (
            _interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),
            _interaction(2, _turn(1, "customer", "hi"), _turn(2, "brand", "Unfortunately we can't.")),
            _interaction(3, _turn(1, "customer", "hi"), _turn(2, "brand", "Here's the answer.")),
            _interaction(4, _turn(1, "customer", "hi"), _turn(2, "brand", "Can you DM us?")),
        )

        _, report = closure_mod.label_closures(found)

        assert report.total == 4
        assert report.resolved == 1
        assert report.unresolved == 1
        assert report.uncertain == 1
        assert report.needs_adjudication == 1
        assert report.labeled == 3
        assert report.label_rate == 0.75

    def test_empty_input_reports_zero_rate(self):
        labels, report = closure_mod.label_closures(())

        assert labels == ()
        assert report.total == 0
        assert report.label_rate == 0.0

    def test_output_order_matches_input(self):
        found = (
            _interaction(10, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),
            _interaction(20, _turn(1, "customer", "hi"), _turn(2, "brand", "Can you DM us?")),
            _interaction(30, _turn(1, "customer", "hi"), _turn(2, "brand", "Here's the answer.")),
        )

        labels, _ = closure_mod.label_closures(found)

        assert [label.interaction_id for label in labels] == [10, 20, 30]

    def test_deterministic_across_runs(self):
        found = (
            _interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),
            _interaction(2, _turn(1, "customer", "hi"), _turn(2, "brand", "Can you DM us?")),
        )

        first, first_report = closure_mod.label_closures(found)
        second, second_report = closure_mod.label_closures(found)

        assert first == second
        assert first_report == second_report


class TestLabelClosureCli:
    def _write_input(self, tmp_path, interactions):
        return interactions_mod.write_interactions_jsonl(
            interactions, tmp_path / "rag-pool.jsonl"
        )

    def test_command_writes_labels_and_report(self, tmp_path, capsys):
        found = (
            _interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),
            _interaction(2, _turn(1, "customer", "hi"), _turn(2, "brand", "Can you DM us?")),
            _interaction(3, _turn(1, "customer", "hi"), _turn(2, "brand", "Here's the answer.")),
            _interaction(4, _turn(1, "customer", "hi"), _turn(2, "brand", "Unfortunately we can't.")),
        )
        input_path = self._write_input(tmp_path, found)
        out_path = tmp_path / "closure-labels.jsonl"
        report_path = tmp_path / "closure-report.json"

        assert cli.main(
            [
                "label-closure",
                "--in",
                str(input_path),
                "--out",
                str(out_path),
                "--report",
                str(report_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert f"input: {input_path} (4 interactions)" in output
        assert "resolved: 1" in output
        assert "uncertain: 1" in output
        assert "unresolved: 1" in output
        assert "needs adjudication: 1 (25.00%)" in output
        assert f"labels written: {out_path}" in output
        assert f"report written: {report_path}" in output

        records = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["interaction_id"] for record in records] == [1, 2, 3, 4]
        assert records[0]["label"] == "resolved"
        assert records[0]["needs_adjudication"] is False
        assert records[1]["label"] is None
        assert records[1]["needs_adjudication"] is True
        assert all(record["reason"] for record in records)

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["total"] == 4
        assert payload["resolved"] == 1
        assert payload["uncertain"] == 1
        assert payload["unresolved"] == 1
        assert payload["needs_adjudication"] == 1
        assert payload["labeled"] == 3
        assert payload["label_rate"] == 0.75
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(self, tmp_path, capsys):
        input_path = self._write_input(
            tmp_path,
            (_interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),),
        )

        assert cli.main(["label-closure", "--in", str(input_path)]) == 0

        output = capsys.readouterr().out
        assert "resolved: 1" in output
        assert "labels written:" not in output
        assert "report written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys):
        assert cli.main(
            ["label-closure", "--in", str(tmp_path / "missing.jsonl")]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_duplicate_interaction_ids_return_1(self, tmp_path, capsys):
        input_path = self._write_input(
            tmp_path,
            (
                _interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),
                _interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),
            ),
        )

        assert cli.main(["label-closure", "--in", str(input_path)]) == 1
        assert "duplicate interaction_id" in capsys.readouterr().err

    def test_output_cannot_overwrite_input_returns_1(self, tmp_path, capsys):
        input_path = self._write_input(
            tmp_path,
            (_interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),),
        )

        assert cli.main(
            ["label-closure", "--in", str(input_path), "--out", str(input_path)]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_write_failure_leaves_outputs_untouched(self, tmp_path, capsys):
        input_path = self._write_input(
            tmp_path,
            (_interaction(1, _turn(1, "customer", "hi"), _turn(2, "brand", "We've fixed it.")),),
        )
        out_path = tmp_path / "closure-labels.jsonl"
        report_path = tmp_path / "closure-report.json"
        out_path.write_text('{"previous": "labels"}\n', encoding="utf-8")
        report_path.write_text('{"previous": "report"}\n', encoding="utf-8")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        assert cli.main(
            [
                "label-closure",
                "--in",
                str(input_path),
                "--out",
                str(blocker / "closure-labels.jsonl"),
                "--report",
                str(report_path),
            ]
        ) == 1

        assert "error:" in capsys.readouterr().err
        assert out_path.read_text(encoding="utf-8") == '{"previous": "labels"}\n'
        assert report_path.read_text(encoding="utf-8") == '{"previous": "report"}\n'
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

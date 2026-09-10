import json
from datetime import datetime, timezone

from query import cli
from query import english as english_mod
from query import interactions as interactions_mod
from query.interactions import Interaction, Turn

BRAND = "SpotifyCares"


def _turn(tweet_id, side, text, author="cust", minute=0):
    return Turn(
        tweet_id=tweet_id,
        author_id=author,
        side=side,
        created_at=datetime(2017, 11, 1, 10, minute, tzinfo=timezone.utc),
        text=text,
    )


def _interaction(interaction_id, text, brand_text=None):
    turns = [_turn(interaction_id, "customer", text)]
    if brand_text is not None:
        turns.append(
            _turn(interaction_id + 1, "brand", brand_text, author=BRAND, minute=1)
        )
    return Interaction(
        interaction_id=interaction_id,
        customer_id="cust",
        brand_id=BRAND,
        turns=tuple(turns),
    )


ENGLISH = "my app keeps crashing on android, can you help"
SPANISH = "Hola, buenos días, necesito ayuda con mi cuenta"
FRENCH = "Bonjour, je ne reçois plus aucun mail de Spotify"
INDONESIAN = "saya mendapat promo 3 bulan cuma Rp 4990 itu benar ga?"
TAGALOG = "Bakit hindi ako nakaka avail? Palaging try again"
CHINESE = "客服提單編號09811169 哪時候可以處理 ?"


class TestDetectLanguage:
    def test_english_detected(self):
        assert english_mod.detect_language(ENGLISH) == "ENGLISH"

    def test_non_english_detected(self):
        assert english_mod.detect_language(SPANISH) == "SPANISH"
        assert english_mod.detect_language(FRENCH) == "FRENCH"
        assert english_mod.detect_language(INDONESIAN) == "INDONESIAN"
        assert english_mod.detect_language(TAGALOG) == "TAGALOG"
        assert english_mod.detect_language(CHINESE) == "CHINESE"

    def test_short_or_contentless_text_has_no_signal(self):
        assert english_mod.detect_language("How?") is None
        assert english_mod.detect_language("😭😭😭") is None
        assert english_mod.detect_language("https://t.co/YxliFtc8p8") is None

    def test_handles_and_links_are_stripped_before_detection(self):
        assert english_mod.detect_language("@115888 @SpotifyCares") is None
        assert english_mod.detect_language(f"@115888 {FRENCH} https://t.co/abc") == "FRENCH"


class TestFilterEnglish:
    def test_keeps_english_interactions(self):
        found = (_interaction(1, ENGLISH), _interaction(2, "How?"))

        retained, report = english_mod.filter_english(found)

        assert [interaction.interaction_id for interaction in retained] == [1, 2]
        assert report.total == 2
        assert report.retained == 2
        assert report.filtered == 0
        assert report.no_signal == 1

    def test_filters_non_english_and_reports_language(self):
        found = (
            _interaction(1, ENGLISH),
            _interaction(2, SPANISH),
            _interaction(3, FRENCH),
            _interaction(4, INDONESIAN),
            _interaction(5, TAGALOG),
            _interaction(6, CHINESE),
        )

        retained, report = english_mod.filter_english(found)

        assert [interaction.interaction_id for interaction in retained] == [1]
        assert report.total == 6
        assert report.retained == 1
        assert report.filtered == 5
        assert report.no_signal == 0
        assert report.filtered_by_language == {
            "CHINESE": 1,
            "FRENCH": 1,
            "INDONESIAN": 1,
            "SPANISH": 1,
            "TAGALOG": 1,
        }

    def test_filter_rate(self):
        found = (_interaction(1, ENGLISH), _interaction(2, SPANISH))

        _, report = english_mod.filter_english(found)

        assert report.filter_rate == 0.5

    def test_empty_input_reports_zero_rate(self):
        retained, report = english_mod.filter_english(())

        assert retained == ()
        assert report.total == 0
        assert report.filter_rate == 0.0

    def test_only_the_opening_message_decides(self):
        # brand turns in another language must not drop an English Interaction
        found = (_interaction(1, ENGLISH, brand_text=SPANISH),)

        retained, report = english_mod.filter_english(found)

        assert [interaction.interaction_id for interaction in retained] == [1]
        assert report.filtered == 0

    def test_retained_order_matches_input(self):
        found = (
            _interaction(10, ENGLISH),
            _interaction(20, SPANISH),
            _interaction(30, "my playlist disappeared overnight"),
        )

        retained, _ = english_mod.filter_english(found)

        assert [interaction.interaction_id for interaction in retained] == [10, 30]

    def test_deterministic_across_runs(self):
        found = (
            _interaction(1, ENGLISH),
            _interaction(2, SPANISH),
            _interaction(3, TAGALOG),
            _interaction(4, "How?"),
        )

        first_retained, first_report = english_mod.filter_english(found)
        second_retained, second_report = english_mod.filter_english(found)

        assert [i.interaction_id for i in first_retained] == [
            i.interaction_id for i in second_retained
        ]
        assert first_report.filtered_by_language == second_report.filtered_by_language


class TestFilterEnglishCli:
    def _write_input(self, tmp_path, interactions):
        return interactions_mod.write_interactions_jsonl(
            interactions, tmp_path / "interactions.jsonl"
        )

    def test_command_writes_interactions_and_report(self, tmp_path, capsys):
        found = (_interaction(1, ENGLISH), _interaction(2, SPANISH))
        input_path = self._write_input(tmp_path, found)
        out_path = tmp_path / "interactions-en.jsonl"
        report_path = tmp_path / "english-report.json"

        assert cli.main(
            [
                "filter-english",
                "--in",
                str(input_path),
                "--out",
                str(out_path),
                "--report",
                str(report_path),
            ]
        ) == 0

        output = capsys.readouterr().out
        assert f"input: {input_path} (2 interactions)" in output
        assert "retained: 1 (english 1, no confident signal 0)" in output
        assert "filtered: 1 (50.00%)" in output
        assert "filtered by language: SPANISH 1" in output
        assert f"interactions written: {out_path}" in output
        assert f"report written: {report_path}" in output

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["interaction_id"] == 1

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["total"] == 2
        assert payload["retained"] == 1
        assert payload["filtered"] == 1
        assert payload["no_signal"] == 0
        assert payload["filter_rate"] == 0.5
        assert payload["filtered_by_language"] == {"SPANISH": 1}
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_report_write_failure_keeps_previous_report(
        self, tmp_path, capsys, monkeypatch
    ):
        input_path = self._write_input(tmp_path, (_interaction(1, ENGLISH),))
        report_path = tmp_path / "english-report.json"
        report_path.write_text('{"previous": true}\n', encoding="utf-8")

        def failing_replace(source, destination):
            raise OSError("disk full")

        monkeypatch.setattr(cli.os, "replace", failing_replace)

        assert cli.main(
            ["filter-english", "--in", str(input_path), "--report", str(report_path)]
        ) == 1
        assert "error:" in capsys.readouterr().err
        assert json.loads(report_path.read_text(encoding="utf-8")) == {"previous": True}
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, (_interaction(1, ENGLISH),))

        assert cli.main(["filter-english", "--in", str(input_path)]) == 0

        output = capsys.readouterr().out
        assert "retained: 1" in output
        assert "interactions written:" not in output
        assert "report written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys):
        assert cli.main(
            ["filter-english", "--in", str(tmp_path / "missing.jsonl")]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_out_write_error_returns_1(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, (_interaction(1, ENGLISH),))
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        assert cli.main(
            [
                "filter-english",
                "--in",
                str(input_path),
                "--out",
                str(blocker / "interactions-en.jsonl"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

    def test_report_write_error_returns_1(self, tmp_path, capsys):
        input_path = self._write_input(tmp_path, (_interaction(1, ENGLISH),))
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        assert cli.main(
            [
                "filter-english",
                "--in",
                str(input_path),
                "--report",
                str(blocker / "english-report.json"),
            ]
        ) == 1
        assert "error:" in capsys.readouterr().err

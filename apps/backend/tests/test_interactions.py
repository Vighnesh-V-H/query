import csv

import pytest

from query import cli
from query import interactions as interactions_mod

BRAND = "SpotifyCares"


def _row(tweet_id, author, inbound, created_at, text, parent=""):
    return f"{tweet_id},{author},{inbound},{created_at},{text},,{parent}"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(interactions_mod.COLUMNS) + "\n")
        fh.write("\n".join(rows) + "\n")


def _dt(hour, minute=0):
    return f"Wed Nov 01 {hour:02d}:{minute:02d}:00 +0000 2017"


class TestBuildInteractions:
    def test_two_turn_interaction(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares my app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust try a reinstall"', 1),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert len(found) == 1
        interaction = found[0]
        assert interaction.interaction_id == 1
        assert interaction.customer_id == "cust"
        assert interaction.brand_id == BRAND
        assert [t.tweet_id for t in interaction.turns] == [1, 2]
        assert [t.side for t in interaction.turns] == ["customer", "brand"]
        assert report.interactions == 1
        assert report.turns_total == 2
        assert report.turns_customer == 1
        assert report.turns_brand == 1

    def test_opening_message_is_first_customer_turn(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 0
        assert report.unanswered_openings == 1
        assert found == ()

    def test_multi_turn_chains_and_brand_customer_alternation(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust what device?"', 1),
            _row(3, "cust", "True", _dt(12), '"@SpotifyCares android"', 2),
            _row(4, BRAND, "False", _dt(13), '"@cust try reinstalling"', 3),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 1
        assert found[0].customer_turns == 2
        assert found[0].brand_turns == 2
        assert [t.side for t in found[0].turns] == ["customer", "brand", "customer", "brand"]

    def test_turns_ordered_by_created_at_not_tweet_id(self, tmp_path):
        # tweet ids inverted relative to reply order; only timestamps are reliable
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(9, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
            _row(8, BRAND, "False", _dt(11), '"@cust hi"', 9),
            _row(7, "cust", "True", _dt(12), '"@SpotifyCares still broken"', 8),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert [t.tweet_id for t in found[0].turns] == [9, 8, 7]
        assert report.turns_total == 3

    def test_self_reply_chain_climbs_to_opening(self, tmp_path):
        # customer self-reply chain: the earliest message is the opening
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares app crashes"', ""),
            _row(2, "cust", "True", _dt(11), '"and it eats my battery"', 1),
            _row(3, BRAND, "False", _dt(12), '"@cust sorry!"', 2),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert len(found) == 1
        assert found[0].interaction_id == 1
        assert [t.tweet_id for t in found[0].turns] == [1, 2, 3]

    def test_interloper_gets_own_interaction(self, tmp_path):
        # third party jumps in; dyad growth must not absorb their chain
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust what device?"', 1),
            _row(3, "other", "True", _dt(12), '"me too!"', 2),
            _row(4, BRAND, "False", _dt(13), '"@other hi"', 3),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 2
        ids = {i.interaction_id for i in found}
        assert ids == {1, 3}
        # original interaction keeps only customer+brand turns
        first = next(i for i in found if i.interaction_id == 1)
        assert [t.tweet_id for t in first.turns] == [1, 2]

    def test_brand_is_customer_of_other_brand(self, tmp_path):
        # a SpotifyCares tweet replying to another brand must not seed an
        # Interaction with SpotifyCares as the Brand
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "AppleSupport", "False", _dt(10), '"@SpotifyCares question"', ""),
            _row(2, BRAND, "True", _dt(11), '"@AppleSupport answer"', 1),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        # SpotifyCares as inbound author replying to another brand is not a
        # customer of itself; tweet 2's parent is the brand AppleSupport, so
        # it seeds, but the dyad has no SpotifyCares brand turn -> unanswered
        assert report.interactions == 0

    def test_dangling_parent_is_tolerated(self, tmp_path):
        # parent id not present in the CSV: seed still works via mention
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', 999),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.unanswered_openings == 1
        assert found == ()

    def test_duplicate_tweet_id_rejected(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
            _row(1, "cust", "True", _dt(11), '"duplicate"', ""),
        ])

        with pytest.raises(interactions_mod.InteractionsError, match="duplicate"):
            interactions_mod.build_interactions(csv_path, BRAND)

    def test_missing_csv_rejected(self, tmp_path):
        with pytest.raises(interactions_mod.InteractionsError, match="does not exist"):
            interactions_mod.build_interactions(tmp_path / "missing.csv", BRAND)

    def test_wrong_header_rejected(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        csv_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        with pytest.raises(interactions_mod.InteractionsError, match="columns"):
            interactions_mod.build_interactions(csv_path, BRAND)

    def test_bad_timestamp_rejected(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", "not a date", '"@SpotifyCares help"', ""),
        ])

        with pytest.raises(interactions_mod.InteractionsError):
            interactions_mod.build_interactions(csv_path, BRAND)

    def test_brand_reply_without_mention_seeds_via_adjacency(self, tmp_path):
        # customer replies to a brand tweet without mentioning the handle
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"my account is locked"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust dm us"', 1),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 1
        assert found[0].interaction_id == 1

    def test_climb_stops_at_brand_rooted_chain(self, tmp_path):
        # customer replies to a brand tweet: chain is rooted at brand tweet,
        # so the customer reply is the opening (brand turn before it cannot
        # be absorbed into the same dyad)
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, BRAND, "False", _dt(10), '"psa: outage ongoing"', ""),
            _row(2, "cust", "True", _dt(11), '"is it fixed yet?"', 1),
            _row(3, BRAND, "False", _dt(12), '"@cust not yet"', 2),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 1
        first = found[0]
        assert first.interaction_id == 2
        assert [t.tweet_id for t in first.turns] == [2, 3]

    def test_mention_match_is_whole_handle(self, tmp_path):
        # @SpotifyCaresHelp is a different handle and must NOT match,
        # even though it starts with the brand handle
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCaresHelp you there?"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust wrong handle, we are @SpotifyCares"', 1),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        # the brand reply seeds via adjacency even though the mention missed
        assert report.interactions == 1
        assert found[0].interaction_id == 1

    def test_mid_text_mention_seeds(self, tmp_path):
        # a whole-handle mention need not start the text (replies put the
        # handle first; fresh mentions often trail the message)
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"my app broke @SpotifyCares, help?"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust on it"', 1),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.seed_count == 1
        assert report.interactions == 1
        assert found[0].interaction_id == 1

    def test_longer_handle_is_not_a_seed(self, tmp_path):
        # a bare mention of the longer handle with no brand adjacency: no seed
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCaresHelp you there?"', ""),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.seed_count == 0
        assert report.interactions == 0
        assert found == ()

    def test_mention_at_end_of_text_still_seeds_via_boundary(self, tmp_path):
        # exact handle followed by punctuation stays a seed
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares: app crashes!"', ""),
        ])

        _, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.seed_count == 1

    def test_self_parenting_row_does_not_loop_forever(self, tmp_path):
        # parent id equals the tweet id itself: climb must stop, not spin
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', 1),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.unanswered_openings == 1
        assert found == ()

    def test_climb_passes_through_two_consecutive_brand_turns(self, tmp_path):
        # brand self-reply before answering the customer: the whole chain is
        # one Interaction rooted at the customer opening
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
            _row(2, BRAND, "False", _dt(11), '"internal note"', 1),
            _row(3, BRAND, "False", _dt(12), '"@cust which device?"', 2),
            _row(4, "cust", "True", _dt(13), '"android"', 3),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 1
        assert found[0].interaction_id == 1
        assert [t.tweet_id for t in found[0].turns] == [1, 2, 3, 4]
        assert [t.side for t in found[0].turns] == ["customer", "brand", "brand", "customer"]

    def test_two_seeds_same_opening_deduplicated(self, tmp_path):
        # two seeds climb to the same opening: one Interaction, not two
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
            _row(2, "cust", "True", _dt(11), '"@SpotifyCares more info"', 1),
            _row(3, BRAND, "False", _dt(12), '"@cust on it"', 2),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 1
        assert found[0].interaction_id == 1
        assert [t.tweet_id for t in found[0].turns] == [1, 2, 3]

    def test_opening_inside_another_dyad_is_absorbed(self, tmp_path):
        # regression (real-data bug): the customer's middle tweet carries no
        # brand mention and no brand reply, so a climb from a later seed stops
        # at it — but dyad growth absorbs by author only, so the later seed's
        # chain is already inside the first dyad. The later opening must be
        # absorbed, and no turn may be counted twice.
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            # opening: customer mentions the brand
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares downloaded songs choppy"', ""),
            # middle: customer detail tweet, NOT a seed (no mention, no brand
            # reply to it, parent is not the brand)
            _row(2, "cust", "True", _dt(11), '"also running iOS 11"', 1),
            # later seed: customer mentions the brand again; climb stops at 2
            _row(3, "cust", "True", _dt(12), '"@SpotifyCares still choppy after reinstall"', 2),
            # brand answers the later tweet, giving that opening a brand turn
            _row(4, BRAND, "False", _dt(13), '"@cust we are on it"', 3),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.opening_count == 2
        assert report.absorbed_openings == 1
        assert report.interactions == 1
        assert found[0].interaction_id == 1
        assert sorted(t.tweet_id for t in found[0].turns) == [1, 2, 3, 4]

    def test_emitted_interactions_are_pairwise_disjoint(self, tmp_path):
        # global invariant: no tweet may appear in two Interactions
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares downloaded songs choppy"', ""),
            _row(2, "cust", "True", _dt(11), '"also running iOS 11"', 1),
            _row(3, "cust", "True", _dt(12), '"@SpotifyCares still choppy after reinstall"', 2),
            _row(4, BRAND, "False", _dt(13), '"@cust we are on it"', 3),
            # an unrelated second interaction for a different customer
            _row(5, "other", "True", _dt(14), '"@SpotifyCares me too"', ""),
            _row(6, BRAND, "False", _dt(15), '"@other dm us"', 5),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 2
        seen: set[int] = set()
        for interaction in found:
            for turn in interaction.turns:
                assert turn.tweet_id not in seen, "tweet in two Interactions"
                seen.add(turn.tweet_id)

    def test_brand_self_reply_chain_in_dyad(self, tmp_path):
        # downward brand->brand self-replies after the customer turn are part
        # of the same Interaction
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares help"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust we are checking"', 1),
            _row(3, BRAND, "False", _dt(12), '"@cust still checking"', 2),
        ])

        found, report = interactions_mod.build_interactions(csv_path, BRAND)

        assert report.interactions == 1
        assert [t.tweet_id for t in found[0].turns] == [1, 2, 3]
        assert found[0].brand_turns == 2


class TestWriteInteractionsJsonl:
    def test_writes_ordered_turns_one_per_line(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares my app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust try a reinstall"', 1),
        ])
        found, _ = interactions_mod.build_interactions(csv_path, BRAND)

        out_path = interactions_mod.write_interactions_jsonl(found, tmp_path / "sub" / "interactions.jsonl")

        import json
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["interaction_id"] == 1
        assert record["customer_id"] == "cust"
        assert record["brand_id"] == BRAND
        assert [t["side"] for t in record["turns"]] == ["customer", "brand"]
        assert [t["tweet_id"] for t in record["turns"]] == [1, 2]
        assert record["turns"][0]["created_at"].endswith("+00:00")

    def test_write_is_atomic_no_temp_files_left(self, tmp_path):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares my app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust try a reinstall"', 1),
        ])
        found, _ = interactions_mod.build_interactions(csv_path, BRAND)

        interactions_mod.write_interactions_jsonl(found, tmp_path / "interactions.jsonl")

        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []
        assert (tmp_path / "interactions.jsonl").is_file()


class TestCli:
    def test_build_interactions_command_prints_counts(self, tmp_path, capsys):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares my app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust try a reinstall"', 1),
        ])

        assert cli.main(["build-interactions", "--twcs", str(csv_path)]) == 0

        out = capsys.readouterr().out
        assert "interactions: 1" in out
        assert "turns: 2 (customer 1, brand 1)" in out

    def test_build_interactions_writes_json_report(self, tmp_path, capsys):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares my app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust try a reinstall"', 1),
        ])
        report_path = tmp_path / "counts.json"

        assert cli.main(
            ["build-interactions", "--twcs", str(csv_path), "--report", str(report_path)]
        ) == 0

        import json
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["interactions"] == 1
        assert payload["turns_total"] == 2

    def test_build_interactions_writes_jsonl_out(self, tmp_path, capsys):
        csv_path = tmp_path / "twcs.csv"
        _write_csv(csv_path, [
            _row(1, "cust", "True", _dt(10), '"@SpotifyCares my app crashes"', ""),
            _row(2, BRAND, "False", _dt(11), '"@cust try a reinstall"', 1),
        ])
        out_path = tmp_path / "interactions.jsonl"

        assert cli.main(
            ["build-interactions", "--twcs", str(csv_path), "--out", str(out_path)]
        ) == 0

        import json
        out = capsys.readouterr().out
        assert "interactions written: " in out
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["interaction_id"] == 1

    def test_build_interactions_command_errors_on_missing_csv(self, tmp_path, capsys):
        assert cli.main(["build-interactions", "--twcs", str(tmp_path / "nope.csv")]) == 1
        assert "error:" in capsys.readouterr().err

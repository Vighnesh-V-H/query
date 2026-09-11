import hashlib
import json
import re
import threading
from datetime import datetime, timezone

import pytest
from query.interactions import Interaction, Turn

from query import cli, discovery, embedding, llm, taxonomy
from query import interactions as interactions_mod

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


def _seed(intent_id, definition=None, message=None):
    return taxonomy.SeedIntent(
        intent_id=intent_id,
        definition=definition or f"definition of {intent_id}",
        representative_message=message or f"representative message for {intent_id}",
    )


def _reply(mapping, justification="Decisive evidence."):
    return json.dumps({"mapping": mapping, "justification": justification})


def _two_group_pool():
    interactions = (
        _interaction(1, "alpha one"),
        _interaction(2, "alpha two"),
        _interaction(3, "alpha three"),
        _interaction(4, "beta one"),
        _interaction(5, "beta two"),
        _interaction(6, "beta three"),
    )
    vectors = {
        "alpha one": (1, 0, 0, 0),
        "alpha two": (1, 0.01, 0, 0),
        "alpha three": (1, -0.01, 0, 0),
        "beta one": (0, 1, 0, 0),
        "beta two": (0.01, 1, 0, 0),
        "beta three": (-0.01, 1, 0, 0),
    }
    return interactions, vectors


class _FakeEmbedder:
    """Stable unit vectors: explicit per text, else SHA-256-derived."""

    def __init__(self, vectors=None, model_name="test/fake"):
        self.model_name = model_name
        self._vectors = dict(vectors or {})

    def embed(self, texts):
        result = []
        for text in texts:
            vector = self._vectors.get(text)
            if vector is None:
                digest = hashlib.sha256(text.encode()).digest()
                vector = tuple(float(byte) for byte in digest[:4])
            result.append(embedding.l2_normalize(vector))
        return tuple(result)


class _SequenceLabeler:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return llm.LLMReply(content=self.replies.pop(0), model="test/labeler")


class _PromptLabeler:
    """Returns a verdict keyed by the cluster id in the prompt, thread-safe."""

    def __init__(
        self, mappings=None, default="new", justification="No seed intent matches."
    ):
        self.mappings = dict(mappings or {})
        self.default = default
        self.justification = justification
        self.prompts = []
        self._lock = threading.Lock()

    def __call__(self, prompt):
        with self._lock:
            self.prompts.append(prompt)
        cluster_id = int(re.search(r"Cluster (\d+)", prompt).group(1))
        mapping = self.mappings.get(cluster_id, self.default)
        return llm.LLMReply(
            content=_reply(mapping, self.justification), model="test/labeler"
        )


class _BoomLabeler:
    def __call__(self, prompt):
        raise AssertionError("labeler must not be called")


class TestDiscoverIntents:
    def test_clusters_partition_the_messages(self):
        interactions, vectors = _two_group_pool()

        clusters, report = discovery.discover_intents(
            interactions,
            (_seed("playback"), _seed("billing")),
            _FakeEmbedder(vectors),
            clusters=2,
            infer=_PromptLabeler({0: "playback"}, default="playback"),
        )

        assert len(clusters) == 2
        assert sorted(cluster.size for cluster in clusters) == [3, 3]
        assert report.messages == 6
        ids = [
            example.interaction_id
            for cluster in clusters
            for example in cluster.examples
        ]
        assert sorted(ids) == [1, 2, 3, 4, 5, 6]

    def test_examples_are_the_messages_closest_to_the_centroid(self):
        interactions = (
            _interaction(1, "tight one"),
            _interaction(2, "tight two"),
            _interaction(3, "tight three"),
            _interaction(4, "outlier"),
        )
        vectors = {
            "tight one": (1, 0, 0, 0),
            "tight two": (1, 0.01, 0, 0),
            "tight three": (1, -0.01, 0, 0),
            "outlier": (0, 1, 0, 0),
        }

        clusters, _ = discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=1,
            examples=4,
            infer=_PromptLabeler({0: "playback"}),
        )

        examples = clusters[0].examples
        assert examples[-1].interaction_id == 4
        assert examples[0].interaction_id in (1, 2, 3)
        assert (
            examples[0].similarity >= examples[1].similarity >= examples[2].similarity
        )

    def test_examples_are_capped_at_the_requested_count(self):
        interactions, vectors = _two_group_pool()

        clusters, _ = discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=2,
            examples=1,
            infer=_PromptLabeler({0: "playback", 1: "playback"}),
        )

        assert all(len(cluster.examples) == 1 for cluster in clusters)

    def test_candidates_rank_seeds_by_embedding_similarity(self):
        interactions = (_interaction(1, "alpha one"), _interaction(2, "alpha two"))
        vectors = {
            "alpha one": (1, 0, 0, 0),
            "alpha two": (0.99, 0.01, 0, 0),
            "definition of playback": (1, 0, 0, 0),
            "representative message for playback": (1, 0, 0, 0),
            "definition of app_technical": (0, 1, 0, 0),
            "representative message for app_technical": (0, 1, 0, 0),
        }

        clusters, _ = discovery.discover_intents(
            interactions,
            (_seed("playback"), _seed("app_technical")),
            _FakeEmbedder(vectors),
            clusters=1,
            candidates=1,
            infer=_PromptLabeler({0: "playback"}),
        )

        assert clusters[0].candidates[0].intent_id == "playback"
        assert clusters[0].candidates[0].similarity == pytest.approx(1.0, abs=1e-4)

    def test_mapping_verdict_and_justification_are_recorded(self):
        interactions, vectors = _two_group_pool()
        labeler = _PromptLabeler(
            {0: "playback"}, default="junk", justification="  Praise   and chatter.  "
        )

        clusters, report = discovery.discover_intents(
            interactions,
            (_seed("playback"), _seed("billing")),
            _FakeEmbedder(vectors),
            clusters=2,
            infer=labeler,
        )

        mappings = [cluster.mapping for cluster in clusters]
        assert sorted(mappings) == ["junk", "playback"]
        junk = next(cluster for cluster in clusters if cluster.mapping == "junk")
        assert junk.justification == "Praise and chatter."
        assert report.mapped_clusters == 1
        assert report.junk_clusters == 1

    def test_new_and_junk_mappings_are_accepted(self):
        interactions, vectors = _two_group_pool()

        clusters, report = discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=2,
            infer=_PromptLabeler({0: "new"}, default="junk"),
        )

        assert sorted(cluster.mapping for cluster in clusters) == ["junk", "new"]
        assert report.new_clusters == 1
        assert report.junk_clusters == 1
        assert report.mapped_clusters == 0
        assert report.mapped_share == 0.0

    def test_unknown_mapping_is_repaired_once(self):
        interactions, vectors = _two_group_pool()
        labeler = _SequenceLabeler(
            [
                _reply("not-a-seed"),
                _reply("playback", "Plays fail."),
                _reply("playback", "Plays fail."),
            ]
        )

        clusters, _ = discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=2,
            infer=labeler,
        )

        assert len(labeler.prompts) == 3
        assert "previous reply was not valid JSON" in labeler.prompts[1]
        assert [cluster.mapping for cluster in clusters] == ["playback", "playback"]

    def test_unknown_mapping_twice_raises(self):
        interactions, vectors = _two_group_pool()
        labeler = _SequenceLabeler(
            [_reply("not-a-seed"), _reply("still-not-a-seed")]
            + [_reply("playback"), _reply("playback")]
        )

        with pytest.raises(discovery.DiscoveryError, match="invalid mapping"):
            discovery.discover_intents(
                interactions,
                (_seed("playback"),),
                _FakeEmbedder(vectors),
                clusters=1,
                infer=labeler,
            )

        assert len(labeler.prompts) == 2

    def test_empty_justification_twice_raises(self):
        interactions, vectors = _two_group_pool()
        labeler = _SequenceLabeler([_reply("playback", "   "), _reply("playback", "")])

        with pytest.raises(discovery.DiscoveryError, match="empty justification"):
            discovery.discover_intents(
                interactions,
                (_seed("playback"),),
                _FakeEmbedder(vectors),
                clusters=1,
                infer=labeler,
            )

    def test_labeler_failure_is_wrapped(self):
        interactions, vectors = _two_group_pool()

        def broken(prompt):
            raise RuntimeError("upstream down")

        with pytest.raises(discovery.DiscoveryError, match="labeler call failed"):
            discovery.discover_intents(
                interactions,
                (_seed("playback"),),
                _FakeEmbedder(vectors),
                clusters=1,
                infer=broken,
            )

    def test_workers_preserve_cluster_order(self):
        interactions = (
            _interaction(1, "one"),
            _interaction(2, "two"),
            _interaction(3, "three"),
            _interaction(4, "four"),
        )
        vectors = {
            "one": (1, 0, 0, 0),
            "two": (0, 1, 0, 0),
            "three": (0, 0, 1, 0),
            "four": (0, 0, 0, 1),
        }
        labeler = _PromptLabeler({0: "playback", 2: "playback"}, default="new")

        clusters, _ = discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=4,
            workers=4,
            infer=labeler,
        )

        assert [cluster.cluster_id for cluster in clusters] == [0, 1, 2, 3]
        assert [cluster.mapping for cluster in clusters] == [
            "playback",
            "new",
            "playback",
            "new",
        ]

    def test_report_counts_per_intent_and_absent_intents(self):
        interactions, vectors = _two_group_pool()

        _, report = discovery.discover_intents(
            interactions,
            (
                _seed("playback"),
                _seed("billing"),
                _seed("market_availability"),
            ),
            _FakeEmbedder(vectors),
            clusters=2,
            infer=_PromptLabeler({0: "playback"}, default="new"),
        )

        assert report.per_intent["playback"].clusters == 1
        assert report.per_intent["playback"].messages == 3
        assert report.per_intent["billing"].clusters == 0
        assert report.absent_intents == ("billing", "market_availability")
        assert report.models == ("test/labeler",)
        assert report.embedding_model == "test/fake"
        assert report.requested_clusters == 2
        assert report.seed == discovery.DEFAULT_SEED
        assert report.mapped_share == pytest.approx(0.5)

    def test_other_seed_is_not_a_mapping_target(self):
        interactions, vectors = _two_group_pool()
        labeler = _PromptLabeler({0: "playback"}, default="playback")

        discovery.discover_intents(
            interactions,
            (_seed("playback"), _seed(taxonomy.OTHER_INTENT_ID)),
            _FakeEmbedder(vectors),
            clusters=1,
            candidates=5,
            infer=labeler,
        )

        prompt = labeler.prompts[0]
        assert "- other:" not in prompt
        assert "- playback:" in prompt

    def test_prompt_carries_definitions_examples_and_similarity_evidence(self):
        interactions = (_interaction(1, "alpha one"),)
        labeler = _PromptLabeler({0: "playback"})

        discovery.discover_intents(
            interactions,
            (_seed("playback", "Songs that will not play."),),
            _FakeEmbedder({"alpha one": (1, 0, 0, 0)}),
            clusters=1,
            candidates=1,
            infer=labeler,
        )

        prompt = labeler.prompts[0]
        assert "Songs that will not play." in prompt
        assert "alpha one" in prompt
        assert "playback" in prompt


class TestDiscoverIntentsValidation:
    def test_empty_input_raises(self):
        with pytest.raises(discovery.DiscoveryError, match="no interactions"):
            discovery.discover_intents((), (_seed("playback"),), _FakeEmbedder())

    def test_zero_clusters_raises(self):
        interactions, vectors = _two_group_pool()

        with pytest.raises(discovery.DiscoveryError, match="clusters must be at least"):
            discovery.discover_intents(
                interactions, (_seed("playback"),), _FakeEmbedder(vectors), clusters=0
            )

    def test_more_clusters_than_messages_raises(self):
        interactions, vectors = _two_group_pool()

        with pytest.raises(discovery.DiscoveryError, match="cannot exceed"):
            discovery.discover_intents(
                interactions, (_seed("playback"),), _FakeEmbedder(vectors), clusters=7
            )

    def test_duplicate_interaction_ids_raise(self):
        interactions = (_interaction(1, "one"), _interaction(1, "one again"))

        with pytest.raises(discovery.DiscoveryError, match="duplicate interaction_id"):
            discovery.discover_intents(
                interactions,
                (_seed("playback"),),
                _FakeEmbedder(),
                clusters=1,
                infer=_PromptLabeler({0: "playback"}),
            )

    def test_seed_taxonomy_without_support_intents_raises(self):
        interactions, vectors = _two_group_pool()

        with pytest.raises(discovery.DiscoveryError, match="no support intents"):
            discovery.discover_intents(
                interactions,
                (_seed(taxonomy.OTHER_INTENT_ID),),
                _FakeEmbedder(vectors),
                clusters=1,
            )

    def test_zero_examples_candidates_or_workers_raise(self):
        interactions, vectors = _two_group_pool()
        seeds = (_seed("playback"),)

        for keyword, value, message in (
            ("examples", 0, "examples must be at least"),
            ("candidates", 0, "candidates must be at least"),
            ("workers", 0, "workers must be at least"),
        ):
            with pytest.raises(discovery.DiscoveryError, match=message):
                discovery.discover_intents(
                    interactions,
                    seeds,
                    _FakeEmbedder(vectors),
                    clusters=1,
                    **{keyword: value},
                )


class TestDiscoveryCache:
    def test_reuses_verdicts_on_rerun(self, tmp_path):
        interactions, vectors = _two_group_pool()
        seeds = (_seed("playback"), _seed("billing"))
        cache = discovery.DiscoveryCache(tmp_path / "cache.jsonl")
        labeler = _PromptLabeler({0: "playback"}, default="playback")

        first, _ = discovery.discover_intents(
            interactions,
            seeds,
            _FakeEmbedder(vectors),
            clusters=2,
            cache=cache,
            infer=labeler,
        )
        second, _ = discovery.discover_intents(
            interactions,
            seeds,
            _FakeEmbedder(vectors),
            clusters=2,
            cache=cache,
            infer=_BoomLabeler(),
        )

        assert second == first
        verdicts = discovery.DiscoveryCache(tmp_path / "cache.jsonl").verdicts()
        assert set(verdicts) == {0, 1}
        assert verdicts[0].model == "test/labeler"
        assert verdicts[0].prompt_sha256

    def test_changed_prompt_ignores_stale_entry(self, tmp_path):
        interactions, vectors = _two_group_pool()
        path = tmp_path / "cache.jsonl"
        cache = discovery.DiscoveryCache(path)
        discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=2,
            cache=cache,
            infer=_PromptLabeler({0: "playback"}, default="playback"),
        )

        labeler = _PromptLabeler({0: "new"}, default="new")
        clusters, _ = discovery.discover_intents(
            interactions,
            (
                _seed("playback", "A different definition."),
                _seed("billing"),
            ),
            _FakeEmbedder(vectors),
            clusters=2,
            cache=discovery.DiscoveryCache(path),
            infer=labeler,
        )

        assert {cluster.mapping for cluster in clusters} == {"new"}
        assert len(labeler.prompts) == 2

    def test_cached_verdict_still_checked_against_the_contract(self, tmp_path):
        interactions, vectors = _two_group_pool()
        path = tmp_path / "cache.jsonl"
        discovery.discover_intents(
            interactions,
            (_seed("playback"),),
            _FakeEmbedder(vectors),
            clusters=1,
            cache=discovery.DiscoveryCache(path),
            infer=_PromptLabeler({0: "playback"}),
        )
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        record["mapping"] = "mystery"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(discovery.DiscoveryError, match="violates the contract"):
            discovery.discover_intents(
                interactions,
                (_seed("playback"),),
                _FakeEmbedder(vectors),
                clusters=1,
                cache=discovery.DiscoveryCache(path),
                infer=_BoomLabeler(),
            )


def _valid_cluster_json(
    cluster_id=0,
    size=1,
    mapping="playback",
    justification="Decisive evidence.",
):
    return {
        "cluster_id": cluster_id,
        "size": size,
        "mapping": mapping,
        "justification": justification,
        "candidates": [{"intent_id": "playback", "similarity": 0.9}],
        "examples": [{"interaction_id": 1, "message": "alpha one", "similarity": 0.9}],
    }


def _write_clusters(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


class TestIntentClusterArtifacts:
    def _clusters(self):
        interactions, vectors = _two_group_pool()
        clusters, _ = discovery.discover_intents(
            interactions,
            (_seed("playback"), _seed("billing")),
            _FakeEmbedder(vectors),
            clusters=2,
            examples=2,
            infer=_PromptLabeler({0: "playback"}, default="new"),
        )
        return clusters

    def test_stage_and_read_round_trip(self, tmp_path):
        clusters = self._clusters()

        temporary = discovery.stage_intent_clusters_jsonl(
            clusters, tmp_path / "clusters.jsonl"
        )
        temporary.replace(tmp_path / "clusters.jsonl")
        loaded = discovery.read_intent_clusters_jsonl(
            tmp_path / "clusters.jsonl",
            intent_ids=("playback", "billing"),
        )

        assert loaded == clusters

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(discovery.DiscoveryError, match="does not exist"):
            discovery.read_intent_clusters_jsonl(tmp_path / "missing.jsonl")

    def test_unknown_seed_mapping_raises_with_seed_list(self, tmp_path):
        path = _write_clusters(
            tmp_path / "clusters.jsonl", [_valid_cluster_json(mapping="mystery")]
        )

        with pytest.raises(discovery.DiscoveryError, match="unknown mapping"):
            discovery.read_intent_clusters_jsonl(path, intent_ids=("playback",))

    def test_unknown_mapping_is_accepted_without_seed_list(self, tmp_path):
        path = _write_clusters(
            tmp_path / "clusters.jsonl", [_valid_cluster_json(mapping="mystery")]
        )

        loaded = discovery.read_intent_clusters_jsonl(path)

        assert loaded[0].mapping == "mystery"

    def test_new_and_junk_mappings_always_pass(self, tmp_path):
        path = _write_clusters(
            tmp_path / "clusters.jsonl",
            [
                _valid_cluster_json(cluster_id=0, mapping="new"),
                _valid_cluster_json(cluster_id=1, mapping="junk"),
            ],
        )

        loaded = discovery.read_intent_clusters_jsonl(path, intent_ids=("playback",))

        assert [cluster.mapping for cluster in loaded] == ["new", "junk"]

    def test_duplicate_cluster_id_raises(self, tmp_path):
        path = _write_clusters(
            tmp_path / "clusters.jsonl",
            [_valid_cluster_json(cluster_id=0), _valid_cluster_json(cluster_id=0)],
        )

        with pytest.raises(discovery.DiscoveryError, match="duplicate cluster_id"):
            discovery.read_intent_clusters_jsonl(path)

    def test_zero_size_raises(self, tmp_path):
        path = _write_clusters(
            tmp_path / "clusters.jsonl", [_valid_cluster_json(size=0)]
        )

        with pytest.raises(discovery.DiscoveryError, match="invalid size"):
            discovery.read_intent_clusters_jsonl(path)

    def test_empty_examples_raise(self, tmp_path):
        record = _valid_cluster_json()
        record["examples"] = []
        path = _write_clusters(tmp_path / "clusters.jsonl", [record])

        with pytest.raises(discovery.DiscoveryError, match="no examples"):
            discovery.read_intent_clusters_jsonl(path)

    def test_non_int_example_interaction_id_raises(self, tmp_path):
        record = _valid_cluster_json()
        record["examples"][0]["interaction_id"] = "1"
        path = _write_clusters(tmp_path / "clusters.jsonl", [record])

        with pytest.raises(discovery.DiscoveryError, match="invalid example entry"):
            discovery.read_intent_clusters_jsonl(path)

    def test_boolean_candidate_similarity_raises(self, tmp_path):
        record = _valid_cluster_json()
        record["candidates"][0]["similarity"] = True
        path = _write_clusters(tmp_path / "clusters.jsonl", [record])

        with pytest.raises(discovery.DiscoveryError, match="invalid candidate entry"):
            discovery.read_intent_clusters_jsonl(path)

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "clusters.jsonl"
        path.write_text("{not json\n", encoding="utf-8")

        with pytest.raises(discovery.DiscoveryError, match="malformed JSON"):
            discovery.read_intent_clusters_jsonl(path)


class TestReviewMarkdown:
    def test_contains_mapping_summary_and_cluster_evidence(self):
        interactions, vectors = _two_group_pool()
        seeds = (_seed("playback"), _seed("billing"))
        clusters, report = discovery.discover_intents(
            interactions,
            seeds,
            _FakeEmbedder(vectors),
            clusters=2,
            infer=_PromptLabeler(
                {0: "playback"}, default="junk", justification="Praise and chatter."
            ),
        )

        markdown = discovery.render_review_markdown(clusters, report, seeds)

        assert "# Intent discovery review" in markdown
        assert "| Mapping | Clusters | Messages |" in markdown
        assert "| `playback` | 1 | 3 |" in markdown
        assert "| `junk` | 1 | 3 |" in markdown
        assert (
            "### Cluster 0 — `playback`" in markdown
            or "### Cluster 1 — `playback`" in markdown
        )
        assert "Praise and chatter." in markdown
        assert "alpha one" in markdown


def _taxonomy_document(*intents):
    lines = [
        "| # | Intent id | One-line definition | Rough signal |",
        "|---|-----------|---------------------|--------------|",
    ]
    for number, (intent_id, definition) in enumerate(intents, 1):
        lines.append(f"| {number} | `{intent_id}` | {definition} | ~1% |")
    lines.append("")
    for intent_id, _ in intents:
        lines.append(f"- `{intent_id}` [1]: message for {intent_id}")
    return "\n".join(lines) + "\n"


class TestDiscoverIntentsCli:
    def _write_pool(self, tmp_path, interactions):
        return interactions_mod.write_interactions_jsonl(
            interactions, tmp_path / "rag-pool.jsonl"
        )

    def _write_taxonomy(self, tmp_path):
        path = tmp_path / "taxonomy.md"
        path.write_text(
            _taxonomy_document(
                ("playback", "Songs that will not play."),
                ("billing", "Charges and refunds."),
            ),
            encoding="utf-8",
        )
        return path

    def _fake_embedder(self):
        _, vectors = _two_group_pool()
        return _FakeEmbedder(vectors)

    def test_command_writes_clusters_report_and_review(
        self, tmp_path, capsys, monkeypatch
    ):
        interactions, _ = _two_group_pool()
        input_path = self._write_pool(tmp_path, interactions)
        taxonomy_path = self._write_taxonomy(tmp_path)
        out_path = tmp_path / "intent-clusters.jsonl"
        report_path = tmp_path / "intent-discovery-report.json"
        review_path = tmp_path / "intent-discovery.md"
        monkeypatch.setattr(embedding, "MiniLMEmbedder", self._fake_embedder)
        monkeypatch.setattr(discovery, "call_labeler", _PromptLabeler({0: "playback"}))

        assert (
            cli.main(
                [
                    "discover-intents",
                    "--in",
                    str(input_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--out",
                    str(out_path),
                    "--report",
                    str(report_path),
                    "--review",
                    str(review_path),
                    "--clusters",
                    "2",
                ]
            )
            == 0
        )

        output = capsys.readouterr().out
        assert f"input: {input_path} (6 interactions)" in output
        assert "clusters: 2 (k=2, seed=42)" in output
        assert "mapped: 1 clusters, 3 messages (50.00%)" in output
        assert "new: 1 clusters, 3 messages" in output
        assert "seed intents with no mapped cluster: billing" in output
        assert f"clusters written: {out_path}" in output
        assert f"report written: {report_path}" in output
        assert f"review written: {review_path}" in output

        records = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 2
        assert sum(record["size"] for record in records) == 6
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["clusters"] == 2
        assert payload["messages"] == 6
        assert payload["mapped_clusters"] == 1
        assert payload["new_clusters"] == 1
        assert payload["junk_clusters"] == 0
        assert payload["absent_intents"] == ["billing"]
        assert payload["models"] == ["test/labeler"]
        assert (
            payload["per_intent"]["playback"]["clusters"] == 1
            and payload["per_intent"]["playback"]["messages"] == 3
        )
        assert "### Cluster" in review_path.read_text(encoding="utf-8")
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_command_without_outputs_prints_stats_only(
        self, tmp_path, capsys, monkeypatch
    ):
        interactions, _ = _two_group_pool()
        input_path = self._write_pool(tmp_path, interactions)
        taxonomy_path = self._write_taxonomy(tmp_path)
        monkeypatch.setattr(embedding, "MiniLMEmbedder", self._fake_embedder)
        monkeypatch.setattr(discovery, "call_labeler", _PromptLabeler({0: "playback"}))

        assert (
            cli.main(
                [
                    "discover-intents",
                    "--in",
                    str(input_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--clusters",
                    "2",
                ]
            )
            == 0
        )

        output = capsys.readouterr().out
        assert "clusters written:" not in output
        assert "report written:" not in output
        assert "review written:" not in output

    def test_missing_input_returns_1(self, tmp_path, capsys, monkeypatch):
        taxonomy_path = self._write_taxonomy(tmp_path)
        monkeypatch.setattr(embedding, "MiniLMEmbedder", self._fake_embedder)

        assert (
            cli.main(
                [
                    "discover-intents",
                    "--in",
                    str(tmp_path / "missing.jsonl"),
                    "--taxonomy",
                    str(taxonomy_path),
                ]
            )
            == 1
        )
        assert "error:" in capsys.readouterr().err

    def test_missing_taxonomy_returns_1(self, tmp_path, capsys, monkeypatch):
        interactions, _ = _two_group_pool()
        input_path = self._write_pool(tmp_path, interactions)
        monkeypatch.setattr(embedding, "MiniLMEmbedder", self._fake_embedder)

        assert (
            cli.main(
                [
                    "discover-intents",
                    "--in",
                    str(input_path),
                    "--taxonomy",
                    str(tmp_path / "missing.md"),
                ]
            )
            == 1
        )
        assert "error:" in capsys.readouterr().err

    def test_output_cannot_overwrite_input_returns_1(self, tmp_path, capsys):
        interactions, _ = _two_group_pool()
        input_path = self._write_pool(tmp_path, interactions)
        taxonomy_path = self._write_taxonomy(tmp_path)

        assert (
            cli.main(
                [
                    "discover-intents",
                    "--in",
                    str(input_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--out",
                    str(input_path),
                ]
            )
            == 1
        )
        assert "error:" in capsys.readouterr().err

    def test_write_failure_leaves_outputs_untouched(
        self, tmp_path, capsys, monkeypatch
    ):
        interactions, _ = _two_group_pool()
        input_path = self._write_pool(tmp_path, interactions)
        taxonomy_path = self._write_taxonomy(tmp_path)
        out_path = tmp_path / "intent-clusters.jsonl"
        report_path = tmp_path / "report.json"
        out_path.write_text("previous clusters\n", encoding="utf-8")
        report_path.write_text("previous report\n", encoding="utf-8")
        monkeypatch.setattr(embedding, "MiniLMEmbedder", self._fake_embedder)
        monkeypatch.setattr(discovery, "call_labeler", _PromptLabeler({0: "playback"}))
        real_replace = cli.os.replace

        def failing_replace(source, destination):
            if str(destination) == str(report_path):
                raise OSError("disk full")
            return real_replace(source, destination)

        monkeypatch.setattr(cli.os, "replace", failing_replace)

        assert (
            cli.main(
                [
                    "discover-intents",
                    "--in",
                    str(input_path),
                    "--taxonomy",
                    str(taxonomy_path),
                    "--out",
                    str(out_path),
                    "--report",
                    str(report_path),
                    "--clusters",
                    "2",
                ]
            )
            == 1
        )

        assert out_path.read_text(encoding="utf-8") == "previous clusters\n"
        assert report_path.read_text(encoding="utf-8") == "previous report\n"
        assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

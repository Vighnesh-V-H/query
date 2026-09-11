"""Cluster RAG-pool Customer Messages and reconcile them with the seed taxonomy.

Ticket 9's seed taxonomy (``docs/intent-seed-taxonomy.md``) is domain knowledge
about what intents we expect; this stage is the data's answer (decision 12). It
embeds every opening Customer Message of the RAG pool with local MiniLM
embeddings (decision 9), clusters the messages with deterministic KMeans, and
maps each cluster back to the seed list: to one seed intent id, or to ``new``
when the cluster is a coherent support theme the seed does not cover, or to
``junk`` when the messages are not a support issue at all.

The mapping is not a similarity lookup. Embedding similarity only ranks a few
candidate seeds as evidence; the configured ``labeler`` role makes the call
from the cluster's examples and a one-line justification is stored beside every
decision, mirroring closure adjudication (ADR-0007). Verdicts are cached by
prompt hash, so an interrupted run resumes without paying for finished
clusters, and clustering is deterministic for a fixed model, seed, and library
version.

The stage emits three artifacts from the same records: a JSON Lines file that
ticket 11 consumes, a JSON report of cluster and mapping counts — including the
seed intents no cluster maps to, the evidence for a merge, split, or drop — and
a Markdown review with examples per cluster. Clusters are computed over the RAG
pool only: the holdout stays reserved for the Golden Set.
"""

import json
import os
import tempfile
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from query import cachefile, closure, embedding, llm, taxonomy
from query.interactions import Interaction

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_PATH = closure.DEFAULT_INPUT_PATH
DEFAULT_CLUSTERS_PATH = REPO_ROOT / "data" / "intent-clusters.jsonl"
DEFAULT_REVIEW_PATH = REPO_ROOT / "docs" / "intent-discovery.md"

DEFAULT_CLUSTERS = 30
DEFAULT_SEED = 42
DEFAULT_EXAMPLES = 8
DEFAULT_CANDIDATES = 5
MAPPING_NEW = "new"
MAPPING_JUNK = "junk"
FLAG_MAPPINGS = (MAPPING_NEW, MAPPING_JUNK)
SIMILARITY_DECIMALS = 6

SYSTEM_PROMPT = (
    "You are a precise taxonomy analyst for customer-support research. "
    "You map one cluster of customer messages to a candidate intent and reply "
    "with raw JSON only."
)


class DiscoveryError(Exception):
    """Raised when intent discovery cannot run or its artifacts cannot be read."""


@dataclass(frozen=True)
class SeedCandidate:
    """One seed intent's embedding similarity to a cluster centroid."""

    intent_id: str
    similarity: float


@dataclass(frozen=True)
class ClusterExample:
    """One representative message of a cluster, closest to its centroid first."""

    interaction_id: int
    message: str
    similarity: float


@dataclass(frozen=True)
class IntentCluster:
    """One discovered cluster with its mapping verdict and the evidence."""

    cluster_id: int
    size: int
    mapping: str
    justification: str
    candidates: tuple[SeedCandidate, ...]
    examples: tuple[ClusterExample, ...]


@dataclass(frozen=True)
class IntentCounts:
    """The clusters and messages mapped to one seed intent."""

    clusters: int
    messages: int


@dataclass(frozen=True)
class DiscoveryReport:
    """Counts of the discovery run: clusters and messages per mapping."""

    clusters: int
    messages: int
    requested_clusters: int
    seed: int
    embedding_model: str
    mapped_clusters: int
    new_clusters: int
    junk_clusters: int
    mapped_messages: int
    new_messages: int
    junk_messages: int
    per_intent: dict[str, IntentCounts]
    absent_intents: tuple[str, ...]
    models: tuple[str, ...]

    @property
    def mapped_share(self) -> float:
        return self.mapped_messages / self.messages if self.messages else 0.0


@dataclass(frozen=True)
class CachedClusterVerdict:
    """A labeler mapping verdict stored for resuming an interrupted run."""

    cluster_id: int
    prompt_sha256: str
    mapping: str
    justification: str
    model: str


class DiscoveryCache:
    """Append-only JSONL cache of cluster verdicts, safe under worker threads.

    Callers load the verdicts once before a run and let every completed mapping
    be recorded immediately, so a run killed mid-way resumes without paying
    for calls it already made.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._cache = cachefile.AppendOnlyCache(
            self.path,
            serialize=cached_cluster_to_json,
            parse=_parse_cache_line,
            key=lambda verdict: verdict.cluster_id,
            error_type=DiscoveryError,
        )

    def verdicts(self) -> dict[int, CachedClusterVerdict]:
        """Read the cached verdicts; a missing cache file is an empty cache."""
        return self._cache.entries()

    def record(self, verdict: CachedClusterVerdict) -> None:
        """Append one verdict, atomically with respect to worker threads."""
        self._cache.record(verdict)


def discover_intents(
    interactions: Sequence[Interaction],
    seed_intents: Sequence[taxonomy.SeedIntent],
    embedder: embedding.Embedder,
    clusters: int = DEFAULT_CLUSTERS,
    seed: int = DEFAULT_SEED,
    examples: int = DEFAULT_EXAMPLES,
    candidates: int = DEFAULT_CANDIDATES,
    workers: int = 1,
    cache: DiscoveryCache | None = None,
    infer: Callable[[str], llm.LLMReply] | None = None,
) -> tuple[tuple[IntentCluster, ...], DiscoveryReport]:
    """Cluster the Interactions' opening messages and map every cluster.

    Returns one cluster per KMeans cluster, ordered by cluster id, and a report
    of the cluster and message counts per mapping plus the seed intents no
    cluster maps to. ``infer`` is the labeler call, kept injectable for tests;
    ``workers`` bounds concurrent labeler calls; ``cache`` reuses and records
    verdicts so interrupted runs can resume.
    """
    _require_arguments(interactions, clusters, examples, candidates, workers)
    _require_unique_ids(interactions)
    support = _support_intents(seed_intents)
    messages = tuple(interaction.opening_message.text for interaction in interactions)
    anchors = _embed_anchors(embedder, support)
    matrix = _as_matrix(embedder.embed(messages), len(messages))
    members = _cluster_messages(matrix, clusters, seed)
    infer = infer if infer is not None else call_labeler
    context = _MappingContext(
        interactions=interactions,
        matrix=matrix,
        anchors=anchors,
        seed_intents=support,
        examples=examples,
        candidates=candidates,
        infer=infer,
        saved=cache.verdicts() if cache is not None else {},
        record=cache.record if cache is not None else None,
    )
    mapped, models = _map_clusters(members, context, workers)
    return mapped, _report(
        mapped, models, interactions, support, clusters, seed, embedder.model_name
    )


def call_labeler(prompt: str) -> llm.LLMReply:
    """Ask the configured labeler role to map one cluster."""
    return llm.call_llm(prompt, role="labeler", system=SYSTEM_PROMPT, temperature=0.0)


def build_mapping_prompt(
    cluster_id: int,
    size: int,
    examples: Sequence[ClusterExample],
    candidates: Sequence[SeedCandidate],
    seed_intents: Sequence[taxonomy.SeedIntent],
) -> str:
    """Render the seed definitions, similarity evidence, and cluster examples."""
    definitions = "\n".join(
        f"- {intent.intent_id}: {intent.definition}" for intent in seed_intents
    )
    candidate_list = ", ".join(
        f"{candidate.intent_id} {candidate.similarity:.2f}" for candidate in candidates
    )
    example_list = "\n".join(
        f"{number}. [interaction {example.interaction_id}] {example.message}"
        for number, example in enumerate(examples, start=1)
    )
    return (
        "Map one cluster of customer-support messages to a candidate intent.\n\n"
        f"Seed intents (the only valid mappings):\n{definitions}\n\n"
        "Embedding-similarity evidence — candidate seed intents for this "
        f"cluster, most similar first: {candidate_list or 'none'}\n\n"
        f"Cluster {cluster_id} holds {size} messages; these are the "
        f"{len(examples)} closest to its centroid:\n{example_list}\n\n"
        "Decide by what the messages say, not by which seed intent is closest "
        "in embedding space: similarity shows topic proximity and is actively "
        "misleading for very short messages.\n\n"
        "Decide:\n"
        '- "junk" when the messages state no issue and no request a support '
        "team would act on: praise, jokes, spam, support-channel chatter, and "
        'bare mentions or one-word replies ("@brand", "help", "How?"). A short '
        'message that names a problem ("app keeps crashing", "charged '
        'twice") is not junk, and a request about the product (update the app '
        "for a device, launch in a country, add a feature) is a support issue, "
        "not junk. Choose junk only when no example names a problem or request.\n"
        "- one seed intent id above when the messages clearly show that "
        "intent's issue;\n"
        '- "new" when the messages show a coherent support issue that none of '
        "the seed intents covers.\n"
        "Boundary rules from the seed taxonomy:\n"
        "- any charge, receipt, or payment-method problem is billing_payment; "
        "a completed payment that did not unlock the plan is billing_payment;\n"
        "- choosing or changing a plan is subscription_plans, and ads on the "
        "Free tier are subscription_plans;\n"
        "- managing Family members after purchase is family_plan;\n"
        "- a fault in the music stream or player is playback; the app or site "
        "failing anywhere is app_technical.\n"
        "The seed list's `other` fallback is not a mapping target: non-support "
        "messages are junk.\n\n"
        "Reply with raw JSON only, no markdown and no prose:\n"
        '{"mapping": "<seed intent id>" | "new" | "junk", '
        '"justification": "one sentence naming the decisive evidence"}'
    )


def parse_mapping_reply(
    cluster_id: int, content: str, intent_ids: Collection[str]
) -> tuple[str, str]:
    """Validate the labeler's mapping reply and normalize the justification."""
    payload = llm.extract_json_object(content)
    if payload is None:
        raise DiscoveryError(
            f"labeler returned no JSON object for cluster {cluster_id}: "
            f"{content.strip()[: llm.MAX_REPLY_EXCERPT]!r}"
        )
    mapping = payload.get("mapping")
    mapping_error = cluster_mapping_error(mapping, intent_ids)
    if mapping_error is not None:
        raise DiscoveryError(
            f"labeler returned an invalid mapping for cluster {cluster_id}: "
            f"{mapping_error} ({mapping!r})"
        )
    justification = payload.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise DiscoveryError(
            f"labeler returned an empty justification for cluster {cluster_id}"
        )
    return mapping, " ".join(justification.split())


def cluster_mapping_error(mapping: object, intent_ids: Collection[str]) -> str | None:
    """Check a mapping verdict against the supported flags and seed ids."""
    if not isinstance(mapping, str) or not mapping:
        return "invalid mapping"
    if mapping in FLAG_MAPPINGS or mapping in intent_ids:
        return None
    return "unknown mapping"


def intent_cluster_error(
    cluster: IntentCluster, intent_ids: Collection[str] | None = None
) -> str | None:
    """Check a cluster against the contract the artifact reader enforces.

    Shared by :func:`discover_intents` and :func:`read_intent_clusters_jsonl`
    so the stage cannot stage a cluster its own reader would reject. When
    ``intent_ids`` is given, a seed mapping outside them is invalid; the
    ``new`` and ``junk`` flags are always valid. Without the seed list only
    the mapping's shape can be checked.
    """
    if type(cluster.cluster_id) is not int or cluster.cluster_id < 0:
        return "invalid cluster_id"
    if type(cluster.size) is not int or cluster.size < 1:
        return "invalid size"
    if not isinstance(cluster.mapping, str) or not cluster.mapping:
        return "invalid mapping"
    if cluster.mapping not in FLAG_MAPPINGS and (
        intent_ids is not None and cluster.mapping not in intent_ids
    ):
        return f"unknown mapping {cluster.mapping!r}"
    if not isinstance(cluster.justification, str) or not cluster.justification.strip():
        return "invalid justification"
    if not cluster.candidates:
        return "no candidate seeds"
    for candidate in cluster.candidates:
        if not isinstance(candidate.intent_id, str) or not candidate.intent_id:
            return "invalid candidate intent_id"
        if not _is_number(candidate.similarity):
            return "invalid candidate similarity"
    if not cluster.examples:
        return "no examples"
    for example in cluster.examples:
        if type(example.interaction_id) is not int:
            return "invalid example interaction_id"
        if not isinstance(example.message, str) or not example.message.strip():
            return "invalid example message"
        if not _is_number(example.similarity):
            return "invalid example similarity"
    return None


def cluster_to_json(cluster: IntentCluster) -> dict:
    """Serialize an IntentCluster to a JSON-compatible mapping."""
    return {
        "cluster_id": cluster.cluster_id,
        "size": cluster.size,
        "mapping": cluster.mapping,
        "justification": cluster.justification,
        "candidates": [
            {"intent_id": candidate.intent_id, "similarity": candidate.similarity}
            for candidate in cluster.candidates
        ],
        "examples": [
            {
                "interaction_id": example.interaction_id,
                "message": example.message,
                "similarity": example.similarity,
            }
            for example in cluster.examples
        ],
    }


def cached_cluster_to_json(verdict: CachedClusterVerdict) -> dict:
    """Serialize a CachedClusterVerdict to a JSON-compatible mapping."""
    return {
        "cluster_id": verdict.cluster_id,
        "prompt_sha256": verdict.prompt_sha256,
        "mapping": verdict.mapping,
        "justification": verdict.justification,
        "model": verdict.model,
    }


def stage_intent_clusters_jsonl(
    clusters: Sequence[IntentCluster], output_path: Path | str
) -> Path:
    """Write the clusters to a temporary sibling, ready to be swapped in.

    Callers that must update several files as one transaction stage every
    output first and then swap the returned paths in together, so a write
    failure cannot leave a partial set of new files behind.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for cluster in clusters:
                handle.write(json.dumps(cluster_to_json(cluster)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_intent_clusters_jsonl(
    input_path: Path | str = DEFAULT_CLUSTERS_PATH,
    intent_ids: Collection[str] | None = None,
) -> tuple[IntentCluster, ...]:
    """Read the cluster artifact back from the JSON Lines format written above.

    Inverse of :func:`stage_intent_clusters_jsonl`: parses every record and
    validates the shared cluster contract, plus unique cluster ids. When
    ``intent_ids`` is given, every seed mapping must be one of them; the
    ``new`` and ``junk`` flags are always accepted. Returns the clusters in
    file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise DiscoveryError(f"intent clusters JSONL does not exist: {input_path}")
    clusters = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DiscoveryError(f"malformed JSON on {location}: {exc}") from exc
            cluster = _parse_cluster_record(decoded, location)
            error = intent_cluster_error(cluster, intent_ids)
            if error is not None:
                raise DiscoveryError(f"malformed cluster on {location}: {error}")
            if cluster.cluster_id in seen_ids:
                raise DiscoveryError(
                    f"duplicate cluster_id {cluster.cluster_id} on {location}"
                )
            seen_ids.add(cluster.cluster_id)
            clusters.append(cluster)
    return tuple(clusters)


def render_review_markdown(
    clusters: Sequence[IntentCluster],
    report: DiscoveryReport,
    seed_intents: Sequence[taxonomy.SeedIntent],
) -> str:
    """Render the human-reviewable Markdown with the evidence per cluster."""
    lines = [
        "# Intent discovery review",
        "",
        (
            "Generated by `query discover-intents` (ticket 10) over "
            f"{report.messages} RAG-pool Customer Messages, embedded with "
            f"`{report.embedding_model}` and clustered into {report.clusters} "
            f"clusters (k={report.requested_clusters}, seed={report.seed}). Each "
            "cluster was mapped to a seed intent from "
            "`docs/intent-seed-taxonomy.md`, or flagged `new`/`junk`, by the "
            "labeler role with embedding similarity as evidence; ticket 11 "
            "reconciles this with the seed list and records the final merge and "
            "split decisions."
        ),
        "",
        "## Mapping summary",
        "",
        (
            f"- clusters: {report.clusters} (mapped {report.mapped_clusters}, "
            f"new {report.new_clusters}, junk {report.junk_clusters})"
        ),
        (
            f"- messages: {report.messages} "
            f"(mapped {report.mapped_messages}, new {report.new_messages}, "
            f"junk {report.junk_messages})"
        ),
        f"- mapped share of messages: {report.mapped_share:.2%}",
        (
            "- seed intents with no mapped cluster: "
            + (
                ", ".join(f"`{intent_id}`" for intent_id in report.absent_intents)
                if report.absent_intents
                else "none"
            )
        ),
        "",
        "| Mapping | Clusters | Messages |",
        "|---------|---------:|---------:|",
    ]
    for intent in seed_intents:
        counts = report.per_intent.get(intent.intent_id)
        if counts is None:
            continue
        lines.append(
            f"| `{intent.intent_id}` | {counts.clusters} | {counts.messages} |"
        )
    lines.append(f"| `{MAPPING_NEW}` | {report.new_clusters} | {report.new_messages} |")
    lines.append(
        f"| `{MAPPING_JUNK}` | {report.junk_clusters} | {report.junk_messages} |"
    )
    lines.extend(["", "## Clusters"])
    for cluster in clusters:
        lines.extend(
            [
                "",
                (
                    f"### Cluster {cluster.cluster_id} — `{cluster.mapping}` "
                    f"({cluster.size} messages)"
                ),
                "",
                "Candidate seeds: "
                + ", ".join(
                    f"`{candidate.intent_id}` {candidate.similarity:.2f}"
                    for candidate in cluster.candidates
                ),
                "",
                f"Justification: {cluster.justification}",
                "",
                "Examples closest to the centroid:",
                "",
            ]
        )
        lines.extend(
            f"{number}. `[{example.interaction_id}]` {example.message}"
            for number, example in enumerate(cluster.examples, start=1)
        )
    lines.append("")
    return "\n".join(lines)


def stage_review_markdown(
    clusters: Sequence[IntentCluster],
    report: DiscoveryReport,
    seed_intents: Sequence[taxonomy.SeedIntent],
    output_path: Path | str,
) -> Path:
    """Write the review Markdown to a temporary sibling, ready to be swapped in."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_review_markdown(clusters, report, seed_intents))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def _require_arguments(
    interactions: Sequence[Interaction],
    clusters: int,
    examples: int,
    candidates: int,
    workers: int,
) -> None:
    if not interactions:
        raise DiscoveryError("no interactions to cluster")
    if clusters < 1:
        raise DiscoveryError(f"clusters must be at least 1, got {clusters}")
    if clusters > len(interactions):
        raise DiscoveryError(
            f"clusters ({clusters}) cannot exceed the {len(interactions)} messages"
        )
    if examples < 1:
        raise DiscoveryError(f"examples must be at least 1, got {examples}")
    if candidates < 1:
        raise DiscoveryError(f"candidates must be at least 1, got {candidates}")
    if workers < 1:
        raise DiscoveryError(f"workers must be at least 1, got {workers}")


def _require_unique_ids(interactions: Sequence[Interaction]) -> None:
    seen: set[int] = set()
    for interaction in interactions:
        if interaction.interaction_id in seen:
            raise DiscoveryError(
                f"duplicate interaction_id {interaction.interaction_id} in the input"
            )
        seen.add(interaction.interaction_id)


def _support_intents(
    seed_intents: Sequence[taxonomy.SeedIntent],
) -> tuple[taxonomy.SeedIntent, ...]:
    support = tuple(
        intent
        for intent in seed_intents
        if intent.intent_id != taxonomy.OTHER_INTENT_ID
    )
    if not support:
        raise DiscoveryError("seed taxonomy has no support intents to map against")
    return support


def _embed_anchors(
    embedder: embedding.Embedder, seed_intents: Sequence[taxonomy.SeedIntent]
) -> dict[str, np.ndarray]:
    """Embed each intent's definition and representative message as anchors."""
    texts = [
        text
        for intent in seed_intents
        for text in (intent.definition, intent.representative_message)
    ]
    vectors = _as_matrix(embedder.embed(texts), len(texts))
    return {
        intent.intent_id: vectors[index * 2 : index * 2 + 2]
        for index, intent in enumerate(seed_intents)
    }


def _as_matrix(vectors: Sequence[Sequence[float]], expected: int) -> np.ndarray:
    if len(vectors) != expected:
        raise DiscoveryError(
            f"embedder returned {len(vectors)} vectors for {expected} texts"
        )
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except ValueError as exc:
        raise DiscoveryError(f"embedder returned malformed vectors: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise DiscoveryError("embedder returned malformed vectors")
    return matrix


def _cluster_messages(
    matrix: np.ndarray, clusters: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Partition message indices into clusters with seeded KMeans."""
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=10, algorithm="lloyd")
    labels = model.fit_predict(matrix)
    members: list[list[int]] = [[] for _ in range(clusters)]
    for index, label in enumerate(labels):
        members[int(label)].append(index)
    if any(not group for group in members):
        raise DiscoveryError(
            f"KMeans left an empty cluster; lower --clusters (currently {clusters})"
        )
    return tuple(tuple(group) for group in members)


@dataclass(frozen=True)
class _MappingContext:
    """Everything one cluster mapping needs beyond its member indices."""

    interactions: Sequence[Interaction]
    matrix: np.ndarray
    anchors: dict[str, np.ndarray]
    seed_intents: Sequence[taxonomy.SeedIntent]
    examples: int
    candidates: int
    infer: Callable[[str], llm.LLMReply]
    saved: dict[int, CachedClusterVerdict]
    record: Callable[[CachedClusterVerdict], None] | None


def _map_clusters(
    members: Sequence[Sequence[int]],
    context: _MappingContext,
    workers: int,
) -> tuple[tuple[IntentCluster, ...], tuple[str, ...]]:
    """Map every cluster, preserving cluster order; also return the models used."""
    if workers == 1 or len(members) <= 1:
        results = [
            _map_cluster(cluster_id, group, context)
            for cluster_id, group in enumerate(members)
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_map_cluster, cluster_id, group, context)
                for cluster_id, group in enumerate(members)
            ]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    clusters = tuple(cluster for cluster, _ in results)
    models: list[str] = []
    for _, model in results:
        if model not in models:
            models.append(model)
    return clusters, tuple(models)


def _map_cluster(
    cluster_id: int,
    member_indices: Sequence[int],
    context: _MappingContext,
) -> tuple[IntentCluster, str]:
    """Map one cluster, reusing a matching cached verdict.

    The labeler is retried once with a repair prompt when a reply is not valid
    JSON. Fresh verdicts are recorded immediately so an interrupted run can
    resume.
    """
    exemplars, centroid = _centroid_and_examples(
        context.matrix, member_indices, context.interactions, context.examples
    )
    seed_candidates = _candidates(centroid, context.anchors, context.candidates)
    prompt = build_mapping_prompt(
        cluster_id,
        len(member_indices),
        exemplars,
        seed_candidates,
        context.seed_intents,
    )
    cache_key = llm.prompt_sha256(SYSTEM_PROMPT, prompt)
    cached = context.saved.get(cluster_id)
    intent_ids = tuple(intent.intent_id for intent in context.seed_intents)
    if cached is not None and cached.prompt_sha256 == cache_key:
        cluster = IntentCluster(
            cluster_id=cluster_id,
            size=len(member_indices),
            mapping=cached.mapping,
            justification=cached.justification,
            candidates=seed_candidates,
            examples=exemplars,
        )
        model = cached.model
    else:
        reply = _call_labeler(context.infer, prompt, cluster_id)
        try:
            mapping, justification = parse_mapping_reply(
                cluster_id, reply.content, intent_ids
            )
        except DiscoveryError:
            repair_prompt = llm.build_repair_prompt(prompt, reply.content)
            reply = _call_labeler(context.infer, repair_prompt, cluster_id)
            mapping, justification = parse_mapping_reply(
                cluster_id, reply.content, intent_ids
            )
        cluster = IntentCluster(
            cluster_id=cluster_id,
            size=len(member_indices),
            mapping=mapping,
            justification=justification,
            candidates=seed_candidates,
            examples=exemplars,
        )
        model = reply.model
        if context.record is not None:
            context.record(
                CachedClusterVerdict(
                    cluster_id=cluster_id,
                    prompt_sha256=cache_key,
                    mapping=mapping,
                    justification=justification,
                    model=reply.model,
                )
            )
    error = intent_cluster_error(cluster, intent_ids)
    if error is not None:
        raise DiscoveryError(f"cluster {cluster_id} violates the contract: {error}")
    return cluster, model


def _centroid_and_examples(
    matrix: np.ndarray,
    member_indices: Sequence[int],
    interactions: Sequence[Interaction],
    examples: int,
) -> tuple[tuple[ClusterExample, ...], np.ndarray]:
    """Pick the examples closest to the cluster centroid, and the centroid."""
    members = matrix[list(member_indices)]
    centroid = _normalize_row(members.mean(axis=0))
    similarities = members @ centroid
    order = sorted(
        range(len(member_indices)),
        key=lambda position: (
            -similarities[position],
            interactions[member_indices[position]].interaction_id,
        ),
    )
    picked = []
    for position in order[:examples]:
        interaction = interactions[member_indices[position]]
        picked.append(
            ClusterExample(
                interaction_id=interaction.interaction_id,
                message=interaction.opening_message.text,
                similarity=round(float(similarities[position]), SIMILARITY_DECIMALS),
            )
        )
    return tuple(picked), centroid


def _candidates(
    centroid: np.ndarray, anchors: dict[str, np.ndarray], candidates: int
) -> tuple[SeedCandidate, ...]:
    """Rank seed intents by their best anchor similarity to the centroid."""
    scored = [
        SeedCandidate(
            intent_id=intent_id,
            similarity=round(float((vectors @ centroid).max()), SIMILARITY_DECIMALS),
        )
        for intent_id, vectors in anchors.items()
    ]
    scored.sort(key=lambda candidate: (-candidate.similarity, candidate.intent_id))
    return tuple(scored[:candidates])


def _normalize_row(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _report(
    clusters: Sequence[IntentCluster],
    models: Sequence[str],
    interactions: Sequence[Interaction],
    seed_intents: Sequence[taxonomy.SeedIntent],
    requested_clusters: int,
    seed: int,
    embedding_model: str,
) -> DiscoveryReport:
    per_intent = {
        intent.intent_id: IntentCounts(clusters=0, messages=0)
        for intent in seed_intents
    }
    mapped_clusters = new_clusters = junk_clusters = 0
    mapped_messages = new_messages = junk_messages = 0
    for cluster in clusters:
        if cluster.mapping in FLAG_MAPPINGS:
            if cluster.mapping == MAPPING_NEW:
                new_clusters += 1
                new_messages += cluster.size
            else:
                junk_clusters += 1
                junk_messages += cluster.size
            continue
        mapped_clusters += 1
        mapped_messages += cluster.size
        counts = per_intent[cluster.mapping]
        per_intent[cluster.mapping] = IntentCounts(
            clusters=counts.clusters + 1, messages=counts.messages + cluster.size
        )
    absent = tuple(
        intent_id for intent_id, counts in per_intent.items() if counts.clusters == 0
    )
    return DiscoveryReport(
        clusters=len(clusters),
        messages=len(interactions),
        requested_clusters=requested_clusters,
        seed=seed,
        embedding_model=embedding_model,
        mapped_clusters=mapped_clusters,
        new_clusters=new_clusters,
        junk_clusters=junk_clusters,
        mapped_messages=mapped_messages,
        new_messages=new_messages,
        junk_messages=junk_messages,
        per_intent=per_intent,
        absent_intents=absent,
        models=tuple(models),
    )


def _parse_cluster_record(record: object, location: str) -> IntentCluster:
    if not isinstance(record, dict):
        raise DiscoveryError(f"malformed cluster on {location}: expected an object")
    cluster_id = record.get("cluster_id")
    if type(cluster_id) is not int:
        raise DiscoveryError(f"malformed cluster on {location}: invalid cluster_id")
    size = record.get("size")
    if type(size) is not int:
        raise DiscoveryError(f"malformed cluster on {location}: invalid size")
    mapping = record.get("mapping")
    if not isinstance(mapping, str):
        raise DiscoveryError(f"malformed cluster on {location}: invalid mapping")
    justification = record.get("justification")
    if not isinstance(justification, str):
        raise DiscoveryError(f"malformed cluster on {location}: invalid justification")
    candidates = _parse_candidates(record.get("candidates"), location)
    examples = _parse_examples(record.get("examples"), location)
    return IntentCluster(
        cluster_id=cluster_id,
        size=size,
        mapping=mapping,
        justification=justification,
        candidates=candidates,
        examples=examples,
    )


def _parse_candidates(value: object, location: str) -> tuple[SeedCandidate, ...]:
    if not isinstance(value, list):
        raise DiscoveryError(f"malformed cluster on {location}: invalid candidates")
    candidates = []
    for candidate in value:
        if not isinstance(candidate, dict):
            raise DiscoveryError(
                f"malformed cluster on {location}: invalid candidate entry"
            )
        intent_id = candidate.get("intent_id")
        similarity = candidate.get("similarity")
        if not isinstance(intent_id, str) or not _is_number(similarity):
            raise DiscoveryError(
                f"malformed cluster on {location}: invalid candidate entry"
            )
        candidates.append(
            SeedCandidate(intent_id=intent_id, similarity=float(similarity))
        )
    return tuple(candidates)


def _parse_examples(value: object, location: str) -> tuple[ClusterExample, ...]:
    if not isinstance(value, list):
        raise DiscoveryError(f"malformed cluster on {location}: invalid examples")
    examples = []
    for example in value:
        if not isinstance(example, dict):
            raise DiscoveryError(
                f"malformed cluster on {location}: invalid example entry"
            )
        interaction_id = example.get("interaction_id")
        message = example.get("message")
        similarity = example.get("similarity")
        if (
            type(interaction_id) is not int
            or not isinstance(message, str)
            or not _is_number(similarity)
        ):
            raise DiscoveryError(
                f"malformed cluster on {location}: invalid example entry"
            )
        examples.append(
            ClusterExample(
                interaction_id=interaction_id,
                message=message,
                similarity=float(similarity),
            )
        )
    return tuple(examples)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_cache_line(line: str, location: str) -> CachedClusterVerdict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DiscoveryError(f"malformed JSON in cache on {location}: {exc}") from exc
    if not isinstance(record, dict):
        raise DiscoveryError(
            f"malformed cache record on {location}: expected an object"
        )
    cluster_id = record.get("cluster_id")
    if type(cluster_id) is not int:
        raise DiscoveryError(
            f"malformed cache record on {location}: invalid cluster_id"
        )
    prompt_sha256 = record.get("prompt_sha256")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise DiscoveryError(
            f"malformed cache record on {location}: invalid prompt_sha256"
        )
    mapping = record.get("mapping")
    if not isinstance(mapping, str) or not mapping:
        raise DiscoveryError(f"malformed cache record on {location}: invalid mapping")
    justification = record.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise DiscoveryError(
            f"malformed cache record on {location}: invalid justification"
        )
    model = record.get("model")
    if not isinstance(model, str) or not model:
        raise DiscoveryError(f"malformed cache record on {location}: invalid model")
    return CachedClusterVerdict(
        cluster_id=cluster_id,
        prompt_sha256=prompt_sha256,
        mapping=mapping,
        justification=justification,
        model=model,
    )


def _call_labeler(
    infer: Callable[[str], llm.LLMReply], prompt: str, cluster_id: int
) -> llm.LLMReply:
    try:
        return infer(prompt)
    except Exception as exc:
        raise DiscoveryError(
            f"labeler call failed for cluster {cluster_id}: {exc}"
        ) from exc

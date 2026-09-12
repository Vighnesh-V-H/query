"""Semantic (MiniLM) retrieval over Resolved Cases.

Ticket 16 builds the semantic half of the historical retrieval in
``docs/specs-v0.md`` §5: a cosine-similarity index over the
retrieval-eligible records of the resolution dataset (ADR-0008) that returns
ranked Historical Cases for a query Customer Message.

Only Resolved Cases enter the index (ADR-0001, decision 4): the build reads
``ResolutionRecord`` values and embeds exactly those with
``retrieval_eligible`` true, counting the rest as skipped. The indexed text
is the opening Customer Message — the same shape as an inference-time query —
so brand replies cannot leak into similarity scores; the full thread stays
reachable through the returned ``interaction_id``.

Embeddings come from :mod:`query.embedding` (decision 9): local
all-MiniLM-L6-v2 vectors, computed offline through fastembed/ONNX with no API
dependency. Vectors are L2-normalized at build time, so cosine similarity is
a plain dot product. The first embedding call downloads the ~90 MB model
file; later runs work offline. Callers that already hold an embedder (the
CLI builds the index and embeds the query in one process) pass it in so the
model loads once; tests pass fakes through the same parameter and never
touch the network.

The index persists as versioned JSON so ticket 17's hybrid retrieval can
reuse it without re-embedding; ``build_time_s`` is recorded with
``perf_counter`` at build time and stored in both the index and the report.
Unlike BM25, retrieval returns the top-K cases unconditionally (apart from
an empty query or an empty index): embedding space always has a nearest
neighbor, and the raw similarity score is the evidence-weakness signal the
escalation policy (ticket 21) consumes downstream.
"""

import json
import math
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from query import embedding, resolution

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_PATH = resolution.DEFAULT_DATASET_PATH
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "minilm-index.json"

INDEX_VERSION = 1
DEFAULT_TOP_K = 5


class MiniLMError(Exception):
    """Raised when a MiniLM index cannot be built, read, or queried."""


@dataclass(frozen=True)
class RetrievedCase:
    """One ranked Historical Case for a query."""

    interaction_id: int
    score: float


@dataclass(frozen=True)
class MiniLMReport:
    """Counts and timings of one index build, for honest reporting."""

    index_version: int
    total_records: int
    indexed: int
    skipped: int
    embedding_model: str
    embedding_dim: int
    build_time_s: float


@dataclass
class MiniLMIndex:
    """An in-memory cosine-similarity index over Resolved Cases.

    Mutable only through construction helpers; query paths never mutate it.
    ``vectors`` maps interaction id to its L2-normalized embedding, so the
    cosine similarity of a normalized query is a plain dot product.
    """

    doc_ids: tuple[int, ...]
    vectors: dict[int, tuple[float, ...]]
    embedding_model: str
    embedding_dim: int
    build_time_s: float = 0.0
    version: int = INDEX_VERSION


def document_text(record: resolution.ResolutionRecord) -> str:
    """Return the indexed text of one record: its opening Customer Message."""
    return record.interaction.opening_message.text


def build_minilm_index(
    records: Sequence[resolution.ResolutionRecord],
    embedder: embedding.Embedder | None = None,
) -> tuple[MiniLMIndex, MiniLMReport]:
    """Embed the retrieval-eligible records and report the build.

    Only records with ``retrieval_eligible`` true enter the index; the rest
    are counted as skipped. Every Resolved Case stays addressable, even one
    whose text embeds to a zero vector — it simply scores zero against every
    query. Returns the index and a report holding the wall-clock build time
    in ``build_time_s`` (embedding compute included; the first run also pays
    the one-time model download inside the embedder).

    ``embedder`` defaults to a lazy :class:`MiniLMEmbedder` that loads on
    first use; tests pass fakes through the same parameter. Embed failures
    propagate as :class:`EmbeddingError`; malformed embedder output (wrong
    count, ragged or empty vectors, non-finite values) raises
    :class:`MiniLMError`.
    """
    eligible = [record for record in records if record.retrieval_eligible]
    seen: set[int] = set()
    for record in eligible:
        if record.interaction_id in seen:
            raise MiniLMError(
                f"duplicate interaction_id {record.interaction_id} in the input"
            )
        seen.add(record.interaction_id)
    active = embedder if embedder is not None else embedding.MiniLMEmbedder()
    texts = [document_text(record) for record in eligible]
    started = time.perf_counter()
    try:
        raw_vectors = active.embed(texts)
    except embedding.EmbeddingError:
        raise
    except Exception as exc:
        raise MiniLMError(f"embedder failed on {len(texts)} texts: {exc}") from exc
    vectors = _normalize_vectors(raw_vectors, len(texts))
    build_time_s = time.perf_counter() - started
    dim = len(vectors[0]) if vectors else 0
    index = MiniLMIndex(
        doc_ids=tuple(record.interaction_id for record in eligible),
        vectors={
            record.interaction_id: vector
            for record, vector in zip(eligible, vectors)
        },
        embedding_model=active.model_name,
        embedding_dim=dim,
        build_time_s=build_time_s,
    )
    report = MiniLMReport(
        index_version=INDEX_VERSION,
        total_records=len(records),
        indexed=len(eligible),
        skipped=len(records) - len(eligible),
        embedding_model=active.model_name,
        embedding_dim=dim,
        build_time_s=build_time_s,
    )
    return index, report


def retrieve_minilm(
    index: MiniLMIndex,
    query: str,
    embedder: embedding.Embedder | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[RetrievedCase, ...]:
    """Rank the indexed cases for a query message, best score first.

    The query is embedded with the same model family as the index; a
    dimension mismatch raises :class:`MiniLMError` rather than scoring
    garbage. Ties break by ascending ``interaction_id`` so rankings are
    deterministic across machines. An empty query or an empty index returns
    no cases rather than raising. Otherwise the top-K cases are returned
    unconditionally — even low-similarity ones, with their scores intact for
    the escalation policy to judge as weak evidence.
    """
    if top_k < 1:
        raise MiniLMError(f"invalid top_k {top_k!r}: must be >= 1")
    if not query.strip() or not index.doc_ids:
        return ()
    active = embedder if embedder is not None else embedding.MiniLMEmbedder()
    try:
        raw = active.embed([query])
    except embedding.EmbeddingError:
        raise
    except Exception as exc:
        raise MiniLMError(f"embedder failed on the query: {exc}") from exc
    if len(raw) != 1:
        raise MiniLMError(
            f"embedder returned {len(raw)} vectors for 1 query text"
        )
    (query_vector,) = _normalize_vectors(raw, 1)
    if len(query_vector) != index.embedding_dim:
        raise MiniLMError(
            f"query embedding dim {len(query_vector)} does not match "
            f"the index dim {index.embedding_dim} "
            f"(index model {index.embedding_model!r})"
        )
    scored = [
        RetrievedCase(
            interaction_id=doc_id,
            score=sum(
                component * query_component
                for component, query_component in zip(
                    index.vectors[doc_id], query_vector
                )
            ),
        )
        for doc_id in index.doc_ids
    ]
    scored.sort(key=lambda case: (-case.score, case.interaction_id))
    return tuple(scored[:top_k])


def _normalize_vectors(
    raw_vectors: Sequence[Sequence[float]], expected: int
) -> list[tuple[float, ...]]:
    """Validate embedder output and return one unit-length vector per text."""
    if len(raw_vectors) != expected:
        raise MiniLMError(
            f"embedder returned {len(raw_vectors)} vectors for {expected} texts"
        )
    normalized: list[tuple[float, ...]] = []
    dim: int | None = None
    for position, raw in enumerate(raw_vectors):
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise MiniLMError(
                f"embedder returned a malformed vector at position {position}"
            )
        values = []
        for component in raw:
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(component)
            ):
                raise MiniLMError(
                    f"embedder returned a non-finite vector at position {position}"
                )
            values.append(float(component))
        if not values:
            raise MiniLMError(
                f"embedder returned an empty vector at position {position}"
            )
        if dim is None:
            dim = len(values)
        elif len(values) != dim:
            raise MiniLMError(
                f"embedder returned ragged vectors "
                f"({len(values)} != {dim} at position {position})"
            )
        normalized.append(embedding.l2_normalize(values))
    return normalized


def index_to_json(index: MiniLMIndex) -> dict:
    """Serialize an index to a JSON-compatible mapping."""
    return {
        "index_version": index.version,
        "embedding_model": index.embedding_model,
        "embedding_dim": index.embedding_dim,
        "build_time_s": index.build_time_s,
        "docs": [
            {
                "interaction_id": doc_id,
                "vector": list(index.vectors[doc_id]),
            }
            for doc_id in index.doc_ids
        ],
    }


def index_from_json(payload: object, location: str = "minilm index") -> MiniLMIndex:
    """Parse an index written by :func:`index_to_json`, validating its shape."""
    if not isinstance(payload, dict):
        raise MiniLMError(f"malformed {location}: expected an object")
    version = payload.get("index_version")
    if version != INDEX_VERSION:
        raise MiniLMError(
            f"malformed {location}: unsupported index_version {version!r} "
            f"(expected {INDEX_VERSION})"
        )
    model = payload.get("embedding_model")
    if not isinstance(model, str) or not model:
        raise MiniLMError(f"malformed {location}: invalid embedding_model")
    dim = payload.get("embedding_dim")
    if type(dim) is not int or dim < 1:
        raise MiniLMError(f"malformed {location}: invalid embedding_dim")
    try:
        build_time_s = float(payload.get("build_time_s", 0.0))
    except (TypeError, ValueError) as exc:
        raise MiniLMError(f"malformed {location}: {exc}") from exc
    raw_docs = payload.get("docs")
    if not isinstance(raw_docs, list):
        raise MiniLMError(f"malformed {location}: docs must be a list")
    doc_ids: list[int] = []
    vectors: dict[int, tuple[float, ...]] = {}
    for position, raw in enumerate(raw_docs):
        where = f"{location} doc {position}"
        if not isinstance(raw, dict):
            raise MiniLMError(f"malformed {where}: expected an object")
        doc_id = raw.get("interaction_id")
        vector = raw.get("vector")
        if type(doc_id) is not int:
            raise MiniLMError(f"malformed {where}: bad interaction_id")
        if doc_id in vectors:
            raise MiniLMError(
                f"malformed {where}: duplicate interaction_id {doc_id}"
            )
        if not isinstance(vector, list) or len(vector) != dim:
            raise MiniLMError(
                f"malformed {where}: vector must be a list of {dim} numbers"
            )
        values: list[float] = []
        for component in vector:
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(component)
            ):
                raise MiniLMError(f"malformed {where}: bad vector entry")
            values.append(float(component))
        doc_ids.append(doc_id)
        vectors[doc_id] = tuple(values)
    return MiniLMIndex(
        doc_ids=tuple(doc_ids),
        vectors=vectors,
        embedding_model=model,
        embedding_dim=dim,
        build_time_s=build_time_s,
    )


def stage_minilm_index_json(index: MiniLMIndex, output_path: Path | str) -> Path:
    """Write an index to a temporary sibling, ready to be swapped in.

    Mirrors the JSONL stagers: callers stage every output first and then swap
    the returned paths in together, so a write failure cannot leave a partial
    set of new files behind.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(index_to_json(index), indent=2) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_minilm_index_json(input_path: Path | str) -> MiniLMIndex:
    """Read a persisted index, failing loudly on missing or malformed files."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise MiniLMError(f"minilm index JSON does not exist: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MiniLMError(f"malformed JSON in {input_path}: {exc}") from exc
    return index_from_json(payload, location=str(input_path))

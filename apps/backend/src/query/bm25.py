"""Lexical (BM25) retrieval over Resolved Cases.

Ticket 15 builds the lexical half of the historical retrieval in
``docs/specs-v0.md`` §5: a BM25 Okapi index over the retrieval-eligible
records of the resolution dataset (ADR-0008) that returns ranked Historical
Cases for a query Customer Message.

Only Resolved Cases enter the index (ADR-0001, decision 4): the build reads
``ResolutionRecord`` values and indexes exactly those with
``retrieval_eligible`` true, counting the rest as skipped. The indexed text
is the opening Customer Message — the same shape as an inference-time query —
so brand replies cannot leak into lexical scores; the full thread stays
reachable through the returned ``interaction_id``.

The implementation is dependency-free pure Python (BM25 Okapi, ``k1=1.5``,
``b=0.75``) rather than a new third-party package: the math is ~30 lines,
it stays offline and deterministic, and it keeps the 15-minute reproduction
story clear of extra downloads. Tokenization lowercases, strips @mentions
and links (they carry no lexical signal, same as the English filter), and
keeps ``[a-z0-9]+`` runs. The index persists as versioned JSON so ticket 17's
hybrid retrieval can reuse it without rebuilding; ``build_time_s`` is
recorded with ``perf_counter`` at build time and stored in both the index
and the report.
"""

import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from query import resolution

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_PATH = resolution.DEFAULT_DATASET_PATH
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "bm25-index.json"

INDEX_VERSION = 1
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_TOP_K = 5
TOKENIZER = "lowercase-alnum-mentions-urls-stripped"

URL_PATTERN = re.compile(r"https?://\S+|t\.co/\S+")
MENTION_PATTERN = re.compile(r"@\w+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class BM25Error(Exception):
    """Raised when a BM25 index cannot be built, read, or queried."""


@dataclass(frozen=True)
class RetrievedCase:
    """One ranked Historical Case for a query."""

    interaction_id: int
    score: float


@dataclass(frozen=True)
class BM25Report:
    """Counts and timings of one index build, for honest reporting."""

    index_version: int
    total_records: int
    indexed: int
    skipped: int
    avgdl: float
    build_time_s: float
    k1: float
    b: float
    tokenizer: str = TOKENIZER


@dataclass
class BM25Index:
    """An in-memory BM25 Okapi index over Resolved Cases.

    Mutable only through construction helpers; query paths never mutate it.
    ``doc_term_freqs`` maps interaction id -> {token -> count} and
    ``doc_freqs`` maps token -> number of indexed docs containing it.
    """

    doc_ids: tuple[int, ...]
    doc_lens: dict[int, int]
    doc_term_freqs: dict[int, dict[str, int]]
    doc_freqs: dict[str, int]
    avgdl: float
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    tokenizer: str = TOKENIZER
    build_time_s: float = 0.0
    version: int = INDEX_VERSION


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into lowercase alphanumeric tokens.

    Mentions and links are stripped first — they carry no lexical signal
    (same choice as the English filter). Returns an empty tuple for
    contentless messages rather than raising.
    """
    stripped = MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", text))
    return tuple(TOKEN_PATTERN.findall(stripped.lower()))


def document_text(record: resolution.ResolutionRecord) -> str:
    """Return the indexed text of one record: its opening Customer Message."""
    return record.interaction.opening_message.text


def build_bm25_index(
    records: Sequence[resolution.ResolutionRecord],
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> tuple[BM25Index, BM25Report]:
    """Index the retrieval-eligible records and report the build.

    Only records with ``retrieval_eligible`` true enter the index; the rest
    are counted as skipped. Eligible records whose tokenized text is empty
    are still indexed (as zero-length docs) so every Resolved Case stays
    addressable — they simply never score above zero. Returns the index and
    a report holding the wall-clock build time in ``build_time_s``.
    """
    if not math.isfinite(k1) or k1 < 0:
        raise BM25Error(f"invalid k1 {k1!r}: must be >= 0")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise BM25Error(f"invalid b {b!r}: must be in [0, 1]")
    started = time.perf_counter()
    eligible = [record for record in records if record.retrieval_eligible]
    seen: set[int] = set()
    for record in eligible:
        if record.interaction_id in seen:
            raise BM25Error(
                f"duplicate interaction_id {record.interaction_id} in the input"
            )
        seen.add(record.interaction_id)
    doc_ids: list[int] = []
    doc_lens: dict[int, int] = {}
    doc_term_freqs: dict[int, dict[str, int]] = {}
    doc_freqs: dict[str, int] = {}
    total_length = 0
    for record in eligible:
        tokens = tokenize(document_text(record))
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token in counts:
            doc_freqs[token] = doc_freqs.get(token, 0) + 1
        doc_ids.append(record.interaction_id)
        doc_lens[record.interaction_id] = len(tokens)
        doc_term_freqs[record.interaction_id] = counts
        total_length += len(tokens)
    avgdl = total_length / len(eligible) if eligible else 0.0
    build_time_s = time.perf_counter() - started
    index = BM25Index(
        doc_ids=tuple(doc_ids),
        doc_lens=doc_lens,
        doc_term_freqs=doc_term_freqs,
        doc_freqs=doc_freqs,
        avgdl=avgdl,
        k1=k1,
        b=b,
        build_time_s=build_time_s,
    )
    report = BM25Report(
        index_version=INDEX_VERSION,
        total_records=len(records),
        indexed=len(eligible),
        skipped=len(records) - len(eligible),
        avgdl=avgdl,
        build_time_s=build_time_s,
        k1=k1,
        b=b,
    )
    return index, report


def retrieve_bm25(
    index: BM25Index,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[RetrievedCase, ...]:
    """Rank the indexed cases for a query message, best score first.

    Ties break by ascending ``interaction_id`` so rankings are deterministic
    across machines. Returns at most ``top_k`` cases with positive scores;
    an empty query, an empty index, or a query with no overlapping terms
    returns no cases rather than raising.
    """
    if top_k < 1:
        raise BM25Error(f"invalid top_k {top_k!r}: must be >= 1")
    query_counts: dict[str, int] = {}
    for token in tokenize(query):
        query_counts[token] = query_counts.get(token, 0) + 1
    if not query_counts or not index.doc_ids:
        return ()
    scored: list[RetrievedCase] = []
    for doc_id in index.doc_ids:
        score = _score_document(index, doc_id, query_counts)
        if score > 0:
            scored.append(RetrievedCase(interaction_id=doc_id, score=score))
    scored.sort(key=lambda case: (-case.score, case.interaction_id))
    return tuple(scored[:top_k])


def _score_document(
    index: BM25Index, doc_id: int, query_counts: Mapping[str, int]
) -> float:
    """Score one document with BM25 Okapi over the query's distinct terms."""
    doc_len = index.doc_lens[doc_id]
    if index.avgdl == 0:
        return 0.0
    total = 0.0
    term_freqs = index.doc_term_freqs[doc_id]
    num_docs = len(index.doc_ids)
    for term in query_counts:
        freq = term_freqs.get(term)
        if not freq:
            continue
        doc_freq = index.doc_freqs.get(term, 0)
        idf = math.log(1 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        denom = freq + index.k1 * (1 - index.b + index.b * doc_len / index.avgdl)
        total += idf * (freq * (index.k1 + 1) / denom)
    return total


def index_to_json(index: BM25Index) -> dict:
    """Serialize an index to a JSON-compatible mapping."""
    return {
        "index_version": index.version,
        "tokenizer": index.tokenizer,
        "k1": index.k1,
        "b": index.b,
        "avgdl": index.avgdl,
        "build_time_s": index.build_time_s,
        "docs": [
            {
                "interaction_id": doc_id,
                "length": index.doc_lens[doc_id],
                "terms": dict(
                    sorted(index.doc_term_freqs[doc_id].items())
                ),
            }
            for doc_id in index.doc_ids
        ],
    }


def index_from_json(payload: object, location: str = "bm25 index") -> BM25Index:
    """Parse an index written by :func:`index_to_json`, validating its shape."""
    if not isinstance(payload, dict):
        raise BM25Error(f"malformed {location}: expected an object")
    version = payload.get("index_version")
    if version != INDEX_VERSION:
        raise BM25Error(
            f"malformed {location}: unsupported index_version {version!r} "
            f"(expected {INDEX_VERSION})"
        )
    try:
        k1 = float(payload["k1"])
        b = float(payload["b"])
        avgdl = float(payload["avgdl"])
        build_time_s = float(payload.get("build_time_s", 0.0))
        raw_docs = payload["docs"]
    except (KeyError, TypeError, ValueError) as exc:
        raise BM25Error(f"malformed {location}: {exc}") from exc
    if not math.isfinite(k1) or k1 < 0:
        raise BM25Error(f"malformed {location}: invalid k1 {k1!r}")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise BM25Error(f"malformed {location}: invalid b {b!r}")
    if not math.isfinite(avgdl) or avgdl < 0:
        raise BM25Error(f"malformed {location}: invalid avgdl {avgdl!r}")
    if not isinstance(raw_docs, list):
        raise BM25Error(f"malformed {location}: docs must be a list")
    tokenizer = payload.get("tokenizer", TOKENIZER)
    doc_ids: list[int] = []
    doc_lens: dict[int, int] = {}
    doc_term_freqs: dict[int, dict[str, int]] = {}
    doc_freqs: dict[str, int] = {}
    for position, raw in enumerate(raw_docs):
        where = f"{location} doc {position}"
        if not isinstance(raw, dict):
            raise BM25Error(f"malformed {where}: expected an object")
        doc_id = raw.get("interaction_id")
        length = raw.get("length")
        terms = raw.get("terms")
        if (
            type(doc_id) is not int
            or type(length) is not int
            or not isinstance(terms, dict)
            or length < 0
        ):
            raise BM25Error(f"malformed {where}: bad interaction_id/length/terms")
        if doc_id in doc_lens:
            raise BM25Error(f"malformed {where}: duplicate interaction_id {doc_id}")
        counts: dict[str, int] = {}
        for token, count in terms.items():
            if (
                type(token) is not str
                or type(count) is not int
                or not token
                or count < 1
            ):
                raise BM25Error(f"malformed {where}: bad term entry {token!r}")
            counts[token] = count
        if sum(counts.values()) != length:
            raise BM25Error(
                f"malformed {where}: length {length} does not match term counts"
            )
        doc_ids.append(doc_id)
        doc_lens[doc_id] = length
        doc_term_freqs[doc_id] = counts
        for token in counts:
            doc_freqs[token] = doc_freqs.get(token, 0) + 1
    return BM25Index(
        doc_ids=tuple(doc_ids),
        doc_lens=doc_lens,
        doc_term_freqs=doc_term_freqs,
        doc_freqs=doc_freqs,
        avgdl=avgdl,
        k1=k1,
        b=b,
        tokenizer=tokenizer if isinstance(tokenizer, str) else TOKENIZER,
        build_time_s=build_time_s,
    )


def stage_bm25_index_json(index: BM25Index, output_path: Path | str) -> Path:
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


def read_bm25_index_json(input_path: Path | str) -> BM25Index:
    """Read a persisted index, failing loudly on missing or malformed files."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise BM25Error(f"bm25 index JSON does not exist: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BM25Error(f"malformed JSON in {input_path}: {exc}") from exc
    return index_from_json(payload, location=str(input_path))

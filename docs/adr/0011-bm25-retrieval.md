# Lexical retrieval is dependency-free BM25 over opening messages, persisted for hybrid reuse

Ticket 15 builds the lexical half of `docs/specs-v0.md` §5: a BM25 Okapi index over the retrieval-eligible records of the resolution dataset (ADR-0008), queryable with `query build-bm25-index` and `query retrieve-bm25`.

Only Resolved Cases enter the index (ADR-0001, decision 4). The build reads `ResolutionRecord` values and indexes exactly those with `retrieval_eligible` true, counting the rest as skipped in the report — the same filter-then-count shape as the English filter stage, so the report states honestly how much of the dataset is retrieval evidence (215 of 3,000 in the recorded snapshot).

The indexed text is the opening Customer Message only, not the full thread. The inference-time query is a single customer message, so indexing brand replies would score queries against text the query side can never look like and leak resolution wording into lexical ranks. The full thread stays reachable through the returned `interaction_id`.

BM25 Okapi (`k1=1.5`, `b=0.75`) is implemented in ~30 lines of dependency-free Python instead of adding a `rank-bm25` package: the math is small, it stays offline and deterministic (ties break by ascending `interaction_id`), and it keeps the 15-minute reproduction story clear of extra downloads. Tokenization lowercases, strips @mentions and links (no lexical signal, same as ADR-0004), and keeps `[a-z0-9]+` runs; queries that tokenize to nothing return no cases instead of raising.

The index persists as versioned JSON (`index_version: 1`, tokenizer, params, per-doc term counts, `build_time_s` measured with `perf_counter`) so ticket 17's hybrid retrieval reuses it without rebuilding. `query retrieve-bm25` accepts either `--index` (a persisted file) or `--in` (a dataset for a one-shot ephemeral build) — never both — plus `--query`, `--top-k`, and an optional `--report` of ranked `{interaction_id, score}` pairs.

## Considered Options

- **Depend on `rank-bm25`**: rejected. A new dependency for standard BM25 arithmetic buys nothing over the local implementation and complicates offline reproduction.
- **Index the full thread (brand replies included)**: rejected. Brand wording would dominate lexical scores and does not match the query shape; evidence content is still one `interaction_id` lookup away.
- **TF-IDF cosine instead of BM25**: rejected. Ticket 13 already owns TF-IDF as the intent baseline; BM25's saturation and length normalization are the standard lexical complement to ticket 16's semantic index, which is what the hybrid stage needs.
- **In-memory only, no persisted index**: rejected. Hybrid retrieval would rebuild both indexes on every call; the versioned JSON file is the contract between tickets 15–17.
- **Empty queries / no-overlap queries raise**: rejected. Returning no cases is the honest signal the escalation policy (ticket 21) consumes as weak evidence.

## Consequences

- `apps/backend/src/query/bm25.py` owns build, query, and persistence; `query/cli.py` gains `build-bm25-index` and `retrieve-bm25` following the existing stage-then-commit output pattern.
- `data/bm25-index.json` is the lexical artifact; its report records `indexed`, `skipped`, `avgdl`, and `build_time_s`.
- Ticket 17 consumes this index plus ticket 16's semantic index; ticket 21 consumes empty-result signals as weak-evidence escalations.

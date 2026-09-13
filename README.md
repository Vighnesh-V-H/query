# Query

AI support agent for SpotifyCares — intent classification, grounded replies, auto/escalate decisions. Monorepo:

```text
apps/
├── backend/    Python pipeline (uv workspace member, package "query")
└── frontend/   Next.js minimal chat UI (mocked pipeline; see apps/frontend/README.md)
```

## Backend setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12+ is managed automatically). From the repo root:

```sh
uv sync --all-packages --all-extras
cp apps/backend/.env.example .env   # then set NVIDIA_API_KEY (repo root)
```

## Smoke test the LLM connection

```sh
uv run --package query query smoke-llm --role generator
```

## Run tests

```sh
uv run pytest   # from repo root or apps/backend/
```

## Read the source archive

With the provided `data/archive.zip` in place, extract and inspect the raw CSV files:

```sh
uv run --package query query read-source
```

The extracted files and their schema/count manifest are cached in `data/raw/`; rerunning the command with the same archive and an intact cache reports the cache without extracting again. A changed archive or modified cached CSV file is extracted again. Use `--archive` and `--extract-dir` to override the defaults.

## Build SpotifyCares Interactions

Stitch the raw tweets into normalized Interactions (one customer, one brand, ordered turns) and report counts for every stitching stage:

```sh
uv run --package query query build-interactions            # prints counts
uv run --package query query build-interactions --out data/interactions.jsonl --report data/interactions-report.json
```

The run over the provided dataset reports ~29.8K Interactions and ~89K turns. `--out` writes one Interaction per line as JSON Lines (the normalized format downstream stages consume); `--report` writes the stage counts as JSON. `--twcs` and `--brand` override the source CSV path and brand author id.

## Filter to English Interactions

Keep only Interactions whose opening customer message is English (see `docs/adr/0004-english-filter.md`) and report the retained vs filtered volume:

```sh
uv run --package query query filter-english --in data/interactions.jsonl --out data/interactions-en.jsonl --report data/english-filter-report.json
```

The run over the provided dataset retains 29,130 of 29,796 Interactions: 666 (2.24%) are dropped as confidently non-English (mostly Indonesian, Tagalog, Dutch, and French), and 2,938 short or link-only messages stay because the detector has no confident signal — the filter only drops positive non-English evidence. `--out` writes the retained Interactions in the same JSON Lines format; `--report` writes the counts by detected language as JSON. The stage is deterministic and re-runnable; `--in` defaults to `data/interactions.jsonl`.

## Sample and split

Draw a deterministic 4,000-Interaction sample from the English set and split it into the RAG pool (3,000) and the holdout reserved for the Golden Set (1,000):

```sh
uv run --package query query sample-split --in data/interactions-en.jsonl --rag-out data/rag-pool.jsonl --holdout-out data/holdout.jsonl --report data/sample-report.json
```

Selection ranks every Interaction by a SHA-256 of the seed and its interaction id (`--seed`, default 42), so the same seed yields the same sample on any machine and regardless of input-file order. The first 1,000 of the ranked sample become the holdout, the remaining 3,000 the RAG pool; the pools are written to separate files and never overlap. The holdout is reserved for the Golden Set — retrieval indexes only the RAG pool. `--sample-size` and `--holdout-size` override the defaults; when the input holds fewer Interactions than requested, the sample covers everything and the holdout scales down with it. `--in` defaults to `data/interactions-en.jsonl`; `--rag-out` and `--holdout-out` must be given together and must not point at the input or at each other, and `--report` is optional but must not collide with any of those paths.

## Label closure heuristics

Label the obvious closures in the RAG pool and flag the ambiguous ones for adjudication (see `docs/adr/0006-closure-heuristic-labels.md`):

```sh
uv run --package query query label-closure --in data/rag-pool.jsonl --out data/closure-labels.jsonl --report data/closure-report.json
```

Each output line is one Interaction's `{interaction_id, label, reason, needs_adjudication}`. The heuristics label 1,694 of the 3,000 RAG-pool Interactions definitively (190 Resolved, 1,317 Uncertain, 187 Unresolved); 1,306 (43.5%) are flagged for adjudication, 94% of them involving a move to DMs — the ambiguous middle reserved for ticket 7. A flagged record has `label: null` — it is not evidence of anything, and downstream stages must adjudicate it before it can be used. `--in` defaults to `data/rag-pool.jsonl`; `--out` and `--report` are optional and must not collide with `--in` or each other.

## Adjudicate flagged closures

Resolve the ambiguous endings the heuristics flagged — mostly DM deflections — with the configured labeler role and write the full labeled sample (see `docs/adr/0007-closure-llm-adjudication.md`):

```sh
uv run --package query query adjudicate-closures --in data/rag-pool.jsonl --labels data/closure-labels.jsonl --out data/closure-labels-final.jsonl --report data/closure-adjudication-report.json --workers 8 --cache data/closure-adjudication-cache.jsonl
```

Each output line is one Interaction's final `{interaction_id, label, source, reason, flag_reason, model}`: flagged cases get a labeler verdict with a one-line justification, everything else keeps its heuristic label. The recorded snapshot labels all 3,000 RAG-pool Interactions — 215 Resolved, 2,569 Uncertain, 216 Unresolved — recovering 25 Resolved and 29 Unresolved from the 1,306 flagged cases. The labeler is the only non-deterministic stage, so rerunning can move the ambiguous middle. `--workers` bounds concurrent calls and `--cache` records each verdict as it completes so an interrupted run resumes without re-calling; `--in` and `--labels` default to the RAG pool and heuristic labels, and all paths must be distinct.

## Build the resolution dataset

Join the final closure labels back onto the RAG pool and flag only Resolved Cases as retrieval-eligible Historical Cases (see `docs/adr/0008-resolution-dataset.md`):

```sh
uv run --package query query build-resolution-dataset --in data/rag-pool.jsonl --labels data/closure-labels-final.jsonl --out data/resolution-dataset.jsonl --report data/resolution-report.json
```

Each line is one Interaction's full record: `{interaction_id, dataset_version, label, retrieval_eligible, source, reason, flag_reason, model, customer_id, brand_id, turns}`. `retrieval_eligible` is derived from the label, so it is true only for Resolved Cases; Uncertain and Unresolved records stay in the dataset as evaluation negatives (ADR-0001's decision 4). The recorded snapshot holds all 3,000 RAG-pool Interactions with 215 retrieval-eligible (190 heuristic, 25 labeler); the report counts each category and the heuristic vs labeler split. The build fails if the labels do not cover the input exactly. `--in` and `--labels` default to the RAG pool and final labels; `--out` and `--report` are optional and must not collide with the inputs or each other.

## Discover intents from the data

Cluster the RAG pool's opening Customer Messages and reconcile each cluster against the seed taxonomy with a labeler verdict (see `docs/adr/0009-intent-discovery.md`):

```sh
uv run --package query query discover-intents --in data/rag-pool.jsonl --out data/intent-clusters.jsonl --report data/intent-discovery-report.json --review docs/intent-discovery.md --clusters 30 --workers 8 --cache data/intent-discovery-cache.jsonl
```

Every opening message is embedded locally with all-MiniLM-L6-v2 (ONNX via fastembed; the ~90 MB model downloads once, then works offline) and clustered with seeded KMeans. Each cluster carries embedding-similarity candidate seeds, the examples closest to its centroid, and one labeler verdict with a one-line justification: a seed intent id, `new` for a coherent support theme the seed misses, or `junk` for non-support messages. The recorded snapshot puts all 3,000 messages in 30 clusters — 26 mapped (2,501 messages), 1 new (23, concert presale codes), 3 junk (476) — with `account_admin` and `devices_connectivity` unmapped; `docs/intent-discovery.md` is the review artifact with examples per cluster and feeds ticket 11's final taxonomy. `--workers` bounds concurrent labeler calls and `--cache` records each verdict so an interrupted run resumes; the labeler is non-deterministic, so the review is one snapshot. `--in` defaults to the RAG pool, `--taxonomy` to the seed document, and `--clusters`, `--seed`, `--examples`, and `--candidates` tune the run; all paths must be distinct.

## Finalize the intent taxonomy

Reconcile the discovery clusters with the seed list into the versioned final taxonomy (see `docs/adr/0010-final-intent-taxonomy.md`) and check every recorded decision against the cluster artifact:

```sh
uv run --package query query reconcile-intents --report data/intent-reconciliation-report.json
```

`docs/intent-taxonomy.md` is the versioned single source of truth (v1: 12 support intents plus an `other` fallback), and later stages import it with `query.taxonomy.read_final_taxonomy()`; `docs/intent-seed-taxonomy.md` stays as input evidence. The recorded decisions: `account_access` + `account_admin` merge into `account`, `devices_connectivity` folds into `app_technical`, and discovery's `new` cluster (concert presale codes) is promoted as `presale_codes`. `query reconcile-intents` routes every cluster of `data/intent-clusters.jsonl` through those decisions and fails on drift — a dropped seed intent that receives mapped clusters, a `new` cluster without a decision, or a final intent left without cluster evidence. The cluster artifact is committed alongside the taxonomy: it is the recorded labeler snapshot the decisions cite and cannot be exactly reproduced, so the command runs on a fresh checkout; `--in` accepts a regenerated snapshot instead. The recorded run covers all 30 clusters over 3,000 messages: 2,524 (84.13%) route to a support intent and 476 stay as `other`. `--in` defaults to the cluster artifact, `--seed` to the seed document, and `--taxonomy` to the final taxonomy; `--report` writes the per-intent coverage as JSON, and all paths must be distinct.

## Model configuration

Model roles (generator / judge / labeler) and the NVIDIA NIM base URL live in `apps/backend/configs/models.yaml`. Each role can be overridden with a `QUERY_<ROLE>_MODEL` env var (e.g. `QUERY_GENERATOR_MODEL`). Set `QUERY_CONFIG_PATH` to load model roles from an alternate config file.

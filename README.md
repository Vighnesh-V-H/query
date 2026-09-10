# Query

AI support agent for SpotifyCares — intent classification, grounded replies, auto/escalate decisions. Monorepo:

```text
apps/
├── backend/    Python pipeline (uv workspace member, package "query")
└── frontend/   Next.js chat UI (planned)
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

Each output line is one Interaction's `{interaction_id, label, reason, needs_adjudication}`. The heuristics label 1,712 of the 3,000 RAG-pool Interactions definitively (211 Resolved, 1,231 Uncertain, 270 Unresolved); 1,288 (42.9%) are flagged for adjudication, 94% of them brand replies that move the conversation to DMs. A flagged record has `label: null` — it is not evidence of anything, and downstream stages must adjudicate it (ticket 7) before it can be used. `--in` defaults to `data/rag-pool.jsonl`; `--out` and `--report` are optional and must not collide with `--in` or each other.

## Model configuration

Model roles (generator / judge / labeler) and the NVIDIA NIM base URL live in `apps/backend/configs/models.yaml`. Each role can be overridden with a `QUERY_<ROLE>_MODEL` env var (e.g. `QUERY_GENERATOR_MODEL`). Set `QUERY_CONFIG_PATH` to load model roles from an alternate config file.

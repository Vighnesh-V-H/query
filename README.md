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

## Model configuration

Model roles (generator / judge / labeler) and the NVIDIA NIM base URL live in `apps/backend/configs/models.yaml`. Each role can be overridden with a `QUERY_<ROLE>_MODEL` env var (e.g. `QUERY_GENERATOR_MODEL`). Set `QUERY_CONFIG_PATH` to load model roles from an alternate config file.

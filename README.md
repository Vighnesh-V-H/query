# Query

AI support agent for SpotifyCares — intent classification, grounded replies, auto/escalate decisions. Monorepo:

```text
apps/
├── backend/    Python pipeline (uv workspace member, package "query")
└── frontend/   Next.js chat UI (planned)
```

## Backend setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11+ is managed automatically). From the repo root:

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

The extracted files and their schema/count manifest are cached in `data/raw/`; rerunning the command reports the cache without extracting again. Use `--archive` and `--extract-dir` to override the defaults.

## Model configuration

Model roles (generator / judge / labeler) and the NVIDIA NIM base URL live in `apps/backend/configs/models.yaml`. Each role can be overridden with a `QUERY_<ROLE>_MODEL` env var (e.g. `QUERY_GENERATOR_MODEL`). Set `QUERY_CONFIG_PATH` to load model roles from an alternate config file.

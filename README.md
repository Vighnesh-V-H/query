Query
=====

A customer support agent (AI support agent for SpotifyCares — intent classification, grounded replies, auto/escalate decisions).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11+ is managed automatically).

```sh
uv sync --extra dev
cp .env.example .env   # then set NVIDIA_API_KEY
```

## Smoke test the LLM connection

```sh
uv run query smoke-llm --role generator
```

## Run tests

```sh
uv run pytest
```

## Model configuration

Model roles (generator / judge / labeler) and the NVIDIA NIM base URL live in `configs/models.yaml`. Each role can be overridden with a `QUERY_<ROLE>_MODEL` env var (e.g. `QUERY_GENERATOR_MODEL`). Set `QUERY_CONFIG_PATH` to load model roles from an alternate config file instead of `configs/models.yaml`.

Query
=====

A customer support agent (AI support agent for SpotifyCares — intent classification, grounded replies, auto/escalate decisions).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11+ is managed automatically).

```sh
uv sync --extra dev
cp .env.example .env   # then set OPENROUTER_API_KEY
```

## Smoke test the LLM connection

```sh
uv run query smoke-llm --role labeler
```

## Run tests

```sh
uv run pytest
```

Model roles (generator / judge / labeler) live in `configs/models.yaml` and can be overridden with `QUERY_<ROLE>_MODEL` env vars.

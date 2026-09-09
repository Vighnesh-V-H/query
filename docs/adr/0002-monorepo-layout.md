# Monorepo layout: apps/backend + apps/frontend under a uv workspace

Query is developed as a monorepo: the Python support-agent pipeline lives at `apps/backend` (the `query` package, a uv workspace member), and the Next.js chat UI will live at `apps/frontend`. The workspace root `pyproject.toml` is virtual (`package = false`); one lockfile and one virtualenv serve the whole repo, and backend commands run via `uv run --package query ...` from the root.

We chose this over separate repositories because backend and frontend change together (the frontend is a thin demo surface over the pipeline's interface), and a single workspace keeps one clone, one lockfile, and one review path through the no-mistakes gate.

## Consequences

- Python tests and the CLI run from the repo root with `uv run` — paths inside `apps/backend` are workspace-relative, not repo-relative.
- The frontend needs no Node tooling at the root; its toolchain stays inside `apps/frontend` and is added only when that work starts (ADR to follow at that point).
- New backend subpackages belong under `apps/backend/src/query/`; do not add Python packages at the workspace root.

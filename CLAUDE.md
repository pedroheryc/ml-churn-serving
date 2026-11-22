# CLAUDE.md

Churn baseline (scikit-learn) → MLflow tracking → versioned artifact → FastAPI.

## Commands (run from repo root)

```bash
uv venv && uv pip install -r requirements.lock && uv pip install -e . --no-deps
uv run --no-sync pytest                       # must pass before any commit
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync churn-train                  # writes artifacts/models/ and mlflow.db
uv run --no-sync uvicorn churn_serving.api:app --port 8000
```

After changing `pyproject.toml` dependencies, relock:
`uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 -o requirements.lock`

## Structure

- `src/churn_serving/data.py` — seeded synthetic data; feature lists are the source of truth
- `src/churn_serving/train.py` — pipeline, metrics, MLflow logging, CLI
- `src/churn_serving/model_store.py` — immutable `vNNNN/` dirs + `registry.json` pointer
- `src/churn_serving/schemas.py`, `api.py` — request contract and endpoints
- `tests/` — tests train into tmp dirs; never depend on `artifacts/`
- `docs/decisions.md` — append a decision when making a non-obvious choice

## Conventions

- English for code, docs, issues, PRs. Python 3.11+, ruff, line length 100.
- One logical change per commit; imperative subject (`Add model rollback endpoint`).
  No `Co-Authored-By` trailers.
- Work on `claude/<issue-number>-<slug>` branches; open a PR that links the issue.
  Never push to `main`, never merge.
- If the README example output changes, update it in the same commit and explain why.

## Publication limits

- Only synthetic or public data with a compatible license. No employer code, data,
  names or infrastructure.
- Never commit `artifacts/`, `mlflow.db`, `.venv/`, secrets or tokens.
- Commit author/email come from the repo-local git config set by the owner; do not
  change them. Do not invent activity: every commit must be a useful, tested change.

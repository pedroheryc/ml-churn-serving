# churn-serving

Train a churn baseline with scikit-learn, track it in MLflow, and serve a versioned
model artifact behind a validated FastAPI endpoint. Everything runs on a laptop CPU
in under a minute, offline, from a clean clone.

Part of a five-project MLOps portfolio — see [ROADMAP.md](ROADMAP.md).

## Problem

A subscription business wants a churn score per customer that downstream systems can
call. The interesting part is not the model, it is the contract around it: a
reproducible training run, an artifact you can identify and roll back, and an API that
rejects bad input instead of scoring it.

## Architecture

```
generate_customers(seed)          synthetic data, known ground truth
        │
        ▼
train.py ── Pipeline(scaler + one-hot + LogisticRegression)
   │  ├── MLflow (sqlite:///mlflow.db): params, metrics, dataset fingerprint
   │  └── model_store: artifacts/models/v0001/{model.joblib, metadata.json}
   │                   artifacts/models/registry.json  {"active": "v0001"}
   ▼
api.py (FastAPI) ── loads the active version at startup
   ├── GET  /health   status + served version
   ├── GET  /model    metadata of the served version
   └── POST /predict  pydantic schema (strict: ranges, enums, no extra fields)
```

Design decisions and trade-offs are recorded in [docs/decisions.md](docs/decisions.md).

## Demo (reproducible)

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/pedroheryc/ml-churn-serving.git
cd ml-churn-serving
uv venv
uv pip install -r requirements.lock
uv pip install -e . --no-deps

uv run --no-sync churn-train                 # -> saved model v0001 metrics={...}
uv run --no-sync uvicorn churn_serving.api:app --port 8000
```

In a second terminal:

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" -d @examples/request.json
```

Expected output (seed 42, locked dependencies):

```json
{"model_version":"v0001","threshold":0.5,"predictions":[
  {"churn_probability":0.934748,"churn":true},
  {"churn_probability":0.005052,"churn":false}]}
```

Inspect experiments with `uv run --no-sync mlflow ui --backend-store-uri sqlite:///mlflow.db`
(requires the full `mlflow` package for the UI; tracking itself uses `mlflow-skinny`).

## Results

Held-out 20% split, 5,000 synthetic customers, seed 42:

| metric | value |
|---|---|
| ROC AUC | 0.8455 |
| Average precision | 0.686 |
| Brier score | 0.1421 |
| Test positive rate | 0.292 |

## Commands

| task | command |
|---|---|
| tests | `uv run --no-sync pytest` |
| lint | `uv run --no-sync ruff check . && uv run --no-sync ruff format --check .` |
| train (no MLflow) | `uv run --no-sync churn-train --no-mlflow` |
| serve a pinned version | `CHURN_MODEL_VERSION=v0001 uv run --no-sync uvicorn churn_serving.api:app` |
| relock dependencies | `uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 -o requirements.lock` |

Configuration: `CHURN_MODEL_DIR` (default `artifacts/models`), `CHURN_MODEL_VERSION`
(default: active version), `CHURN_THRESHOLD` (default `0.5`).

## Limitations

- Synthetic data: metrics show the pipeline works, not that it would on real customers.
- The 0.5 threshold is not tuned to any business cost.
- The model loads once at startup; switching versions needs a restart (hot swap and
  rollback are the next milestone).
- No authentication, rate limiting or containerization yet.

## License

MIT.

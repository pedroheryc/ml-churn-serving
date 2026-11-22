"""Train the churn baseline, log it to MLflow and save a versioned artifact."""

from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_serving import model_store
from churn_serving.data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    generate_customers,
)

EXPERIMENT_NAME = "churn-baseline"


def build_pipeline(c: float = 1.0, seed: int = 42) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ]
    )
    model = LogisticRegression(C=c, max_iter=1000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def dataset_fingerprint(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


def train(
    n_rows: int = 5000,
    seed: int = 42,
    c: float = 1.0,
    model_dir: Path | None = None,
    tracking_uri: str | None = "sqlite:///mlflow.db",
) -> dict[str, Any]:
    """Train, evaluate on a held-out split, persist a new version and return its metadata.

    ``tracking_uri=None`` skips MLflow (used by fast unit tests).
    """
    model_dir = model_dir or model_store.store_root()
    df = generate_customers(n_rows=n_rows, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=seed, stratify=df[TARGET]
    )
    pipeline = build_pipeline(c=c, seed=seed).fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "average_precision": round(float(average_precision_score(y_test, proba)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }
    params = {"n_rows": n_rows, "seed": seed, "C": c, "model": "LogisticRegression"}
    metadata: dict[str, Any] = {
        "params": params,
        "metrics": metrics,
        "features": FEATURES,
        "dataset_sha256_16": dataset_fingerprint(df),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "mlflow_run_id": None,
    }

    if tracking_uri is not None:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset_sha256_16", metadata["dataset_sha256_16"])
            metadata["mlflow_run_id"] = run.info.run_id
            version = model_store.save_model(pipeline, metadata, model_dir)
            mlflow.set_tag("model_version", version)
            mlflow.log_artifact(str(model_dir / version / model_store.METADATA_FILE))
    else:
        version = model_store.save_model(pipeline, metadata, model_dir)

    return {"version": version, **metadata}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", dest="c", type=float, default=1.0)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    args = parser.parse_args(argv)
    result = train(
        n_rows=args.rows,
        seed=args.seed,
        c=args.c,
        model_dir=args.model_dir,
        tracking_uri=None if args.no_mlflow else args.tracking_uri,
    )
    print(f"saved model {result['version']} metrics={result['metrics']}")


if __name__ == "__main__":
    main()

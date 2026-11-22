import json

import pytest

from churn_serving import model_store
from churn_serving.data import FEATURES, TARGET, generate_customers
from churn_serving.train import train


def test_generator_is_deterministic_and_has_expected_columns():
    a = generate_customers(500, seed=7)
    b = generate_customers(500, seed=7)
    assert a.equals(b)
    assert list(a.columns) == FEATURES + [TARGET]
    assert 0.1 < a[TARGET].mean() < 0.6  # realistic, non-degenerate class balance


def test_baseline_beats_chance_and_metadata_is_complete(model_dir):
    meta = model_store.load_model(model_dir).metadata
    assert meta["metrics"]["roc_auc"] > 0.75
    for key in ("version", "params", "dataset_sha256_16", "sklearn_version", "features"):
        assert key in meta


def test_each_training_run_creates_a_new_immutable_version(tmp_path):
    first = train(n_rows=800, model_dir=tmp_path, tracking_uri=None)
    second = train(n_rows=800, model_dir=tmp_path, tracking_uri=None)
    assert (first["version"], second["version"]) == ("v0001", "v0002")
    assert model_store.active_version(tmp_path) == "v0002"
    # Same seed and data -> same fingerprint and metrics: training is reproducible.
    assert first["dataset_sha256_16"] == second["dataset_sha256_16"]
    assert first["metrics"] == second["metrics"]
    registry = json.loads((tmp_path / "registry.json").read_text())
    assert registry["versions"] == ["v0001", "v0002"]


def test_set_active_rejects_unknown_version(tmp_path):
    train(n_rows=800, model_dir=tmp_path, tracking_uri=None)
    with pytest.raises(KeyError):
        model_store.set_active("v0099", tmp_path)


def test_training_logs_run_to_mlflow(tmp_path):
    result = train(
        n_rows=800,
        model_dir=tmp_path / "models",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    assert result["mlflow_run_id"]
    meta = model_store.load_model(tmp_path / "models").metadata
    assert meta["mlflow_run_id"] == result["mlflow_run_id"]

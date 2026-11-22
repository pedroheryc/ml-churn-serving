import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn_serving.train import train

EXAMPLE_REQUEST = Path(__file__).resolve().parents[1] / "examples" / "request.json"


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    train(model_dir=root, tracking_uri=None)
    return root


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_DIR", str(model_dir))
    from churn_serving.api import app

    with TestClient(app) as c:  # context manager runs the lifespan (model load)
        yield c


@pytest.fixture(scope="session")
def example_request() -> dict:
    return json.loads(EXAMPLE_REQUEST.read_text(encoding="utf-8"))

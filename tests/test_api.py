import copy

import pytest


def test_health_reports_active_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "v0001"}


def test_predict_example_request(client, example_request):
    response = client.post("/predict", json=example_request)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v0001"
    high_risk, low_risk = (p["churn_probability"] for p in body["predictions"])
    assert 0.0 <= low_risk < high_risk <= 1.0
    # The example rows sit at opposite ends of the known ground truth.
    assert body["predictions"][0]["churn"] is True
    assert body["predictions"][1]["churn"] is False


def test_prediction_is_deterministic(client, example_request):
    first = client.post("/predict", json=example_request).json()
    second = client.post("/predict", json=example_request).json()
    assert first == second


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(contract_type="weekly"),  # unknown category
        lambda c: c.update(tenure_months=-1),  # out of range
        lambda c: c.update(monthly_charges="cheap"),  # wrong type
        lambda c: c.pop("payment_method"),  # missing field
        lambda c: c.update(tenure=5),  # unexpected (typo'd) field
    ],
)
def test_invalid_input_is_rejected_with_422(client, example_request, mutate):
    payload = copy.deepcopy(example_request)
    mutate(payload["customers"][0])
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict", json={"customers": []}).status_code == 422


def test_api_reports_503_without_a_model(tmp_path, monkeypatch, example_request):
    from fastapi.testclient import TestClient

    from churn_serving.api import app

    monkeypatch.setenv("CHURN_MODEL_DIR", str(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "no-model"
        assert c.post("/predict", json=example_request).status_code == 503

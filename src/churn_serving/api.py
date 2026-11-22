"""FastAPI app serving the active model version.

Run with: ``uvicorn churn_serving.api:app``
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn_serving import model_store
from churn_serving.data import FEATURES
from churn_serving.schemas import Prediction, PredictRequest, PredictResponse

DEFAULT_THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = model_store.store_root()
    try:
        app.state.model = model_store.load_model(root, os.environ.get("CHURN_MODEL_VERSION"))
    except FileNotFoundError:
        app.state.model = None  # /health reports not-ready instead of crashing on boot
    app.state.threshold = float(os.environ.get("CHURN_THRESHOLD", DEFAULT_THRESHOLD))
    yield


app = FastAPI(title="churn-serving", version="0.1.0", lifespan=lifespan)


def _model(request: Request) -> model_store.LoadedModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded; run churn-train first")
    return model


@app.get("/health")
def health(request: Request) -> dict:
    model = request.app.state.model
    return {"status": "ok" if model else "no-model", "model_version": model and model.version}


@app.get("/model")
def model_info(request: Request) -> dict:
    return _model(request).metadata


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model = _model(request)
    threshold = request.app.state.threshold
    frame = pd.DataFrame([c.model_dump() for c in body.customers], columns=FEATURES)
    proba = model.pipeline.predict_proba(frame)[:, 1]
    return PredictResponse(
        model_version=model.version,
        threshold=threshold,
        predictions=[
            Prediction(churn_probability=round(float(p), 6), churn=bool(p >= threshold))
            for p in proba
        ],
    )

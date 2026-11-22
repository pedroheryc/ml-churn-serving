"""Request/response contracts for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """One customer. Unknown fields are rejected so typos fail loudly instead of
    silently falling back to defaults."""

    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=600)
    monthly_charges: float = Field(gt=0, le=1000)
    support_tickets_90d: int = Field(ge=0, le=100)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    payment_method: Literal["credit-card", "bank-transfer", "electronic-check", "mailed-check"]
    internet_service: Literal["none", "dsl", "fiber"]
    paperless_billing: bool


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    churn: bool


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]

"""Synthetic, seeded churn dataset.

The data is generated rather than downloaded so that a clean clone trains offline,
with no license questions, and the ground-truth relationship is known (useful for
sanity-checking the model and, later, for simulating drift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_TYPES = ("month-to-month", "one-year", "two-year")
PAYMENT_METHODS = ("credit-card", "bank-transfer", "electronic-check", "mailed-check")
INTERNET_SERVICES = ("none", "dsl", "fiber")

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets_90d"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]
BOOLEAN_FEATURES = ["paperless_billing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
TARGET = "churned"


def generate_customers(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Return ``n_rows`` synthetic customers with a ``churned`` label.

    Churn follows a known logistic model: short tenure, month-to-month contracts,
    electronic-check payments, fiber service, high charges and many support tickets
    increase churn probability.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENT_METHODS, size=n_rows, p=[0.25, 0.25, 0.35, 0.15])
    internet = rng.choice(INTERNET_SERVICES, size=n_rows, p=[0.20, 0.35, 0.45])
    tenure = rng.integers(0, 73, size=n_rows)
    base_charge = np.select(
        [internet == "none", internet == "dsl", internet == "fiber"], [25.0, 55.0, 85.0]
    )
    monthly = np.clip(base_charge + rng.normal(0, 12, size=n_rows), 18.0, 130.0).round(2)
    tickets = rng.poisson(1.2, size=n_rows)
    paperless = rng.random(n_rows) < 0.6

    logit = (
        -1.0
        - 0.045 * tenure
        + 0.018 * (monthly - 65.0)
        + 0.35 * tickets
        + np.select([contract == "month-to-month", contract == "one-year"], [1.3, 0.0], -1.2)
        + np.where(payment == "electronic-check", 0.6, 0.0)
        + np.where(internet == "fiber", 0.4, 0.0)
        + np.where(paperless, 0.2, 0.0)
    )
    churned = rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))

    return pd.DataFrame(
        {
            "tenure_months": tenure.astype(int),
            "monthly_charges": monthly.astype(float),
            "support_tickets_90d": tickets.astype(int),
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "paperless_billing": paperless.astype(bool),
            TARGET: churned.astype(int),
        }
    )

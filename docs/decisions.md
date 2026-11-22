# Design decisions

Short records of choices worth defending in a review or interview.

## D1 — Synthetic, seeded data instead of a public churn dataset

The popular public churn CSVs have unclear licensing and require a download step that
breaks offline reproduction. A generator with a known logistic ground truth gives a
clean license, deterministic data from a seed, and lets tests assert behaviour
("long-tenure two-year customer scores low"). Trade-off: metrics are not evidence of
real-world performance. Later, the same generator can inject drift for the
observability project.

## D2 — Logistic regression baseline in one sklearn `Pipeline`

A baseline must be cheap, explainable and hard to get wrong. Preprocessing lives in the
same pipeline object as the model, so training and serving cannot apply different
transformations (no train/serve skew from duplicated feature code). Stronger models
are only worth adding once they beat this baseline on the same split.

## D3 — Filesystem model store as the serving contract; MLflow for history

MLflow records params, metrics and the dataset fingerprint for every run. The API,
however, loads from `artifacts/models/<version>/` and `registry.json`, so serving does
not depend on a tracking server being up. Versions are immutable directories; "active"
is a pointer updated with an atomic file replace, which makes rollback a pointer move
(next milestone).

## D4 — MLflow on SQLite via `mlflow-skinny`

MLflow 3.x puts the file store in maintenance mode and raises by default. SQLite is the
recommended local backend. `mlflow-skinny` + SQLAlchemy + Alembic keeps the install far
smaller than full `mlflow` (no UI server dependencies) while supporting it.

## D5 — Strict input schema

`extra="forbid"`, enumerated categories and numeric ranges return 422 for bad input.
The one-hot encoder uses `handle_unknown="ignore"`, which would otherwise silently score
an unknown category as "all zeros" — the API rejects it before it reaches the model.

## D6 — Locked dependencies with `uv pip compile --universal`

One lock file resolves for Linux CI and Windows/macOS laptops. The expected prediction
in the README is only guaranteed under the locked versions.

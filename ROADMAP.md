# Portfolio roadmap

Five public projects showing the lifecycle of ML systems: serving, forecasting,
features, observability and LLM evaluation. All run on CPU with public or synthetic data.

| repo | problem | first demo |
|---|---|---|
| [ml-churn-serving](https://github.com/pedroheryc/ml-churn-serving) | serve a versioned churn model behind a validated API | weeks 1–8 |
| [llm-agent-evaluation](https://github.com/pedroheryc/llm-agent-evaluation) | catch regressions in a tool-using agent | weeks 1–8 |
| [ml-demand-forecasting](https://github.com/pedroheryc/ml-demand-forecasting) | leakage-safe backtest: seasonal baseline vs candidate | weeks 9–20 |
| [ml-feature-platform](https://github.com/pedroheryc/ml-feature-platform) | point-in-time correct features with Feast | weeks 9–20 |
| [ml-model-observability](https://github.com/pedroheryc/ml-model-observability) | detect data quality issues, drift and latency regressions | weeks 9–20 |

## Phases

- **Weeks 1–8**: churn serving demo; agent evaluation demo with mocked CI responses.
- **Weeks 9–20**: first demos of forecasting, feature store and observability.
- **Weeks 21–36**: reprocessing, failure recovery, versioning, integration and
  regression tests where they add value (e.g. churn model served from Feast features,
  monitored by the observability replay).
- **Weeks 37–52**: releases, reproducible benchmarks, architecture docs, improvements
  driven by results.

## Acceptance criteria per project

1. **ml-churn-serving** — from a clean clone: train, start the API, reproduce an example
   prediction. Next: switch and roll back model versions.
2. **ml-demand-forecasting** — reproducible backtest on identical windows for both
   models; a test detects future information leaking into features.
3. **ml-feature-platform** — a timestamp test proves each training row only receives
   features available at prediction time; a train/serve consistency check.
4. **ml-model-observability** — a reproducible degraded replay triggers the expected
   alert and produces evidence for the runbook.
5. **llm-agent-evaluation** — a regression introduced in a test case fails the check
   with an understandable diagnosis.

## Status — churn-serving

- [x] Trainable baseline, MLflow tracking, versioned artifact, validated API, tests, CI
- [ ] Version switch and rollback without restart
- [ ] Container image and load test

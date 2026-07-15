# Store Sales Time-Series Forecasting

This project develops a reproducible workflow for forecasting daily product-family sales across 54 grocery stores in Ecuador. It follows the full path from raw-data inspection to a validated 16-day forecast, with particular attention to chronological evaluation, leakage-safe feature construction, and clear reporting.

## Problem and data

Each observation is identified by `date`, `store_nbr`, and `family`; the response is daily `sales`. The supporting data describe promotions, store type and cluster, transactions, holidays, and oil prices. Historical sales run from 2013-01-01 to 2017-08-15, and the forecast horizon covers the following 16 days.

The raw files remain unchanged in `data/raw/`. Generated tables, figures, reports, and forecasts are kept separately so that every analytical result can be traced back to code.

## Approach

The workflow is deliberately incremental:

1. **Audit the inputs.** Check schemas, keys, missing values, duplicates, date coverage, and consistency across files.
2. **Understand the series.** Examine seasonality, store and family scale, promotions, transactions, holidays, earthquake-related changes, and oil-price patterns.
3. **Establish honest evaluation.** Reserve the final 16 observed days as a chronological holdout and compare simple seasonal forecasts before fitting machine-learning models.
4. **Build leakage-safe features.** Combine calendar variables, store metadata, promotions, and lagged or rolling sales summaries. Target-derived validation and forecast features use only observations available before the prediction period.
5. **Compare compact models.** Evaluate Ridge regression and HistGradientBoosting on the same rows and the same preprocessed feature matrix.
6. **Produce the forecast.** Refit the selected specification once on the shifted 365-day training window, verify the output, and preserve the final 16-day predictions.

The shared feature matrix is built once per experiment stage; configurations differ by column selection rather than repeated feature-pipeline execution.

## Validation and findings

Random train/test splitting would let future observations influence an earlier forecasting task, so the project uses 2017-07-31 through 2017-08-15 as a fixed holdout. RMSLE is used because sales vary greatly across store-family series and the metric reduces the dominance of the largest values.

| Method | Validation RMSLE |
|---|---:|
| 8-week weekday mean | 0.520631 |
| Ridge regression | 0.494938 |
| HistGradientBoosting | **0.449788** |

The experiments indicate that recent sales history—especially lag and rolling features—contains the most useful incremental signal. HistGradientBoosting improves on the linear model for 30 of 33 families and 52 of 54 stores, suggesting that nonlinear relationships help across most of the dataset rather than only a few large series. Holiday and oil variables, in their simplified form, did not improve the controlled feature check and were excluded from the final specification.

These results are predictive, not causal. The single holdout window also means that performance may vary in other seasonal or event-driven periods.

## Repository guide

```text
.
|-- data/raw/                 # unchanged source data
|-- notebooks/               # readable analysis from EDA to final forecast
|-- reports/                 # concise findings, generated tables, and figures
|-- src/
|   |-- audit_data.py        # input validation and audit report
|   |-- baselines.py         # chronological split, metric, and seasonal baselines
|   |-- features.py          # shared leakage-safe feature construction
|   |-- models.py            # preprocessing and controlled model comparison
|   `-- final_forecast.py    # one final fit, forecast, and output checks
|-- submissions/             # generated prediction file
|-- requirements.txt
`-- README.md
```

For a quick overview, read [`reports/final_project_summary.md`](reports/final_project_summary.md). For implementation details, follow the source modules in their dependency order: `audit_data.py`, `baselines.py`, `features.py`, `models.py`, then `final_forecast.py`. The notebooks provide a narrative companion to each analytical step.

## Reproduce the workflow

Create a Python environment, install the dependencies, and run the modules from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.audit_data
python -m src.baselines
python -m src.features
python -m src.models
python -m src.final_forecast
```

The last command writes `submissions/store_sales_hgb_submission.csv` and a set of integrity checks to `reports/tables/final_submission_checks.csv`. The generated file contains 28,512 finite, nonnegative predictions in the original test-ID order.

## Main limitations and next steps

- Evaluation currently uses one 16-day chronological window; several carefully chosen backtesting windows would provide stronger evidence of temporal stability.
- Holiday features do not yet model locale, transfers, or event interactions in full detail.
- Intermittent and spike-prone product families remain difficult and may benefit from specialized demand models or additional error analysis.
- The model comparison intentionally uses fixed, lightweight specifications; future work should expand it cautiously while preserving the same leakage controls.

Detailed findings are available in [`reports/data_audit.md`](reports/data_audit.md), [`reports/eda_summary.md`](reports/eda_summary.md), [`reports/baseline_results.md`](reports/baseline_results.md), [`reports/feature_engineering.md`](reports/feature_engineering.md), and [`reports/model_comparison.md`](reports/model_comparison.md).

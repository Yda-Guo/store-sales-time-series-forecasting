# Store Sales Time-Series Forecasting

This repository is a reproducible, leakage-tested solution for Kaggle's 16-day Ecuador grocery-sales task: 54 stores x 33 product families x 16 forecast horizons.

## Forecast design

The implementation uses pooled direct forecasting. One supervised row represents `forecast_origin x store x family x horizon`; a single model learns all horizons with `horizon` as an explicit feature. Every target-history value is frozen at the origin. Training, development, final holdout, and Kaggle inference all call the same feature constructor.

The evaluation contract is fixed:

- Development origins: 2017-03-31, 2017-05-15, and 2017-06-30.
- Final holdout origin: 2017-07-30, covering 2017-07-31 through 2017-08-15.
- Metric: RMSLE.
- Models: eight-week weekday seasonal baseline, Ridge, and HistGradientBoosting.
- HGB iteration count: selected only by mean development-fold RMSLE; random row-level early stopping is disabled.

The prior single-holdout HGB score of `0.449788` is retained only as historical context. It is not directly comparable because the former pipeline changed lag semantics between fitting and prediction and reused the same window for development.

## Results

HGB achieved mean development RMSLE **0.447500**, beating Ridge and the seasonal baseline in all three development folds. On the untouched final holdout, RMSLE was **0.433648** for HGB, **0.520631** for the seasonal baseline, and **0.635423** for Ridge. HGB beat Ridge at all 16 horizons and all 33 families on paired mean squared-log loss. Its horizon RMSLE ranged from 0.388 to 0.515; the weak positive horizon/error correlation (0.226) does not support a claim of monotonic degradation.

## Repository map

```text
src/audit_data.py       raw-file contract checks
src/eda.py              compact reproducible EDA
src/baselines.py        RMSLE and honest seasonal reference
src/features.py         sole forecast-origin feature implementation
src/backtesting.py      rolling-origin comparison and final holdout
src/models.py           compatibility entry point for backtesting
src/final_forecast.py   frozen fit and Kaggle CSV generation
tests/                  synthetic temporal-integrity tests
notebooks/              concise narrative companions
reports/                generated findings, tables, and figures
```

## Reproduce

The tracked competition archive supplies `train.csv`; the raw CSV is intentionally ignored. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.audit_data
python -m src.eda
python -m src.baselines
python -m unittest discover -s tests -v
python -m src.features
python -m src.backtesting
python -m src.final_forecast
```

The last command creates `submissions/store_sales_hgb_submission.csv` locally. The generated submission is intentionally ignored by Git; its structural checks are tracked in `reports/tables/final_submission_checks.csv`.

The workflow was tested on Python 3.13 with pandas 2.3, NumPy 2.2, scikit-learn 1.7, Matplotlib 3.10, and nbformat 5.x. Compatible minor-version ranges are recorded in `requirements.txt`.

Start with [the final summary](reports/final_project_summary.md), then inspect [the model comparison](reports/model_comparison.md) and [feature design](reports/feature_engineering.md). Results are predictive, not causal, and a compact set of historical origins cannot represent every future retail regime.


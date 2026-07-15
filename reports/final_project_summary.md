# Final Project Summary

## 1. Project objective

Forecast daily sales for 54 Ecuadorian grocery stores and 33 product families while demonstrating a clear, leakage-safe, reproducible undergraduate forecasting workflow.

## 2. Dataset and forecasting task

The training target covers 2013-01-01 through 2017-08-15. The Kaggle test set covers 2017-08-16 through 2017-08-31 (16 days and 28512 rows).

## 3. Data audit

Stage 1 verified keys, dimensions, dates, missing oil values, holiday duplication rules, and transaction coverage without changing raw data.

## 4. Main exploratory findings

Stage 2 found weekly structure, changing sales levels, strong family/store scale differences, promotion associations, and special-event periods. These are descriptive rather than causal findings.

## 5. Validation design

Stages 3-5 use one strict chronological 16-day holdout (2017-07-31 through 2017-08-15) with 28,512 rows. Random splitting is avoided because it would leak later observations into an unrealistic forecasting task.

## 6. Baseline results

The best simple weekday mean baseline achieved RMSLE **0.520631**.

## 7. Feature engineering

The final moderate set contains calendar, store type/cluster, promotion, lag 7/14/28, rolling mean 7/28, rolling standard deviation 28, and existing log-history transformations. Validation/test target-derived inputs use pre-period sales only.

## 8. Model comparison

Ridge achieved **0.494938** RMSLE and HistGradientBoosting achieved **0.449788**. The nonlinear gain was broad across families and stores while remaining computationally modest.

## 9. Final model

The exact Stage 5 HistGradientBoosting specification is fitted once on 648,648 rows from 2016-08-16 through 2017-08-15, using a `log1p` target and 29 encoded inputs.

## 10. Final submission generation

The submission contains 28512 rows with exact test/sample ID order. Predictions are fractional, nonnegative, finite, and were not manually rescaled. Kaggle upload remains manual.

![Test-period predictions](figures/final/01_test_daily_predictions.png)

## 11. Main lessons

- Chronological validation is essential.
- Explicit leakage tests make multi-step target-history features defensible.
- Simple seasonal baselines provide meaningful reference points.
- Moderate lag/rolling features add substantial predictive information.
- A lightweight nonlinear model can improve broadly without an extensive search.

## 12. Limitations

Validation uses one 16-day window, holiday handling remains simplified and excluded from the final model, intermittent families remain difficult, and no Kaggle leaderboard score is recorded. Predictive associations are not causal evidence.

## 13. Possible future extensions

Test a small number of additional chronological windows, improve locale-aware holiday handling, study intermittent-demand families, record the manual Kaggle score, and carefully assess one or two further fixed specifications.

## 14. Reproduction instructions

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.audit_data
python -m src.baselines
python -m src.features
python -m src.models
python -m src.final_forecast
```

The manual-upload file is `submissions/store_sales_hgb_submission.csv`.

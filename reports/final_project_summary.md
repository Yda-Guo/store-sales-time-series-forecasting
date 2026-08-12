# Final Project Summary

The final workflow uses one pooled direct model for all 54 stores, 33 product families, and 16 forecast horizons. Every target-history feature is frozen at its forecast origin. Three development origins selected 120 boosting iterations; the separate 2017-07-30 origin was evaluated exactly once and achieved RMSLE **0.433648**.

The generated submission has 28,512 rows in the sample submission's original ID order. It is intentionally ignored by Git because it is a reproducible generated artifact, not a source file.

![Test-period predictions](figures/final/test_daily_predictions.png)

## Reproduce

Run `python -m src.audit_data`, `python -m src.eda`, `python -m unittest discover -s tests -v`, `python -m src.backtesting`, and `python -m src.final_forecast` from the repository root.

## Limits

This is a predictive benchmark, not a causal analysis. The compact rolling-origin design cannot cover every retail regime, and abrupt events remain difficult. Kaggle test labels are unavailable locally, so the final CSV is structurally validated but not assigned a local test score.


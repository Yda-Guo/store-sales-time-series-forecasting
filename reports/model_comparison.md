# Lightweight Model Comparison

## 1. Purpose

Stage 5 compares two fitted, interpretable-to-explain models against the established weekday baseline. It does not tune extensively, ensemble models, or forecast the Kaggle test set.

## 2. Validation and feature setup

The unchanged validation period is **2017-07-31 through 2017-08-15** (16 days, 28,512 rows). The shared Stage 4 matrix uses the final 365 pre-validation days and 29 encoded features: calendar, store, promotion, lag, rolling, and existing log-history columns.

## 3. Models compared

- Weekday mean baseline: reused Stage 3 score.
- Ridge: Stage 4 preprocessing and alpha 1.0.
- HistGradientBoosting: one fixed, moderate nonlinear specification with early stopping.

## 4. Computational controls

Features are built once, preprocessing is fitted once, and both fitted models reuse the same encoded rows. No search, cross-validation, random forest, or repeat fitting is used. Training times are approximate wall-clock measurements, not formal benchmarks.

## 5. Overall results

| Rank | Model | RMSLE | vs weekday | vs Ridge | Fit time |
|---:|---|---:|---:|---:|---:|
| 1 | HistGradientBoosting | 0.449788 | -0.070843 | -0.045150 | 3.53 s |
| 2 | Ridge regression | 0.494938 | -0.025693 | +0.000000 | 0.10 s |
| 3 | Weekday mean baseline | 0.520631 | +0.000000 | +0.025693 | 0.00 s |

![Model ranking](figures/models/01_model_ranking.png)

HistGradientBoosting beats Ridge by **0.045150 RMSLE**. Both fitted models beat the weekday baseline.

## 6. Comparison with the Stage 3 baseline

The selected model improves on `0.520631` by **0.070843**. Ridge scores 0.494938, compared with its Stage 4 check of `0.494938`; small timing or floating-point differences may occur, but the implementation is equivalent.

## 7. Error patterns by family, store, and date

HistGradientBoosting has lower family RMSLE than Ridge for **30 of 33 families** and lower store RMSLE for **52 of 54 stores**. Difficult families under the selected model include SCHOOL AND OFFICE SUPPLIES, GROCERY II, LINGERIE; difficult stores include 50, 47, 44. Gains are therefore broad.

![Family errors](figures/models/02_family_errors.png)

![Validation-date errors](figures/models/03_date_errors.png)

## 8. Representative forecasts

Low-, medium-, and high-volume series are selected mechanically using the Stage 3 rule. The plots show that both models can miss intermittent changes and spikes even when aggregate RMSLE improves.

![Representative forecasts](figures/models/04_representative_forecasts.png)

## 9. Model interpretation

Stage 4 ablations show that lag and rolling history provide the decisive information. The nonlinear model can represent thresholds and curved relationships that Ridge cannot, but predictive gains do not establish causal effects.

## 10. Preferred model for Stage 6

**HistGradientBoosting** is selected with RMSLE **0.449788** because it has the lowest validation error, remains computationally modest, and uses the established shared pipeline. No ensemble is needed.

## 11. Limitations

This comparison uses one 16-day holdout, one fixed setting per model, conservative horizon-safe target-history features, and approximate timing. Validation leadership does not guarantee Kaggle leaderboard leadership.

## 12. Conclusion

The controlled comparison identifies HistGradientBoosting as the Stage 6 candidate while preserving the simple baseline and Ridge as transparent references.

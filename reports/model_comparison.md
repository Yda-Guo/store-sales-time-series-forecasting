# Horizon-Aware Model Comparison

## Protocol

Each supervised row is `forecast_origin × store × family × horizon`. Horizons 1–16 are pooled in one model with an explicit horizon feature. Target-history variables are fixed at the forecast origin, so no future true sale inside a 16-day block can enter another prediction.

The transparent reference forecast is the mean sale for the matching weekday over the eight weeks ending at each forecast origin. Ridge provides a regularized linear comparison using the same information set; HistGradientBoosting tests whether modest nonlinearity improves forecast accuracy.

Development origins were fixed before comparison: 2017-03-31, 2017-05-15, 2017-06-30. Each fold uses 16 earlier origins on a fixed 14-day grid, skipping incomplete calendar blocks. The final 2017-07-30 origin was untouched until the iteration count and feature set were frozen.

## Development backtests

| model | mean | median | min | max |
| --- | --- | --- | --- | --- |
| HistGradientBoosting (120 iter) | 0.447501 | 0.421821 | 0.421393 | 0.499288 |
| HistGradientBoosting (80 iter) | 0.449594 | 0.425129 | 0.423265 | 0.500388 |
| Ridge | 0.645909 | 0.623045 | 0.603951 | 0.710732 |
| Seasonal baseline | 0.501865 | 0.511431 | 0.427461 | 0.566702 |

The predefined HGB choice was 120 iterations, selected by mean development RMSLE with random internal early stopping disabled. HGB beat Ridge in 3 of 3 development folds.

## One-time final holdout

| model | RMSLE |
| --- | --- |
| Seasonal baseline | 0.520631 |
| Ridge | 0.635423 |
| HistGradientBoosting (120 iter) | 0.433648 |

## Horizon and family behavior

![Final holdout error by horizon](figures/models/horizon_errors.png)

HGB final-holdout RMSLE ranges from 0.388 to 0.515 across horizons. It has lower paired squared-log loss than Ridge for 16 of 16 horizons and 33 of 33 families. The family table relates this gain to historical zero-sales frequency without making a causal claim.

## Interpretation and limitations

The comparison asks whether modest nonlinearity improves on transparent alternatives under identical forecast-origin information. Results are predictive, not causal. Three development origins and one final origin cannot represent every future regime; intermittent families and event-driven spikes remain difficult.

## Final forecast artifact

After the specification is frozen, `python -m src.final_forecast` fits HGB on 24 legitimate historical origins and generates 28,512 nonnegative Kaggle-test predictions in the sample submission's original ID order. The submission CSV is a reproducible local artifact and is intentionally ignored by Git; `reports/tables/final_submission_checks.csv` records its structural validation. No Kaggle leaderboard score is claimed.


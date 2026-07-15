# Leakage-Safe Feature Engineering

## 1. Purpose

Stage 4 creates a moderate feature set and uses one fixed Ridge regression as an information check. It does not perform the Stage 5 model comparison or generate Kaggle predictions.

## 2. Validation design

The unchanged holdout runs from **2017-07-31** through **2017-08-15**, containing **16 days** and **28,512 rows**. The shared training matrix uses the final 365 pre-validation days.

## 3. Leakage prevention

Target-derived validation features use sales dated before validation only. Horizon-safe lag lookups step back by complete weeks whenever a normal lag would enter validation; rolling summaries are fixed from pre-validation history. The explicit perturbation check passed: replacing all validation targets did not change any lag or rolling input.

## 4. Feature groups

The shared matrix contains 28 raw input columns: calendar, store type/cluster, promotion, lags, rolling summaries, log-transformed target-history features, simplified national holiday/transfer flags, and oil price. The best preprocessing configuration produces 29 encoded columns.

## 5. Missing-value handling

Target-derived gaps fall back to a pre-validation store-family mean, then family mean, global mean, and zero. Store categories use `Unknown`; oil is filled forward and backward within the ordered oil series. Final validation model inputs contain **0 missing values**.

## 6. Lightweight Ridge check

All four checks reuse one shared feature build. They use a 365-day training window, `log1p(sales)`, Ridge alpha 1.0, `expm1` predictions, and zero clipping. Only selected columns differ.

## 7. Results

| Rank | Configuration | Encoded features | RMSLE | Difference from 0.520631 |
|---:|---|---:|---:|---:|
| 1 | Add lag and rolling | 29 | 0.494938 | -0.025693 |
| 2 | Add holiday and oil | 32 | 0.495017 | -0.025614 |
| 3 | Add promotion | 17 | 1.545483 | +1.024852 |
| 4 | Calendar and store | 14 | 2.481887 | +1.961256 |

![Feature-check RMSLE](figures/features/01_feature_check_scores.png)

Promotion features materially improve this linear check relative to calendar/store inputs alone. Adding lag and rolling history produces the decisive improvement. Adding the simplified holiday and oil features slightly worsens RMSLE, so no benefit is established here.

## 8. Comparison with the Stage 3 baseline

The best Ridge check is **Add lag and rolling** with RMSLE **0.494938**, improving on `0.520631` by **0.025693**. This suggests the feature set contains useful information, but Ridge remains only a feature check.

## 9. Features retained for Stage 5

Retain calendar, store type and cluster, promotion, lag 7/14/28, rolling mean 7/28, rolling standard deviation 28, and the corresponding log transforms.

## 10. Features excluded or deferred

Defer oil and simplified holiday flags because the full configuration scored slightly worse. City/state expansion, local holiday matching, interactions, and additional lags are also deferred to preserve scope.

## 11. Limitations

This is one 16-day holdout and one fixed linear model. Horizon-safe lag backtracking is deliberately conservative, simplified holidays omit local/regional application, and coefficient interpretation is not used to claim causality. Results do not guarantee Kaggle performance.

## 12. Conclusion

The leakage-safe lag and rolling group improves the Ridge check beyond the Stage 3 benchmark. A moderate 29-column encoded set is recommended for Stage 5, while holiday and oil features are deferred.

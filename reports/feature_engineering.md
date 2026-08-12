# Forecast-Origin Feature Design

`src/features.py` is the only feature implementation. It is called identically for historical training origins, development folds, the final holdout, and the Kaggle test period.

Each row identifies a target date, store, family, and horizon from 1 through 16. Known-at-target inputs include calendar fields and `onpromotion`. Target-history inputs are all explicitly origin-relative: sales at the origin; lags 7, 14, and 28 days before the origin; 7- and 28-day means; 28-day standard deviation, median, and zero fraction; and an eight-week matching-weekday mean. Store type and cluster are joined by store number.

No feature may read a true sale after the row's forecast origin. The automated suite verifies future-target perturbation invariance, store-family isolation, exact rolling-window boundaries, and horizon-1/horizon-16 semantics. The runtime feature check repeats the perturbation test on the real panel before model comparison.

Known-future promotions are legitimate inputs because Kaggle supplies them for every test row. Transactions, oil, and holidays are excluded from the final compact specification: their future availability and preprocessing require additional assumptions that are unnecessary for the benchmark being tested here.


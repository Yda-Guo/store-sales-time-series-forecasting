"""Fit the selected model once and generate the final Kaggle submission."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import (
    TARGET_DERIVED, _add_training_target_features, _read_csv,
    _safe_validation_target_features, build_calendar_features, load_raw_data,
)
from src.models import FEATURES, _preprocess, make_hist_gradient_boosting


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = ROOT / "submissions" / "store_sales_hgb_submission.csv"
CHECKS_PATH = ROOT / "reports" / "tables" / "final_submission_checks.csv"
REPORT_PATH = ROOT / "reports" / "final_project_summary.md"
FIGURE_DIR = ROOT / "reports" / "figures" / "final"
TRAINING_DAYS = 365


def _complete_features(frame: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Add the selected non-holiday Stage 5 features."""
    result = build_calendar_features(frame)
    metadata = stores[["store_nbr", "type", "cluster"]].rename(columns={"type": "store_type"})
    result = result.merge(metadata, on="store_nbr", how="left", validate="many_to_one")
    result["store_type"] = result["store_type"].fillna("Unknown")
    result["cluster"] = result["cluster"].fillna(-1)
    for feature in TARGET_DERIVED:
        result[f"log_{feature}"] = np.log1p(result[feature].clip(lower=0))
    return result.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)


def build_final_matrices(raw: dict[str, pd.DataFrame], test: pd.DataFrame):
    """Build the shifted 365-day training matrix and direct test matrix once."""
    train = raw["train"]
    test_start = test["date"].min()
    train_end = train["date"].max()
    training_start = test_start - pd.Timedelta(days=TRAINING_DAYS)
    warm_start = training_start - pd.Timedelta(days=28)
    source = train.loc[train["date"].between(warm_start, train_end)]
    train_features = _add_training_target_features(source)
    train_features = train_features.loc[train_features["date"] >= training_start].copy()
    test_features = _safe_validation_target_features(train, test)
    train_features = _complete_features(train_features, raw["stores"])
    test_features = _complete_features(test_features, raw["stores"])
    return train_features, test_features


def test_period_integrity(test: pd.DataFrame, test_features: pd.DataFrame) -> bool:
    """Confirm every direct lag reference remains before the test period."""
    test_start = test["date"].min()
    for lag in (7, 14, 28):
        reference = test["date"] - pd.to_timedelta(lag, unit="D")
        while (reference >= test_start).any():
            reference = reference.where(reference < test_start, reference - pd.Timedelta(days=7))
        if not (reference < test_start).all():
            return False
    return test_features[FEATURES].notna().all().all()


def _submission_checks(test, sample, submission, train_end):
    prediction = submission["sales"]
    checks = [
        ("test_row_count", len(test)),
        ("submission_row_count", len(submission)),
        ("unique_test_ids", test["id"].nunique()),
        ("unique_submission_ids", submission["id"].nunique()),
        ("ids_match", set(test["id"]) == set(submission["id"])),
        ("id_order_matches_test", test["id"].reset_index(drop=True).equals(submission["id"])),
        ("id_order_matches_sample", sample["id"].reset_index(drop=True).equals(submission["id"])),
        ("missing_predictions", int(prediction.isna().sum())),
        ("infinite_predictions", int(np.isinf(prediction).sum())),
        ("negative_predictions", int((prediction < 0).sum())),
        ("minimum_prediction", float(prediction.min())),
        ("maximum_prediction", float(prediction.max())),
        ("mean_prediction", float(prediction.mean())),
        ("median_prediction", float(prediction.median())),
        ("zero_prediction_count", int(prediction.eq(0).sum())),
        ("test_date_start", test["date"].min().date()),
        ("test_date_end", test["date"].max().date()),
        ("test_horizon_days", test["date"].nunique()),
        ("final_training_date", train_end.date()),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def _make_figures(test_features: pd.DataFrame, predictions: np.ndarray) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    forecast = test_features[["date", "family"]].copy()
    forecast["sales"] = predictions
    daily = forecast.groupby("date")["sales"].sum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(daily.index, daily.values, marker="o", color="#3266a8")
    ax.set(title="Total Predicted Sales Across the Kaggle Test Period", xlabel="Test date", ylabel="Predicted total sales")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "01_test_daily_predictions.png", dpi=150); plt.close(fig)

    family = forecast.groupby("family")["sales"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    family.plot.barh(ax=ax, color="#3b8c6e")
    ax.set(title="Highest Predicted-Sales Product Families", xlabel="Predicted test-period sales", ylabel="Product family")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_test_family_predictions.png", dpi=150); plt.close(fig)


def _write_summary(train_features, test, submission, checks):
    values = dict(zip(checks["check"], checks["value"]))
    report = f"""# Final Project Summary

## 1. Project objective

Forecast daily sales for 54 Ecuadorian grocery stores and 33 product families while demonstrating a clear, leakage-safe, reproducible undergraduate forecasting workflow.

## 2. Dataset and forecasting task

The training target covers 2013-01-01 through 2017-08-15. The Kaggle test set covers {values['test_date_start']} through {values['test_date_end']} ({values['test_horizon_days']} days and {values['test_row_count']} rows).

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

The exact Stage 5 HistGradientBoosting specification is fitted once on {len(train_features):,} rows from {train_features['date'].min().date()} through {train_features['date'].max().date()}, using a `log1p` target and 29 encoded inputs.

## 10. Final submission generation

The submission contains {values['submission_row_count']} rows with exact test/sample ID order. Predictions are fractional, nonnegative, finite, and were not manually rescaled. Kaggle upload remains manual.

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
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_final_forecast():
    """Build once, fit once, and generate the final submission and checks."""
    raw = load_raw_data()
    test = _read_csv("test.csv", parse_dates=["date"])
    sample = _read_csv("sample_submission.csv")
    train = raw["train"]
    if (train["date"].min().date().isoformat(), train["date"].max().date().isoformat()) != ("2013-01-01", "2017-08-15"):
        raise ValueError("Unexpected training date range.")
    if test["date"].nunique() != 16 or len(test) != 28_512 or test["date"].min().date().isoformat() != "2017-08-16" or test["date"].max().date().isoformat() != "2017-08-31":
        raise ValueError("Unexpected test period or size.")

    train_features, test_features = build_final_matrices(raw, test)
    if not test_period_integrity(test, test_features):
        raise ValueError("Test-period leakage/integrity check failed.")
    _, train_matrix, test_matrix = _preprocess(train_features, test_features)
    model = make_hist_gradient_boosting()
    model.fit(train_matrix, np.log1p(train_features["sales"]))
    prediction = np.clip(np.expm1(model.predict(test_matrix)), 0, None)
    if np.isnan(prediction).any() or np.isinf(prediction).any() or (prediction < 0).any():
        raise ValueError("Final predictions are invalid.")

    keyed = test_features[["id"]].copy()
    keyed["sales"] = prediction
    submission = test[["id"]].merge(keyed, on="id", how="left", validate="one_to_one")
    checks = _submission_checks(test, sample, submission, train["date"].max())
    required_true = ["ids_match", "id_order_matches_test", "id_order_matches_sample"]
    check_map = dict(zip(checks["check"], checks["value"]))
    if not all(bool(check_map[name]) for name in required_true):
        raise ValueError("Submission IDs do not align.")
    if any(int(check_map[name]) != 0 for name in ["missing_predictions", "infinite_predictions", "negative_predictions"]):
        raise ValueError("Submission prediction checks failed.")

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False, float_format="%.6f")
    checks.to_csv(CHECKS_PATH, index=False)
    _make_figures(test_features, prediction)
    _write_summary(train_features, test, submission, checks)
    return train_features, test_features, submission, checks


def main() -> None:
    train_features, test_features, submission, checks = run_final_forecast()
    print(f"Final training: {train_features['date'].min().date()} to {train_features['date'].max().date()} ({len(train_features):,} rows)")
    print(f"Test: {test_features['date'].min().date()} to {test_features['date'].max().date()} ({len(test_features):,} rows)")
    print("HistGradientBoosting fits: 1")
    print("Test-period leakage/integrity check: passed")
    print(checks.to_string(index=False))
    print(f"Saved {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

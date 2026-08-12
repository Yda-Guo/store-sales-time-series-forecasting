"""Fit the frozen pooled-direct model and create a Kaggle-ready submission."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtesting import BACKTEST_PATH, HGB_ITERATIONS, _fit_hgb
from src.features import build_origin_features, build_supervised_origins, load_raw_data, sample_training_origins


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = ROOT / "submissions" / "store_sales_hgb_submission.csv"
CHECKS_PATH = ROOT / "reports" / "tables" / "final_submission_checks.csv"
FIGURE_DIR = ROOT / "reports" / "figures" / "final"
REPORT_PATH = ROOT / "reports" / "final_project_summary.md"
TEST_ORIGIN = pd.Timestamp("2017-08-15")


def selected_iterations() -> int:
    scores = pd.read_csv(BACKTEST_PATH)
    hgb = scores[scores["model"].str.startswith("HistGradientBoosting")]
    name = hgb.groupby("model")["rmsle"].mean().idxmin()
    value = int(name.split("(")[1].split()[0])
    if value not in HGB_ITERATIONS:
        raise ValueError("Backtest selected an unrecognized HGB configuration")
    return value


def main() -> None:
    raw = load_raw_data()
    iterations = selected_iterations()
    origins = sample_training_origins(TEST_ORIGIN, count=24, train=raw["train"])
    supervised = build_supervised_origins(raw["train"], raw["stores"], origins)
    test_features = build_origin_features(raw["train"], raw["test"], raw["stores"], TEST_ORIGIN)
    predictions = _fit_hgb(supervised, test_features, iterations)

    predicted = test_features[["id", "date", "store_nbr", "family"]].copy()
    predicted["sales"] = predictions
    submission = raw["sample"][["id"]].merge(predicted[["id", "sales"]], on="id", how="left", validate="one_to_one")
    if submission["sales"].isna().any() or (submission["sales"] < 0).any():
        raise ValueError("Submission contains missing or negative predictions")
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False, float_format="%.6f")

    checks = pd.DataFrame([
        {"check": "row_count_matches_sample", "passed": len(submission) == len(raw["sample"]), "value": len(submission)},
        {"check": "id_order_matches_sample", "passed": submission["id"].equals(raw["sample"]["id"]), "value": submission["id"].nunique()},
        {"check": "predictions_are_finite_and_nonnegative", "passed": submission["sales"].notna().all() and (submission["sales"] >= 0).all(), "value": float(submission["sales"].max())},
        {"check": "covers_16_test_days", "passed": predicted["date"].nunique() == 16, "value": predicted["date"].nunique()},
    ])
    CHECKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(CHECKS_PATH, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    daily = predicted.groupby("date")["sales"].sum()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    daily.plot(ax=ax, marker="o")
    ax.set(title="Forecast Sales Across the Kaggle Test Window", xlabel="Date", ylabel="Predicted sales")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "test_daily_predictions.png", dpi=150); plt.close(fig)

    final_scores = pd.read_csv(ROOT / "reports" / "tables" / "final_holdout_scores.csv")
    best = final_scores.loc[final_scores["model"].str.startswith("HistGradientBoosting")].iloc[0]
    REPORT_PATH.write_text(f"""# Final Project Summary

The final workflow uses one pooled direct model for all 54 stores, 33 product families, and 16 forecast horizons. Every target-history feature is frozen at its forecast origin. Three development origins selected {iterations} boosting iterations; the separate 2017-07-30 origin was evaluated exactly once and achieved RMSLE **{best['rmsle']:.6f}**.

The generated submission has {len(submission):,} rows in the sample submission's original ID order. It is intentionally ignored by Git because it is a reproducible generated artifact, not a source file.

![Test-period predictions](figures/final/test_daily_predictions.png)

## Reproduce

Run `python -m src.audit_data`, `python -m src.eda`, `python -m unittest discover -s tests -v`, `python -m src.backtesting`, and `python -m src.final_forecast` from the repository root.

## Limits

This is a predictive benchmark, not a causal analysis. The compact rolling-origin design cannot cover every retail regime, and abrupt events remain difficult. Kaggle test labels are unavailable locally, so the final CSV is structurally validated but not assigned a local test score.
""", encoding="utf-8")
    print(checks.to_string(index=False))
    print(f"Submission: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()


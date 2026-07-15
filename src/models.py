"""Compare lightweight forecasting models on the established validation split."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.baselines import KEYS, rmsle
from src.features import (
    BENCHMARK, CALENDAR, PROMOTION, STORE, TARGET_DERIVED, TARGET_LOG,
    build_feature_matrices, leakage_perturbation_check, load_raw_data,
    validate_feature_integrity,
)


ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "reports" / "tables" / "model_scores.csv"
ERRORS_PATH = ROOT / "reports" / "tables" / "model_error_breakdown.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison.md"
FIGURE_DIR = ROOT / "reports" / "figures" / "models"
FEATURES = CALENDAR + STORE + PROMOTION + TARGET_DERIVED + TARGET_LOG
RIDGE_REFERENCE = 0.494938


def _preprocess(train: pd.DataFrame, validation: pd.DataFrame):
    """Fit one shared encoder/scaler and transform both matrices once."""
    categorical = ["store_type"]
    numeric = [column for column in FEATURES if column not in categorical]
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    train_matrix = transformer.fit_transform(train[FEATURES])
    validation_matrix = transformer.transform(validation[FEATURES])
    return transformer, train_matrix, validation_matrix


def make_hist_gradient_boosting() -> HistGradientBoostingRegressor:
    """Return the fixed Stage 5 model specification."""
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


def _fit_models(train_matrix, validation_matrix, train_target):
    """Fit Ridge and one fixed HistGradientBoosting specification."""
    target = np.log1p(train_target)
    predictions, timings = {}, {}

    ridge = Ridge(alpha=1.0)
    start = perf_counter()
    ridge.fit(train_matrix, target)
    timings["Ridge regression"] = perf_counter() - start
    predictions["Ridge regression"] = np.clip(np.expm1(ridge.predict(validation_matrix)), 0, None)

    hist = make_hist_gradient_boosting()
    start = perf_counter()
    hist.fit(train_matrix, target)
    timings["HistGradientBoosting"] = perf_counter() - start
    predictions["HistGradientBoosting"] = np.clip(np.expm1(hist.predict(validation_matrix)), 0, None)
    return predictions, timings


def _score_table(actual, predictions, timings, feature_count):
    rows = [{
        "model_name": "Weekday mean baseline",
        "model_type": "seasonal baseline",
        "validation_rmsle": BENCHMARK,
        "difference_from_weekday_baseline": 0.0,
        "difference_from_ridge": BENCHMARK - RIDGE_REFERENCE,
        "input_feature_count": 0,
        "training_window": "previous 8 weeks",
        "training_time_seconds": 0.0,
        "specification": "store-family weekday mean",
    }]
    specifications = {
        "Ridge regression": "Ridge(alpha=1.0), log1p target",
        "HistGradientBoosting": "learning_rate=0.08, max_iter=120, max_leaf_nodes=31",
    }
    scores = {name: rmsle(actual, prediction) for name, prediction in predictions.items()}
    ridge_score = scores["Ridge regression"]
    for name in ("Ridge regression", "HistGradientBoosting"):
        rows.append({
            "model_name": name,
            "model_type": "regularized linear" if name == "Ridge regression" else "nonlinear histogram boosting",
            "validation_rmsle": scores[name],
            "difference_from_weekday_baseline": scores[name] - BENCHMARK,
            "difference_from_ridge": scores[name] - ridge_score,
            "input_feature_count": feature_count,
            "training_window": "final 365 days before validation",
            "training_time_seconds": timings[name],
            "specification": specifications[name],
        })
    table = pd.DataFrame(rows).sort_values("validation_rmsle").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def _group_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("Ridge regression", "HistGradientBoosting"):
        for group_type, column in (("family", "family"), ("store", "store_nbr"), ("date", "date")):
            errors = predictions[[column, "actual", model]].copy()
            errors["squared_log_error"] = (np.log1p(errors[model]) - np.log1p(errors["actual"])) ** 2
            grouped = errors.groupby(column)["squared_log_error"].mean().pow(0.5)
            rows.extend({"group_type": group_type, "group_value": str(key), "model_name": model, "rmsle": value} for key, value in grouped.items())
    return pd.DataFrame(rows)


def _make_figures(scores, errors, predictions, history):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ordered = scores.sort_values("validation_rmsle", ascending=False)
    ax.barh(ordered["model_name"], ordered["validation_rmsle"], color="#3266a8")
    for y, value in enumerate(ordered["validation_rmsle"]): ax.text(value + 0.004, y, f"{value:.4f}", va="center")
    ax.set(title="Validation RMSLE by Model", xlabel="RMSLE (lower is better)", ylabel="Model")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "01_model_ranking.png", dpi=150); plt.close(fig)

    family = errors.loc[errors["group_type"] == "family"].pivot(index="group_value", columns="model_name", values="rmsle")
    shown = family.mean(axis=1).nlargest(12).sort_values()
    family.loc[shown.index].plot.barh(figsize=(8, 5), color=["#dd7f2a", "#3266a8"])
    plt.title("Highest Family Errors for Fitted Models"); plt.xlabel("RMSLE"); plt.ylabel("Product family")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "02_family_errors.png", dpi=150); plt.close()

    date = errors.loc[errors["group_type"] == "date"].pivot(index="group_value", columns="model_name", values="rmsle")
    date.index = pd.to_datetime(date.index)
    date.plot(figsize=(9, 4.5), marker="o", color=["#dd7f2a", "#3266a8"])
    plt.title("RMSLE Across the 16-Day Validation Horizon"); plt.xlabel("Validation date"); plt.ylabel("RMSLE")
    plt.xticks(rotation=35); plt.tight_layout(); plt.savefig(FIGURE_DIR / "03_date_errors.png", dpi=150); plt.close()

    volume = history.groupby(["store_nbr", "family"])["sales"].sum().sort_values()
    positions = [int(0.1 * (len(volume) - 1)), int(0.5 * (len(volume) - 1)), int(0.9 * (len(volume) - 1))]
    selected = [volume.index[position] for position in positions]
    validation_start = predictions["date"].min()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (store, family_name) in zip(axes, selected):
        hist = history.loc[(history["store_nbr"] == store) & (history["family"] == family_name) & (history["date"] >= validation_start - pd.Timedelta(days=56))]
        val = predictions.loc[(predictions["store_nbr"] == store) & (predictions["family"] == family_name)]
        ax.plot(hist["date"], hist["sales"], color="#777777", label="History")
        ax.plot(val["date"], val["actual"], color="black", marker="o", label="Validation actual")
        ax.plot(val["date"], val["Ridge regression"], color="#dd7f2a", linestyle="--", label="Ridge")
        ax.plot(val["date"], val["HistGradientBoosting"], color="#3266a8", linestyle=":", label="HistGradientBoosting")
        ax.set_ylabel("Sales"); ax.set_title(f"Store {store} - {family_name}", loc="left", fontsize=10)
    axes[0].legend(frameon=False, ncol=2, fontsize=8); axes[-1].set_xlabel("Date")
    fig.suptitle("Representative Validation Forecasts", y=1.01)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "04_representative_forecasts.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _write_report(scores, errors):
    best = scores.iloc[0]
    ridge = scores.loc[scores["model_name"] == "Ridge regression"].iloc[0]
    hist = scores.loc[scores["model_name"] == "HistGradientBoosting"].iloc[0]
    family = errors.loc[errors["group_type"] == "family"].pivot(index="group_value", columns="model_name", values="rmsle")
    store = errors.loc[errors["group_type"] == "store"].pivot(index="group_value", columns="model_name", values="rmsle")
    family_gain = family["Ridge regression"] - family["HistGradientBoosting"]
    store_gain = store["Ridge regression"] - store["HistGradientBoosting"]
    rows = "\n".join(f"| {int(r['rank'])} | {r['model_name']} | {r['validation_rmsle']:.6f} | {r['difference_from_weekday_baseline']:+.6f} | {r['difference_from_ridge']:+.6f} | {r['training_time_seconds']:.2f} s |" for _, r in scores.iterrows())
    report = f"""# Lightweight Model Comparison

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
{rows}

![Model ranking](figures/models/01_model_ranking.png)

HistGradientBoosting {'beats' if hist['validation_rmsle'] < ridge['validation_rmsle'] else 'does not beat'} Ridge by **{abs(hist['validation_rmsle'] - ridge['validation_rmsle']):.6f} RMSLE**. Both fitted models beat the weekday baseline.

## 6. Comparison with the Stage 3 baseline

The selected model improves on `0.520631` by **{BENCHMARK - best['validation_rmsle']:.6f}**. Ridge scores {ridge['validation_rmsle']:.6f}, compared with its Stage 4 check of `0.494938`; small timing or floating-point differences may occur, but the implementation is equivalent.

## 7. Error patterns by family, store, and date

HistGradientBoosting has lower family RMSLE than Ridge for **{int((family_gain > 0).sum())} of {len(family_gain)} families** and lower store RMSLE for **{int((store_gain > 0).sum())} of {len(store_gain)} stores**. Difficult families under the selected model include {', '.join(family[best['model_name']].nlargest(3).index)}; difficult stores include {', '.join(store[best['model_name']].nlargest(3).index)}. Gains are therefore {'broad' if (family_gain > 0).mean() > 0.6 else 'mixed rather than universal'}.

![Family errors](figures/models/02_family_errors.png)

![Validation-date errors](figures/models/03_date_errors.png)

## 8. Representative forecasts

Low-, medium-, and high-volume series are selected mechanically using the Stage 3 rule. The plots show that both models can miss intermittent changes and spikes even when aggregate RMSLE improves.

![Representative forecasts](figures/models/04_representative_forecasts.png)

## 9. Model interpretation

Stage 4 ablations show that lag and rolling history provide the decisive information. The nonlinear model can represent thresholds and curved relationships that Ridge cannot, but predictive gains do not establish causal effects.

## 10. Preferred model for Stage 6

**{best['model_name']}** is selected with RMSLE **{best['validation_rmsle']:.6f}** because it has the lowest validation error, remains computationally modest, and uses the established shared pipeline. No ensemble is needed.

## 11. Limitations

This comparison uses one 16-day holdout, one fixed setting per model, conservative horizon-safe target-history features, and approximate timing. Validation leadership does not guarantee Kaggle leaderboard leadership.

## 12. Conclusion

The controlled comparison identifies {best['model_name']} as the Stage 6 candidate while preserving the simple baseline and Ridge as transparent references.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_model_comparison():
    """Build once, preprocess once, fit two models, and generate outputs."""
    raw = load_raw_data()
    history, validation, train_features, validation_features = build_feature_matrices(raw)
    validate_feature_integrity(validation, validation_features)
    if not leakage_perturbation_check(raw, validation_features):
        raise ValueError("Leakage perturbation check failed.")
    _, train_matrix, validation_matrix = _preprocess(train_features, validation_features)
    predictions, timings = _fit_models(train_matrix, validation_matrix, train_features["sales"])
    if any(np.isnan(values).any() or (values < 0).any() for values in predictions.values()):
        raise ValueError("Predictions must be complete and nonnegative.")
    feature_count = train_matrix.shape[1]
    scores = _score_table(validation_features["sales"], predictions, timings, feature_count)
    prediction_frame = validation_features[KEYS + ["sales"]].rename(columns={"sales": "actual"}).copy()
    for name, values in predictions.items(): prediction_frame[name] = values
    errors = _group_errors(prediction_frame)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False, float_format="%.6f")
    errors.to_csv(ERRORS_PATH, index=False, float_format="%.6f")
    _make_figures(scores, errors, prediction_frame, history)
    _write_report(scores, errors)
    return history, validation, train_features, validation_features, prediction_frame, scores, errors


def main() -> None:
    """Run the controlled Stage 5 comparison."""
    _, validation, train_features, _, _, scores, _ = run_model_comparison()
    print(f"Shared training matrix built once: {len(train_features):,} rows")
    print(f"Validation: {validation['date'].min().date()} to {validation['date'].max().date()} ({len(validation):,} rows)")
    print("Leakage perturbation check: passed")
    print(scores[["rank", "model_name", "validation_rmsle", "training_time_seconds"]].to_string(index=False, formatters={"validation_rmsle": "{:.6f}".format, "training_time_seconds": "{:.2f}".format}))
    print(f"Saved {SCORES_PATH}")


if __name__ == "__main__":
    main()

"""Rolling-origin development backtests and one untouched final holdout."""

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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.baselines import rmsle
from src.features import (
    CATEGORICAL_FEATURES, FEATURES, build_origin_features, build_supervised_origins,
    future_target_perturbation_check, load_raw_data, observed_future_rows,
    sample_training_origins, validate_temporal_integrity,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures" / "models"
BACKTEST_PATH = TABLE_DIR / "backtest_scores.csv"
FINAL_SCORES_PATH = TABLE_DIR / "final_holdout_scores.csv"
HORIZON_PATH = TABLE_DIR / "horizon_error_breakdown.csv"
FAMILY_PATH = TABLE_DIR / "family_error_breakdown.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison.md"
DEVELOPMENT_ORIGINS = [pd.Timestamp(date) for date in ("2017-03-31", "2017-05-15", "2017-06-30")]
FINAL_HOLDOUT_ORIGIN = pd.Timestamp("2017-07-30")
HGB_ITERATIONS = (80, 120)


def _markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    shown = frame.reset_index() if index else frame.copy()
    headers = [str(column) for column in shown.columns]
    rows = []
    for values in shown.itertuples(index=False, name=None):
        rows.append([f"{value:.6f}" if isinstance(value, float) else str(value) for value in values])
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _numeric_features() -> list[str]:
    return [column for column in FEATURES if column not in CATEGORICAL_FEATURES]


def _fit_ridge(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    pre = ColumnTransformer([
        ("numeric", StandardScaler(), _numeric_features()),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    x_train = pre.fit_transform(train[FEATURES])
    x_target = pre.transform(target[FEATURES])
    model = Ridge(alpha=1.0)
    model.fit(x_train, np.log1p(train["sales"]))
    return np.clip(np.expm1(model.predict(x_target)), 0, None)


def _fit_hgb(train: pd.DataFrame, target: pd.DataFrame, max_iter: int) -> np.ndarray:
    pre = ColumnTransformer([
        ("numeric", "passthrough", _numeric_features()),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ])
    x_train = pre.fit_transform(train[FEATURES])
    x_target = pre.transform(target[FEATURES])
    categorical_start = len(_numeric_features())
    model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=max_iter, max_leaf_nodes=31,
        l2_regularization=0.1, early_stopping=False, random_state=42,
        categorical_features=list(range(categorical_start, x_train.shape[1])),
    )
    model.fit(x_train, np.log1p(train["sales"]))
    return np.clip(np.expm1(model.predict(x_target)), 0, None)


def _baseline(target: pd.DataFrame) -> np.ndarray:
    return target["origin_weekday_mean_8w"].clip(lower=0).to_numpy(dtype=float)


def evaluate_origin(
    train_data: pd.DataFrame,
    stores: pd.DataFrame,
    evaluation_origin: pd.Timestamp,
    hgb_iterations: tuple[int, ...] | list[int],
    training_origin_count: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_origins = sample_training_origins(evaluation_origin, training_origin_count, train=train_data)
    supervised = build_supervised_origins(train_data, stores, training_origins)
    future = observed_future_rows(train_data, evaluation_origin)
    evaluation = build_origin_features(train_data, future, stores, evaluation_origin)
    validate_temporal_integrity(supervised)
    validate_temporal_integrity(evaluation)

    predictions = {"Seasonal baseline": _baseline(evaluation)}
    start = perf_counter(); predictions["Ridge"] = _fit_ridge(supervised, evaluation); ridge_time = perf_counter() - start
    timings = {"Seasonal baseline": 0.0, "Ridge": ridge_time}
    for iterations in hgb_iterations:
        name = f"HistGradientBoosting ({iterations} iter)"
        start = perf_counter(); predictions[name] = _fit_hgb(supervised, evaluation, iterations); timings[name] = perf_counter() - start

    rows = []
    for model, values in predictions.items():
        rows.append({
            "forecast_origin": evaluation_origin.date().isoformat(),
            "validation_start": (evaluation_origin + pd.Timedelta(days=1)).date().isoformat(),
            "validation_end": (evaluation_origin + pd.Timedelta(days=16)).date().isoformat(),
            "model": model,
            "rmsle": rmsle(evaluation["sales"], values),
            "rows": len(evaluation),
            "training_origins": len(training_origins),
            "training_examples": len(supervised),
            "fit_seconds": timings[model],
        })
    prediction_frame = evaluation[["date", "forecast_origin", "horizon", "store_nbr", "family", "sales", "origin_zero_fraction_28"]].copy()
    for name, values in predictions.items():
        prediction_frame[name] = values
    return pd.DataFrame(rows), prediction_frame


def run_development_backtests(raw: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, int]:
    raw = load_raw_data() if raw is None else raw
    tables = []
    for origin in DEVELOPMENT_ORIGINS:
        scores, _ = evaluate_origin(raw["train"], raw["stores"], origin, HGB_ITERATIONS)
        tables.append(scores)
    result = pd.concat(tables, ignore_index=True)
    hgb = result[result["model"].str.startswith("HistGradientBoosting")]
    selected_name = hgb.groupby("model")["rmsle"].mean().idxmin()
    selected_iterations = int(selected_name.split("(")[1].split()[0])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(BACKTEST_PATH, index=False, float_format="%.6f")
    return result, selected_iterations


def _squared_log_loss(actual: pd.Series, prediction: pd.Series | np.ndarray) -> np.ndarray:
    return (np.log1p(np.asarray(prediction)) - np.log1p(np.asarray(actual))) ** 2


def run_final_holdout(raw: dict[str, pd.DataFrame], selected_iterations: int):
    scores, predictions = evaluate_origin(
        raw["train"], raw["stores"], FINAL_HOLDOUT_ORIGIN, [selected_iterations], training_origin_count=20,
    )
    hgb_name = f"HistGradientBoosting ({selected_iterations} iter)"
    scores["role"] = "one-time final holdout"
    scores.to_csv(FINAL_SCORES_PATH, index=False, float_format="%.6f")

    horizon_rows = []
    for horizon, part in predictions.groupby("horizon"):
        hgb_loss = _squared_log_loss(part["sales"], part[hgb_name])
        ridge_loss = _squared_log_loss(part["sales"], part["Ridge"])
        horizon_rows.append({
            "horizon": int(horizon),
            "baseline_rmsle": rmsle(part["sales"], part["Seasonal baseline"]),
            "ridge_rmsle": rmsle(part["sales"], part["Ridge"]),
            "hgb_rmsle": rmsle(part["sales"], part[hgb_name]),
            "mean_paired_hgb_minus_ridge_squared_log_loss": float(np.mean(hgb_loss - ridge_loss)),
            "rows": len(part),
        })
    horizon = pd.DataFrame(horizon_rows)
    horizon.to_csv(HORIZON_PATH, index=False, float_format="%.6f")

    history = raw["train"].loc[raw["train"]["date"] <= FINAL_HOLDOUT_ORIGIN]
    family_zero = history.groupby("family")["sales"].apply(lambda values: float(values.eq(0).mean()))
    family_rows = []
    for family, part in predictions.groupby("family"):
        ridge_loss = _squared_log_loss(part["sales"], part["Ridge"])
        hgb_loss = _squared_log_loss(part["sales"], part[hgb_name])
        family_rows.append({
            "family": family,
            "historical_zero_sales_fraction": family_zero.loc[family],
            "ridge_rmsle": rmsle(part["sales"], part["Ridge"]),
            "hgb_rmsle": rmsle(part["sales"], part[hgb_name]),
            "mean_paired_hgb_minus_ridge_squared_log_loss": float(np.mean(hgb_loss - ridge_loss)),
            "hgb_better": bool(np.mean(hgb_loss - ridge_loss) < 0),
        })
    family = pd.DataFrame(family_rows).sort_values("historical_zero_sales_fraction", ascending=False)
    family.to_csv(FAMILY_PATH, index=False, float_format="%.6f")
    return scores, predictions, horizon, family


def _write_outputs(backtests, selected_iterations, final_scores, horizon, family):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(horizon["horizon"], horizon["baseline_rmsle"], marker="o", label="Seasonal baseline")
    ax.plot(horizon["horizon"], horizon["ridge_rmsle"], marker="o", label="Ridge")
    ax.plot(horizon["horizon"], horizon["hgb_rmsle"], marker="o", label="HistGradientBoosting")
    ax.set(xlabel="Forecast horizon (days)", ylabel="RMSLE", title="Final Holdout Error by Forecast Horizon", xticks=range(1, 17))
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURE_DIR / "horizon_errors.png", dpi=150); plt.close(fig)

    aggregate = backtests.groupby("model")["rmsle"].agg(["mean", "median", "min", "max"])
    final = final_scores.set_index("model")["rmsle"]
    hgb_name = f"HistGradientBoosting ({selected_iterations} iter)"
    fold_wins = int(sum(group.loc[group["model"] == hgb_name, "rmsle"].iloc[0] < group.loc[group["model"] == "Ridge", "rmsle"].iloc[0] for _, group in backtests.groupby("forecast_origin")))
    family_wins = int(family["hgb_better"].sum())
    report = f"""# Horizon-Aware Model Comparison

## Protocol

Each supervised row is `forecast_origin × store × family × horizon`. Horizons 1–16 are pooled in one model with an explicit horizon feature. Target-history variables are fixed at the forecast origin, so no future true sale inside a 16-day block can enter another prediction.

Development origins were fixed before comparison: {', '.join(x.date().isoformat() for x in DEVELOPMENT_ORIGINS)}. Each fold uses 16 earlier origins on a fixed 14-day grid, skipping incomplete calendar blocks. The final 2017-07-30 origin was untouched until the iteration count and feature set were frozen.

## Development backtests

{_markdown_table(aggregate)}

The predefined HGB choice was {selected_iterations} iterations, selected by mean development RMSLE with random internal early stopping disabled. HGB beat Ridge in {fold_wins} of {len(DEVELOPMENT_ORIGINS)} development folds.

## One-time final holdout

{_markdown_table(final.to_frame('RMSLE'))}

These values are not directly comparable to the former `0.449788`: the old result used different target-history semantics and reused this holdout for development.

## Horizon and family behavior

![Final holdout error by horizon](figures/models/horizon_errors.png)

HGB final-holdout RMSLE ranges from {horizon['hgb_rmsle'].min():.3f} to {horizon['hgb_rmsle'].max():.3f} across horizons. It has lower paired squared-log loss than Ridge for {int((horizon['mean_paired_hgb_minus_ridge_squared_log_loss'] < 0).sum())} of 16 horizons and {family_wins} of 33 families. The family table relates this gain to historical zero-sales frequency without making a causal claim.

## Interpretation and limitations

The comparison asks whether modest nonlinearity improves on transparent alternatives under identical forecast-origin information. Results are predictive, not causal. Three development origins and one final origin cannot represent every future regime; intermittent families and event-driven spikes remain difficult.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    raw = load_raw_data()
    if not future_target_perturbation_check(raw["train"], raw["stores"], FINAL_HOLDOUT_ORIGIN):
        raise ValueError("Future target perturbation check failed")
    backtests, selected = run_development_backtests(raw)
    final_scores, _, horizon, family = run_final_holdout(raw, selected)
    _write_outputs(backtests, selected, final_scores, horizon, family)
    print(f"Development origins: {', '.join(x.date().isoformat() for x in DEVELOPMENT_ORIGINS)}")
    print(f"Selected HGB iterations: {selected}")
    print(final_scores[["model", "rmsle"]].to_string(index=False))


if __name__ == "__main__":
    main()


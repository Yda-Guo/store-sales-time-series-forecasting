"""Leakage-safe validation and simple baseline forecasts for Store Sales."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SCORES_PATH = ROOT / "reports" / "tables" / "baseline_scores.csv"
KEYS = ["date", "store_nbr", "family"]
SERIES_KEYS = ["store_nbr", "family"]


def _find_train_file() -> Path:
    """Locate train.csv directly or inside a ZIP archive."""
    direct = RAW_DIR / "train.csv"
    if direct.exists():
        return direct
    for archive in sorted(RAW_DIR.glob("*.zip")):
        with zipfile.ZipFile(archive) as zipped:
            if "train.csv" in {Path(name).name for name in zipped.namelist()}:
                return archive
    raise FileNotFoundError("Could not find train.csv or a ZIP containing it.")


def load_training_data() -> pd.DataFrame:
    """Load the columns required for baseline evaluation."""
    path = _find_train_file()
    columns = ["date", "store_nbr", "family", "sales", "onpromotion"]
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, usecols=columns, parse_dates=["date"])
    with zipfile.ZipFile(path) as zipped:
        member = next(name for name in zipped.namelist() if Path(name).name == "train.csv")
        with zipped.open(member) as stream:
            return pd.read_csv(stream, usecols=columns, parse_dates=["date"])


def make_validation_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the final 16 unique dates as a chronological holdout."""
    dates = np.sort(data["date"].unique())
    if len(dates) < 17:
        raise ValueError("At least 17 unique dates are required.")
    validation_dates = dates[-16:]
    validation_start = pd.Timestamp(validation_dates[0])
    history = data.loc[data["date"] < validation_start].copy()
    validation = data.loc[data["date"].isin(validation_dates)].copy()
    validation = validation.sort_values(KEYS).reset_index(drop=True)
    if validation["date"].nunique() != 16:
        raise ValueError("Validation must contain exactly 16 dates.")
    if validation.duplicated(KEYS).any():
        raise ValueError("Validation keys are not unique.")
    return history, validation


def rmsle(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    """Calculate RMSLE after clipping predictions to zero."""
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.clip(np.asarray(prediction, dtype=float), 0, None)
    if np.any(actual_values < 0):
        raise ValueError("RMSLE requires nonnegative actual sales.")
    return float(np.sqrt(np.mean((np.log1p(prediction_values) - np.log1p(actual_values)) ** 2)))


def _lookup_with_fallback(
    validation: pd.DataFrame,
    series_values: pd.Series,
    family_values: pd.Series,
    global_value: float,
    extra_key: str | None = None,
) -> pd.Series:
    """Apply series, family, global, then zero fallback estimates."""
    series_index = SERIES_KEYS + ([extra_key] if extra_key else [])
    family_index = ["family"] + ([extra_key] if extra_key else [])
    series_keys = pd.MultiIndex.from_frame(validation[series_index])
    family_keys = pd.MultiIndex.from_frame(validation[family_index])
    prediction = pd.Series(series_values.reindex(series_keys).to_numpy(), index=validation.index, dtype=float)
    family_fallback = family_values.reindex(family_keys).to_numpy()
    prediction = prediction.fillna(pd.Series(family_fallback, index=validation.index))
    return prediction.fillna(global_value).fillna(0).clip(lower=0)


def _simple_group_baseline(history: pd.DataFrame, validation: pd.DataFrame, method: str) -> pd.Series:
    series_values = history.groupby(SERIES_KEYS)["sales"].agg(method)
    family_values = history.groupby("family")["sales"].agg(method)
    global_value = float(history["sales"].agg(method))
    return _lookup_with_fallback(validation, series_values, family_values, global_value)


def zero_forecast(validation: pd.DataFrame) -> pd.Series:
    """Predict zero for every row."""
    return pd.Series(0.0, index=validation.index)


def last_value_forecast(history: pd.DataFrame, validation: pd.DataFrame) -> pd.Series:
    """Repeat each series' final pre-validation observation."""
    latest = history.sort_values("date").groupby(SERIES_KEYS, as_index=False).tail(1)
    series_values = latest.set_index(SERIES_KEYS)["sales"]
    family_values = latest.groupby("family")["sales"].mean()
    return _lookup_with_fallback(validation, series_values, family_values, float(latest["sales"].mean()))


def recent_mean_forecast(history: pd.DataFrame, validation: pd.DataFrame, days: int) -> pd.Series:
    """Repeat a series mean from the final fixed historical window."""
    cutoff = history["date"].max() - pd.Timedelta(days=days - 1)
    window = history.loc[history["date"] >= cutoff]
    return _simple_group_baseline(window, validation, "mean")


def weekday_mean_forecast(history: pd.DataFrame, validation: pd.DataFrame, weeks: int = 8) -> pd.Series:
    """Use recent pre-validation means for each series and weekday."""
    cutoff = history["date"].max() - pd.Timedelta(days=weeks * 7 - 1)
    window = history.loc[history["date"] >= cutoff].copy()
    window["weekday"] = window["date"].dt.dayofweek
    target = validation.copy()
    target["weekday"] = target["date"].dt.dayofweek
    series_values = window.groupby(SERIES_KEYS + ["weekday"])["sales"].mean()
    family_values = window.groupby(["family", "weekday"])["sales"].mean()
    return _lookup_with_fallback(target, series_values, family_values, float(window["sales"].mean()), "weekday")


def repeated_week_forecast(history: pd.DataFrame, validation: pd.DataFrame) -> pd.Series:
    """Repeat the final complete seven-day pre-validation pattern."""
    pattern_end = history["date"].max()
    pattern_start = pattern_end - pd.Timedelta(days=6)
    pattern = history.loc[history["date"].between(pattern_start, pattern_end)].copy()
    pattern["pattern_day"] = (pattern["date"] - pattern_start).dt.days
    target = validation.copy()
    target["pattern_day"] = (target["date"] - target["date"].min()).dt.days % 7
    series_values = pattern.set_index(SERIES_KEYS + ["pattern_day"])["sales"]
    family_values = pattern.groupby(["family", "pattern_day"])["sales"].mean()
    return _lookup_with_fallback(target, series_values, family_values, float(pattern["sales"].mean()), "pattern_day")


def generate_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create all baseline predictions from pre-validation history only."""
    history, validation = make_validation_split(data)
    predictions = validation[KEYS + ["sales"]].rename(columns={"sales": "actual"}).copy()
    predictions["Zero forecast"] = zero_forecast(validation)
    predictions["Last observed value"] = last_value_forecast(history, validation)
    predictions["Recent mean (7 days)"] = recent_mean_forecast(history, validation, 7)
    predictions["Recent mean (28 days)"] = recent_mean_forecast(history, validation, 28)
    predictions["Weekday mean (8 weeks)"] = weekday_mean_forecast(history, validation, 8)
    predictions["Repeated final week"] = repeated_week_forecast(history, validation)
    return history, validation, predictions


def evaluate_baselines(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate, rank, and save the baseline score table."""
    history, validation, predictions = generate_predictions(data)
    rules = {
        "Zero forecast": "Predict 0",
        "Last observed value": "Repeat final pre-validation value",
        "Recent mean (7 days)": "Repeat mean of final 7 historical dates",
        "Recent mean (28 days)": "Repeat mean of final 28 historical dates",
        "Weekday mean (8 weeks)": "Series-weekday mean from prior 8 weeks",
        "Repeated final week": "Repeat final complete 7-day historical pattern",
    }
    rows = []
    for name, rule in rules.items():
        pred = predictions[name].clip(lower=0)
        rows.append({
            "baseline": name,
            "rmsle": rmsle(predictions["actual"], pred),
            "mae": float(np.mean(np.abs(predictions["actual"] - pred))),
            "prediction_rule": rule,
        })
    scores = pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)
    scores.insert(0, "rank", np.arange(1, len(scores) + 1))
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False, float_format="%.6f")
    return history, validation, predictions, scores


def main() -> None:
    """Run baseline evaluation and print a concise ranking."""
    data = load_training_data()
    history, validation, predictions, scores = evaluate_baselines(data)
    if len(validation) != 28_512 or validation["date"].nunique() != 16:
        raise ValueError("Expected 28,512 rows across 16 validation dates.")
    if predictions.drop(columns=KEYS + ["actual"]).isna().any().any():
        raise ValueError("Baseline predictions contain missing values.")
    if (predictions.drop(columns=KEYS + ["actual"]) < 0).any().any():
        raise ValueError("Baseline predictions contain negative values.")
    print(f"History: {history['date'].min().date()} to {history['date'].max().date()}")
    print(f"Validation: {validation['date'].min().date()} to {validation['date'].max().date()} ({len(validation):,} rows)")
    print(scores[["rank", "baseline", "rmsle"]].to_string(index=False, formatters={"rmsle": "{:.6f}".format}))
    print(f"Saved {SCORES_PATH}")


if __name__ == "__main__":
    main()

"""Forecast-origin-safe features for pooled direct 16-day forecasting."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SERIES_KEYS = ["store_nbr", "family"]
HORIZONS = tuple(range(1, 17))
TARGET_HISTORY_FEATURES = [
    "origin_sales", "origin_lag_7", "origin_lag_14", "origin_lag_28",
    "origin_mean_7", "origin_mean_28", "origin_std_28",
    "origin_median_28", "origin_zero_fraction_28", "origin_weekday_mean_8w",
]
CALENDAR_FEATURES = [
    "horizon", "target_day_of_week", "target_day_of_month", "target_month",
    "target_year", "target_is_weekend", "target_is_month_start",
    "target_is_month_end", "target_is_payday",
]
PROMOTION_FEATURES = ["onpromotion", "is_promoted", "log_onpromotion"]
CATEGORICAL_FEATURES = ["store_nbr", "family", "store_type", "cluster"]
FEATURES = CATEGORICAL_FEATURES + CALENDAR_FEATURES + PROMOTION_FEATURES + TARGET_HISTORY_FEATURES


def _find_csv(csv_name: str) -> Path:
    """Find a direct CSV or an archive containing it, including legacy names."""
    direct = RAW_DIR / csv_name
    if direct.exists():
        return direct
    for archive in sorted(RAW_DIR.glob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as zipped:
                if csv_name in {Path(name).name for name in zipped.namelist()}:
                    return archive
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError(f"Could not find {csv_name} directly or in data/raw/*.zip")


def _read_csv(csv_name: str, **kwargs) -> pd.DataFrame:
    path = _find_csv(csv_name)
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, **kwargs)
    with zipfile.ZipFile(path) as zipped:
        member = next(name for name in zipped.namelist() if Path(name).name == csv_name)
        with zipped.open(member) as stream:
            return pd.read_csv(stream, **kwargs)


def load_raw_data() -> dict[str, pd.DataFrame]:
    train = _read_csv(
        "train.csv",
        usecols=["id", "date", "store_nbr", "family", "sales", "onpromotion"],
        parse_dates=["date"],
    )
    return {
        "train": train.sort_values(["date", *SERIES_KEYS]).reset_index(drop=True),
        "test": _read_csv("test.csv", parse_dates=["date"]),
        "sample": _read_csv("sample_submission.csv"),
        "stores": _read_csv("stores.csv"),
    }


def _series_stat(history: pd.DataFrame, column: str, statistic: str) -> pd.Series:
    grouped = history.groupby(SERIES_KEYS, sort=False)[column]
    if statistic == "zero_fraction":
        return grouped.apply(lambda values: float(values.eq(0).mean()))
    return grouped.agg(statistic)


def _map_series(rows: pd.DataFrame, values: pd.Series, fallback: float = 0.0) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(rows[SERIES_KEYS])
    return values.reindex(keys).fillna(fallback).to_numpy(dtype=float)


def build_origin_features(
    sales_history: pd.DataFrame,
    future_rows: pd.DataFrame,
    stores: pd.DataFrame,
    forecast_origin: pd.Timestamp | str,
) -> pd.DataFrame:
    """Build one 16-day direct forecast matrix using sales through origin only.

    This is the single feature implementation used for historical training origins,
    development backtests, the final holdout, and the Kaggle test period.
    """
    origin = pd.Timestamp(forecast_origin)
    history = sales_history.loc[sales_history["date"] <= origin].copy()
    if history.empty or history["date"].max() > origin:
        raise ValueError("Target history must end on or before the forecast origin")

    result = future_rows.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["forecast_origin"] = origin
    result["horizon"] = (result["date"] - origin).dt.days
    if not result["horizon"].isin(HORIZONS).all():
        raise ValueError("Every target must be 1 to 16 days after its forecast origin")
    if result.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Future date/store/family keys must be unique")

    lookup = history.set_index(["date", *SERIES_KEYS])["sales"]
    for days, name in ((0, "origin_sales"), (7, "origin_lag_7"), (14, "origin_lag_14"), (28, "origin_lag_28")):
        dates = pd.Series(origin - pd.Timedelta(days=days), index=result.index)
        keys = pd.MultiIndex.from_arrays(
            [dates, result["store_nbr"], result["family"]],
            names=["date", *SERIES_KEYS],
        )
        result[name] = lookup.reindex(keys).to_numpy(dtype=float)

    fallback = float(history["sales"].mean())
    for days in (7, 28):
        window = history.loc[history["date"].between(origin - pd.Timedelta(days=days - 1), origin)]
        result[f"origin_mean_{days}"] = _map_series(result, _series_stat(window, "sales", "mean"), fallback)
    window28 = history.loc[history["date"].between(origin - pd.Timedelta(days=27), origin)]
    result["origin_std_28"] = _map_series(result, _series_stat(window28, "sales", "std"), 0.0)
    result["origin_median_28"] = _map_series(result, _series_stat(window28, "sales", "median"), fallback)
    result["origin_zero_fraction_28"] = _map_series(result, _series_stat(window28, "sales", "zero_fraction"), 0.0)

    weekday_history = history.loc[history["date"] >= origin - pd.Timedelta(days=55)].copy()
    weekday_history["target_day_of_week"] = weekday_history["date"].dt.dayofweek
    weekday_values = weekday_history.groupby([*SERIES_KEYS, "target_day_of_week"])["sales"].mean()
    weekday_keys = pd.MultiIndex.from_arrays(
        [result["store_nbr"], result["family"], result["date"].dt.dayofweek],
        names=[*SERIES_KEYS, "target_day_of_week"],
    )
    result["origin_weekday_mean_8w"] = weekday_values.reindex(weekday_keys).fillna(fallback).to_numpy(dtype=float)

    series_means = history.groupby(SERIES_KEYS)["sales"].mean()
    series_fallback = _map_series(result, series_means, fallback)
    for column in ["origin_sales", "origin_lag_7", "origin_lag_14", "origin_lag_28"]:
        result[column] = result[column].fillna(pd.Series(series_fallback, index=result.index))

    target_dates = result["date"].dt
    result["target_day_of_week"] = target_dates.dayofweek
    result["target_day_of_month"] = target_dates.day
    result["target_month"] = target_dates.month
    result["target_year"] = target_dates.year
    result["target_is_weekend"] = (target_dates.dayofweek >= 5).astype("int8")
    result["target_is_month_start"] = target_dates.is_month_start.astype("int8")
    result["target_is_month_end"] = target_dates.is_month_end.astype("int8")
    result["target_is_payday"] = ((target_dates.day == 15) | target_dates.is_month_end).astype("int8")
    result["is_promoted"] = result["onpromotion"].gt(0).astype("int8")
    result["log_onpromotion"] = np.log1p(result["onpromotion"].clip(lower=0))

    metadata = stores[["store_nbr", "type", "cluster"]].rename(columns={"type": "store_type"})
    result = result.merge(metadata, on="store_nbr", how="left", validate="many_to_one")
    result["store_type"] = result["store_type"].fillna("Unknown")
    result["cluster"] = result["cluster"].fillna(-1).astype(str)
    result["store_nbr"] = result["store_nbr"].astype(str)
    result["family"] = result["family"].astype(str)
    if result[FEATURES].isna().any().any():
        missing = result[FEATURES].isna().sum()
        raise ValueError(f"Origin feature matrix has missing values: {missing[missing > 0].to_dict()}")
    return result.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)


def observed_future_rows(train: pd.DataFrame, origin: pd.Timestamp | str) -> pd.DataFrame:
    origin = pd.Timestamp(origin)
    rows = train.loc[train["date"].between(origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=16))].copy()
    expected = 16 * train["store_nbr"].nunique() * train["family"].nunique()
    if len(rows) != expected:
        raise ValueError(f"Origin {origin.date()} has {len(rows):,} targets; expected {expected:,}")
    return rows


def build_supervised_origins(
    train: pd.DataFrame,
    stores: pd.DataFrame,
    origins: list[pd.Timestamp | str],
) -> pd.DataFrame:
    frames = []
    for origin in origins:
        future = observed_future_rows(train, origin)
        frames.append(build_origin_features(train, future, stores, origin))
    return pd.concat(frames, ignore_index=True)


def sample_training_origins(
    evaluation_origin: pd.Timestamp | str,
    count: int = 16,
    spacing_days: int = 14,
    train: pd.DataFrame | None = None,
) -> list[pd.Timestamp]:
    """Select earlier origins on a fixed grid, skipping incomplete target blocks."""
    evaluation_origin = pd.Timestamp(evaluation_origin)
    latest = evaluation_origin - pd.Timedelta(days=16)
    candidates = latest - pd.to_timedelta(np.arange(count * 3) * spacing_days, unit="D")
    if train is None:
        return sorted(candidates[:count])
    expected = 16 * train["store_nbr"].nunique() * train["family"].nunique()
    selected = []
    for candidate in candidates:
        rows = train["date"].between(candidate + pd.Timedelta(days=1), candidate + pd.Timedelta(days=16)).sum()
        if rows == expected:
            selected.append(candidate)
        if len(selected) == count:
            return sorted(selected)
    raise ValueError(f"Could not find {count} complete training origins before {evaluation_origin.date()}")


def validate_temporal_integrity(features: pd.DataFrame) -> None:
    if not features["horizon"].isin(HORIZONS).all():
        raise ValueError("Invalid horizon")
    if not (features["date"] == features["forecast_origin"] + pd.to_timedelta(features["horizon"], unit="D")).all():
        raise ValueError("Target date and horizon disagree")
    if features[FEATURES].isna().any().any():
        raise ValueError("Missing model inputs")


def future_target_perturbation_check(train: pd.DataFrame, stores: pd.DataFrame, origin: pd.Timestamp | str) -> bool:
    future = observed_future_rows(train, origin)
    original = build_origin_features(train, future, stores, origin)
    changed = train.copy()
    changed.loc[changed["date"] > pd.Timestamp(origin), "sales"] = 999_999.0
    rebuilt = build_origin_features(changed, future, stores, origin)
    return original[FEATURES].equals(rebuilt[FEATURES])


def main() -> None:
    raw = load_raw_data()
    origin = pd.Timestamp("2017-07-30")
    features = build_origin_features(raw["train"], observed_future_rows(raw["train"], origin), raw["stores"], origin)
    validate_temporal_integrity(features)
    if not future_target_perturbation_check(raw["train"], raw["stores"], origin):
        raise ValueError("Future-target perturbation changed origin-safe features")
    print(f"Origin: {origin.date()}; rows: {len(features):,}; horizons: 1-16")
    print(f"Model inputs: {len(FEATURES)}; future-target perturbation: passed")


if __name__ == "__main__":
    main()


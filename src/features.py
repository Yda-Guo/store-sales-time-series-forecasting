"""Build leakage-safe Store Sales features and run a lightweight Ridge check."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.baselines import KEYS, make_validation_split, rmsle


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SUMMARY_PATH = ROOT / "reports" / "tables" / "feature_summary.csv"
SCORES_PATH = ROOT / "reports" / "tables" / "feature_check_scores.csv"
BENCHMARK = 0.520631
TRAINING_DAYS = 365

CALENDAR = ["day_of_week", "day_of_month", "month", "year", "is_weekend", "is_month_start", "is_month_end", "is_payday"]
STORE = ["store_type", "cluster"]
PROMOTION = ["onpromotion", "is_promoted", "log_onpromotion"]
TARGET_DERIVED = ["lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28", "rolling_std_28"]
TARGET_LOG = [f"log_{feature}" for feature in TARGET_DERIVED]
HOLIDAY_OIL = ["is_national_holiday", "is_transfer_day", "oil_price"]
ALL_FEATURES = CALENDAR + STORE + PROMOTION + TARGET_DERIVED + TARGET_LOG + HOLIDAY_OIL


def _find_csv(csv_name: str) -> Path:
    direct = RAW_DIR / csv_name
    if direct.exists():
        return direct
    for archive in sorted(RAW_DIR.glob("*.zip")):
        with zipfile.ZipFile(archive) as zipped:
            if csv_name in {Path(name).name for name in zipped.namelist()}:
                return archive
    raise FileNotFoundError(f"Could not find {csv_name}")


def _read_csv(csv_name: str, **kwargs) -> pd.DataFrame:
    path = _find_csv(csv_name)
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, **kwargs)
    with zipfile.ZipFile(path) as zipped:
        member = next(name for name in zipped.namelist() if Path(name).name == csv_name)
        with zipped.open(member) as stream:
            return pd.read_csv(stream, **kwargs)


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load only raw files needed by Stage 4."""
    train = _read_csv(
        "train.csv",
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        parse_dates=["date"],
    )
    return {
        "train": train,
        "stores": _read_csv("stores.csv"),
        "holidays": _read_csv("holidays_events.csv", parse_dates=["date"]),
        "oil": _read_csv("oil.csv", parse_dates=["date"]),
    }


def build_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar and payday variables."""
    result = frame.copy()
    dates = result["date"].dt
    result["day_of_week"] = dates.dayofweek
    result["day_of_month"] = dates.day
    result["month"] = dates.month
    result["year"] = dates.year
    result["is_weekend"] = (dates.dayofweek >= 5).astype("int8")
    result["is_month_start"] = dates.is_month_start.astype("int8")
    result["is_month_end"] = dates.is_month_end.astype("int8")
    result["is_payday"] = ((dates.day == 15) | dates.is_month_end).astype("int8")
    result["is_promoted"] = (result["onpromotion"] > 0).astype("int8")
    result["log_onpromotion"] = np.log1p(result["onpromotion"].clip(lower=0))
    return result


def build_date_features(holidays: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    """Create one simplified holiday/oil row per date."""
    holiday = holidays.assign(
        is_national_holiday=(
            holidays["locale"].eq("National")
            & ~holidays["transferred"].fillna(False)
            & ~holidays["type"].eq("Transfer")
        ),
        is_transfer_day=holidays["type"].eq("Transfer"),
    )
    holiday = holiday.groupby("date", as_index=False)[["is_national_holiday", "is_transfer_day"]].max()
    oil_clean = oil.sort_values("date").copy()
    oil_clean["oil_price"] = oil_clean["dcoilwtico"].ffill().bfill()
    oil_clean = oil_clean[["date", "oil_price"]]
    dates = pd.DataFrame({"date": pd.date_range(oil_clean["date"].min(), oil_clean["date"].max())})
    dates = dates.merge(oil_clean, on="date", how="left")
    dates["oil_price"] = dates["oil_price"].ffill().bfill()
    dates = dates.merge(holiday, on="date", how="left")
    for column in ("is_national_holiday", "is_transfer_day"):
        dates[column] = dates[column].eq(True).astype("int8")
    return dates


def _add_training_target_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add shifted lags and rolling statistics to historical rows."""
    result = rows.sort_values(KEYS).copy()
    groups = result.groupby(["store_nbr", "family"], sort=False)["sales"]
    for lag in (7, 14, 28):
        result[f"lag_{lag}"] = groups.shift(lag)
    shifted = groups.shift(1)
    group_index = [result["store_nbr"], result["family"]]
    result["rolling_mean_7"] = shifted.groupby(group_index, sort=False).rolling(7).mean().reset_index(level=[0, 1], drop=True)
    result["rolling_mean_28"] = shifted.groupby(group_index, sort=False).rolling(28).mean().reset_index(level=[0, 1], drop=True)
    result["rolling_std_28"] = shifted.groupby(group_index, sort=False).rolling(28).std().reset_index(level=[0, 1], drop=True)
    return result


def _safe_validation_target_features(history: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Build validation target features using pre-validation sales only."""
    result = validation.copy()
    validation_start = result["date"].min()
    history_lookup = history.set_index(KEYS)["sales"]
    offsets = (result["date"] - validation_start).dt.days
    for lag in (7, 14, 28):
        reference = result["date"] - pd.to_timedelta(lag, unit="D")
        while (reference >= validation_start).any():
            reference = reference.where(reference < validation_start, reference - pd.Timedelta(days=7))
        index = pd.MultiIndex.from_arrays([reference, result["store_nbr"], result["family"]], names=KEYS)
        result[f"lag_{lag}"] = history_lookup.reindex(index).to_numpy()

    recent = history.loc[history["date"] >= validation_start - pd.Timedelta(days=28)]
    series_mean = history.groupby(["store_nbr", "family"])["sales"].mean()
    family_mean = history.groupby("family")["sales"].mean()
    global_mean = float(history["sales"].mean())
    keys = pd.MultiIndex.from_frame(result[["store_nbr", "family"]])
    fallback = pd.Series(series_mean.reindex(keys).to_numpy(), index=result.index)
    fallback = fallback.fillna(result["family"].map(family_mean)).fillna(global_mean).fillna(0)
    for window in (7, 28):
        part = recent.loc[recent["date"] >= validation_start - pd.Timedelta(days=window)]
        values = part.groupby(["store_nbr", "family"])["sales"].mean()
        result[f"rolling_mean_{window}"] = pd.Series(values.reindex(keys).to_numpy(), index=result.index).fillna(fallback)
    std_values = recent.groupby(["store_nbr", "family"])["sales"].std()
    result["rolling_std_28"] = pd.Series(std_values.reindex(keys).to_numpy(), index=result.index).fillna(0)
    for column in ["lag_7", "lag_14", "lag_28"]:
        result[column] = result[column].fillna(fallback)
    return result


def build_feature_matrices(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the shared train and validation matrices exactly once."""
    history, validation = make_validation_split(raw["train"])
    validation_start = validation["date"].min()
    training_start = validation_start - pd.Timedelta(days=TRAINING_DAYS)
    warm_start = training_start - pd.Timedelta(days=28)
    source = raw["train"].loc[raw["train"]["date"].between(warm_start, history["date"].max())]
    train_features = _add_training_target_features(source)
    train_features = train_features.loc[train_features["date"] >= training_start].copy()
    validation_features = _safe_validation_target_features(history, validation)

    date_features = build_date_features(raw["holidays"], raw["oil"])
    stores = raw["stores"][["store_nbr", "type", "cluster"]].rename(columns={"type": "store_type"})
    completed = []
    for frame in (train_features, validation_features):
        frame = build_calendar_features(frame)
        frame = frame.merge(stores, on="store_nbr", how="left", validate="many_to_one")
        frame = frame.merge(date_features, on="date", how="left", validate="many_to_one")
        frame["store_type"] = frame["store_type"].fillna("Unknown")
        frame["cluster"] = frame["cluster"].fillna(-1)
        frame[HOLIDAY_OIL] = frame[HOLIDAY_OIL].fillna({"is_national_holiday": 0, "is_transfer_day": 0, "oil_price": float(date_features["oil_price"].median())})
        for feature in TARGET_DERIVED:
            frame[f"log_{feature}"] = np.log1p(frame[feature].clip(lower=0))
        completed.append(frame.sort_values(KEYS).reset_index(drop=True))
    return history, validation, completed[0], completed[1]


def validate_feature_integrity(validation: pd.DataFrame, features: pd.DataFrame) -> None:
    """Check validation shape, keys, and final input completeness."""
    if validation["date"].nunique() != 16 or len(validation) != 28_512:
        raise ValueError("Expected 16 validation dates and 28,512 rows.")
    if not validation[KEYS].reset_index(drop=True).equals(features[KEYS]):
        raise ValueError("Feature keys do not match validation keys.")
    if features[ALL_FEATURES].isna().any().any():
        raise ValueError("Final feature matrix contains missing values.")


def leakage_perturbation_check(raw: dict[str, pd.DataFrame], validation_features: pd.DataFrame) -> bool:
    """Confirm validation target changes cannot alter target-derived inputs."""
    changed = raw["train"].copy()
    validation_start = validation_features["date"].min()
    changed.loc[changed["date"] >= validation_start, "sales"] = 999_999
    history, validation = make_validation_split(changed)
    rebuilt = _safe_validation_target_features(history, validation)
    for feature in TARGET_DERIVED:
        rebuilt[f"log_{feature}"] = np.log1p(rebuilt[feature].clip(lower=0))
    return validation_features[TARGET_DERIVED + TARGET_LOG].equals(rebuilt[TARGET_DERIVED + TARGET_LOG])


def _ridge_score(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> tuple[float, int, np.ndarray]:
    categorical = [column for column in columns if column == "store_type"]
    numeric = [column for column in columns if column not in categorical]
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    model = Pipeline([("preprocess", transformer), ("ridge", Ridge(alpha=1.0))])
    model.fit(train[columns], np.log1p(train["sales"]))
    prediction = np.clip(np.expm1(model.predict(validation[columns])), 0, None)
    if np.isnan(prediction).any() or (prediction < 0).any():
        raise ValueError("Invalid Ridge predictions.")
    count = len(model.named_steps["preprocess"].get_feature_names_out())
    return rmsle(validation["sales"], prediction), count, prediction


def run_feature_checks(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Reuse shared matrices and select columns for four Ridge checks."""
    configurations = [
        ("Calendar and store", CALENDAR + STORE, "calendar; store"),
        ("Add promotion", CALENDAR + STORE + PROMOTION, "calendar; store; promotion"),
        ("Add lag and rolling", CALENDAR + STORE + PROMOTION + TARGET_DERIVED + TARGET_LOG, "calendar; store; promotion; lag; rolling"),
        ("Add holiday and oil", ALL_FEATURES, "calendar; store; promotion; lag; rolling; holiday; oil"),
    ]
    rows, predictions = [], {}
    for name, columns, groups in configurations:
        score, count, prediction = _ridge_score(train, validation, columns)
        predictions[name] = prediction
        rows.append({
            "configuration": name,
            "included_feature_groups": groups,
            "training_window": f"final {TRAINING_DAYS} days before validation",
            "model_specification": "Ridge(alpha=1.0), log1p target",
            "encoded_feature_count": count,
            "validation_rmsle": score,
            "difference_from_baseline": score - BENCHMARK,
        })
    scores = pd.DataFrame(rows).sort_values("validation_rmsle").reset_index(drop=True)
    scores.insert(0, "rank", np.arange(1, len(scores) + 1))
    return scores, predictions


def build_feature_summary(validation: pd.DataFrame) -> pd.DataFrame:
    """Describe the final shared feature columns."""
    groups = {**{x: "calendar" for x in CALENDAR}, **{x: "store" for x in STORE}, **{x: "promotion" for x in PROMOTION}, **{x: "target-derived" for x in TARGET_DERIVED + TARGET_LOG}, **{x: "holiday/oil" for x in HOLIDAY_OIL}}
    sources = {**{x: "date" for x in CALENDAR}, "store_type": "stores.csv", "cluster": "stores.csv", **{x: "train onpromotion" for x in PROMOTION}, **{x: "pre-validation sales" for x in TARGET_DERIVED + TARGET_LOG}, "is_national_holiday": "holidays_events.csv", "is_transfer_day": "holidays_events.csv", "oil_price": "oil.csv"}
    rows = []
    for feature in ALL_FEATURES:
        target_derived = feature in TARGET_DERIVED + TARGET_LOG
        rows.append({
            "feature_name": feature,
            "feature_group": groups[feature],
            "data_type": str(validation[feature].dtype),
            "source": sources[feature],
            "target_derived": target_derived,
            "leakage_note": "pre-validation sales only" if target_derived else "known or external at forecast time",
            "validation_missing_count": int(validation[feature].isna().sum()),
        })
    return pd.DataFrame(rows)


def run_pipeline(raw: dict[str, pd.DataFrame] | None = None):
    """Build the shared matrices once, then run all column-selection checks."""
    raw = load_raw_data() if raw is None else raw
    history, validation, train_features, validation_features = build_feature_matrices(raw)
    validate_feature_integrity(validation, validation_features)
    scores, predictions = run_feature_checks(train_features, validation_features)
    summary = build_feature_summary(validation_features)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    scores.to_csv(SCORES_PATH, index=False, float_format="%.6f")
    return history, validation, train_features, validation_features, summary, scores, predictions


def main() -> None:
    """Generate Stage 4 feature tables and print the Ridge ranking."""
    raw = load_raw_data()
    _, validation, train_features, validation_features, _, scores, _ = run_pipeline(raw)
    leakage_passed = leakage_perturbation_check(raw, validation_features)
    if not leakage_passed:
        raise ValueError("Leakage perturbation check failed.")
    print(f"Shared training matrix: {len(train_features):,} rows x {len(ALL_FEATURES)} input columns")
    print(f"Validation: {validation['date'].min().date()} to {validation['date'].max().date()} ({len(validation_features):,} rows)")
    print("Leakage perturbation check: passed")
    print(scores[["rank", "configuration", "encoded_feature_count", "validation_rmsle", "difference_from_baseline"]].to_string(index=False, formatters={"validation_rmsle": "{:.6f}".format, "difference_from_baseline": "{:+.6f}".format}))
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {SCORES_PATH}")


if __name__ == "__main__":
    main()

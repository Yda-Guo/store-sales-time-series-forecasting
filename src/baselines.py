"""Metric and seasonal reference model for the declared development folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "reports" / "tables" / "baseline_scores.csv"
DEVELOPMENT_ORIGINS = tuple(pd.Timestamp(value) for value in ("2017-03-31", "2017-05-15", "2017-06-30"))


def rmsle(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.clip(np.asarray(prediction, dtype=float), 0, None)
    if np.any(actual_values < 0):
        raise ValueError("RMSLE requires nonnegative actual sales")
    return float(np.sqrt(np.mean((np.log1p(prediction_values) - np.log1p(actual_values)) ** 2)))


def main() -> None:
    from src.features import build_origin_features, load_raw_data, observed_future_rows

    raw = load_raw_data(); rows = []
    for origin in DEVELOPMENT_ORIGINS:
        future = observed_future_rows(raw["train"], origin)
        features = build_origin_features(raw["train"], future, raw["stores"], origin)
        prediction = features["origin_weekday_mean_8w"].clip(lower=0)
        rows.append({"forecast_origin": origin.date().isoformat(), "model": "Seasonal baseline",
                     "rmsle": rmsle(features["sales"], prediction), "rows": len(features)})
    scores = pd.DataFrame(rows)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True); scores.to_csv(SCORES_PATH, index=False, float_format="%.6f")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()


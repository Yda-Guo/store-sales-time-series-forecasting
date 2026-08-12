import unittest

import numpy as np
import pandas as pd

from src.features import FEATURES, build_origin_features


class TemporalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2020-01-01", periods=90)
        rows = []
        for store in (1, 2):
            for family_index, family in enumerate(("A", "B")):
                for day, date in enumerate(dates):
                    rows.append({"id": len(rows), "date": date, "store_nbr": store, "family": family,
                                 "sales": float(1000 * store + 100 * family_index + day), "onpromotion": day % 3})
        cls.train = pd.DataFrame(rows)
        cls.stores = pd.DataFrame({"store_nbr": [1, 2], "type": ["X", "Y"], "cluster": [1, 2]})
        cls.origin = pd.Timestamp("2020-03-10")
        cls.future = cls.train[cls.train["date"].between(cls.origin + pd.Timedelta(days=1), cls.origin + pd.Timedelta(days=16))].copy()

    def features(self, train=None):
        return build_origin_features(self.train if train is None else train, self.future, self.stores, self.origin)

    def test_future_sales_perturbation_cannot_change_features(self):
        changed = self.train.copy()
        changed.loc[changed["date"] > self.origin, "sales"] = 999999
        pd.testing.assert_frame_equal(self.features()[FEATURES], self.features(changed)[FEATURES])

    def test_series_are_isolated(self):
        features = self.features()
        row = features[(features["store_nbr"] == "2") & (features["family"] == "B")].iloc[0]
        expected = 2000 + 100 + (self.origin - pd.Timestamp("2020-01-01")).days
        self.assertEqual(row["origin_sales"], expected)

    def test_rolling_window_ends_at_origin(self):
        features = self.features()
        row = features[(features["store_nbr"] == "1") & (features["family"] == "A")].iloc[0]
        history = self.train[(self.train["store_nbr"] == 1) & (self.train["family"] == "A") & self.train["date"].between(self.origin - pd.Timedelta(days=6), self.origin)]
        self.assertAlmostEqual(row["origin_mean_7"], history["sales"].mean())

    def test_origin_relative_lags_use_exact_dates(self):
        features = self.features()
        row = features[(features["store_nbr"] == "1") & (features["family"] == "A")].iloc[0]
        series = self.train[(self.train["store_nbr"] == 1) & (self.train["family"] == "A")].set_index("date")["sales"]
        self.assertEqual(row["origin_lag_7"], series.loc[self.origin - pd.Timedelta(days=7)])
        self.assertEqual(row["origin_lag_28"], series.loc[self.origin - pd.Timedelta(days=28)])

    def test_horizon_one_and_sixteen_have_origin_frozen_history(self):
        features = self.features()
        series = features[(features["store_nbr"] == "1") & (features["family"] == "A")].sort_values("horizon")
        self.assertEqual(series.iloc[0]["horizon"], 1)
        self.assertEqual(series.iloc[-1]["horizon"], 16)
        self.assertEqual(series.iloc[0]["origin_sales"], series.iloc[-1]["origin_sales"])
        self.assertEqual(series.iloc[0]["date"], self.origin + pd.Timedelta(days=1))
        self.assertEqual(series.iloc[-1]["date"], self.origin + pd.Timedelta(days=16))

    def test_repeated_construction_is_identical(self):
        pd.testing.assert_frame_equal(self.features(), self.features())


if __name__ == "__main__":
    unittest.main()


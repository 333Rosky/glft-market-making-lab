import unittest

import numpy as np

from glft_lab.walkforward import monthly_splits, walk_forward_monthly


def monthly_state(month_distances, rows_per_month=20):
    timestamps = []
    distances = []
    sides = []
    for month, distance in enumerate(month_distances, start=1):
        for row in range(rows_per_month):
            day = row // 2 + 1
            timestamps.append(f"2024-{month:02d}-{day:02d}T12:00:00")
            distances.append(float(distance))
            sides.append("bid" if row % 2 == 0 else "ask")
    n = len(timestamps)
    return {
        "distance": np.asarray(distances),
        "spread": np.full(n, 0.5),
        "imbalance": np.resize(np.array([-0.2, 0.2]), n),
        "ofi": np.resize(np.array([-0.1, 0.1]), n),
        "volatility": np.full(n, 0.02),
        "queue_ahead": np.full(n, 2.0),
        "order_age": np.full(n, 0.1),
        "timestamp": np.asarray(timestamps, dtype="datetime64[s]"),
        "side": np.asarray(sides),
    }


class RecordingHazard:
    def fit(self, data, count, exposure):
        self.train_timestamps = np.asarray(data["timestamp"]).copy()
        self.training_distance_mean = float(np.mean(data["distance"]))
        return self

    def predict_rate(self, data):
        return np.full(len(data["distance"]), 0.25)


class MonthlySplitTests(unittest.TestCase):
    def test_expanding_splits_are_strictly_chronological_when_rows_are_unsorted(self):
        timestamp = np.array(
            [
                "2024-03-03",
                "2024-01-02",
                "2024-02-04",
                "2024-01-03",
                "2024-03-04",
                "2024-02-05",
            ],
            dtype="datetime64[D]",
        )
        splits = monthly_splits(timestamp, mode="expanding", min_train_months=1)

        self.assertEqual([str(split.test_month) for split in splits], ["2024-02", "2024-03"])
        np.testing.assert_array_equal(splits[0].train_indices, [1, 3])
        np.testing.assert_array_equal(splits[0].test_indices, [2, 5])
        for split in splits:
            train_month = timestamp[split.train_indices].astype("datetime64[M]")
            self.assertTrue(np.all(train_month < split.test_month))
            self.assertFalse(np.intersect1d(split.train_indices, split.test_indices).size)

    def test_rolling_splits_keep_only_requested_history(self):
        timestamp = np.array(
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            dtype="datetime64[D]",
        )
        splits = monthly_splits(
            timestamp,
            mode="rolling",
            min_train_months=2,
            train_window_months=2,
        )
        self.assertEqual(len(splits), 2)
        self.assertEqual(tuple(map(str, splits[-1].train_months)), ("2024-02", "2024-03"))
        self.assertEqual(str(splits[-1].test_month), "2024-04")


class WalkForwardTests(unittest.TestCase):
    def test_models_never_receive_their_test_month(self):
        data = monthly_state([1.0, 100.0, 10_000.0], rows_per_month=8)
        n = len(data["timestamp"])
        count = np.resize(np.array([0, 1]), n)
        exposure = np.ones(n)

        result = walk_forward_monthly(
            data,
            count,
            exposure,
            model_factory=RecordingHazard,
            min_train_months=1,
        )

        self.assertEqual(len(result.folds), 2)
        for fold in result.folds:
            train_months = fold.model.train_timestamps.astype("datetime64[M]")
            self.assertTrue(np.all(train_months < fold.test_month))
        self.assertEqual(result.folds[0].model.training_distance_mean, 1.0)
        self.assertEqual(result.folds[1].model.training_distance_mean, 50.5)
        self.assertTrue(np.all(np.isnan(result.rate[:8])))
        self.assertTrue(np.all(result.rate[8:] == 0.25))
        np.testing.assert_allclose(result.probability[8:], 1.0 - np.exp(-0.25 * exposure[8:]))

    def test_feature_scaling_is_fitted_on_train_months_only(self):
        data = monthly_state([1.0, 100.0, 1_000_000.0])
        n = len(data["timestamp"])
        count = np.resize(np.array([0, 1, 1, 0]), n)
        result = walk_forward_monthly(data, count, np.ones(n))

        first_center = result.folds[0].model.feature_builder.center_[0]
        second_center = result.folds[1].model.feature_builder.center_[0]
        self.assertEqual(first_center, 1.0)
        self.assertEqual(second_center, 50.5)
        self.assertNotEqual(second_center, 1_000_000.0)

    def test_walk_forward_returns_oos_metrics_and_diagnostics(self):
        data = monthly_state([1.0, 2.0, 3.0], rows_per_month=8)
        n = len(data["timestamp"])
        count = np.resize(np.array([0, 1]), n)
        markout = np.where(count > 0, np.resize(np.array([1.0, -0.5]), n), np.nan)
        pnl = np.resize(np.array([1.0, -2.0, 0.5]), n)
        inventory = np.resize(np.array([0.0, 1.0, -1.0]), n)

        result = walk_forward_monthly(
            data,
            count,
            np.ones(n),
            model_factory=RecordingHazard,
            markout=markout,
            pnl=pnl,
            inventory=inventory,
            max_residual_lag=2,
        )

        self.assertTrue(result.diagnostics["out_of_sample"])
        self.assertIn("poisson_deviance", result.metrics)
        self.assertIn("brier_score", result.metrics)
        self.assertIn("mean_markout", result.metrics)
        self.assertIn("net_pnl", result.metrics)
        self.assertIn("inventory_rms", result.metrics)
        self.assertIn("decision", result.hawkes_gate)
        self.assertEqual(result.metrics["n_observations"], int(result.oos_mask.sum()))


if __name__ == "__main__":
    unittest.main()

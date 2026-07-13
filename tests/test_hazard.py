import unittest

import numpy as np

from glft_lab.hazard import (
    FEATURE_NAMES,
    HazardFeatureBuilder,
    StateDependentFillHazard,
    aggregate_oos_metrics,
    competing_risk_labels,
    fill_probability,
    hawkes_decision_gate,
    poisson_deviance,
    poisson_log_likelihood,
    residual_diagnostics,
)


def state(rows, *, distance=None, side=None, timestamp=None):
    rows = int(rows)
    return {
        "distance": np.ones(rows) if distance is None else np.asarray(distance),
        "spread": np.full(rows, 0.5),
        "imbalance": np.linspace(-0.2, 0.2, rows),
        "ofi": np.linspace(0.1, -0.1, rows),
        "volatility": np.full(rows, 0.02),
        "queue_ahead": np.full(rows, 3.0),
        "order_age": np.full(rows, 0.1),
        "timestamp": (
            np.arange(rows).astype("timedelta64[m]") + np.datetime64("2024-01-01")
            if timestamp is None
            else np.asarray(timestamp)
        ),
        "side": (np.resize(np.array(["bid", "ask"]), rows) if side is None else np.asarray(side)),
    }


class HazardFeatureTests(unittest.TestCase):
    def test_builder_uses_training_statistics_only(self):
        train = state(3, distance=[1.0, 2.0, 3.0])
        test = state(1, distance=[1_000_000.0])

        builder = HazardFeatureBuilder(clip=5.0).fit(train)
        transformed = builder.transform(test)

        self.assertEqual(transformed.shape, (1, len(FEATURE_NAMES)))
        self.assertEqual(builder.center_[0], 2.0)
        self.assertEqual(transformed[0, 0], 1.0)
        self.assertEqual(transformed[0, 1], 5.0)

    def test_cyclic_time_of_day_is_continuous_at_midnight(self):
        data = state(
            2,
            timestamp=np.array(
                ["2024-01-01T23:59:59", "2024-01-02T00:00:01"],
                dtype="datetime64[s]",
            ),
        )
        raw = HazardFeatureBuilder.raw_matrix(data)
        self.assertLess(np.linalg.norm(raw[0, -2:] - raw[1, -2:]), 0.001)

    def test_millisecond_epoch_uses_correct_time_of_day(self):
        timestamp = np.array([np.datetime64("2024-01-01T12:00:00", "ms").astype(np.int64)])
        data = state(1, timestamp=timestamp)
        raw = HazardFeatureBuilder.raw_matrix(data)
        self.assertAlmostEqual(raw[0, -2], 0.0, places=12)
        self.assertAlmostEqual(raw[0, -1], -1.0, places=12)

    def test_invalid_physical_state_is_rejected(self):
        data = state(2)
        data["queue_ahead"] = np.array([1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "queue_ahead"):
            HazardFeatureBuilder.raw_matrix(data)


class HazardFormulaTests(unittest.TestCase):
    def test_fill_probability_uses_integrated_hazard(self):
        rate = np.array([0.0, 0.5, 2.0])
        exposure = np.array([10.0, 2.0, 0.25])
        expected = 1.0 - np.exp(-rate * exposure)
        np.testing.assert_allclose(fill_probability(rate, exposure), expected)

    def test_poisson_log_likelihood_and_deviance(self):
        count = np.array([0, 1])
        rate = np.array([0.5, 0.5])
        exposure = np.ones(2)
        expected_log_likelihood = -0.5 + np.log(0.5) - 0.5
        expected_deviance = 2.0 * np.log(2.0)

        self.assertAlmostEqual(
            poisson_log_likelihood(count, rate, exposure), expected_log_likelihood
        )
        self.assertAlmostEqual(poisson_deviance(count, rate, exposure), expected_deviance)

    def test_side_specific_glm_recovers_exposure_adjusted_rates(self):
        rng = np.random.default_rng(7)
        n = 4_000
        sides = np.resize(np.array(["bid", "ask"]), n)
        distance = rng.uniform(0.0, 1.0, n)
        exposure = rng.uniform(0.25, 2.0, n)
        true_rate = np.where(
            sides == "bid",
            np.exp(-0.3 - 1.0 * distance),
            np.exp(-1.3 + 0.2 * distance),
        )
        count = rng.poisson(true_rate * exposure)
        data = state(n, distance=distance, side=sides)

        model = StateDependentFillHazard(l2=1e-3).fit(data, count, exposure)
        probe = state(
            2,
            distance=[0.5, 0.5],
            side=["bid", "ask"],
            timestamp=np.array(["2024-01-10T12:00", "2024-01-10T12:00"], dtype="datetime64[m]"),
        )
        predicted = model.predict_rate(probe)

        self.assertGreater(predicted[0], predicted[1])
        self.assertTrue(model.fit_results_["bid"].converged)
        self.assertTrue(model.fit_results_["ask"].converged)
        np.testing.assert_allclose(
            model.predict_probability(probe, [0.5, 2.0]),
            1.0 - np.exp(-predicted * np.array([0.5, 2.0])),
        )


class HazardEvaluationTests(unittest.TestCase):
    def test_competing_risk_labels_use_first_event(self):
        labels = competing_risk_labels(
            fill=[True, True, False, False],
            adverse_move=[False, True, True, False],
            cancel=[False, False, False, True],
            fill_time=[1.0, 4.0, 99.0, 99.0],
            adverse_move_time=[99.0, 2.0, 1.0, 99.0],
            cancel_time=[99.0, 99.0, 99.0, 1.0],
        )
        np.testing.assert_array_equal(labels, ["fill", "adverse_move", "adverse_move", "cancel"])

    def test_overlapping_risks_without_times_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "event-time"):
            competing_risk_labels([True], [True], [False])

    def test_non_event_times_may_be_missing(self):
        labels = competing_risk_labels(
            [True],
            [True],
            [False],
            fill_time=[1.0],
            adverse_move_time=[2.0],
            cancel_time=[np.nan],
        )
        np.testing.assert_array_equal(labels, ["fill"])

    def test_oos_metrics_cover_calibration_and_trading_risk(self):
        metrics = aggregate_oos_metrics(
            count=[0, 1, 0, 1],
            rate=[0.2, 0.8, 0.2, 0.8],
            exposure=[1.0, 1.0, 1.0, 1.0],
            markout=[1.0, -2.0],
            pnl=[1.0, -3.0, 2.0, 1.0],
            inventory=[0.0, 2.0, -1.0, 0.0],
        )
        for name in (
            "poisson_log_likelihood",
            "poisson_deviance",
            "brier_score",
            "calibration",
            "mean_markout",
            "net_pnl",
            "inventory_rms",
            "pnl_expected_shortfall",
            "max_drawdown",
        ):
            self.assertIn(name, metrics)
        self.assertEqual(metrics["net_pnl"], 1.0)
        self.assertEqual(metrics["max_absolute_inventory"], 2.0)

    def test_hawkes_gate_requires_oos_residual_dependence(self):
        count = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        rate = np.full(len(count), 0.5)
        side = np.resize(np.array(["bid", "ask"]), len(count))
        timestamp = np.repeat(
            np.arange(4).astype("timedelta64[s]") + np.datetime64("2024-01-01"),
            2,
        )
        diagnostics = residual_diagnostics(
            count,
            rate,
            np.ones(len(count)),
            side,
            timestamp=timestamp,
            max_lag=2,
            out_of_sample=True,
        )
        decision = hawkes_decision_gate(diagnostics)

        self.assertAlmostEqual(diagnostics["cross_side_residual_correlation"], 1.0)
        self.assertTrue(decision["hawkes_warranted"])
        in_sample = dict(diagnostics, out_of_sample=False)
        self.assertFalse(hawkes_decision_gate(in_sample)["hawkes_warranted"])


if __name__ == "__main__":
    unittest.main()

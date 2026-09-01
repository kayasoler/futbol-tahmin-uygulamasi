import unittest

import numpy as np

from analysis import (
    _canonical_source_url,
    market_odds_context,
    normalized_market_probabilities,
    odds_similarity_label,
    totals_odds_movement,
)

from calibration import (
    _bootstrap_improvement_probability,
    _fit_weights,
    _value_band_rows,
    _value_rows,
)


class CalibrationRobustnessTests(unittest.TestCase):
    def test_totals_movement_removes_two_way_margin(self):
        rows = totals_odds_movement(1.90, 1.90, 1.70, 2.10)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0]["Hareket"], 0)
        self.assertAlmostEqual(sum(row["Güncel olasılık"] for row in rows), 1.0)

    def test_manual_opening_odds_are_exposed_as_non_model_context(self):
        context = market_odds_context({
            "opening_b365_home": 1.90,
            "opening_b365_draw": 3.50,
            "opening_b365_away": 4.20,
            "b365_home": 1.70,
            "b365_draw": 3.80,
            "b365_away": 5.20,
            "opening_b365_over_25": 1.95,
            "opening_b365_under_25": 1.85,
            "b365_over_25": 1.80,
            "b365_under_25": 2.00,
        })
        self.assertIn("piyasa", context["note"].casefold())
        self.assertEqual(len(context["one_x_two"]["de_vigged_movement"]), 3)
        self.assertEqual(len(context["total_2_5"]["de_vigged_movement"]), 2)

    def test_empty_opening_odds_create_no_context(self):
        self.assertEqual(market_odds_context({"b365_home": 1.70}), {})
    def test_odds_similarity_uses_probability_points(self):
        probabilities = normalized_market_probabilities((1.70, 3.80, 5.20))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(odds_similarity_label(0.0149), "Çok yakın")
        self.assertEqual(odds_similarity_label(0.04), "Geniş")
        self.assertIsNone(odds_similarity_label(0.051))
    def test_canonical_source_url_removes_tracking(self):
        first = _canonical_source_url("https://Example.com/news/?utm_source=x&id=1#part")
        second = _canonical_source_url("https://example.com/news?id=1")
        self.assertEqual(first, second)

    def test_detects_consistent_improvement(self):
        actual = np.array([0, 1, 2] * 40)
        current = np.tile([0.45, 0.30, 0.25], (len(actual), 1))
        calibrated = np.eye(3)[actual] * 0.70 + 0.10

        probability = _bootstrap_improvement_probability(current, calibrated, actual)

        self.assertGreaterEqual(probability, 0.95)

    def test_rejects_consistent_regression(self):
        actual = np.array([0, 1, 2] * 40)
        current = np.eye(3)[actual] * 0.70 + 0.10
        calibrated = np.tile([0.45, 0.30, 0.25], (len(actual), 1))

        probability = _bootstrap_improvement_probability(current, calibrated, actual)

        self.assertLess(probability, 0.05)

    def test_value_selection_checks_all_outcomes(self):
        probabilities = np.array([[0.50, 0.30, 0.20]])
        odds = np.array([[1.80, 4.00, 6.00]])
        actual = np.array([1])

        rows = _value_rows("Test", probabilities, actual, odds)
        plus_three = next(row for row in rows if row["Değer eşiği"] == "+3 puan")

        self.assertEqual(plus_three["Sanal bahis"], 1)
        self.assertEqual(plus_three["Doğru"], 1)

    def test_value_selection_can_choose_away_result(self):
        probabilities = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[1.70, 3.50, 8.00]])
        actual = np.array([2])

        rows = _value_rows("Test", probabilities, actual, odds)
        plus_five = next(row for row in rows if row["Değer eşiği"] == "+5 puan")

        self.assertEqual(plus_five["Sanal bahis"], 1)
        self.assertEqual(plus_five["Doğru"], 1)

    def test_weight_fitting_returns_normalized_weights(self):
        actual = np.array([0, 1, 2] * 10)
        component = np.eye(3)[actual] * 0.70 + 0.10
        components = np.repeat(component[:, np.newaxis, :], 4, axis=1)
        training = {
            "components": components,
            "strengths": np.ones((len(actual), 4)),
            "actual": actual,
        }

        weights, temperature = _fit_weights(training)

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertIn(temperature, (0.8, 1.0, 1.2, 1.4, 1.6, 1.8))

    def test_value_band_diagnostics_separate_longshots(self):
        probabilities = np.array([[0.45, 0.30, 0.25], [0.55, 0.25, 0.20]])
        odds = np.array([[1.80, 4.00, 6.00], [1.70, 3.50, 8.00]])
        actual = np.array([1, 2])

        rows = _value_band_rows(probabilities, actual, odds)
        longshots = next(row for row in rows if row["Oran aralığı"] == "5.00+")

        self.assertEqual(longshots["Sanal bahis"], 2)
        self.assertEqual(longshots["Doğru"], 1)


if __name__ == "__main__":
    unittest.main()

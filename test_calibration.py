import unittest

import numpy as np

from calibration import _bootstrap_improvement_probability, _fit_weights, _value_rows


class CalibrationRobustnessTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

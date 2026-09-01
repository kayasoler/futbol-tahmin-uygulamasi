import unittest

import numpy as np

from calibration import _bootstrap_improvement_probability


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


if __name__ == "__main__":
    unittest.main()

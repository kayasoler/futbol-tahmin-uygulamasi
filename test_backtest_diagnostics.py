import unittest

from backtest import (
    _dixon_coles_ms_probabilities,
    dixon_coles_diagnostics,
    ms_confusion_rows,
    ms_draw_rule_diagnostics,
    ms_threshold_diagnostics,
)


def diagnostic_record(
    index: int,
    *,
    probability: float,
    margin: float,
    correct: bool,
) -> dict[str, object]:
    predicted = "1"
    actual = predicted if correct else "X"
    return {
        "date": f"2026-01-{index + 1:02d}",
        "division": "E0",
        "predicted": predicted,
        "actual": actual,
        "max_probability": probability,
        "margin": margin,
        "odds": 1.80,
        "correct": correct,
    }


class BacktestDiagnosticsTests(unittest.TestCase):
    def test_confusion_matrix_separates_actual_one_x_two(self):
        rows = ms_confusion_rows(
            [
                {"actual": "1", "predicted": "1"},
                {"actual": "1", "predicted": "X"},
                {"actual": "X", "predicted": "1"},
                {"actual": "2", "predicted": "2"},
            ]
        )

        by_actual = {row["Gerçek sonuç"]: row for row in rows}
        self.assertEqual(by_actual["1"]["Tahmin 1"], 1)
        self.assertEqual(by_actual["1"]["Tahmin X"], 1)
        self.assertEqual(by_actual["1"]["Doğru oran"], 0.5)
        self.assertEqual(by_actual["X"]["Tahmin 1"], 1)
        self.assertEqual(by_actual["2"]["Doğru oran"], 1.0)

    def test_threshold_is_chosen_on_old_matches_and_scored_on_new_matches(self):
        records = [
            diagnostic_record(index, probability=0.70, margin=0.40, correct=True)
            for index in range(6)
        ]
        records.extend(
            diagnostic_record(
                index,
                probability=0.50,
                margin=0.10,
                correct=index % 2 == 0,
            )
            for index in range(6, 14)
        )
        records.extend(
            diagnostic_record(
                index,
                probability=0.70 if index < 17 else 0.50,
                margin=0.40 if index < 17 else 0.10,
                correct=index in {14, 15, 17},
            )
            for index in range(14, 20)
        )

        result = ms_threshold_diagnostics(
            records,
            train_ratio=0.70,
            minimum_training_bets=4,
        )
        selected = result["selected"]

        self.assertEqual(result["training_count"], 14)
        self.assertEqual(result["holdout_count"], 6)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["Eğitim seçimi"], 6)
        self.assertEqual(selected["Yeni %30 seçim"], 3)
        self.assertAlmostEqual(selected["Yeni %30 doğruluk"], 2 / 3)
        self.assertAlmostEqual(selected["Yeni %30 kapsama"], 0.5)

    def test_temporal_split_keeps_matches_from_same_date_together(self):
        records = [
            {
                **diagnostic_record(
                    index,
                    probability=0.70,
                    margin=0.40,
                    correct=True,
                ),
                "date": date,
            }
            for index, date in enumerate(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                ]
            )
        ]

        result = ms_threshold_diagnostics(
            records,
            train_ratio=0.70,
            minimum_training_bets=1,
        )

        self.assertEqual(result["training_count"], 5)
        self.assertEqual(result["holdout_count"], 1)

    def test_draw_rule_is_selected_on_training_and_compared_on_holdout(self):
        records = []
        for index in range(20):
            is_close_draw = index < 5 or index in {14, 15}
            records.append(
                {
                    "date": f"2026-01-{index + 1:02d}",
                    "division": "E0",
                    "predicted": "1",
                    "actual": "X" if is_close_draw else "1",
                    "probabilities": (
                        {"1": 0.36, "X": 0.34, "2": 0.30}
                        if is_close_draw
                        else {"1": 0.60, "X": 0.25, "2": 0.15}
                    ),
                    "odds_by_result": {"1": 1.80, "X": 3.20, "2": 4.00},
                }
            )

        result = ms_draw_rule_diagnostics(
            records,
            train_ratio=0.70,
            minimum_training_overrides=4,
        )
        selected = result["selected"]

        self.assertIsNotNone(selected)
        self.assertAlmostEqual(result["baseline"]["Yeni %30 doğruluk"], 4 / 6)
        self.assertAlmostEqual(selected["Yeni %30 doğruluk"], 1.0)
        self.assertAlmostEqual(selected["Yeni %30 doğruluk farkı"], 2 / 6)
        self.assertAlmostEqual(selected["Yeni %30 X yakalama"], 1.0)
        self.assertEqual(selected["Yeni %30 X müdahale"], 2)

    def test_negative_dixon_coles_rho_increases_draw_probability(self):
        independent = _dixon_coles_ms_probabilities(1.0, 1.0, 0.0)
        corrected = _dixon_coles_ms_probabilities(1.0, 1.0, -0.10)

        self.assertIsNotNone(independent)
        self.assertIsNotNone(corrected)
        self.assertGreater(corrected["X"], independent["X"])
        self.assertAlmostEqual(sum(corrected.values()), 1.0)

    def test_dixon_coles_candidate_is_fitted_only_on_old_matches(self):
        independent = _dixon_coles_ms_probabilities(1.0, 1.0, 0.0)
        records = []
        for index in range(20):
            home_goals = index % 2
            away_goals = home_goals
            records.append(
                {
                    "date": f"2026-01-{index + 1:02d}",
                    "division": "E0",
                    "actual": "X",
                    "actual_home_goals": home_goals,
                    "actual_away_goals": away_goals,
                    "expected_home_goals": 1.0,
                    "expected_away_goals": 1.0,
                    "current_probabilities": independent,
                    "odds": {"1": 2.80, "X": 3.20, "2": 2.80},
                    "components": {
                        "Poisson + son saha formu": {
                            "probabilities": independent,
                            "weight": 1.0,
                            "sample": 10,
                        }
                    },
                }
            )

        result = dixon_coles_diagnostics(
            records,
            train_ratio=0.70,
            minimum_low_score_matches=5,
        )

        self.assertEqual(result["candidate"]["matches"], 6)
        self.assertLess(result["league_rows"][0]["Rho"], 0)
        self.assertGreater(result["differences"]["accuracy"], 0)
        self.assertLess(result["differences"]["brier"], 0)
        self.assertLess(result["differences"]["log_loss"], 0)
        self.assertTrue(result["passes"])


if __name__ == "__main__":
    unittest.main()

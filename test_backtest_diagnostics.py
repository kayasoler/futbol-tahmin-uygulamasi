import unittest

from analysis import build_report
from backtest import (
    _dixon_coles_ms_probabilities,
    dixon_coles_diagnostics,
    ms_confusion_rows,
    ms_draw_rule_diagnostics,
    ms_threshold_diagnostics,
    totals_25_market_independent_diagnostics,
    totals_25_threshold_diagnostics,
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

    def test_totals_25_threshold_is_selected_only_on_old_matches(self):
        records = []
        for index in range(20):
            high_confidence = index < 6 or 14 <= index < 17
            if index < 14:
                correct = high_confidence or index % 2 == 0
            else:
                correct = index in {14, 15, 17}
            records.append(
                {
                    "date": f"2026-01-{index + 1:02d}",
                    "division": "E0",
                    "predicted": "Üst",
                    "actual": "Üst" if correct else "Alt",
                    "over_probability": 0.70 if high_confidence else 0.52,
                    "odds": {"Üst": 1.80, "Alt": 2.10},
                }
            )

        result = totals_25_threshold_diagnostics(
            records,
            train_ratio=0.70,
            minimum_training_priced_bets=4,
        )

        self.assertEqual(result["training_count"], 14)
        self.assertEqual(result["holdout_count"], 6)
        self.assertIsNotNone(result["selected"])
        self.assertEqual(result["selected"]["Eğitim seçimi"], 6)
        self.assertEqual(result["candidate"]["matches"], 3)
        self.assertAlmostEqual(result["candidate"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["candidate"]["coverage"], 0.5)
        self.assertAlmostEqual(result["candidate"]["roi"], 0.2)
        self.assertEqual(result["market"]["matches"], 3)

    def test_totals_25_uses_probability_of_selected_under_side(self):
        records = [
            {
                "date": f"2026-01-{index + 1:02d}",
                "division": "E0",
                "predicted": "Alt",
                "actual": "Alt",
                "over_probability": 0.30,
                "odds": {"Üst": 2.20, "Alt": 1.80},
            }
            for index in range(10)
        ]

        result = totals_25_threshold_diagnostics(
            records,
            train_ratio=0.70,
            minimum_training_priced_bets=1,
        )

        self.assertIsNotNone(result["selected"])
        under_probability_row = next(
            row
            for row in result["rows"]
            if row["Olasılık eşiği"] == 0.68 and row["Değer eşiği"] == -1.0
        )
        self.assertEqual(under_probability_row["Eğitim seçimi"], 7)
        self.assertEqual(under_probability_row["Yeni %30 seçim"], 3)
        self.assertEqual(result["candidate"]["matches"], 3)
        self.assertEqual(result["candidate"]["accuracy"], 1.0)

    def test_statistical_totals_probability_does_not_change_with_market_odds(self):
        league_rows = [
            {
                "match_date": f"2025-01-{index + 1:02d}",
                "division": "E0",
                "home_team": "Home" if index % 2 == 0 else "Other",
                "away_team": "Away" if index % 2 == 0 else "Third",
                "full_time_home_goals": 2 if index % 3 else 1,
                "full_time_away_goals": 1 if index % 4 else 0,
            }
            for index in range(20)
        ]
        match = {
            "match_date": "2026-01-01",
            "division": "E0",
            "home_team": "Home",
            "away_team": "Away",
            "b365_over_25": 1.45,
            "b365_under_25": 2.80,
        }
        opposite_market = {
            **match,
            "b365_over_25": 2.80,
            "b365_under_25": 1.45,
        }

        first = build_report(match, [], [], league_rows)
        second = build_report(opposite_market, [], [], league_rows)
        first_total = first["predictions"]["totals"]["2.5"]
        second_total = second["predictions"]["totals"]["2.5"]

        self.assertAlmostEqual(
            first_total["statistical_probability"],
            second_total["statistical_probability"],
        )
        self.assertNotAlmostEqual(
            first_total["probability"],
            second_total["probability"],
        )

    def test_market_independent_candidate_uses_same_matches_for_comparison(self):
        records = []
        for index in range(20):
            high_confidence = index < 6 or 14 <= index < 17
            if index < 14:
                actual = "Üst" if high_confidence or index % 2 == 0 else "Alt"
            else:
                actual = "Üst" if index in {14, 15, 17} else "Alt"
            records.append(
                {
                    "date": f"2026-01-{index + 1:02d}",
                    "division": "E0",
                    "predicted": "Alt",
                    "over_probability": 0.45,
                    "statistical_predicted": "Üst",
                    "statistical_over_probability": (
                        0.70 if high_confidence else 0.52
                    ),
                    "actual": actual,
                    "odds": {"Üst": 2.20, "Alt": 1.70},
                }
            )

        result = totals_25_market_independent_diagnostics(
            records,
            train_ratio=0.70,
            minimum_training_priced_bets=4,
        )

        self.assertEqual(result["training_count"], 14)
        self.assertEqual(result["holdout_count"], 6)
        self.assertIsNotNone(result["selected"])
        self.assertEqual(result["candidate_shadow"]["matches"], 3)
        self.assertEqual(result["candidate_current"]["matches"], 3)
        self.assertEqual(result["candidate_market"]["matches"], 3)
        self.assertGreater(
            result["candidate_shadow"]["roi"],
            result["candidate_current"]["roi"],
        )
        self.assertGreater(
            result["candidate_shadow"]["accuracy"],
            result["candidate_market"]["accuracy"],
        )


if __name__ == "__main__":
    unittest.main()

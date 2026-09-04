import unittest

from analysis_store import (
    analysis_match_key,
    compact_report,
    evaluate_analysis,
    match_snapshot,
    restore_report_snapshot,
)


class AnalysisStoreTests(unittest.TestCase):
    def test_match_key_is_stable_and_team_specific(self):
        base = {"division": "E0", "match_date": "2026-09-05", "home_team": "A", "away_team": "B"}
        self.assertEqual(analysis_match_key(base), analysis_match_key(dict(base)))
        changed = dict(base, away_team="C")
        self.assertNotEqual(analysis_match_key(base), analysis_match_key(changed))

    def test_compact_report_excludes_historical_rows(self):
        compact = compact_report({"predictions": {"ms": "1"}, "h2h": [{"large": "row"}]})
        self.assertEqual(compact["predictions"]["ms"], "1")
        self.assertNotIn("h2h", compact)

    def test_compact_report_keeps_bounded_display_evidence(self):
        evidence = {
            "h2h_rows": [{"home_team": "A", "away_team": "B"}],
            "same_league": {"sample_count": 40, "rows": [{"match_date": "2026-01-01"}]},
        }

        compact = compact_report({"predictions": {"ms": "MS 1"}, "evidence": evidence})

        self.assertEqual(compact["snapshot_version"], 2)
        self.assertEqual(compact["evidence"], evidence)

    def test_restore_report_snapshot_requires_predictions(self):
        self.assertIsNone(restore_report_snapshot({"report_snapshot": {"warnings": []}}))
        self.assertIsNone(restore_report_snapshot({"report_snapshot": {"predictions": {}}}))

        restored = restore_report_snapshot({
            "report_snapshot": {
                "snapshot_version": 2,
                "predictions": {"ms": "MS X"},
                "evidence": {"h2h_rows": []},
            }
        })

        self.assertEqual(restored["predictions"]["ms"], "MS X")
        self.assertEqual(restored["evidence"]["h2h_rows"], [])

    def test_snapshot_keeps_manual_and_csv_odds_separate(self):
        snapshot = match_snapshot({"b365_home": 1.70, "csv_b365_home": 1.82})
        self.assertEqual(snapshot["b365_home"], 1.70)
        self.assertEqual(snapshot["csv_b365_home"], 1.82)

    def test_evaluates_stored_prediction_against_score(self):
        analysis = {"report_snapshot": {"predictions": {
            "ms": "MS 1", "score": "2-1", "btts_prediction": "KG Var",
            "ht_ms": "İY X / MS 1", "totals": {"2.5": {"prediction": "Üst"}},
        }}}
        result = {
            "full_time_home": 2, "full_time_away": 1,
            "half_time_home": 0, "half_time_away": 0,
        }

        rows = evaluate_analysis(analysis, result)

        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["Doğru"] for row in rows))


if __name__ == "__main__":
    unittest.main()

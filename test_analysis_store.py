import unittest

from analysis_store import analysis_match_key, compact_report, match_snapshot


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

    def test_snapshot_keeps_manual_and_csv_odds_separate(self):
        snapshot = match_snapshot({"b365_home": 1.70, "csv_b365_home": 1.82})
        self.assertEqual(snapshot["b365_home"], 1.70)
        self.assertEqual(snapshot["csv_b365_home"], 1.82)


if __name__ == "__main__":
    unittest.main()

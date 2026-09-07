import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from analysis_store import (
    analysis_match_key,
    compact_report,
    evaluate_analysis,
    latest_analysis_versions,
    match_snapshot,
    pending_manual_result_analyses,
    restore_report_snapshot,
    save_match_result,
)


class AnalysisStoreTests(unittest.TestCase):
    def test_match_key_is_stable_and_team_specific(self):
        base = {"division": "E0", "match_date": "2026-09-05", "home_team": "A", "away_team": "B"}
        self.assertEqual(analysis_match_key(base), analysis_match_key(dict(base)))
        changed = dict(base, away_team="C")
        self.assertNotEqual(analysis_match_key(base), analysis_match_key(changed))

    def test_compact_report_excludes_historical_rows(self):
        compact = compact_report({
            "model_revision": "stat-v1",
            "predictions": {"ms": "1"},
            "evidence": {"h2h_rows": [{"bounded": "row"}]},
            "h2h": [{"large": "row"}],
        })
        self.assertEqual(compact["predictions"]["ms"], "1")
        self.assertEqual(compact["model_revision"], "stat-v1")
        self.assertEqual(compact["snapshot_version"], 2)
        self.assertEqual(compact["evidence"]["h2h_rows"][0]["bounded"], "row")
        self.assertNotIn("h2h", compact)

    def test_restore_report_snapshot_rejects_missing_predictions(self):
        self.assertIsNone(restore_report_snapshot(None))
        self.assertIsNone(restore_report_snapshot({"report_snapshot": {"warnings": []}}))
        restored = restore_report_snapshot({
            "report_snapshot": {"predictions": {"ms": "MS 1"}}
        })
        self.assertEqual(restored["predictions"]["ms"], "MS 1")

    def test_snapshot_keeps_manual_and_csv_odds_separate(self):
        snapshot = match_snapshot({
            "b365_home": 1.70, "csv_b365_home": 1.82, "entry_method": "manual"
        })
        self.assertEqual(snapshot["b365_home"], 1.70)
        self.assertEqual(snapshot["csv_b365_home"], 1.82)
        self.assertEqual(snapshot["entry_method"], "manual")

    def test_snapshot_keeps_api_football_traceability(self):
        snapshot = match_snapshot({
            "entry_method": "api-football",
            "api_fixture_id": 42,
            "league_id": 39,
            "league": "Premier League",
            "country": "England",
        })
        self.assertEqual(snapshot["api_fixture_id"], 42)
        self.assertEqual(snapshot["league_id"], 39)

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

    def test_daily_loader_keeps_latest_version_per_match(self):
        rows = [
            {"match_key": "a", "version": 1, "kickoff_time": "18:00"},
            {"match_key": "a", "version": 2, "kickoff_time": "18:00"},
            {"match_key": "b", "version": 1, "kickoff_time": "20:00"},
        ]
        latest = latest_analysis_versions(rows)
        self.assertEqual([(row["match_key"], row["version"]) for row in latest], [("a", 2), ("b", 1)])

    def test_manual_result_list_contains_only_finished_matches_without_results(self):
        analyses = [
            {"match_key": "pending", "match_date": "2026-09-07", "kickoff_time": "18:00:00"},
            {"match_key": "resolved", "match_date": "2026-09-07", "kickoff_time": "17:00:00"},
            {"match_key": "playing", "match_date": "2026-09-07", "kickoff_time": "21:00:00"},
            {"match_key": "future", "match_date": "2026-09-08", "kickoff_time": "18:00:00"},
            {"match_key": "invalid", "match_date": "", "kickoff_time": ""},
        ]
        results = {"resolved": {"full_time_home": 2, "full_time_away": 1}}

        pending = pending_manual_result_analyses(
            analyses,
            results,
            datetime(2026, 9, 7, 22, 0, tzinfo=ZoneInfo("Europe/Istanbul")),
        )

        self.assertEqual([row["match_key"] for row in pending], ["pending"])

    def test_manual_result_requires_both_half_time_scores(self):
        error = save_match_result(
            None,
            {"match_key": "a"},
            full_time_home=2,
            full_time_away=1,
            half_time_home=1,
        )

        self.assertEqual(error, "İlk yarı skorunun iki takım için de girilmesi gerekir.")

    def test_half_time_score_cannot_exceed_full_time_score(self):
        error = save_match_result(
            None,
            {"match_key": "a"},
            full_time_home=1,
            full_time_away=1,
            half_time_home=2,
            half_time_away=0,
        )

        self.assertEqual(error, "İlk yarı skoru maç sonu skorundan büyük olamaz.")


if __name__ == "__main__":
    unittest.main()

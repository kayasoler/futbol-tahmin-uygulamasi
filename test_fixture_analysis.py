import unittest
from unittest.mock import patch

from fixture_analysis import (
    analyze_world_fixture,
    prediction_summary,
    prepare_world_fixture_match,
    world_fixture_table_row,
)


class FixtureAnalysisTests(unittest.TestCase):
    def test_prepares_fixture_with_historical_team_names(self):
        fixture = {
            "api_fixture_id": 42,
            "match_date": "2026-09-06",
            "kickoff_time": "20:00:00",
            "home_team": "Arsenal FC",
            "away_team": "Chelsea FC",
            "status": "NS",
        }
        league_rows = [{"home_team": "Arsenal", "away_team": "Chelsea"}]

        match, error = prepare_world_fixture_match(fixture, "E0", league_rows)

        self.assertIsNone(error)
        self.assertEqual(match["home_team"], "Arsenal")
        self.assertEqual(match["away_team"], "Chelsea")
        self.assertEqual(match["entry_method"], "api-football-batch")

    def test_prediction_summary_contains_requested_table_markets(self):
        summary = prediction_summary({"predictions": {
            "ms": "MS 1",
            "btts_prediction": "KG Var",
            "totals": {"2.5": {"prediction": "Üst"}},
            "score": "2-1",
            "confidence": "Orta",
        }})

        self.assertEqual(summary, {
            "ms": "MS 1", "btts": "KG Var", "total_25": "Üst",
            "score": "2-1", "confidence": "Orta",
        })

    @patch("fixture_analysis.fetch_league_rows")
    def test_unsupported_league_is_listed_without_database_reads(self, fetch_league_rows):
        fixture = {"league_id": 999999, "status": "NS", "home_team": "A", "away_team": "B"}

        outcome = analyze_world_fixture(object(), fixture, {"E0"}, {})
        row = world_fixture_table_row(outcome)

        self.assertEqual(outcome["status"], "Desteklenmiyor")
        self.assertEqual(row["Analiz durumu"], "Desteklenmiyor")
        fetch_league_rows.assert_not_called()

    @patch("fixture_analysis.fetch_league_rows")
    def test_started_fixture_is_not_analyzed(self, fetch_league_rows):
        fixture = {"league_id": 39, "status": "1H", "home_team": "Arsenal", "away_team": "Chelsea"}

        outcome = analyze_world_fixture(object(), fixture, {"E0"}, {})

        self.assertEqual(outcome["status"], "Maç başlamış")
        fetch_league_rows.assert_not_called()

    @patch("fixture_analysis.restore_report_snapshot")
    @patch("fixture_analysis.load_latest_analysis")
    @patch("fixture_analysis.fetch_league_rows")
    def test_existing_snapshot_is_reused_without_recalculation(
        self, fetch_league_rows, load_latest_analysis, restore_report_snapshot
    ):
        fixture = {
            "api_fixture_id": 1, "league_id": 39, "status": "NS",
            "match_date": "2026-09-06", "kickoff_time": "20:00:00",
            "home_team": "Arsenal", "away_team": "Chelsea",
        }
        fetch_league_rows.return_value = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        load_latest_analysis.return_value = ({"id": 7, "version": 1}, None)
        restore_report_snapshot.return_value = {"predictions": {"ms": "MS 1"}}

        with patch("fixture_analysis.build_report") as build_report:
            outcome = analyze_world_fixture(object(), fixture, {"E0"}, {})

        self.assertEqual(outcome["status"], "Kayıtlı analiz")
        build_report.assert_not_called()

    @patch("fixture_analysis.save_analysis_version")
    @patch("fixture_analysis.build_analysis_evidence", return_value={"h2h_rows": []})
    @patch("fixture_analysis.build_report")
    @patch("fixture_analysis.fetch_team_form_rows", return_value=[])
    @patch("fixture_analysis.fetch_same_odds_rows", return_value=[])
    @patch("fixture_analysis.fetch_h2h_rows", return_value=[])
    @patch("fixture_analysis.load_latest_analysis", return_value=(None, None))
    @patch("fixture_analysis.fetch_league_rows")
    def test_new_fixture_is_analyzed_and_saved(
        self,
        fetch_league_rows,
        _load_latest,
        _fetch_h2h,
        _fetch_same_odds,
        _fetch_form,
        build_report,
        _build_evidence,
        save_analysis_version,
    ):
        fixture = {
            "api_fixture_id": 1, "league_id": 39, "status": "NS",
            "match_date": "2026-09-06", "kickoff_time": "20:00:00",
            "home_team": "Arsenal", "away_team": "Chelsea",
        }
        fetch_league_rows.return_value = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        build_report.return_value = {
            "predictions": {
                "ms": "MS 1", "btts_prediction": "KG Var", "score": "2-1",
                "confidence": "Orta", "totals": {"2.5": {"prediction": "Üst"}},
            }
        }
        save_analysis_version.return_value = ({"id": 8, "version": 1}, None)

        outcome = analyze_world_fixture(object(), fixture, {"E0"}, {})

        self.assertEqual(outcome["status"], "Yeni kaydedildi")
        save_analysis_version.assert_called_once()
        self.assertEqual(outcome["report"]["evidence"], {"h2h_rows": []})


if __name__ == "__main__":
    unittest.main()

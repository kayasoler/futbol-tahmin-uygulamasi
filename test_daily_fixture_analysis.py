import unittest
from unittest.mock import patch

from daily_fixture_analysis import (
    analyze_daily_fixture,
    fixture_table_row,
    normalize_upcoming_fixture,
    prepare_fixture_match,
    resolve_fixture_duplicates,
)


class DailyFixtureAnalysisTests(unittest.TestCase):
    def test_manual_duplicate_wins_over_football_data(self):
        common = {
            "division": "E0", "match_date": "2026-09-06",
            "home_team": "Arsenal", "away_team": "Chelsea",
        }
        football_data = dict(common, id="fd-1", entry_method="football-data-live", b365_home=1.80)
        manual = dict(common, id=9, entry_method="manual", b365_home=1.65)

        winners, dropped = resolve_fixture_duplicates([football_data, manual])

        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["entry_method"], "manual")
        self.assertEqual(winners[0]["b365_home"], 1.65)
        self.assertEqual(dropped[0]["entry_method"], "football-data-live")

    def test_sources_remain_separate_when_matches_are_different(self):
        fixtures = [
            {"division": "E0", "match_date": "2026-09-06", "home_team": "A", "away_team": "B", "entry_method": "manual"},
            {"division": "E0", "match_date": "2026-09-06", "home_team": "C", "away_team": "D", "entry_method": "csv"},
            {"division": "E0", "match_date": "2026-09-06", "home_team": "E", "away_team": "F", "entry_method": "football-data-live"},
        ]
        winners, _ = resolve_fixture_duplicates(fixtures)
        self.assertEqual({row["entry_method"] for row in winners}, {"manual", "csv", "football-data-live"})

    def test_normalizes_opening_odds_from_manual_raw_data(self):
        fixture = normalize_upcoming_fixture({
            "entry_method": "manual",
            "raw_data": {"opening_odds": {"opening_b365_home": 2.10}},
        })
        self.assertEqual(fixture["opening_b365_home"], 2.10)
        self.assertEqual(fixture["analysis_odds_source"], "Manuel analiz anı oranı")

    def test_missing_historical_team_is_unanalyzable(self):
        fixture = {"home_team": "Unknown", "away_team": "Chelsea", "match_status": "NS"}
        match, reason = prepare_fixture_match(
            fixture, [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        )
        self.assertIsNone(match)
        self.assertIn("Unknown", reason)

    @patch("daily_fixture_analysis.fetch_league_rows", side_effect=RuntimeError("database unavailable"))
    def test_one_match_failure_is_returned_as_an_outcome(self, fetch_league_rows):
        fixture = {
            "division": "E0", "match_date": "2026-09-06",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "entry_method": "football-data-live", "match_status": "NS",
        }
        outcome = analyze_daily_fixture(object(), fixture, {})
        self.assertEqual(outcome["status"], "Analiz edilemedi")
        self.assertIn("database unavailable", outcome["reason"])

    @patch("daily_fixture_analysis.restore_report_snapshot")
    @patch("daily_fixture_analysis.load_latest_analysis")
    @patch("daily_fixture_analysis.fetch_league_rows")
    def test_existing_snapshot_is_reused_without_recalculation(
        self, fetch_league_rows, load_latest_analysis, restore_report_snapshot
    ):
        fixture = {
            "id": "fd-1", "division": "E0", "match_date": "2026-09-06",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "entry_method": "football-data-live", "match_status": "NS",
        }
        fetch_league_rows.return_value = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        load_latest_analysis.return_value = ({"id": 7, "version": 1}, None)
        restore_report_snapshot.return_value = {
            "predictions": {"ms": "MS 1", "ms_probabilities": {"1": 0.55}}
        }
        with patch("daily_fixture_analysis.build_report") as build_report:
            outcome = analyze_daily_fixture(object(), fixture, {})
        self.assertEqual(outcome["status"], "Kayıtlı analiz")
        build_report.assert_not_called()
        self.assertEqual(fixture_table_row(outcome)["Olasılık"], "%55.0")

    @patch("daily_fixture_analysis.save_analysis_version")
    @patch("daily_fixture_analysis.build_analysis_evidence", return_value={"h2h_rows": []})
    @patch("daily_fixture_analysis.build_report")
    @patch("daily_fixture_analysis.fetch_team_form_rows", return_value=[])
    @patch("daily_fixture_analysis.fetch_same_odds_rows", return_value=[])
    @patch("daily_fixture_analysis.fetch_h2h_rows", return_value=[])
    @patch("daily_fixture_analysis.fetch_league_rows")
    def test_explicit_refresh_creates_new_version_without_loading_latest(
        self, fetch_league_rows, fetch_h2h, fetch_same, fetch_form,
        build_report, build_evidence, save_analysis_version,
    ):
        fixture = {
            "id": 4, "division": "E0", "match_date": "2026-09-06",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "entry_method": "manual", "match_status": "NS",
        }
        fetch_league_rows.return_value = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        build_report.return_value = {"predictions": {"ms": "MS 1"}}
        save_analysis_version.return_value = ({"id": 10, "version": 2}, None)
        with patch("daily_fixture_analysis.load_latest_analysis") as load_latest:
            outcome = analyze_daily_fixture(object(), fixture, {}, force_refresh=True)
        load_latest.assert_not_called()
        save_analysis_version.assert_called_once()
        self.assertEqual(outcome["status"], "Yenilendi")


if __name__ == "__main__":
    unittest.main()

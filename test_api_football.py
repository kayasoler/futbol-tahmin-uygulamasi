import unittest
from unittest.mock import patch

from api_football import (
    fetch_bet365_odds_for_date,
    match_final_result,
    normalize_api_keys,
    normalize_bet365_odds,
    normalize_bet365_odds_by_fixture,
    normalize_final_result,
    normalize_fixture,
)


class ApiFootballTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_api_keys(self):
        self.assertEqual(normalize_api_keys(" first,second;first "), ["first", "second"])

    def test_normalizes_fixture_for_analysis(self):
        row = normalize_fixture(
            {
                "fixture": {
                    "id": 42,
                    "date": "2026-09-01T20:45:00+03:00",
                    "status": {"short": "NS", "long": "Not Started"},
                    "venue": {"name": "Test Arena", "city": "Izmir"},
                },
                "league": {"id": 39, "name": "Premier League", "country": "England"},
                "teams": {
                    "home": {"id": 1, "name": "Home FC"},
                    "away": {"id": 2, "name": "Away FC"},
                },
            }
        )

        self.assertEqual(row["api_fixture_id"], 42)
        self.assertEqual(row["match_date"], "2026-09-01")
        self.assertEqual(row["kickoff_time"], "20:45:00")
        self.assertEqual(row["home_team"], "Home FC")
        self.assertEqual(row["away_team"], "Away FC")
        self.assertEqual(row["entry_method"], "api-football")
        self.assertEqual(row["match_status"], "NS")

    def test_extracts_bet365_match_winner(self):
        odds = normalize_bet365_odds([{
            "update": "2026-09-01T12:00:00Z",
            "bookmakers": [{"id": 8, "name": "Bet365", "bets": [{
                "name": "Match Winner", "values": [
                    {"value": "Home", "odd": "1.70"},
                    {"value": "Draw", "odd": "3.80"},
                    {"value": "Away", "odd": "5.20"},
                ]
            }]}]
        }])
        self.assertEqual(odds["b365_home"], 1.70)
        self.assertEqual(odds["b365_away"], 5.20)

    def test_indexes_date_odds_by_fixture(self):
        response = [{
            "fixture": {"id": 77},
            "bookmakers": [{"id": 8, "name": "Bet365", "bets": [{
                "name": "Match Winner", "values": [
                    {"value": "Home", "odd": "1.80"},
                    {"value": "Draw", "odd": "3.40"},
                    {"value": "Away", "odd": "4.20"},
                ],
            }]}],
        }]
        indexed = normalize_bet365_odds_by_fixture(response)
        self.assertEqual(indexed["77"]["b365_draw"], 3.40)

    @patch("api_football._get")
    def test_fetches_all_bounded_odds_pages(self, api_get):
        api_get.side_effect = [
            {"response": [], "quota": {"daily_remaining": "98"}, "paging": {"total": 2}},
            {"response": [], "quota": {"daily_remaining": "97"}, "paging": {"total": 2}},
        ]
        result = fetch_bet365_odds_for_date(("key",), "2026-09-06")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(api_get.call_count, 2)

    def test_rejects_live_api_football_score(self):
        item = {
            "fixture": {"id": 5, "date": "2026-09-06T20:00:00+03:00", "status": {"short": "2H"}},
            "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
            "goals": {"home": 2, "away": 1},
            "score": {"fulltime": {"home": None, "away": None}},
        }
        self.assertIsNone(normalize_final_result(item))

    def test_accepts_only_regular_time_score_from_final_fixture(self):
        item = {
            "fixture": {"id": 5, "date": "2026-09-06T20:00:00+03:00", "status": {"short": "PEN"}},
            "teams": {"home": {"name": "Arsenal FC"}, "away": {"name": "Chelsea"}},
            "goals": {"home": 5, "away": 4},
            "score": {
                "halftime": {"home": 0, "away": 0},
                "fulltime": {"home": 1, "away": 1},
                "penalty": {"home": 4, "away": 3},
            },
        }
        result = normalize_final_result(item)
        self.assertEqual((result["full_time_home"], result["full_time_away"]), (1, 1))
        self.assertEqual(result["half_time_home"], 0)

    def test_matches_api_result_with_common_team_suffix(self):
        result = match_final_result(
            [{
                "match_date": "2026-09-06", "home_team": "Arsenal FC",
                "away_team": "Chelsea FC", "full_time_home": 2, "full_time_away": 0,
            }],
            {"match_date": "2026-09-06", "home_team": "Arsenal", "away_team": "Chelsea"},
        )
        self.assertEqual(result["full_time_home"], 2)

    def test_matches_api_result_with_provider_alias(self):
        result = match_final_result(
            [{
                "match_date": "2026-09-06", "home_team": "Wolverhampton Wanderers",
                "away_team": "Chelsea FC", "full_time_home": 1, "full_time_away": 1,
            }],
            {"match_date": "2026-09-06", "home_team": "Wolves", "away_team": "Chelsea"},
        )
        self.assertEqual(result["full_time_away"], 1)


if __name__ == "__main__":
    unittest.main()

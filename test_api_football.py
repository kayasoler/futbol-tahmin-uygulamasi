import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from api_football import (
    fetch_match_context,
    fetch_match_reference,
    fetch_bet365_odds_for_date,
    match_fixture,
    match_final_result,
    normalize_api_keys,
    normalize_bet365_odds,
    normalize_bet365_odds_by_fixture,
    normalize_final_result,
    normalize_fixture,
    normalize_injuries,
    normalize_lineups,
)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class ApiFootballTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_api_keys(self):
        self.assertEqual(normalize_api_keys(" first,second;first "), ["first", "second"])

    @patch("api_football.urlopen")
    def test_uses_second_key_after_first_connection_failure(self, opener):
        opener.side_effect = [
            URLError("temporary block"),
            FakeResponse(b'{"errors": [], "response": [], "paging": {}}'),
        ]
        result = fetch_bet365_odds_for_date(
            ("first-key", "second-key"), "2026-09-06", max_pages=1
        )
        self.assertEqual(result["quota"]["key_number"], 2)
        self.assertEqual(opener.call_count, 2)

    @patch("api_football.urlopen")
    def test_uses_second_key_after_first_invalid_response(self, opener):
        opener.side_effect = [
            FakeResponse(b'not-json'),
            FakeResponse(b'{"errors": [], "response": [], "paging": {}}'),
        ]
        result = fetch_bet365_odds_for_date(
            ("first-key", "second-key"), "2026-09-06", max_pages=1
        )
        self.assertEqual(result["quota"]["key_number"], 2)
        self.assertEqual(opener.call_count, 2)

    @patch("api_football.urlopen")
    def test_uses_second_key_after_first_temporary_server_error(self, opener):
        opener.side_effect = [
            HTTPError(
                "https://example.test", 503, "unavailable", {}, BytesIO(b'temporary')
            ),
            FakeResponse(b'{"errors": [], "response": [], "paging": {}}'),
        ]
        result = fetch_bet365_odds_for_date(
            ("first-key", "second-key"), "2026-09-06", max_pages=1
        )
        self.assertEqual(result["quota"]["key_number"], 2)
        self.assertEqual(opener.call_count, 2)

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

    def test_matches_selected_fixture_with_provider_suffixes(self):
        match = match_fixture(
            [{"home_team": "Arsenal FC", "away_team": "Chelsea FC", "api_fixture_id": 42}],
            "Arsenal",
            "Chelsea",
        )
        self.assertEqual(match["api_fixture_id"], 42)

    def test_normalizes_player_status_and_lineups(self):
        injuries = normalize_injuries([{
            "team": {"name": "Arsenal"},
            "player": {"name": "Test Player", "type": "Missing Fixture", "reason": "Injury"},
        }])
        lineups = normalize_lineups([{
            "team": {"id": 1, "name": "Arsenal"}, "formation": "4-3-3",
            "startXI": [{"player": {"name": "Starter"}}],
            "substitutes": [{"player": {"name": "Sub"}}],
        }])
        self.assertEqual(injuries[0]["Oyuncu"], "Test Player")
        self.assertEqual(lineups[0]["starting"], ["Starter"])

    @patch("api_football._get")
    def test_fetches_context_only_for_selected_match(self, api_get):
        api_get.side_effect = [
            {"response": [{
                "fixture": {"id": 42, "date": "2026-09-06T20:00:00+03:00", "status": {"short": "NS"}},
                "league": {"name": "Premier League"},
                "teams": {"home": {"id": 1, "name": "Arsenal FC"}, "away": {"id": 2, "name": "Chelsea FC"}},
            }], "quota": {}, "paging": {}},
            {"response": [], "quota": {}, "paging": {}},
            {"response": [], "quota": {}, "paging": {}},
            {"response": [], "quota": {}, "paging": {}},
        ]
        context = fetch_match_context(("key",), "2026-09-06", "Arsenal", "Chelsea")
        self.assertEqual(context["match"]["api_fixture_id"], 42)
        self.assertEqual(api_get.call_count, 4)

    @patch("api_football._get")
    def test_lineup_reference_uses_only_one_fixture_request(self, api_get):
        api_get.return_value = {
            "response": [{
                "fixture": {"id": 42, "date": "2026-09-06T20:00:00+03:00", "status": {"short": "NS"}},
                "league": {"name": "Premier League"},
                "teams": {"home": {"id": 1, "name": "Arsenal FC"}, "away": {"id": 2, "name": "Chelsea FC"}},
            }],
            "quota": {},
            "paging": {},
        }
        reference = fetch_match_reference(("key",), "2026-09-06", "Arsenal", "Chelsea")
        self.assertEqual(reference["match"]["api_fixture_id"], 42)
        api_get.assert_called_once()

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

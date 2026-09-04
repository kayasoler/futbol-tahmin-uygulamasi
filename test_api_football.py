import unittest

from api_football import normalize_api_keys, normalize_bet365_odds, normalize_fixture


class ApiFootballTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_api_keys(self):
        self.assertEqual(normalize_api_keys(" first,second;first "), ["first", "second"])

    def test_normalizes_toml_table_and_nested_key_objects(self):
        configured = {
            "primary": {"token": "first"},
            "backup": {"api_key": "second"},
        }

        self.assertEqual(normalize_api_keys(configured), ["first", "second"])

    def test_rejects_field_names_and_api_error_payloads(self):
        configured = [
            "token",
            {"token": "Error/Missing application key. Go to documentation."},
            " valid-key ",
        ]

        self.assertEqual(normalize_api_keys(configured), ["valid-key"])

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


if __name__ == "__main__":
    unittest.main()

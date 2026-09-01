import unittest

from api_football import normalize_api_keys, normalize_fixture


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


if __name__ == "__main__":
    unittest.main()

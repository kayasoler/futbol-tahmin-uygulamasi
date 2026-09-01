import unittest

from football_data_live import parse_fixtures_csv
from highlightly import find_match


class LiveSourceTests(unittest.TestCase):
    def test_parses_football_data_fixture(self):
        rows = parse_fixtures_csv(b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,02/09/2026,20:00,West Ham,Wolves,1.70,3.80,5.20\n")
        self.assertEqual(rows[0]["division"], "E0")
        self.assertEqual(rows[0]["b365_home"], 1.70)

    def test_matches_highlightly_teams(self):
        match = find_match({"data": [{"id": 1, "homeTeam": {"name": "West Ham United"}, "awayTeam": {"name": "Wolverhampton Wanderers"}}]}, "West Ham", "Wolverhampton")
        self.assertEqual(match["id"], 1)


if __name__ == "__main__":
    unittest.main()

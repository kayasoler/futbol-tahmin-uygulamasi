import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from football_data_live import parse_fixtures_csv
from highlightly import REQUEST_HEADERS, find_match


class LiveSourceTests(unittest.TestCase):
    def test_highlightly_request_has_browser_signature(self):
        self.assertIn("Mozilla/5.0", REQUEST_HEADERS["User-Agent"])
        self.assertEqual(REQUEST_HEADERS["Accept"], "application/json, text/plain, */*")
    def test_parses_football_data_fixture(self):
        rows = parse_fixtures_csv(
            b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,02/09/2026,20:00,West Ham,Wolves,1.70,3.80,5.20\n",
            datetime(2026, 9, 2, 12, 0, tzinfo=ZoneInfo("Europe/Istanbul")),
        )
        self.assertEqual(rows[0]["division"], "E0")
        self.assertEqual(rows[0]["b365_home"], 1.70)
        self.assertEqual(rows[0]["kickoff_time"], "22:00:00")

    def test_removes_finished_and_started_fixtures(self):
        rows = parse_fixtures_csv(
            b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,FTHG,FTAG,FTR\nE0,02/09/2026,10:00,A,B,1.7,3.8,5.2,,,\nE0,02/09/2026,20:00,C,D,1.8,3.5,4.8,2,1,H\nE0,02/09/2026,21:00,E,F,1.9,3.4,4.2,,,\n",
            datetime(2026, 9, 2, 18, 0, tzinfo=ZoneInfo("Europe/Istanbul")),
        )
        self.assertEqual([row["home_team"] for row in rows], ["E"])

    def test_matches_highlightly_teams(self):
        match = find_match({"data": [{"id": 1, "homeTeam": {"name": "West Ham United"}, "awayTeam": {"name": "Wolverhampton Wanderers"}}]}, "West Ham", "Wolverhampton")
        self.assertEqual(match["id"], 1)


if __name__ == "__main__":
    unittest.main()

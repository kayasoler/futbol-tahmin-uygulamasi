import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from zoneinfo import ZoneInfo

from football_data_live import (
    fetch_current_fixtures,
    parse_fixtures_csv,
    parse_uploaded_fixtures,
)
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

    def test_uploaded_fixture_is_kept_as_csv_source(self):
        rows = parse_uploaded_fixtures(
            b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,02/09/2026,20:00,A,B,1.70,3.80,5.20\n",
            datetime(2026, 9, 2, 12, 0, tzinfo=ZoneInfo("Europe/Istanbul")),
        )
        self.assertEqual(rows[0]["entry_method"], "csv")
        self.assertTrue(rows[0]["id"].startswith("upload-fd-"))

    @patch("football_data_live.time.sleep")
    @patch("football_data_live.urlopen")
    def test_retries_transient_source_failure(self, urlopen, sleep):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,09/09/2026,20:00,A,B,1.70,3.80,5.20\n"
        )
        urlopen.side_effect = [URLError("temporary"), response]

        rows = fetch_current_fixtures(attempts=3, retry_delay=0)

        self.assertEqual(len(rows), 1)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_not_called()

    @patch("football_data_live.time.sleep")
    @patch("football_data_live.urlopen", side_effect=URLError("offline"))
    def test_reports_failure_after_all_retries(self, urlopen, sleep):
        with self.assertRaisesRegex(RuntimeError, "Football-Data bağlantı hatası"):
            fetch_current_fixtures(attempts=3, retry_delay=0)
        self.assertEqual(urlopen.call_count, 3)

    def test_matches_highlightly_teams(self):
        match = find_match({"data": [{"id": 1, "homeTeam": {"name": "West Ham United"}, "awayTeam": {"name": "Wolverhampton Wanderers"}}]}, "West Ham", "Wolverhampton")
        self.assertEqual(match["id"], 1)


if __name__ == "__main__":
    unittest.main()

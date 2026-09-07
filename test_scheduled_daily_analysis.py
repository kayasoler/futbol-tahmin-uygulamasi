import os
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from jobs.daily_analysis import create_job_client, run_daily_analysis, select_daily_fixtures


ISTANBUL = ZoneInfo("Europe/Istanbul")


def fixture(
    home: str,
    away: str,
    *,
    source: str = "football-data-live",
    kickoff: str = "20:00:00",
    match_date: str = "2026-09-08",
    odd: float = 1.80,
):
    return {
        "division": "E0",
        "match_date": match_date,
        "kickoff_time": kickoff,
        "home_team": home,
        "away_team": away,
        "entry_method": source,
        "match_status": "NS",
        "b365_home": odd,
    }


class ScheduledDailyAnalysisTests(unittest.TestCase):
    def test_selects_only_today_future_kickoffs_and_keeps_manual_priority(self):
        live = [
            fixture("Arsenal", "Chelsea", kickoff="09:00:00"),
            fixture("Liverpool", "Everton", kickoff="20:00:00"),
            fixture("Tomorrow", "Match", match_date="2026-09-09"),
        ]
        stored = [
            fixture(
                "Liverpool", "Everton", source="manual", kickoff="20:00:00", odd=1.60
            )
        ]

        selected, duplicates = select_daily_fixtures(
            live,
            stored,
            match_date="2026-09-08",
            now=datetime(2026, 9, 8, 10, 0, tzinfo=ISTANBUL),
        )

        self.assertEqual(duplicates, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["entry_method"], "manual")
        self.assertEqual(selected[0]["b365_home"], 1.60)

    def test_job_creates_missing_and_reuses_existing_without_refresh(self):
        live = [fixture("Arsenal", "Chelsea"), fixture("Liverpool", "Everton")]
        analyzer_calls = []

        def analyzer(client, row, cache, *, force_refresh):
            analyzer_calls.append((row["home_team"], force_refresh, cache))
            return {
                "fixture": row,
                "status": (
                    "Kayıtlı analiz" if row["home_team"] == "Arsenal" else "Yeni kaydedildi"
                ),
                "reason": "",
            }

        summary = run_daily_analysis(
            object(),
            now=datetime(2026, 9, 8, 10, 0, tzinfo=ISTANBUL),
            fixture_loader=lambda: live,
            stored_loader=lambda client, match_date: [],
            analyzer=analyzer,
        )

        self.assertEqual(summary["selected_matches"], 2)
        self.assertEqual(summary["created"], 1)
        self.assertEqual(summary["reused"], 1)
        self.assertEqual(summary["unanalyzable"], 0)
        self.assertTrue(all(force_refresh is False for _, force_refresh, _ in analyzer_calls))
        self.assertIs(analyzer_calls[0][2], analyzer_calls[1][2])

    def test_job_reports_unanalyzable_match_without_aborting_batch(self):
        live = [fixture("Unknown", "Chelsea"), fixture("Liverpool", "Everton")]

        def analyzer(client, row, cache, *, force_refresh):
            if row["home_team"] == "Unknown":
                return {"fixture": row, "status": "Analiz edilemedi", "reason": "Takım yok"}
            return {"fixture": row, "status": "Yeni kaydedildi", "reason": ""}

        summary = run_daily_analysis(
            object(),
            now=datetime(2026, 9, 8, 10, 0, tzinfo=ISTANBUL),
            fixture_loader=lambda: live,
            stored_loader=lambda client, match_date: [],
            analyzer=analyzer,
        )

        self.assertEqual(summary["created"], 1)
        self.assertEqual(summary["unanalyzable"], 1)
        self.assertEqual(summary["unanalyzable_matches"][0]["reason"], "Takım yok")

    def test_missing_job_secret_is_reported_without_printing_values(self):
        with patch.dict(
            os.environ, {"SUPABASE_URL": "https://private-project.example"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "SUPABASE_SERVICE_ROLE_KEY") as error:
                create_job_client()
        self.assertNotIn("private-project", str(error.exception))


if __name__ == "__main__":
    unittest.main()

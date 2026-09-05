import unittest
from urllib.error import URLError

from results_api import fetch_match_result, parse_event_result, result_sync_page


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class ResultsApiTests(unittest.TestCase):
    def test_extracts_matching_completed_event(self):
        payload = {"event": [{
            "idEvent": "42", "dateEvent": "2026-09-02",
            "strHomeTeam": "West Bromwich Albion", "strAwayTeam": "Charlton Athletic",
            "intHomeScore": "2", "intAwayScore": "1", "strStatus": "Match Finished",
        }]}
        match = {
            "match_date": "2026-09-02", "home_team": "West Brom", "away_team": "Charlton",
        }

        result = parse_event_result(payload, match)

        self.assertEqual(result["full_time_home"], 2)
        self.assertEqual(result["full_time_away"], 1)

    def test_rejects_wrong_date_or_teams(self):
        payload = {"event": [{
            "dateEvent": "2026-09-01", "strHomeTeam": "Other",
            "strAwayTeam": "Teams", "intHomeScore": "1", "intAwayScore": "0",
        }]}
        match = {"match_date": "2026-09-02", "home_team": "West Brom", "away_team": "Charlton"}

        self.assertIsNone(parse_event_result(payload, match))

    def test_rejects_live_score_until_event_is_final(self):
        payload = {"event": [{
            "dateEvent": "2026-09-02", "strHomeTeam": "West Brom",
            "strAwayTeam": "Charlton", "intHomeScore": "2", "intAwayScore": "1",
            "strStatus": "Second Half", "strProgress": "78",
        }]}
        match = {"match_date": "2026-09-02", "home_team": "West Brom", "away_team": "Charlton"}

        self.assertIsNone(parse_event_result(payload, match))

    def test_accepts_short_final_status(self):
        payload = {"events": [{
            "dateEvent": "2026-09-02", "strHomeTeam": "West Brom",
            "strAwayTeam": "Charlton", "intHomeScore": "2", "intAwayScore": "1",
            "strStatus": "FT",
        }]}
        match = {"match_date": "2026-09-02", "home_team": "West Brom", "away_team": "Charlton"}

        self.assertEqual(parse_event_result(payload, match)["full_time_home"], 2)

    def test_retries_transient_connection_error(self):
        calls = []
        responses = iter([
            URLError("temporary"),
            FakeResponse(b'{"event": []}'),
        ])

        def opener(request, timeout):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        result = fetch_match_result(
            "123", "2026-09-02", "A", "B",
            opener=opener, sleeper=calls.append, base_delay=0.25,
        )

        self.assertIsNone(result)
        self.assertEqual(calls, [0.25])

    def test_result_sync_pages_rotate_without_starving_later_matches(self):
        candidates = [{"match_key": str(index)} for index in range(5)]

        first, cursor = result_sync_page(candidates, cursor=0, page_size=2)
        second, cursor = result_sync_page(candidates, cursor=cursor, page_size=2)
        third, cursor = result_sync_page(candidates, cursor=cursor, page_size=2)

        self.assertEqual([row["match_key"] for row in first], ["0", "1"])
        self.assertEqual([row["match_key"] for row in second], ["2", "3"])
        self.assertEqual([row["match_key"] for row in third], ["4", "0"])
        self.assertEqual(cursor, 1)


if __name__ == "__main__":
    unittest.main()

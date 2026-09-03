import unittest

from results_api import parse_event_result


class ResultsApiTests(unittest.TestCase):
    def test_extracts_matching_completed_event(self):
        payload = {"event": [{
            "idEvent": "42", "dateEvent": "2026-09-02",
            "strHomeTeam": "West Bromwich Albion", "strAwayTeam": "Charlton Athletic",
            "intHomeScore": "2", "intAwayScore": "1",
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


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

from supabase_resilience import fetch_team_catalog_with_retry


class FakeQuery:
    def __init__(self, pages=None, error=None):
        self.pages = list(pages or [])
        self.error = error
        self.page_index = 0

    def select(self, _columns):
        return self

    def range(self, _start, _end):
        return self

    def execute(self):
        if self.error:
            raise self.error
        rows = self.pages[self.page_index] if self.page_index < len(self.pages) else []
        self.page_index += 1
        return SimpleNamespace(data=rows)


class FakeClient:
    def __init__(self, query):
        self.query = query

    def table(self, name):
        if name != "historical_matches":
            raise AssertionError(name)
        return self.query


class SupabaseResilienceTests(unittest.TestCase):
    def test_collects_unique_sorted_teams_and_divisions(self):
        client = FakeClient(FakeQuery(pages=[[
            {"division": "E0", "home_team": "Chelsea", "away_team": "Arsenal"},
            {"division": "E0", "home_team": "Arsenal", "away_team": "Liverpool"},
        ]]))

        teams, divisions = fetch_team_catalog_with_retry(
            lambda: client, attempts=1, page_size=1000
        )

        self.assertEqual(teams, ["Arsenal", "Chelsea", "Liverpool"])
        self.assertEqual(divisions, ["E0"])

    def test_retries_with_a_fresh_client_after_connection_termination(self):
        clients = iter([
            FakeClient(FakeQuery(error=ConnectionError("ConnectionTerminated"))),
            FakeClient(FakeQuery(pages=[[
                {"division": "E0", "home_team": "Arsenal", "away_team": "Chelsea"}
            ]])),
        ])
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return next(clients)

        teams, divisions = fetch_team_catalog_with_retry(factory, attempts=2)

        self.assertEqual(factory_calls, 2)
        self.assertEqual(teams, ["Arsenal", "Chelsea"])
        self.assertEqual(divisions, ["E0"])

    def test_rejects_zero_attempts(self):
        with self.assertRaisesRegex(ValueError, "attempts"):
            fetch_team_catalog_with_retry(lambda: object(), attempts=0)


if __name__ == "__main__":
    unittest.main()

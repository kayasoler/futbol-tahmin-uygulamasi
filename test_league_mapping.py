import unittest

from league_mapping import division_for_api_league, match_team_name


class LeagueMappingTests(unittest.TestCase):
    def test_uses_mapping_only_when_historical_division_exists(self):
        self.assertEqual(division_for_api_league(39, {"E0", "SP1"}), "E0")
        self.assertIsNone(division_for_api_league(78, {"E0", "SP1"}))

    def test_matches_common_team_suffix_variants(self):
        name, score = match_team_name("Arsenal FC", ["Arsenal", "Chelsea"])
        self.assertEqual(name, "Arsenal")
        self.assertEqual(score, 1.0)

    def test_rejects_unrelated_team(self):
        name, _ = match_team_name("Unknown Athletic", ["Arsenal", "Chelsea"])
        self.assertIsNone(name)


if __name__ == "__main__":
    unittest.main()

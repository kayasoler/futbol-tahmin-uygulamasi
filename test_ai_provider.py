import unittest
from unittest.mock import patch

import analysis


class AiProviderTests(unittest.TestCase):
    def setUp(self):
        self.match = {
            "division": "E1",
            "home_team": "Home",
            "away_team": "Away",
            "match_date": "2026-09-03",
            "kickoff_time": "20:00",
        }
        self.report = {
            "predictions": {
                "ms": "MS 1",
                "ms_probabilities": {"1": 0.5, "X": 0.3, "2": 0.2},
                "totals": {},
            }
        }
        self.structured = {
            "live_research": [{"claim": "Güncel bilgi", "source_ids": [1]}],
            "combined_interpretation": "Temkinli yorum.",
            "predictions": {"ms": "1", "ms_probabilities": {"1": 50, "X": 30, "2": 20}},
        }
        self.sources = [{"category": "Maç haberi", "title": "Haber", "url": "https://example.com", "published_date": "", "content": "Özet"}]

    @patch("analysis._fetch_match_news")
    @patch("analysis._groq_completion")
    @patch("analysis._gemini_completion")
    def test_uses_groq_before_gemini(self, gemini, groq, news):
        news.return_value = (self.sources, [])
        groq.return_value = (analysis.json.dumps(self.structured), "openai/gpt-oss-20b")

        result = analysis.generate_grounded_analysis("groq", "tavily", self.match, self.report, "gemini")

        self.assertEqual(result["provider"], "Groq")
        gemini.assert_not_called()

    @patch("analysis._fetch_match_news")
    @patch("analysis._groq_completion")
    @patch("analysis._gemini_completion")
    def test_falls_back_to_gemini_when_groq_fails(self, gemini, groq, news):
        news.return_value = (self.sources, [])
        groq.side_effect = RuntimeError("quota")
        gemini.return_value = (analysis.json.dumps(self.structured), "gemini-test")

        result = analysis.generate_grounded_analysis("groq", "tavily", self.match, self.report, "gemini")

        self.assertEqual(result["provider"], "Gemini")
        gemini.assert_called_once()

    @patch("analysis._fetch_match_news")
    @patch("analysis._groq_completion")
    def test_continues_without_tavily_results(self, groq, news):
        news.return_value = ([], [])
        groq.return_value = (analysis.json.dumps(self.structured), "openai/gpt-oss-20b")

        result = analysis.generate_grounded_analysis("groq", "tavily", self.match, self.report)

        self.assertEqual(result["provider"], "Groq")
        self.assertIn("yalnızca mevcut istatistik", result["search_warnings"][0])


if __name__ == "__main__":
    unittest.main()

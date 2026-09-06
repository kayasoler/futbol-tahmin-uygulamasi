import unittest

from performance import (
    build_performance_events,
    grouped_summary,
    sample_status,
    select_pre_match_analyses,
    summarize_events,
)


def analysis(
    *,
    version=1,
    analyzed_at="2026-09-06T08:00:00+00:00",
    kickoff_time="14:00:00",
    match_key="match-1",
    model_revision="stat-v1",
):
    return {
        "match_key": match_key,
        "version": version,
        "division": "E0",
        "match_date": "2026-09-06",
        "kickoff_time": kickoff_time,
        "analyzed_at": analyzed_at,
        "match_snapshot": {
            "b365_home": 2.0,
            "b365_draw": 3.2,
            "b365_away": 4.0,
            "b365_over_25": 1.9,
            "b365_under_25": 1.95,
        },
        "report_snapshot": {
            "model_revision": model_revision,
            "predictions": {
                "ms": "MS 1",
                "ms_probabilities": {"1": 0.6, "X": 0.25, "2": 0.15},
                "totals": {"2.5": {"prediction": "Üst", "probability": 0.65}},
                "btts_prediction": "KG Var",
                "btts_probability": 0.7,
                "confidence": "Yüksek",
            },
        },
    }


class PerformanceTests(unittest.TestCase):
    def test_selects_last_analysis_before_kickoff(self):
        rows = [
            analysis(version=1, analyzed_at="2026-09-06T08:00:00+00:00"),
            analysis(version=2, analyzed_at="2026-09-06T10:59:00+00:00"),
            analysis(version=3, analyzed_at="2026-09-06T12:00:00+00:00"),
        ]

        selected, stats = select_pre_match_analyses(rows)

        self.assertEqual([row["version"] for row in selected], [2])
        self.assertEqual(stats["selected_matches"], 1)

    def test_excludes_post_match_only_and_unverifiable_matches(self):
        rows = [
            analysis(match_key="post", analyzed_at="2026-09-06T12:00:00+00:00"),
            analysis(match_key="missing-time", kickoff_time=""),
        ]

        selected, stats = select_pre_match_analyses(rows)

        self.assertEqual(selected, [])
        self.assertEqual(stats["post_match_only"], 1)
        self.assertEqual(stats["timing_unverifiable"], 1)

    def test_scores_supported_markets_and_roi(self):
        rows = [analysis()]
        results = {"match-1": {"full_time_home": 2, "full_time_away": 1}}

        events = build_performance_events(rows, results)

        self.assertEqual([event["market"] for event in events], [
            "Maç sonucu", "2.5 Alt/Üst", "Karşılıklı gol",
        ])
        self.assertTrue(all(event["correct"] for event in events))
        self.assertAlmostEqual(events[0]["profit"], 1.0)
        self.assertAlmostEqual(events[1]["profit"], 0.9)
        self.assertIsNone(events[2]["profit"])
        self.assertAlmostEqual(events[1]["brier"], (0.65 - 1) ** 2)

    def test_supports_percentage_probabilities_and_legacy_revision(self):
        row = analysis(model_revision="")
        row["report_snapshot"].pop("model_revision")
        row["report_snapshot"]["predictions"]["ms_probabilities"] = {
            "1": 60, "X": 25, "2": 15,
        }
        events = build_performance_events(
            [row], {"match-1": {"full_time_home": 2, "full_time_away": 1}}
        )
        self.assertEqual(events[0]["model_revision"], "Legacy")
        self.assertLess(events[0]["log_loss"], 1)

    def test_summaries_report_sample_status_and_roi_coverage(self):
        events = build_performance_events(
            [analysis()], {"match-1": {"full_time_home": 2, "full_time_away": 1}}
        )

        summary = summarize_events(events)
        markets = grouped_summary(events, "market")

        self.assertEqual(summary["match_count"], 1)
        self.assertEqual(summary["prediction_count"], 3)
        self.assertEqual(summary["roi_count"], 2)
        self.assertEqual(summary["sample_status"], "Yetersiz")
        self.assertEqual(markets[0]["market"], "Maç sonucu")
        self.assertEqual(sample_status(30), "Ön değerlendirme")
        self.assertEqual(sample_status(100), "Yeterli")


if __name__ == "__main__":
    unittest.main()

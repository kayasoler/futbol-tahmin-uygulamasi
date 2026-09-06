from __future__ import annotations

from collections import defaultdict
from datetime import datetime, time
import math
import re
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ISTANBUL = ZoneInfo("Europe/Istanbul")
MARKET_ORDER = ("Maç sonucu", "2.5 Alt/Üst", "Karşılıklı gol")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _probability(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 1 < number <= 100:
        number /= 100
    if not 0 <= number <= 1:
        return None
    return min(1 - 1e-12, max(1e-12, number))


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ISTANBUL)
    return parsed.astimezone(ISTANBUL)


def _kickoff_timestamp(analysis: dict[str, Any]) -> datetime | None:
    date_text = str(analysis.get("match_date") or "").strip()
    kickoff_text = str(analysis.get("kickoff_time") or "").strip()[:8]
    if not date_text or not kickoff_text:
        return None
    try:
        match_date = datetime.fromisoformat(date_text).date()
        kickoff_time = time.fromisoformat(kickoff_text)
    except ValueError:
        return None
    return datetime.combine(match_date, kickoff_time, tzinfo=ISTANBUL)


def select_pre_match_analyses(
    analyses: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Choose the last verifiable analysis created no later than kickoff."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for analysis in analyses:
        key = str(analysis.get("match_key") or "").strip()
        if key:
            groups[key].append(analysis)

    selected: list[dict[str, Any]] = []
    stats = {
        "total_matches": len(groups),
        "selected_matches": 0,
        "timing_unverifiable": 0,
        "post_match_only": 0,
    }
    for rows in groups.values():
        valid_rows: list[tuple[datetime, int, dict[str, Any]]] = []
        has_verifiable_timing = False
        for row in rows:
            kickoff = _kickoff_timestamp(row)
            analyzed_at = _parse_timestamp(row.get("analyzed_at"))
            if kickoff is None or analyzed_at is None:
                continue
            has_verifiable_timing = True
            if analyzed_at <= kickoff:
                valid_rows.append((analyzed_at, int(row.get("version") or 0), row))
        if valid_rows:
            selected.append(max(valid_rows, key=lambda item: (item[0], item[1]))[2])
        elif has_verifiable_timing:
            stats["post_match_only"] += 1
        else:
            stats["timing_unverifiable"] += 1

    selected.sort(
        key=lambda row: (
            str(row.get("match_date") or ""),
            str(row.get("kickoff_time") or ""),
        ),
        reverse=True,
    )
    stats["selected_matches"] = len(selected)
    return selected, stats


def _result_code(home_goals: int, away_goals: int) -> str:
    return "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"


def _prediction_code(value: Any) -> str | None:
    match = re.search(r"(?:MS\s*)?(1|X|2)\b", str(value or ""), re.I)
    return match.group(1).upper() if match else None


def _normalized_ms_probabilities(predictions: dict[str, Any]) -> dict[str, float] | None:
    raw = predictions.get("ms_probabilities")
    if not isinstance(raw, dict):
        return None
    parsed = {label: _probability(raw.get(label)) for label in ("1", "X", "2")}
    if any(value is None for value in parsed.values()):
        return None
    total = sum(float(value) for value in parsed.values())
    if total <= 0:
        return None
    return {label: float(value) / total for label, value in parsed.items()}


def _binary_scores(probability: float, actual: bool) -> tuple[float, float]:
    target = 1.0 if actual else 0.0
    return (probability - target) ** 2, -math.log(probability if actual else 1 - probability)


def _odds_for_ms(snapshot: dict[str, Any], prediction: str) -> float | None:
    field = {"1": "b365_home", "X": "b365_draw", "2": "b365_away"}.get(prediction)
    odds = _number(snapshot.get(field)) if field else None
    return odds if odds is not None and odds > 1 else None


def _event_base(
    analysis: dict[str, Any], predictions: dict[str, Any], market: str
) -> dict[str, Any]:
    snapshot = analysis.get("report_snapshot") or {}
    model_revision = str(snapshot.get("model_revision") or "Legacy")
    return {
        "match_key": str(analysis.get("match_key") or ""),
        "match_date": str(analysis.get("match_date") or ""),
        "division": str(analysis.get("division") or "Bilinmeyen"),
        "market": market,
        "confidence": str(predictions.get("confidence") or "Bilinmeyen").title(),
        "model_revision": model_revision,
        "analysis_version": int(analysis.get("version") or 0),
    }


def build_performance_events(
    analyses: Iterable[dict[str, Any]], results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build one comparable scoring row per supported market and completed match."""
    events: list[dict[str, Any]] = []
    for analysis in analyses:
        result = results.get(str(analysis.get("match_key") or ""))
        if not result:
            continue
        try:
            home_goals = int(result["full_time_home"])
            away_goals = int(result["full_time_away"])
        except (KeyError, TypeError, ValueError):
            continue
        snapshot = analysis.get("report_snapshot") or {}
        predictions = snapshot.get("predictions") if isinstance(snapshot, dict) else None
        if not isinstance(predictions, dict):
            continue
        match_snapshot = analysis.get("match_snapshot") or {}
        if not isinstance(match_snapshot, dict):
            match_snapshot = {}

        actual_ms = _result_code(home_goals, away_goals)
        predicted_ms = _prediction_code(predictions.get("ms"))
        probabilities = _normalized_ms_probabilities(predictions)
        if predicted_ms and probabilities:
            target = {label: 1.0 if label == actual_ms else 0.0 for label in probabilities}
            brier = sum((probabilities[label] - target[label]) ** 2 for label in probabilities) / 3
            log_loss = -math.log(probabilities[actual_ms])
            odds = _odds_for_ms(match_snapshot, predicted_ms)
            correct = predicted_ms == actual_ms
            events.append({
                **_event_base(analysis, predictions, "Maç sonucu"),
                "prediction": predicted_ms,
                "actual": actual_ms,
                "correct": correct,
                "brier": brier,
                "log_loss": log_loss,
                "odds": odds,
                "profit": (odds - 1 if correct else -1.0) if odds else None,
            })

        total_25 = (predictions.get("totals") or {}).get("2.5") or {}
        over_probability = _probability(total_25.get("probability"))
        predicted_total = str(total_25.get("prediction") or "").strip().title()
        if over_probability is not None and predicted_total in {"Üst", "Alt"}:
            actual_over = home_goals + away_goals > 2.5
            brier, log_loss = _binary_scores(over_probability, actual_over)
            correct = (predicted_total == "Üst") == actual_over
            odds_field = "b365_over_25" if predicted_total == "Üst" else "b365_under_25"
            odds = _number(match_snapshot.get(odds_field))
            odds = odds if odds is not None and odds > 1 else None
            events.append({
                **_event_base(analysis, predictions, "2.5 Alt/Üst"),
                "prediction": predicted_total,
                "actual": "Üst" if actual_over else "Alt",
                "correct": correct,
                "brier": brier,
                "log_loss": log_loss,
                "odds": odds,
                "profit": (odds - 1 if correct else -1.0) if odds else None,
            })

        btts_probability = _probability(predictions.get("btts_probability"))
        predicted_btts = str(predictions.get("btts_prediction") or "").strip().casefold()
        if btts_probability is not None and predicted_btts in {"kg var", "kg yok"}:
            actual_btts = home_goals > 0 and away_goals > 0
            brier, log_loss = _binary_scores(btts_probability, actual_btts)
            correct = (predicted_btts == "kg var") == actual_btts
            events.append({
                **_event_base(analysis, predictions, "Karşılıklı gol"),
                "prediction": "KG Var" if predicted_btts == "kg var" else "KG Yok",
                "actual": "KG Var" if actual_btts else "KG Yok",
                "correct": correct,
                "brier": brier,
                "log_loss": log_loss,
                "odds": None,
                "profit": None,
            })
    return events


def sample_status(match_count: int) -> str:
    if match_count >= 100:
        return "Yeterli"
    if match_count >= 30:
        return "Ön değerlendirme"
    return "Yetersiz"


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(events)
    match_count = len({str(row.get("match_key") or "") for row in rows})
    profits = [float(row["profit"]) for row in rows if row.get("profit") is not None]
    return {
        "match_count": match_count,
        "prediction_count": len(rows),
        "accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else None,
        "brier": sum(float(row["brier"]) for row in rows) / len(rows) if rows else None,
        "log_loss": sum(float(row["log_loss"]) for row in rows) / len(rows) if rows else None,
        "roi": sum(profits) / len(profits) if profits else None,
        "roi_count": len(profits),
        "sample_status": sample_status(match_count),
    }


def grouped_summary(
    events: Iterable[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str(event.get(field) or "Bilinmeyen")].append(event)
    rows: list[dict[str, Any]] = []
    order = {name: index for index, name in enumerate(MARKET_ORDER)} if field == "market" else {}
    for label, group in groups.items():
        rows.append({field: label, **summarize_events(group)})
    rows.sort(key=lambda row: (order.get(str(row[field]), 999), -int(row["match_count"]), str(row[field])))
    return rows

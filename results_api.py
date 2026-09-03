from __future__ import annotations

from difflib import SequenceMatcher
import json
import unicodedata
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://www.thesportsdb.com/api/v1/json"


def _normalized(value: Any) -> str:
    plain = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return " ".join(part for part in plain.casefold().replace("-", " ").split() if part not in {"fc", "afc"})


def _team_score(left: Any, right: Any) -> float:
    left_key = _normalized(left)
    right_key = _normalized(right)
    sequence_score = SequenceMatcher(None, left_key, right_key).ratio()
    left_words = left_key.split()
    right_words = right_key.split()
    shorter, longer = (left_words, right_words) if len(left_words) <= len(right_words) else (right_words, left_words)
    if shorter and all(
        any(word == candidate or (len(word) >= 4 and candidate.startswith(word[:4])) for candidate in longer)
        for word in shorter
    ):
        return max(sequence_score, 0.9)
    return sequence_score


def parse_event_result(
    payload: dict[str, Any], match: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a score only when date and both teams match safely."""
    expected_date = str(match.get("match_date") or "")
    home = str(match.get("home_team") or "")
    away = str(match.get("away_team") or "")
    candidates: list[tuple[float, dict[str, Any]]] = []
    for event in payload.get("event") or payload.get("events") or []:
        if not isinstance(event, dict) or str(event.get("dateEvent") or "") != expected_date:
            continue
        home_score = _team_score(home, event.get("strHomeTeam"))
        away_score = _team_score(away, event.get("strAwayTeam"))
        if min(home_score, away_score) < 0.72:
            continue
        try:
            full_home = int(event["intHomeScore"])
            full_away = int(event["intAwayScore"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((home_score + away_score, {
            "full_time_home": full_home,
            "full_time_away": full_away,
            "half_time_home": None,
            "half_time_away": None,
            "source": "thesportsdb",
            "api_event_id": str(event.get("idEvent") or ""),
        }))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def fetch_match_result(
    api_key: str,
    match_date: str,
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    query = urlencode({"e": f"{home_team}_vs_{away_team}", "d": match_date})
    url = f"{BASE_URL}/{api_key or '123'}/searchevents.php?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"TheSportsDB HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"TheSportsDB bağlantı hatası: {exc}") from exc
    return parse_event_result(payload, {
        "match_date": match_date, "home_team": home_team, "away_team": away_team,
    })

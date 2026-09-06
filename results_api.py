from __future__ import annotations

from difflib import SequenceMatcher
import json
import time
import unicodedata
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://www.thesportsdb.com/api/v1/json"
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
FINAL_EVENT_STATUSES = {
    "aet",
    "afterextratime",
    "afterpenalties",
    "final",
    "finished",
    "ft",
    "fulltime",
    "matchfinished",
    "pen",
}


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


def _status_key(value: Any) -> str:
    return "".join(character for character in _normalized(value) if character.isalnum())


def _event_is_final(event: dict[str, Any]) -> bool:
    return any(
        _status_key(event.get(field)) in FINAL_EVENT_STATUSES
        for field in ("strStatus", "strProgress")
    )


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
        if not _event_is_final(event):
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


def result_sync_page(
    candidates: list[dict[str, Any]], cursor: int = 0, page_size: int = 20
) -> tuple[list[dict[str, Any]], int]:
    """Return a bounded rotating page so unresolved early matches cannot starve later ones."""
    if not candidates or page_size <= 0:
        return [], 0
    start = max(int(cursor), 0) % len(candidates)
    ordered = candidates[start:] + candidates[:start]
    page = ordered[:page_size]
    return page, (start + len(page)) % len(candidates)


def _retry_after_seconds(exc: HTTPError, fallback: float) -> float:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(max(float(value), 0.0), 30.0) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _load_json_with_retry(
    request: Request,
    *,
    timeout: float,
    max_attempts: int,
    base_delay: float,
    opener: Callable[..., Any],
    sleeper: Callable[[float], None],
) -> dict[str, Any]:
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            with opener(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("TheSportsDB yanıtı JSON nesnesi değil")
                return payload
        except HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES or attempt + 1 >= attempts:
                raise RuntimeError(f"TheSportsDB HTTP {exc.code}") from exc
            sleeper(_retry_after_seconds(exc, base_delay * (2 ** attempt)))
        except (URLError, TimeoutError) as exc:
            if attempt + 1 >= attempts:
                raise RuntimeError(f"TheSportsDB bağlantı hatası: {exc}") from exc
            sleeper(base_delay * (2 ** attempt))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            if attempt + 1 >= attempts:
                raise RuntimeError(f"TheSportsDB geçersiz yanıtı: {exc}") from exc
            sleeper(base_delay * (2 ** attempt))
    raise RuntimeError("TheSportsDB isteği tamamlanamadı")


def fetch_match_result(
    api_key: str,
    match_date: str,
    home_team: str,
    away_team: str,
    *,
    timeout: float = 30,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any] | None:
    query = urlencode({"e": f"{home_team}_vs_{away_team}", "d": match_date})
    url = f"{BASE_URL}/{api_key or '123'}/searchevents.php?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = _load_json_with_retry(
        request,
        timeout=timeout,
        max_attempts=max_attempts,
        base_delay=base_delay,
        opener=opener,
        sleeper=sleeper,
    )
    return parse_event_result(payload, {
        "match_date": match_date, "home_team": home_team, "away_team": away_team,
    })

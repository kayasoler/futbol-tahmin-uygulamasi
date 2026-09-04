from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://v3.football.api-sports.io"


def normalize_api_keys(value: Any) -> list[str]:
    """Accept string, list, or TOML table secrets without treating field names as keys."""
    candidates: list[str] = []

    def collect(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            candidates.extend(re.split(r"[;,\r\n]+", item))
            return
        if isinstance(item, Mapping) or hasattr(item, "items"):
            mapping = dict(item.items())
            preferred_names = {
                "api_key", "api-football-key", "api_football_key", "key", "token", "value"
            }
            preferred_values = [
                item_value
                for item_name, item_value in mapping.items()
                if str(item_name).strip().casefold() in preferred_names
            ]
            for item_value in preferred_values or list(mapping.values()):
                collect(item_value)
            return
        if isinstance(item, Iterable):
            for nested in item:
                collect(nested)

    collect(value)
    keys: list[str] = []
    for candidate in candidates:
        key = candidate.strip()
        lowered = key.casefold()
        invalid = (
            lowered in {"key", "token", "none", "null", "your_api_key", "api_key"}
            or "missing application key" in lowered
            or key.startswith(("{", "["))
            or any(character.isspace() for character in key)
        )
        if invalid:
            continue
        if key and key not in keys:
            keys.append(key)
    return keys


def _get(api_keys: str | Iterable[str], endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    keys = normalize_api_keys(api_keys)
    if not keys:
        raise RuntimeError("API-Football anahtarı bulunamadı.")
    failures: list[str] = []
    for index, api_key in enumerate(keys):
        request = Request(
            f"{BASE_URL}/{endpoint}?{query}",
            headers={"x-apisports-key": api_key, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=35) as response:
                payload = json.loads(response.read().decode("utf-8"))
                headers = {
                    "daily_limit": response.headers.get("x-ratelimit-requests-limit"),
                    "daily_remaining": response.headers.get("x-ratelimit-requests-remaining"),
                    "key_number": index + 1,
                }
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            failures.append(f"anahtar {index + 1}: HTTP {exc.code} {detail[:160]}")
            if exc.code in {401, 403, 429}:
                continue
            raise RuntimeError(f"API-Football HTTP {exc.code}: {detail[:500]}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"API-Football bağlantı hatası: {exc}") from exc
        errors = payload.get("errors") or []
        if errors:
            error_text = str(errors)
            lowered = error_text.casefold()
            if "missing application key" in lowered or "invalid api key" in lowered:
                failures.append(f"anahtar {index + 1}: kimlik doğrulama reddedildi")
                continue
            failures.append(f"anahtar {index + 1}: {error_text[:160]}")
            if any(word in lowered for word in ("limit", "quota", "key", "request")):
                continue
            raise RuntimeError(f"API-Football hata yanıtı: {errors}")
        return {"response": payload.get("response") or [], "quota": headers}
    raise RuntimeError(
        "API-Football anahtarlarının hiçbiri kullanılamadı. "
        + " | ".join(failures)
        + " Secrets biçimi: API_FOOTBALL_KEY = \"...\" veya "
        "API_FOOTBALL_KEYS = [\"...\", \"...\"]."
    )


def normalize_fixture(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    status = fixture.get("status") or {}
    venue = fixture.get("venue") or {}
    kickoff = str(fixture.get("date") or "")
    match_date = kickoff[:10] if len(kickoff) >= 10 else ""
    kickoff_time = kickoff[11:19] if len(kickoff) >= 19 else None
    return {
        "api_fixture_id": fixture.get("id"),
        "match_date": match_date,
        "kickoff_time": kickoff_time,
        "timestamp": fixture.get("timestamp"),
        "status": status.get("short") or status.get("long") or "NS",
        "status_text": status.get("long") or status.get("short") or "Planlandı",
        "league_id": league.get("id"),
        "league": league.get("name") or "Bilinmeyen lig",
        "country": league.get("country") or "Bilinmeyen ülke",
        "season": league.get("season"),
        "round": league.get("round"),
        "league_logo": league.get("logo"),
        "home_team_id": home.get("id"),
        "home_team": home.get("name") or "Ev sahibi",
        "home_logo": home.get("logo"),
        "away_team_id": away.get("id"),
        "away_team": away.get("name") or "Deplasman",
        "away_logo": away.get("logo"),
        "venue": venue.get("name"),
        "city": venue.get("city"),
        "raw_data": item,
    }


def fetch_fixtures(
    api_key: str | Iterable[str],
    fixture_date: date | str,
    timezone: str = "Europe/Istanbul",
) -> dict[str, Any]:
    result = _get(
        api_key,
        "fixtures",
        {"date": str(fixture_date), "timezone": timezone},
    )
    return {
        "fixtures": [normalize_fixture(item) for item in result["response"]],
        "quota": result["quota"],
    }


def _match_winner_values(bets: list[dict[str, Any]]) -> dict[str, float]:
    for bet in bets:
        if str(bet.get("name") or "").strip().casefold() not in {"match winner", "1x2", "winner"}:
            continue
        values: dict[str, float] = {}
        for item in bet.get("values") or []:
            label = str(item.get("value") or "").strip().casefold()
            try:
                odd = float(item.get("odd"))
            except (TypeError, ValueError):
                continue
            if label in {"home", "draw", "away"} and odd > 1:
                values[label] = odd
        if set(values) == {"home", "draw", "away"}:
            return values
    return {}


def normalize_bet365_odds(response: list[dict[str, Any]]) -> dict[str, Any] | None:
    for fixture_item in response:
        for bookmaker in fixture_item.get("bookmakers") or []:
            name = str(bookmaker.get("name") or "").strip()
            if name.casefold().replace(" ", "") != "bet365":
                continue
            values = _match_winner_values(bookmaker.get("bets") or [])
            if values:
                return {
                    "bookmaker": "Bet365",
                    "bookmaker_id": bookmaker.get("id"),
                    "b365_home": values["home"],
                    "b365_draw": values["draw"],
                    "b365_away": values["away"],
                    "updated_at": fixture_item.get("update"),
                }
    return None


def fetch_bet365_odds(api_keys: str | Iterable[str], fixture_id: int | str) -> dict[str, Any]:
    result = _get(api_keys, "odds", {"fixture": fixture_id, "bookmaker": 8})
    return {"odds": normalize_bet365_odds(result["response"]), "quota": result["quota"]}

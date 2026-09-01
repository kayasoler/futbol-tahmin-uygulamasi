from __future__ import annotations

import json
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://v3.football.api-sports.io"


def _get(api_key: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
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
            }
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API-Football HTTP {exc.code}: {detail[:500]}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"API-Football bağlantı hatası: {exc}") from exc
    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError(f"API-Football hata yanıtı: {errors}")
    return {"response": payload.get("response") or [], "quota": headers}


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
    api_key: str,
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

from __future__ import annotations

import json
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from league_mapping import team_name_key


BASE_URL = "https://v3.football.api-sports.io"
FINAL_FIXTURE_STATUSES = {"FT", "AET", "PEN"}
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


def normalize_api_keys(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        candidates = value.replace(";", ",").split(",")
    else:
        candidates = list(value)
    keys: list[str] = []
    for candidate in candidates:
        key = str(candidate).strip()
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
                if not isinstance(payload, dict):
                    raise ValueError("API-Football yanıtı JSON nesnesi değil")
                headers = {
                    "daily_limit": response.headers.get("x-ratelimit-requests-limit"),
                    "daily_remaining": response.headers.get("x-ratelimit-requests-remaining"),
                    "key_number": index + 1,
                }
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            failures.append(f"anahtar {index + 1}: HTTP {exc.code} {detail[:160]}")
            if exc.code in {401, 403} | RETRYABLE_HTTP_CODES:
                continue
            raise RuntimeError(f"API-Football HTTP {exc.code}: {detail[:500]}") from exc
        except (URLError, TimeoutError) as exc:
            failures.append(f"anahtar {index + 1}: bağlantı hatası {str(exc)[:160]}")
            continue
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            failures.append(f"anahtar {index + 1}: geçersiz yanıt {str(exc)[:160]}")
            continue
        errors = payload.get("errors") or []
        if errors:
            error_text = str(errors)
            failures.append(f"anahtar {index + 1}: {error_text[:160]}")
            lowered = error_text.casefold()
            if any(
                word in lowered
                for word in (
                    "limit", "quota", "key", "request", "plan", "subscription", "access"
                )
            ):
                continue
            raise RuntimeError(f"API-Football hata yanıtı: {errors}")
        return {
            "response": payload.get("response") or [],
            "quota": headers,
            "paging": payload.get("paging") or {},
        }
    raise RuntimeError("API-Football anahtarlarının hiçbiri kullanılamadı. " + " | ".join(failures))


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
        "id": f"api-{fixture.get('id')}",
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
        "entry_method": "api-football",
        "match_status": status.get("short") or status.get("long") or "NS",
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


def match_fixture(
    fixtures: list[dict[str, Any]], home_team: str, away_team: str
) -> dict[str, Any] | None:
    """Resolve one selected match without trusting provider team names exactly."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for fixture in fixtures:
        home_score = _team_score(home_team, fixture.get("home_team"))
        away_score = _team_score(away_team, fixture.get("away_team"))
        if min(home_score, away_score) < 0.72:
            continue
        candidates.append((home_score + away_score, fixture))
    return dict(max(candidates, key=lambda item: item[0])[1]) if candidates else None


def normalize_recent_fixture(item: dict[str, Any], selected_team_id: int) -> dict[str, Any]:
    fixture = item.get("fixture") or {}
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    goals = item.get("goals") or {}
    home_goals = goals.get("home")
    away_goals = goals.get("away")
    score = "—" if home_goals is None or away_goals is None else f"{home_goals}-{away_goals}"
    return {
        "Tarih": str(fixture.get("date") or "")[:10],
        "Ev sahibi": home.get("name") or "—",
        "Deplasman": away.get("name") or "—",
        "Skor": score,
        "Saha": "İç saha" if home.get("id") == selected_team_id else "Deplasman",
    }


def normalize_injuries(response: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response:
        team = item.get("team") or {}
        player = item.get("player") or {}
        rows.append({
            "Takım": team.get("name") or "—",
            "Oyuncu": player.get("name") or "—",
            "Durum": player.get("type") or "—",
            "Neden": player.get("reason") or "—",
        })
    return rows


def normalize_lineups(response: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lineups: list[dict[str, Any]] = []
    for item in response:
        team = item.get("team") or {}

        def player_names(key: str) -> list[str]:
            names: list[str] = []
            for row in item.get(key) or []:
                player = row.get("player") or {}
                name = str(player.get("name") or "").strip()
                if name:
                    names.append(name)
            return names

        lineups.append({
            "team_id": team.get("id"),
            "team_name": team.get("name") or "—",
            "formation": item.get("formation") or "—",
            "starting": player_names("startXI"),
            "substitutes": player_names("substitutes"),
        })
    return lineups


def fetch_match_context(
    api_keys: str | Iterable[str], match_date: date | str, home_team: str, away_team: str
) -> dict[str, Any]:
    """Fetch on-demand form and player availability for one selected fixture."""
    reference = fetch_match_reference(api_keys, match_date, home_team, away_team)
    match = reference.get("match")
    if not match:
        return reference
    home_id = int(match["home_team_id"])
    away_id = int(match["away_team_id"])
    warnings: list[str] = []

    def recent(team_id: int) -> list[dict[str, Any]]:
        try:
            result = _get(
                api_keys,
                "fixtures",
                {"team": team_id, "last": 5, "timezone": "Europe/Istanbul"},
            )
            return [normalize_recent_fixture(item, team_id) for item in result["response"]]
        except Exception as exc:
            warnings.append(f"Son maçlar alınamadı: {exc}")
            return []

    try:
        injury_result = _get(api_keys, "injuries", {"fixture": match["api_fixture_id"]})
        injuries = normalize_injuries(injury_result["response"])
    except Exception as exc:
        warnings.append(f"Oyuncu durumu alınamadı: {exc}")
        injuries = []
    return {
        **reference,
        "home_form": recent(home_id),
        "away_form": recent(away_id),
        "injuries": injuries,
        "standings": [],
        "warnings": warnings,
    }


def fetch_match_reference(
    api_keys: str | Iterable[str], match_date: date | str, home_team: str, away_team: str
) -> dict[str, Any]:
    """Resolve one fixture with a single request for lineup-only actions."""
    fixture_result = fetch_fixtures(api_keys, match_date)
    match = match_fixture(fixture_result["fixtures"], home_team, away_team)
    return {
        "provider": "api-football",
        "match": match,
        "home_form": [],
        "away_form": [],
        "injuries": [],
        "standings": [],
        "warnings": [],
    }


def fetch_match_lineups(
    api_keys: str | Iterable[str], fixture_id: int | str
) -> dict[str, Any]:
    result = _get(api_keys, "fixtures/lineups", {"fixture": fixture_id})
    return {"lineups": normalize_lineups(result["response"]), "quota": result["quota"]}


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


def normalize_bet365_odds_by_fixture(
    response: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index a date-level odds response by API fixture id."""
    indexed: dict[str, dict[str, Any]] = {}
    for fixture_item in response:
        fixture_id = (fixture_item.get("fixture") or {}).get("id")
        odds = normalize_bet365_odds([fixture_item])
        if fixture_id is not None and odds:
            indexed[str(fixture_id)] = odds
    return indexed


def fetch_bet365_odds_for_date(
    api_keys: str | Iterable[str],
    fixture_date: date | str,
    timezone: str = "Europe/Istanbul",
    *,
    max_pages: int = 10,
) -> dict[str, Any]:
    """Fetch Bet365 1-X-2 odds in bounded pages for one selected day."""
    odds_by_fixture: dict[str, dict[str, Any]] = {}
    quota: dict[str, Any] = {}
    page = 1
    while page <= max(1, int(max_pages)):
        result = _get(
            api_keys,
            "odds",
            {
                "date": str(fixture_date),
                "timezone": timezone,
                "bookmaker": 8,
                "page": page,
            },
        )
        odds_by_fixture.update(normalize_bet365_odds_by_fixture(result["response"]))
        quota = result["quota"]
        paging = result.get("paging") or {}
        try:
            total_pages = max(1, int(paging.get("total") or 1))
        except (TypeError, ValueError):
            total_pages = 1
        if page >= total_pages:
            break
        page += 1
    return {"odds_by_fixture": odds_by_fixture, "quota": quota, "pages": page}


def normalize_final_result(item: dict[str, Any]) -> dict[str, Any] | None:
    """Return regular-time scores only for explicitly completed fixtures."""
    fixture = item.get("fixture") or {}
    status = fixture.get("status") or {}
    status_short = str(status.get("short") or "").strip().upper()
    if status_short not in FINAL_FIXTURE_STATUSES:
        return None
    score = item.get("score") or {}
    full_time = score.get("fulltime") or {}
    half_time = score.get("halftime") or {}
    goals = item.get("goals") or {}
    home_value = full_time.get("home")
    away_value = full_time.get("away")
    if status_short == "FT":
        home_value = goals.get("home") if home_value is None else home_value
        away_value = goals.get("away") if away_value is None else away_value
    try:
        full_home = int(home_value)
        full_away = int(away_value)
    except (TypeError, ValueError):
        return None

    def optional_score(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    teams = item.get("teams") or {}
    return {
        "match_date": str(fixture.get("date") or "")[:10],
        "home_team": str((teams.get("home") or {}).get("name") or ""),
        "away_team": str((teams.get("away") or {}).get("name") or ""),
        "full_time_home": full_home,
        "full_time_away": full_away,
        "half_time_home": optional_score(half_time.get("home")),
        "half_time_away": optional_score(half_time.get("away")),
        "source": "api-football",
        "api_fixture_id": fixture.get("id"),
        "status": status_short,
    }


def fetch_final_results_for_date(
    api_keys: str | Iterable[str],
    fixture_date: date | str,
    timezone: str = "Europe/Istanbul",
) -> dict[str, Any]:
    result = _get(
        api_keys,
        "fixtures",
        {"date": str(fixture_date), "timezone": timezone},
    )
    return {
        "results": [
            parsed
            for item in result["response"]
            if (parsed := normalize_final_result(item)) is not None
        ],
        "quota": result["quota"],
    }


def _normalized_team(value: Any) -> str:
    return team_name_key(str(value or ""))


def _team_score(left: Any, right: Any) -> float:
    left_key = _normalized_team(left)
    right_key = _normalized_team(right)
    if not left_key or not right_key:
        return 0.0
    score = SequenceMatcher(None, left_key, right_key).ratio()
    if left_key in right_key or right_key in left_key:
        score = max(score, min(len(left_key), len(right_key)) / max(len(left_key), len(right_key)))
    return score


def match_final_result(
    results: list[dict[str, Any]], match: dict[str, Any]
) -> dict[str, Any] | None:
    """Match a completed API fixture to one stored analysis without trusting date alone."""
    expected_date = str(match.get("match_date") or "")
    candidates: list[tuple[float, dict[str, Any]]] = []
    for result in results:
        if str(result.get("match_date") or "") != expected_date:
            continue
        home_score = _team_score(match.get("home_team"), result.get("home_team"))
        away_score = _team_score(match.get("away_team"), result.get("away_team"))
        if min(home_score, away_score) < 0.72:
            continue
        candidates.append((home_score + away_score, result))
    return dict(max(candidates, key=lambda item: item[0])[1]) if candidates else None

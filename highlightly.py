from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
from typing import Any
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://soccer.highlightly.net"
REQUEST_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Origin": "https://highlightly.net",
    "Referer": "https://highlightly.net/",
}


def _get(api_key: str, endpoint: str, params: dict[str, Any] | None = None) -> Any:
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
    url = f"{BASE_URL}/{endpoint.lstrip('/')}" + (f"?{query}" if query else "")
    request = Request(url, headers={**REQUEST_HEADERS, "x-rapidapi-key": api_key})
    try:
        with urlopen(request, timeout=35) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "1010" in detail:
            raise RuntimeError(
                "Highlightly Cloudflare erişimi Streamlit sunucusunun HTTP imzasını engelledi. "
                "İstek tarayıcı başlıklarıyla gönderildi; sorun sürerse Highlightly desteğinin "
                "Streamlit Cloud çıkışına izin vermesi gerekir."
            ) from exc
        raise RuntimeError(f"Highlightly HTTP {exc.code}: {detail[:400]}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Highlightly bağlantı hatası: {exc}") from exc


def _name_key(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", plain.casefold())
    key = " ".join(word for word in words if word not in {"fc", "afc", "cf", "club"})
    aliases = {
        "wolves": "wolverhampton wanderers",
        "man united": "manchester united",
        "man city": "manchester city",
        "newcastle": "newcastle united",
        "nott m forest": "nottingham forest",
        "sheffield weds": "sheffield wednesday",
        "west brom": "west bromwich albion",
        "west ham": "west ham united",
        "qpr": "queens park rangers",
        "sp lisbon": "sporting lisbon",
        "ath madrid": "atletico madrid",
        "sociedad": "real sociedad",
        "betis": "real betis",
    }
    return aliases.get(key, key)


def _score(left: str, right: str) -> float:
    return SequenceMatcher(None, _name_key(left), _name_key(right)).ratio()


def find_match(matches_payload: Any, home: str, away: str) -> dict[str, Any] | None:
    rows = matches_payload.get("data", []) if isinstance(matches_payload, dict) else matches_payload
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for row in rows or []:
        home_team = row.get("homeTeam") or {}
        away_team = row.get("awayTeam") or {}
        score = (_score(home, str(home_team.get("name") or "")) + _score(away, str(away_team.get("name") or ""))) / 2
        if score > best[0]:
            best = (score, row)
    return best[1] if best[0] >= 0.82 else None


def fetch_match_day(
    api_key: str,
    match_date: str,
    home_team_name: str | None = None,
    away_team_name: str | None = None,
) -> Any:
    return _get(
        api_key,
        "matches",
        {
            "date": match_date,
            "timezone": "Europe/Istanbul",
            "homeTeamName": _name_key(home_team_name) if home_team_name else None,
            "awayTeamName": _name_key(away_team_name) if away_team_name else None,
            "limit": 100,
        },
    )


def fetch_last_five(api_key: str, team_id: int) -> Any:
    return _get(api_key, "last-five-games", {"teamId": team_id})


def fetch_standings(api_key: str, league_id: int, season: int) -> Any:
    return _get(api_key, "standings", {"leagueId": league_id, "season": season})


def fetch_lineups(api_key: str, match_id: int) -> Any:
    return _get(api_key, f"lineups/{match_id}")


def form_rows(payload: Any, selected_team_id: int) -> list[dict[str, Any]]:
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    output: list[dict[str, Any]] = []
    for row in rows or []:
        home = row.get("homeTeam") or {}
        away = row.get("awayTeam") or {}
        state = row.get("state") or {}
        score = (state.get("score") or {}).get("current") or "—"
        output.append({
            "Tarih": str(row.get("date") or "")[:10],
            "Ev sahibi": home.get("name") or "—",
            "Deplasman": away.get("name") or "—",
            "Skor": score,
            "Saha": "İç saha" if home.get("id") == selected_team_id else "Deplasman",
        })
    return output


def selected_standings(payload: Any, team_ids: set[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in (payload or {}).get("groups", []):
        for row in group.get("standings") or []:
            team = row.get("team") or {}
            if team.get("id") not in team_ids:
                continue
            total = row.get("total") or {}
            output.append({
                "Takım": team.get("name"), "Sıra": row.get("position"), "Puan": row.get("points"),
                "Maç": total.get("games"), "G": total.get("wins"), "B": total.get("draws"),
                "M": total.get("loses"), "Gol": f"{total.get('scoredGoals', 0)}-{total.get('receivedGoals', 0)}",
            })
    return output

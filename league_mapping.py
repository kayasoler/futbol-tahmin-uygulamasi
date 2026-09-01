from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata


# API-Football league id -> football-data.co.uk division code.
# A mapping is usable only when that division actually exists in Supabase.
API_LEAGUE_TO_DIVISION = {
    39: "E0",   # England Premier League
    40: "E1",   # England Championship
    41: "E2",   # England League One
    42: "E3",   # England League Two
    43: "EC",   # England National League
    61: "F1",   # France Ligue 1
    62: "F2",   # France Ligue 2
    78: "D1",   # Germany Bundesliga
    79: "D2",   # Germany 2. Bundesliga
    88: "N1",   # Netherlands Eredivisie
    94: "P1",   # Portugal Primeira Liga
    135: "I1",  # Italy Serie A
    136: "I2",  # Italy Serie B
    140: "SP1", # Spain La Liga
    141: "SP2", # Spain Segunda Division
    179: "SC0", # Scotland Premiership
    203: "T1",  # Turkey Super Lig
}


def division_for_api_league(league_id: object, available_divisions: set[str]) -> str | None:
    try:
        division = API_LEAGUE_TO_DIVISION.get(int(league_id))
    except (TypeError, ValueError):
        return None
    return division if division in available_divisions else None


def _name_key(value: str) -> str:
    plain = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", plain.casefold())
    ignored = {"fc", "cf", "afc", "ac", "sc", "fk", "club", "calcio"}
    return " ".join(word for word in words if word not in ignored)


def match_team_name(api_name: str, historical_names: list[str]) -> tuple[str | None, float]:
    target = _name_key(api_name)
    if not target:
        return None, 0.0
    keyed = [(name, _name_key(name)) for name in historical_names]
    exact = next((name for name, key in keyed if key == target), None)
    if exact:
        return exact, 1.0
    best_name = None
    best_score = 0.0
    for name, key in keyed:
        if not key:
            continue
        score = SequenceMatcher(None, target, key).ratio()
        if target in key or key in target:
            score = max(score, min(len(target), len(key)) / max(len(target), len(key)))
        if score > best_score:
            best_name = name
            best_score = score
    return (best_name, best_score) if best_score >= 0.82 else (None, best_score)

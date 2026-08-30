from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


HISTORICAL_COLUMNS = (
    "match_date,division,home_team,away_team,full_time_home_goals,"
    "full_time_away_goals,full_time_result,half_time_result,"
    "b365_home,b365_draw,b365_away,b365_over_25,b365_under_25"
)


def _key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def fetch_h2h_rows(client, match: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch completed meetings in both home/away directions."""
    home_key = _key(match.get("home_team"))
    away_key = _key(match.get("away_team"))
    rows: list[dict[str, Any]] = []
    for home, away in ((home_key, away_key), (away_key, home_key)):
        response = (
            client.table("historical_matches")
            .select(HISTORICAL_COLUMNS)
            .eq("home_team_key", home)
            .eq("away_team_key", away)
            .order("match_date", desc=True)
            .limit(10)
            .execute()
        )
        rows.extend(response.data or [])
    rows.sort(key=lambda row: str(row.get("match_date") or ""), reverse=True)
    return rows[:10]


def fetch_league_rows(client, division: str) -> list[dict[str, Any]]:
    """Read a league in pages so the default Supabase limit is not a hidden cap."""
    rows: list[dict[str, Any]] = []
    page_size = 1000
    start = 0
    while True:
        response = (
            client.table("historical_matches")
            .select(HISTORICAL_COLUMNS)
            .eq("division", division)
            .range(start, start + page_size - 1)
            .execute()
        )
        page = response.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def fetch_same_odds_rows(
    client,
    match: dict[str, Any],
    division_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find historical rows in the same league with identical Bet365 MS odds."""
    target = tuple(
        round(value, 2) if (value := _number(match.get(column))) is not None else None
        for column in ("b365_home", "b365_draw", "b365_away")
    )
    if any(value is None for value in target):
        return []

    matches: list[dict[str, Any]] = []
    for row in division_rows:
        candidate = tuple(
            round(value, 2) if (value := _number(row.get(column))) is not None else None
            for column in ("b365_home", "b365_draw", "b365_away")
        )
        if candidate == target:
            matches.append(row)
    return matches


def _result_label(row: dict[str, Any]) -> str:
    result = str(row.get("full_time_result") or "").strip().upper()
    if result == "H":
        return "Ev sahibi"
    if result == "A":
        return "Deplasman"
    if result == "D":
        return "Beraberlik"
    home = _number(row.get("full_time_home_goals"))
    away = _number(row.get("full_time_away_goals"))
    if home is None or away is None:
        return "Bilinmiyor"
    return "Ev sahibi" if home > away else "Deplasman" if away > home else "Beraberlik"


def _score(row: dict[str, Any]) -> str:
    home = _number(row.get("full_time_home_goals"))
    away = _number(row.get("full_time_away_goals"))
    if home is None or away is None:
        return "—"
    return f"{int(home)}-{int(away)}"


def rows_to_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for row in rows:
        home_goals = _number(row.get("full_time_home_goals"))
        away_goals = _number(row.get("full_time_away_goals"))
        total = None if home_goals is None or away_goals is None else home_goals + away_goals
        output.append(
            {
                "Tarih": row.get("match_date") or "—",
                "Ev sahibi": row.get("home_team") or "—",
                "Deplasman": row.get("away_team") or "—",
                "Skor": _score(row),
                "Kazanan": _result_label(row),
                "Üst 2.5": "Üst" if total is not None and total > 2.5 else "Alt" if total is not None else "—",
            }
        )
    return pd.DataFrame(output)


def _most_common(values: list[str]) -> str:
    clean = [value for value in values if value and value != "Bilinmiyor"]
    return Counter(clean).most_common(1)[0][0] if clean else "Veri yetersiz"


def _poisson_mode(lam: float) -> int:
    return max(0, min(6, int(round(max(0.05, lam)))))


def build_report(
    match: dict[str, Any],
    h2h_rows: list[dict[str, Any]],
    same_odds_rows: list[dict[str, Any]],
    league_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [
        row for row in league_rows
        if _number(row.get("full_time_home_goals")) is not None
        and _number(row.get("full_time_away_goals")) is not None
    ]
    if not valid:
        valid = h2h_rows

    home_odds = _number(match.get("b365_home"))
    draw_odds = _number(match.get("b365_draw"))
    away_odds = _number(match.get("b365_away"))
    odds = [home_odds, draw_odds, away_odds]
    inverse = [1 / value if value and value > 1 else 0 for value in odds]
    inverse_total = sum(inverse)
    implied = [value / inverse_total for value in inverse] if inverse_total else [None] * 3
    labels = ["MS 1", "MS X", "MS 2"]
    best_index = max(range(3), key=lambda index: implied[index] or 0)
    ms_prediction = labels[best_index] if inverse_total else "Veri yetersiz"

    ht_ms = _most_common(
        [
            f"İY {row.get('half_time_result')} / MS {row.get('full_time_result')}"
            for row in valid
            if row.get("half_time_result") and row.get("full_time_result")
        ]
    )

    totals: dict[str, dict[str, Any]] = {}
    for threshold in (0.5, 1.5, 2.5, 3.5):
        values = []
        for row in valid:
            home = _number(row.get("full_time_home_goals"))
            away = _number(row.get("full_time_away_goals"))
            if home is not None and away is not None:
                values.append(home + away > threshold)
        probability = sum(values) / len(values) if values else None
        totals[str(threshold)] = {
            "probability": probability,
            "prediction": "Üst" if probability is not None and probability >= 0.5 else "Alt" if probability is not None else "Veri yetersiz",
        }

    home_goals = [_number(row.get("full_time_home_goals")) for row in valid]
    away_goals = [_number(row.get("full_time_away_goals")) for row in valid]
    home_goals = [value for value in home_goals if value is not None]
    away_goals = [value for value in away_goals if value is not None]
    expected_home = sum(home_goals) / len(home_goals) if home_goals else 1.2
    expected_away = sum(away_goals) / len(away_goals) if away_goals else 1.0
    score_prediction = f"{_poisson_mode(expected_home)}-{_poisson_mode(expected_away)}"

    btts_values = []
    for row in valid:
        home = _number(row.get("full_time_home_goals"))
        away = _number(row.get("full_time_away_goals"))
        if home is not None and away is not None:
            btts_values.append(home > 0 and away > 0)
    btts_probability = sum(btts_values) / len(btts_values) if btts_values else None
    over25 = totals["2.5"]["probability"]
    total_pick = "2.5 Üst" if over25 is not None and over25 >= 0.58 else "2.5 Alt" if over25 is not None and over25 <= 0.42 else "2.5 temkinli"
    coupon = f"{ms_prediction} + {total_pick}"
    comment = (
        f"Bu rapor {len(valid)} tamamlanmış {match.get('division') or ''} lig maçının "
        f"istatistikleri ve mevcut Bet365 oranlarıyla oluşturuldu. "
        f"Oranların ima ettiği en güçlü MS seçimi {ms_prediction}. "
        f"Tahmini skor {score_prediction}; karşılıklı gol olasılığı {_percentage(btts_probability)}. "
        "Geçmiş istatistikler gelecek sonucu garanti etmez; oran, kadro ve haber bilgileri ayrıca değerlendirilmelidir."
    )

    return {
        "h2h": h2h_rows,
        "same_odds": same_odds_rows,
        "predictions": {
            "ms": ms_prediction,
            "ms_probabilities": dict(zip(labels, implied)),
            "ht_ms": ht_ms,
            "totals": totals,
            "score": score_prediction,
            "btts_probability": btts_probability,
            "btts_prediction": "KG Var" if btts_probability is not None and btts_probability >= 0.5 else "KG Yok" if btts_probability is not None else "Veri yetersiz",
            "sample_size": len(valid),
        },
        "comment": comment,
        "coupon": coupon,
    }

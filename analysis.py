from __future__ import annotations

from collections import Counter
import math
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import pandas as pd
import requests


HISTORICAL_COLUMNS = (
    "match_date,division,home_team,away_team,full_time_home_goals,"
    "full_time_away_goals,full_time_result,half_time_result,"
    "b365_home,b365_draw,b365_away,b365_over_25,b365_under_25"
)


def fetch_live_news(home_team: str, away_team: str, max_items: int = 12) -> list[dict[str, str]]:
    """Search Google News RSS without requiring a paid API key."""
    queries = [
        f'"{home_team}" futbol sakatlık ceza kadro',
        f'"{away_team}" futbol sakatlık ceza kadro',
        f'"{home_team}" "{away_team}" maç haber',
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FootballAnalysis/1.0)"}
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for query in queries:
        url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(query)
            + "&hl=tr&gl=TR&ceid=TR:tr"
        )
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            source = (item.findtext("source") or "").strip()
            published = (item.findtext("pubDate") or "").strip()
            if not title or not link or link in seen:
                continue
            seen.add(link)
            found.append(
                {
                    "title": title,
                    "link": link,
                    "source": source or "Google News",
                    "published": published,
                }
            )
            if len(found) >= max_items:
                return found
    return found


def news_signal_summary(
    news: list[dict[str, str]],
    home_team: str,
    away_team: str,
) -> str:
    """Turn headline keywords into a cautious, transparent match note."""
    negative_words = (
        "sakat", "sakatlık", "injur", "ceza", "cezalı", "suspended",
        "eksik", "yok", "doubtful", "out", "kriz", "mağlub", "yenil",
        "loss", "defeat",
    )
    positive_words = (
        "form", "galib", "kazan", "win", "victory", "unbeaten",
        "dönüyor", "hazır", "return", "fit",
    )
    home_key = _key(home_team)
    away_key = _key(away_team)
    scores = {home_key: 0, away_key: 0}
    labels = {home_key: home_team, away_key: away_team}
    matched_titles: list[str] = []

    for item in news:
        title = str(item.get("title") or "")
        title_key = title.casefold()
        team_key = home_key if home_key in title_key else away_key if away_key in title_key else None
        if team_key is None:
            continue
        negative = sum(word in title_key for word in negative_words)
        positive = sum(word in title_key for word in positive_words)
        scores[team_key] += positive - negative
        if negative or positive:
            matched_titles.append(title)

    if not news:
        return "Canlı taramada kullanılabilir bir haber başlığı bulunamadı; yorum yalnızca istatistiklere dayanıyor."

    if not matched_titles:
        return (
            f"{len(news)} haber başlığı bulundu ancak başlıklarda {home_team} veya {away_team} "
            "için belirgin bir olumlu/olumsuz sinyal tespit edilmedi. İstatistiksel tahmin korunuyor."
        )

    home_signal = "olumlu" if scores[home_key] > 0 else "olumsuz" if scores[home_key] < 0 else "belirsiz"
    away_signal = "olumlu" if scores[away_key] > 0 else "olumsuz" if scores[away_key] < 0 else "belirsiz"
    return (
        f"Canlı başlık sinyallerinde {home_team} için {home_signal}, {away_team} için "
        f"{away_signal} görünüm var. Bu sinyal yalnızca haber başlıklarındaki anahtar "
        "kelimelerden çıkarılmıştır; haber metni ve resmi kadro açıklaması ayrıca doğrulanmalıdır."
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
    division: str | None = None,
) -> list[dict[str, Any]]:
    """Find historical rows with identical Bet365 MS odds."""
    target = tuple(
        round(value, 2) if (value := _number(match.get(column))) is not None else None
        for column in ("b365_home", "b365_draw", "b365_away")
    )
    if any(value is None for value in target):
        return []

    query = client.table("historical_matches").select(HISTORICAL_COLUMNS)
    if division:
        query = query.eq("division", division)
    for column, value in zip(
        ("b365_home", "b365_draw", "b365_away"),
        target,
    ):
        query = query.eq(column, value)

    rows: list[dict[str, Any]] = []
    page_size = 1000
    start = 0
    while True:
        page = (
            query.order("match_date", desc=True)
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


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


def h2h_summary_tables(
    rows: list[dict[str, Any]],
    selected_home: str,
    selected_away: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create outcome and goal-line summaries from the selected home team's perspective."""
    selected_home_key = _key(selected_home)
    selected_away_key = _key(selected_away)
    home_wins = draws = away_wins = 0
    total_rows: list[dict[str, Any]] = []
    for threshold in (0.5, 1.5, 2.5, 3.5):
        over_count = 0
        under_count = 0
        for row in rows:
            home_goals = _number(row.get("full_time_home_goals"))
            away_goals = _number(row.get("full_time_away_goals"))
            if home_goals is None or away_goals is None:
                continue
            if home_goals + away_goals > threshold:
                over_count += 1
            else:
                under_count += 1
        total_rows.append(
            {
                "Gol baremi": f"{threshold}",
                "Üst": over_count,
                "Alt": under_count,
                "Üst oranı": f"{over_count / len(rows) * 100:.1f}%" if rows else "—",
            }
        )

    for row in rows:
        result = str(row.get("full_time_result") or "").strip().upper()
        row_home_key = _key(row.get("home_team"))
        if result == "H":
            winner_key = row_home_key
        elif result == "A":
            winner_key = _key(row.get("away_team"))
        elif result == "D":
            winner_key = None
        else:
            home_goals = _number(row.get("full_time_home_goals"))
            away_goals = _number(row.get("full_time_away_goals"))
            if home_goals is None or away_goals is None:
                winner_key = None
            elif home_goals > away_goals:
                winner_key = row_home_key
            elif away_goals > home_goals:
                winner_key = _key(row.get("away_team"))
            else:
                winner_key = None

        if winner_key is None:
            draws += 1
        elif winner_key == selected_home_key:
            home_wins += 1
        elif winner_key == selected_away_key:
            away_wins += 1

    outcome_table = pd.DataFrame(
        [
            {"Sonuç özeti": f"{selected_home} galibiyeti", "Maç sayısı": home_wins},
            {"Sonuç özeti": "Beraberlik", "Maç sayısı": draws},
            {"Sonuç özeti": f"{selected_away} galibiyeti", "Maç sayısı": away_wins},
            {"Sonuç özeti": "Toplam karşılaşma", "Maç sayısı": len(rows)},
        ]
    )
    return outcome_table, pd.DataFrame(total_rows)


def odds_summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize results of a same-odds comparison group."""
    summary: dict[str, Any] = {"Maç sayısı": len(rows)}
    for result, label in (("H", "Ev galibiyeti"), ("D", "Beraberlik"), ("A", "Deplasman galibiyeti")):
        summary[label] = sum(
            str(row.get("full_time_result") or "").strip().upper() == result
            for row in rows
        )
    for threshold in (0.5, 1.5, 2.5, 3.5):
        over_count = 0
        for row in rows:
            home_goals = _number(row.get("full_time_home_goals"))
            away_goals = _number(row.get("full_time_away_goals"))
            if home_goals is not None and away_goals is not None and home_goals + away_goals > threshold:
                over_count += 1
        summary[f"Üst {threshold}"] = over_count
    summary["KG Var"] = sum(
        (_number(row.get("full_time_home_goals")) or 0) > 0
        and (_number(row.get("full_time_away_goals")) or 0) > 0
        for row in rows
    )
    return pd.DataFrame([summary])


def fetch_team_form_rows(
    client,
    team_name: str,
    venue: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch a team's latest home or away matches for match-specific form."""
    key = _key(team_name)
    column = "home_team_key" if venue == "home" else "away_team_key"
    response = (
        client.table("historical_matches")
        .select(HISTORICAL_COLUMNS)
        .eq(column, key)
        .order("match_date", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def _most_common(values: list[str]) -> str:
    clean = [value for value in values if value and value != "Bilinmiyor"]
    return Counter(clean).most_common(1)[0][0] if clean else "Veri yetersiz"


def _poisson_mode(lam: float) -> int:
    return max(0, min(6, int(round(max(0.05, lam)))))


def _poisson_probability(lam: float, goals: int) -> float:
    return (lam ** goals) * math.exp(-lam) / math.factorial(goals)


def build_report(
    match: dict[str, Any],
    h2h_rows: list[dict[str, Any]],
    same_odds_rows: list[dict[str, Any]],
    league_rows: list[dict[str, Any]],
    home_form_rows: list[dict[str, Any]] | None = None,
    away_form_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    valid = [
        row for row in league_rows
        if _number(row.get("full_time_home_goals")) is not None
        and _number(row.get("full_time_away_goals")) is not None
    ]
    if not valid:
        valid = h2h_rows

    home_form_rows = home_form_rows or []
    away_form_rows = away_form_rows or []

    def average(rows: list[dict[str, Any]], scored_column: str, conceded_column: str) -> tuple[float | None, float | None]:
        scored = [_number(row.get(scored_column)) for row in rows]
        conceded = [_number(row.get(conceded_column)) for row in rows]
        scored = [value for value in scored if value is not None]
        conceded = [value for value in conceded if value is not None]
        return (
            sum(scored) / len(scored) if scored else None,
            sum(conceded) / len(conceded) if conceded else None,
        )

    league_home_scored, league_home_conceded = average(
        valid,
        "full_time_home_goals",
        "full_time_away_goals",
    )
    league_away_scored, league_away_conceded = average(
        valid,
        "full_time_away_goals",
        "full_time_home_goals",
    )
    home_scored, home_conceded = average(
        home_form_rows,
        "full_time_home_goals",
        "full_time_away_goals",
    )
    away_scored, away_conceded = average(
        away_form_rows,
        "full_time_away_goals",
        "full_time_home_goals",
    )
    league_home_scored = league_home_scored or 1.25
    league_away_scored = league_away_scored or 1.05
    home_scored = home_scored or league_home_scored
    home_conceded = home_conceded or league_away_scored
    away_scored = away_scored or league_away_scored
    away_conceded = away_conceded or league_home_scored

    expected_home = max(
        0.15,
        0.45 * home_scored + 0.30 * away_conceded + 0.25 * league_home_scored,
    )
    expected_away = max(
        0.15,
        0.45 * away_scored + 0.30 * home_conceded + 0.25 * league_away_scored,
    )

    score_grid = {
        (home_goals, away_goals): _poisson_probability(expected_home, home_goals)
        * _poisson_probability(expected_away, away_goals)
        for home_goals in range(0, 7)
        for away_goals in range(0, 7)
    }
    best_score = max(score_grid, key=score_grid.get)
    score_prediction = f"{best_score[0]}-{best_score[1]}"
    model_probabilities = [
        sum(probability for (home_goals, away_goals), probability in score_grid.items() if home_goals > away_goals),
        sum(probability for (home_goals, away_goals), probability in score_grid.items() if home_goals == away_goals),
        sum(probability for (home_goals, away_goals), probability in score_grid.items() if home_goals < away_goals),
    ]

    home_odds = _number(match.get("b365_home"))
    draw_odds = _number(match.get("b365_draw"))
    away_odds = _number(match.get("b365_away"))
    odds = [home_odds, draw_odds, away_odds]
    inverse = [1 / value if value and value > 1 else 0 for value in odds]
    inverse_total = sum(inverse)
    odds_probabilities = [value / inverse_total for value in inverse] if inverse_total else None
    if odds_probabilities:
        probabilities = [
            0.60 * odds_value + 0.40 * model_value
            for odds_value, model_value in zip(odds_probabilities, model_probabilities)
        ]
    else:
        probabilities = model_probabilities
    labels = ["MS 1", "MS X", "MS 2"]
    best_index = max(range(3), key=lambda index: probabilities[index])
    ms_prediction = labels[best_index] if probabilities else "Veri yetersiz"

    ht_ms = _most_common(
        [
            f"İY {row.get('half_time_result')} / MS {row.get('full_time_result')}"
            for row in valid
            if row.get("half_time_result") and row.get("full_time_result")
        ]
    )

    totals: dict[str, dict[str, Any]] = {}
    total_lambda = expected_home + expected_away
    for threshold in (0.5, 1.5, 2.5, 3.5):
        maximum_below = int(threshold)
        probability = 1 - sum(
            _poisson_probability(total_lambda, goals)
            for goals in range(0, maximum_below + 1)
        )
        totals[str(threshold)] = {
            "probability": probability,
            "prediction": "Üst" if probability >= 0.5 else "Alt",
        }

    btts_probability = (
        1
        - math.exp(-expected_home)
        - math.exp(-expected_away)
        + math.exp(-total_lambda)
    )
    market_candidates: list[tuple[float, str]] = []
    for threshold, data in totals.items():
        if threshold not in {"2.5", "3.5"}:
            continue
        confidence = abs(data["probability"] - 0.5)
        if confidence >= 0.08:
            market_candidates.append((confidence, f"{threshold} {'Üst' if data['probability'] >= 0.5 else 'Alt'}"))
    btts_confidence = abs(btts_probability - 0.5)
    if btts_confidence >= 0.08:
        market_candidates.append((btts_confidence, "KG Var" if btts_probability >= 0.5 else "KG Yok"))
    market_candidates.sort(reverse=True)
    secondary_pick = market_candidates[0][1] if market_candidates else "tek tercih"
    coupon = f"{ms_prediction} + {secondary_pick}" if secondary_pick != "tek tercih" else ms_prediction
    comment = (
        f"Bu rapor {len(valid)} tamamlanmış {match.get('division') or ''} lig maçının "
        f"yanı sıra ev sahibinin son {len(home_form_rows)} iç saha ve deplasmanın son "
        f"{len(away_form_rows)} dış saha maçını dikkate alıyor. "
        f"Oran ve Poisson modelinin ortak MS seçimi {ms_prediction}; tahmini gol ortalamaları "
        f"{expected_home:.2f} - {expected_away:.2f}. Tahmini skor {score_prediction}; "
        f"karşılıklı gol olasılığı {_percentage(btts_probability)}. "
        "Geçmiş istatistikler gelecek sonucu garanti etmez; oran, kadro ve haber bilgileri ayrıca değerlendirilmelidir."
    )

    return {
        "h2h": h2h_rows,
        "same_odds": same_odds_rows,
        "predictions": {
            "ms": ms_prediction,
            "ms_probabilities": dict(zip(labels, probabilities)),
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

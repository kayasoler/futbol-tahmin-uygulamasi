from __future__ import annotations

import re
import unicodedata
from typing import Any

from analysis import (
    build_report,
    exact_odds_diagnostic,
    fetch_h2h_rows,
    fetch_league_rows,
    fetch_same_odds_rows,
    fetch_team_form_rows,
    filter_exact_odds_rows,
    market_odds_context,
    odds_summary_table,
)
from analysis_store import load_latest_analysis, restore_report_snapshot, save_analysis_version


ANALYZABLE_FIXTURE_STATUSES = {"", "NS", "TBD"}
SOURCE_LABELS = {
    "football-data-live": "Football-Data",
    "manual": "Manuel",
    "csv": "CSV",
}
SOURCE_PRIORITY = {"manual": 30, "football-data-live": 20, "csv": 10}


def source_key(fixture: dict[str, Any]) -> str:
    method = str(fixture.get("entry_method") or "csv").strip().casefold()
    return method if method in SOURCE_LABELS else "csv"


def _identity_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "", text)


def fixture_identity(fixture: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _identity_text(fixture.get("division")),
        str(fixture.get("match_date") or ""),
        _identity_text(fixture.get("home_team")),
        _identity_text(fixture.get("away_team")),
    )


def resolve_fixture_duplicates(
    fixtures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one source per match; manual entries always win source conflicts."""
    winners: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for original in fixtures:
        fixture = dict(original)
        identity = fixture_identity(fixture)
        current = winners.get(identity)
        if current is None:
            winners[identity] = fixture
            continue
        if SOURCE_PRIORITY[source_key(fixture)] > SOURCE_PRIORITY[source_key(current)]:
            dropped.append(current)
            winners[identity] = fixture
        else:
            dropped.append(fixture)
    ordered = sorted(
        winners.values(),
        key=lambda row: (
            str(row.get("kickoff_time") or ""),
            str(row.get("division") or ""),
            str(row.get("home_team") or ""),
        ),
    )
    return ordered, dropped


def normalize_upcoming_fixture(row: dict[str, Any]) -> dict[str, Any]:
    fixture = dict(row)
    raw_data = fixture.get("raw_data") or {}
    opening = raw_data.get("opening_odds") if isinstance(raw_data, dict) else {}
    if isinstance(opening, dict):
        for key in (
            "opening_b365_home", "opening_b365_draw", "opening_b365_away",
            "opening_b365_over_25", "opening_b365_under_25",
        ):
            fixture[key] = opening.get(key)
    fixture["entry_method"] = source_key(fixture)
    fixture["analysis_odds_source"] = (
        "Manuel analiz anı oranı" if source_key(fixture) == "manual"
        else "CSV analiz anı oranı"
    )
    return fixture


def comparison_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "summary": odds_summary_table(rows).to_dict(orient="records") if rows else [],
        "rows": rows[:20],
    }


def build_analysis_evidence(
    match: dict[str, Any],
    h2h_rows: list[dict[str, Any]],
    same_league_rows: list[dict[str, Any]],
    same_all_rows: list[dict[str, Any]],
    model_ms: str,
) -> dict[str, Any]:
    exact_league_rows = filter_exact_odds_rows(same_league_rows, match)
    exact_all_rows = filter_exact_odds_rows(same_all_rows, match)
    exact_league = comparison_evidence(exact_league_rows)
    exact_all = comparison_evidence(exact_all_rows)
    exact_league["diagnostic"] = exact_odds_diagnostic(exact_league_rows, model_ms)
    exact_all["diagnostic"] = exact_odds_diagnostic(exact_all_rows, model_ms)
    return {
        "h2h_rows": h2h_rows,
        "same_league": comparison_evidence(same_league_rows),
        "same_all": comparison_evidence(same_all_rows),
        "exact_league": exact_league,
        "exact_all": exact_all,
    }


def _historical_teams(rows: list[dict[str, Any]]) -> dict[str, str]:
    teams: dict[str, str] = {}
    for row in rows:
        for column in ("home_team", "away_team"):
            value = str(row.get(column) or "").strip()
            if value:
                teams.setdefault(_identity_text(value), value)
    return teams


def prepare_fixture_match(
    fixture: dict[str, Any], league_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    status = str(fixture.get("match_status") or "").upper()
    if status not in ANALYZABLE_FIXTURE_STATUSES:
        return None, "Yalnızca başlamamış maçlar önceden analiz edilir."
    if not league_rows:
        return None, "Bu lig için yeterli geçmiş maç verisi bulunmuyor."
    teams = _historical_teams(league_rows)
    home = teams.get(_identity_text(fixture.get("home_team")))
    away = teams.get(_identity_text(fixture.get("away_team")))
    missing = []
    if not home:
        missing.append(str(fixture.get("home_team") or "ev sahibi"))
    if not away:
        missing.append(str(fixture.get("away_team") or "deplasman"))
    if missing:
        return None, "Geçmiş veride takım eşleşmesi bulunamadı: " + ", ".join(missing)
    match = dict(fixture)
    match["home_team"] = home
    match["away_team"] = away
    match["analysis_odds_source"] = match.get("analysis_odds_source") or (
        "Football-Data referans oranı"
        if source_key(match) == "football-data-live"
        else "Kayıtlı analiz anı oranı"
    )
    return match, None


def analyze_daily_fixture(
    client,
    fixture: dict[str, Any],
    league_cache: dict[str, list[dict[str, Any]]],
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    outcome: dict[str, Any] = {
        "fixture": dict(fixture), "analysis_match": None, "analysis_record": None,
        "report": None, "status": "Analiz edilemedi", "reason": "",
        "source": source_key(fixture),
    }
    division = str(fixture.get("division") or "").strip()
    if not division:
        outcome["reason"] = "Lig kodu bulunmuyor."
        return outcome
    try:
        if division not in league_cache:
            league_cache[division] = fetch_league_rows(client, division)
        league_rows = league_cache[division]
        match, error = prepare_fixture_match(fixture, league_rows)
        if not match:
            outcome["reason"] = error or "Geçmiş veri eşleşmesi yapılamadı."
            return outcome
        outcome["analysis_match"] = match

        if not force_refresh:
            latest, load_error = load_latest_analysis(client, match)
            if load_error:
                raise RuntimeError(load_error)
            stored_report = restore_report_snapshot(latest)
            if latest and stored_report:
                outcome.update(
                    analysis_record=latest, report=stored_report, status="Kayıtlı analiz"
                )
                return outcome
            if latest:
                outcome["reason"] = "Kayıtlı analiz snapshot'ı eksik veya bozuk."
                return outcome

        h2h_rows = fetch_h2h_rows(client, match)
        same_league_rows = fetch_same_odds_rows(client, match, division)
        same_all_rows = fetch_same_odds_rows(client, match)
        home_form_rows = fetch_team_form_rows(client, str(match["home_team"]), "home")
        away_form_rows = fetch_team_form_rows(client, str(match["away_team"]), "away")
        report = build_report(
            match, h2h_rows, same_league_rows, league_rows,
            home_form_rows=home_form_rows,
            away_form_rows=away_form_rows,
            same_odds_all_rows=same_all_rows,
        )
        report["evidence"] = build_analysis_evidence(
            match, h2h_rows, same_league_rows, same_all_rows,
            str((report.get("predictions") or {}).get("ms") or ""),
        )
        market_context = market_odds_context(match)
        if market_context:
            report["external_context"] = {"market_odds": market_context}
        saved, save_error = save_analysis_version(client, match, report)
        if save_error or not saved:
            raise RuntimeError(save_error or "Analiz kaydı oluşturulamadı.")
        outcome.update(
            analysis_record=saved, report=report,
            status="Yenilendi" if force_refresh else "Yeni kaydedildi",
        )
    except Exception as exc:
        outcome["status"] = "Analiz edilemedi"
        outcome["reason"] = str(exc)
    return outcome


def prediction_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    predictions = dict((report or {}).get("predictions") or {})
    primary = str(predictions.get("ms") or "—").replace("MS", "").strip()
    probabilities = dict(predictions.get("ms_probabilities") or {})
    probability = probabilities.get(primary)
    return {
        "ms": primary or "—",
        "probability": None if probability is None else float(probability),
        "score": predictions.get("score") or "—",
        "confidence": predictions.get("confidence") or "—",
    }


def fixture_table_row(outcome: dict[str, Any]) -> dict[str, Any]:
    fixture = dict(outcome.get("fixture") or {})
    summary = prediction_summary(outcome.get("report"))
    probability = summary["probability"]
    return {
        "Saat": str(fixture.get("kickoff_time") or "—")[:5],
        "Lig": fixture.get("division") or "—",
        "Maç": f"{fixture.get('home_team') or '—'} — {fixture.get('away_team') or '—'}",
        "MS": summary["ms"],
        "Olasılık": "—" if probability is None else f"%{probability * 100:.1f}",
        "Güven": summary["confidence"],
        "Skor": summary["score"],
        "Durum": outcome.get("status") or "—",
    }

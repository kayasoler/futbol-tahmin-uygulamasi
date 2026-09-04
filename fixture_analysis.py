from __future__ import annotations

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
from league_mapping import division_for_api_league, match_team_name


ANALYZABLE_FIXTURE_STATUSES = {"NS", "TBD"}


def comparison_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Store complete aggregates and only the rows that the UI displays."""
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
    """Build a bounded snapshot of every historical section shown in the UI."""
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


def historical_team_names(league_rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(row.get(column)).strip()
            for row in league_rows
            for column in ("home_team", "away_team")
            if row.get(column)
        },
        key=str.casefold,
    )


def prepare_world_fixture_match(
    fixture: dict[str, Any],
    division: str,
    league_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Map API team names to historical names without accepting weak matches."""
    teams = historical_team_names(league_rows)
    home_name, home_score = match_team_name(str(fixture.get("home_team") or ""), teams)
    away_name, away_score = match_team_name(str(fixture.get("away_team") or ""), teams)
    if not home_name or not away_name:
        reason = (
            "Takım adları geçmiş veriyle güvenilir eşleşmedi: "
            f"{fixture.get('home_team') or '—'} %{home_score * 100:.0f}, "
            f"{fixture.get('away_team') or '—'} %{away_score * 100:.0f}."
        )
        return None, reason
    return {
        "id": f"api-{fixture.get('api_fixture_id')}",
        "division": division,
        "match_date": fixture.get("match_date"),
        "kickoff_time": fixture.get("kickoff_time"),
        "home_team": home_name,
        "away_team": away_name,
        "b365_home": fixture.get("b365_home"),
        "b365_draw": fixture.get("b365_draw"),
        "b365_away": fixture.get("b365_away"),
        "b365_over_25": fixture.get("b365_over_25"),
        "b365_under_25": fixture.get("b365_under_25"),
        "opening_b365_home": fixture.get("opening_b365_home"),
        "opening_b365_draw": fixture.get("opening_b365_draw"),
        "opening_b365_away": fixture.get("opening_b365_away"),
        "opening_b365_over_25": fixture.get("opening_b365_over_25"),
        "opening_b365_under_25": fixture.get("opening_b365_under_25"),
        "analysis_odds_source": fixture.get("analysis_odds_source") or "Oran bulunamadı",
        "entry_method": "api-football-batch",
        "match_status": fixture.get("status"),
    }, None


def prediction_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    predictions = dict((report or {}).get("predictions") or {})
    total_25 = dict((predictions.get("totals") or {}).get("2.5") or {})
    return {
        "ms": predictions.get("ms") or "—",
        "btts": predictions.get("btts_prediction") or "—",
        "total_25": total_25.get("prediction") or "—",
        "score": predictions.get("score") or "—",
        "confidence": predictions.get("confidence") or "—",
    }


def analyze_world_fixture(
    client,
    fixture: dict[str, Any],
    available_divisions: set[str],
    league_cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Analyze one scheduled fixture, persist it, and return a table-ready outcome."""
    outcome: dict[str, Any] = {
        "fixture": dict(fixture),
        "analysis_match": None,
        "analysis_record": None,
        "report": None,
        "status": "Desteklenmiyor",
        "reason": "",
    }
    status = str(fixture.get("status") or "").upper()
    if status not in ANALYZABLE_FIXTURE_STATUSES:
        outcome["status"] = "Maç başlamış"
        outcome["reason"] = "Yalnızca başlamamış maçlar önceden analiz edilir."
        return outcome

    division = division_for_api_league(fixture.get("league_id"), available_divisions)
    if not division:
        outcome["reason"] = "Bu lig için eşleşen geçmiş veri kodu bulunmuyor."
        return outcome

    try:
        if division not in league_cache:
            league_cache[division] = fetch_league_rows(client, division)
        league_rows = league_cache[division]
        analysis_match, mapping_error = prepare_world_fixture_match(fixture, division, league_rows)
        if not analysis_match:
            outcome["status"] = "Takım eşleşmedi"
            outcome["reason"] = mapping_error or "Takım eşleşmesi yapılamadı."
            return outcome
        outcome["analysis_match"] = analysis_match

        latest, load_error = load_latest_analysis(client, analysis_match)
        if load_error:
            raise RuntimeError(load_error)
        stored_report = restore_report_snapshot(latest)
        if latest and stored_report:
            outcome.update(
                analysis_record=latest,
                report=stored_report,
                status="Kayıtlı analiz",
            )
            return outcome
        if latest:
            outcome["status"] = "Snapshot hatası"
            outcome["reason"] = "Kayıtlı analiz snapshot'ı eksik veya bozuk."
            return outcome

        h2h_rows = fetch_h2h_rows(client, analysis_match)
        same_league_rows = fetch_same_odds_rows(client, analysis_match, division)
        same_all_rows = fetch_same_odds_rows(client, analysis_match)
        home_form_rows = fetch_team_form_rows(client, str(analysis_match["home_team"]), "home")
        away_form_rows = fetch_team_form_rows(client, str(analysis_match["away_team"]), "away")
        report = build_report(
            analysis_match,
            h2h_rows,
            same_league_rows,
            league_rows,
            home_form_rows=home_form_rows,
            away_form_rows=away_form_rows,
            same_odds_all_rows=same_all_rows,
        )
        report["evidence"] = build_analysis_evidence(
            analysis_match,
            h2h_rows,
            same_league_rows,
            same_all_rows,
            str((report.get("predictions") or {}).get("ms") or ""),
        )
        context = market_odds_context(analysis_match)
        if context:
            report["external_context"] = {"market_odds": context}
        saved, save_error = save_analysis_version(client, analysis_match, report)
        if save_error or not saved:
            raise RuntimeError(save_error or "Analiz kaydı oluşturulamadı.")
        outcome.update(
            analysis_record=saved,
            report=report,
            status="Yeni kaydedildi",
        )
        return outcome
    except Exception as exc:
        outcome["status"] = "Hata"
        outcome["reason"] = str(exc)
        return outcome


def world_fixture_table_row(outcome: dict[str, Any]) -> dict[str, Any]:
    fixture = dict(outcome.get("fixture") or {})
    analysis_match = dict(outcome.get("analysis_match") or {})
    summary = prediction_summary(outcome.get("report"))
    return {
        "Saat": fixture.get("kickoff_time") or "—",
        "Ülke": fixture.get("country") or "—",
        "Lig": fixture.get("league") or "—",
        "Veri kodu": analysis_match.get("division") or "—",
        "Ev sahibi": fixture.get("home_team") or "—",
        "Deplasman": fixture.get("away_team") or "—",
        "MS": summary["ms"],
        "KG": summary["btts"],
        "2.5": summary["total_25"],
        "Skor": summary["score"],
        "Güven": summary["confidence"],
        "Analiz durumu": outcome.get("status") or "—",
        "Açıklama": outcome.get("reason") or "",
    }

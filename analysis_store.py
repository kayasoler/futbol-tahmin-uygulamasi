from __future__ import annotations

from datetime import date, datetime, time
import hashlib
import json
from typing import Any


def analysis_match_key(match: dict[str, Any]) -> str:
    identity = "|".join(
        str(match.get(key) or "").strip().casefold()
        for key in ("division", "match_date", "home_team", "away_team")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep durable outputs while avoiding duplicate historical/API payload storage."""
    keys = ("predictions", "components", "warnings", "comment", "coupon")
    return json_safe({key: report.get(key) for key in keys})


def match_snapshot(match: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "division", "match_date", "kickoff_time", "home_team", "away_team",
        "b365_home", "b365_draw", "b365_away", "b365_over_25", "b365_under_25",
        "opening_b365_home", "opening_b365_draw", "opening_b365_away",
        "opening_b365_over_25", "opening_b365_under_25",
        "csv_b365_home", "csv_b365_draw", "csv_b365_away",
        "csv_b365_over_25", "csv_b365_under_25", "analysis_odds_source",
    )
    return json_safe({key: match.get(key) for key in keys})


def load_latest_analysis(client, match: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = (
            client.table("match_analyses")
            .select("*")
            .eq("match_key", analysis_match_key(match))
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return (dict(rows[0]) if rows else None), None
    except Exception as exc:
        return None, str(exc)


def save_analysis_version(
    client,
    match: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    match_key = analysis_match_key(match)
    try:
        response = (
            client.table("match_analyses")
            .select("version")
            .eq("match_key", match_key)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        version = int(rows[0]["version"]) + 1 if rows else 1
        payload = {
            "match_key": match_key,
            "version": version,
            "division": str(match.get("division") or ""),
            "match_date": str(match.get("match_date") or ""),
            "kickoff_time": str(match.get("kickoff_time") or "") or None,
            "home_team": str(match.get("home_team") or ""),
            "away_team": str(match.get("away_team") or ""),
            "match_snapshot": match_snapshot(match),
            "report_snapshot": compact_report(report),
            "external_context": json_safe(report.get("external_context") or {}),
        }
        inserted = client.table("match_analyses").insert(payload).execute().data or []
        return (dict(inserted[0]) if inserted else payload), None
    except Exception as exc:
        return None, str(exc)


def update_analysis_artifacts(
    client,
    analysis_id: Any,
    *,
    external_context: dict[str, Any] | None = None,
    gemini_result: dict[str, Any] | None = None,
) -> str | None:
    if not analysis_id:
        return None
    payload: dict[str, Any] = {}
    if external_context is not None:
        payload["external_context"] = json_safe(external_context)
    if gemini_result is not None:
        payload["gemini_result"] = json_safe(gemini_result)
    if not payload:
        return None
    try:
        client.table("match_analyses").update(payload).eq("id", analysis_id).execute()
        return None
    except Exception as exc:
        return str(exc)

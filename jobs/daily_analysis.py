from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
import os
from typing import Any
from zoneinfo import ZoneInfo

from daily_fixture_analysis import (
    analyze_daily_fixture,
    fixture_kickoff_has_passed,
    normalize_upcoming_fixture,
    resolve_fixture_duplicates,
)
from football_data_live import fetch_current_fixtures


ISTANBUL = ZoneInfo("Europe/Istanbul")
UPCOMING_MATCH_COLUMNS = (
    "id,division,match_date,kickoff_time,home_team,away_team,"
    "b365_home,b365_draw,b365_away,b365_over_25,b365_under_25,"
    "entry_method,match_status,raw_data"
)


def fetch_stored_fixtures(client: Any, match_date: str) -> list[dict[str, Any]]:
    """Load persistent manual and imported CSV fixtures for one day."""
    response = (
        client.table("upcoming_matches")
        .select(UPCOMING_MATCH_COLUMNS)
        .eq("match_date", match_date)
        .order("kickoff_time")
        .limit(1000)
        .execute()
    )
    return [normalize_upcoming_fixture(dict(row)) for row in (response.data or [])]


def select_daily_fixtures(
    live_rows: list[dict[str, Any]],
    stored_rows: list[dict[str, Any]],
    *,
    match_date: str,
    now: datetime,
) -> tuple[list[dict[str, Any]], int]:
    """Apply the same source priority and kickoff rules as the Streamlit page."""
    dated_rows = [
        dict(row)
        for row in live_rows + stored_rows
        if str(row.get("match_date") or "") == match_date
    ]
    fixtures, dropped = resolve_fixture_duplicates(dated_rows)
    return (
        [row for row in fixtures if not fixture_kickoff_has_passed(row, now)],
        len(dropped),
    )


def run_daily_analysis(
    client: Any,
    *,
    now: datetime | None = None,
    fixture_loader: Callable[[], list[dict[str, Any]]] = fetch_current_fixtures,
    stored_loader: Callable[[Any, str], list[dict[str, Any]]] = fetch_stored_fixtures,
    analyzer: Callable[..., dict[str, Any]] = analyze_daily_fixture,
) -> dict[str, Any]:
    """Create missing statistical snapshots without refreshing existing versions."""
    current = now or datetime.now(ISTANBUL)
    if current.tzinfo is None:
        current = current.replace(tzinfo=ISTANBUL)
    else:
        current = current.astimezone(ISTANBUL)
    match_date = current.date().isoformat()

    live_rows = list(fixture_loader())
    stored_rows = list(stored_loader(client, match_date))
    fixtures, duplicate_count = select_daily_fixtures(
        live_rows,
        stored_rows,
        match_date=match_date,
        now=current,
    )

    outcomes: list[dict[str, Any]] = []
    league_cache: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        outcomes.append(analyzer(client, fixture, league_cache, force_refresh=False))

    created = sum(row.get("status") == "Yeni kaydedildi" for row in outcomes)
    reused = sum(row.get("status") == "Kayıtlı analiz" for row in outcomes)
    unavailable = [row for row in outcomes if row.get("status") == "Analiz edilemedi"]
    return {
        "date": match_date,
        "football_data_matches": sum(
            str(row.get("match_date") or "") == match_date for row in live_rows
        ),
        "stored_matches": len(stored_rows),
        "selected_matches": len(fixtures),
        "duplicates_removed": duplicate_count,
        "created": created,
        "reused": reused,
        "unanalyzable": len(unavailable),
        "unanalyzable_matches": [
            {
                "match": (
                    f"{(row.get('fixture') or {}).get('home_team', '')} - "
                    f"{(row.get('fixture') or {}).get('away_team', '')}"
                ).strip(" -"),
                "reason": str(row.get("reason") or "Bilinmeyen neden"),
            }
            for row in unavailable
        ],
    }


def create_job_client() -> Any:
    missing = [
        name
        for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
        if not str(os.environ.get(name) or "").strip()
    ]
    if missing:
        raise RuntimeError("Eksik GitHub Secret: " + ", ".join(missing))
    from supabase import create_client

    return create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )


def main() -> int:
    try:
        summary = run_daily_analysis(create_job_client())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "completed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

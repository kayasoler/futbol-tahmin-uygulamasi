from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _fetch_team_catalog_once(
    client: Any, *, page_size: int = 1000
) -> tuple[list[str], list[str]]:
    teams: set[str] = set()
    divisions: set[str] = set()
    start = 0
    while True:
        response = (
            client.table("historical_matches")
            .select("division,home_team,away_team")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            if row.get("division"):
                divisions.add(str(row["division"]).strip())
            if row.get("home_team"):
                teams.add(str(row["home_team"]).strip())
            if row.get("away_team"):
                teams.add(str(row["away_team"]).strip())
        if len(rows) < page_size:
            break
        start += page_size
    return sorted(teams, key=str.casefold), sorted(divisions, key=str.casefold)


def fetch_team_catalog_with_retry(
    client_factory: Callable[[], Any],
    *,
    attempts: int = 2,
    page_size: int = 1000,
) -> tuple[list[str], list[str]]:
    """Retry the complete paginated read with a newly created client each time."""
    if attempts < 1:
        raise ValueError("attempts en az 1 olmalıdır")
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return _fetch_team_catalog_once(client_factory(), page_size=page_size)
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("Takım kataloğu alınamadı.")

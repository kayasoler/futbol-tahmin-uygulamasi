from __future__ import annotations

from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def parse_fixtures_csv(content: bytes) -> list[dict[str, Any]]:
    frame = pd.read_csv(BytesIO(content))
    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Football-Data fikstüründe eksik sütunlar: " + ", ".join(sorted(missing)))
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        parsed_date = pd.to_datetime(item.get("Date"), dayfirst=True, errors="coerce")
        if pd.isna(parsed_date):
            continue
        def number(column: str) -> float | None:
            value = pd.to_numeric(item.get(column), errors="coerce")
            return None if pd.isna(value) or float(value) <= 1 else float(value)
        kickoff = str(item.get("Time") or "").strip()
        if kickoff.casefold() == "nan":
            kickoff = ""
        rows.append({
            "id": f"fd-{item['Div']}-{parsed_date.date()}-{item['HomeTeam']}-{item['AwayTeam']}",
            "division": str(item["Div"]).strip(),
            "match_date": parsed_date.date().isoformat(),
            "kickoff_time": kickoff or None,
            "home_team": str(item["HomeTeam"]).strip(),
            "away_team": str(item["AwayTeam"]).strip(),
            "b365_home": number("B365H"),
            "b365_draw": number("B365D"),
            "b365_away": number("B365A"),
            "b365_over_25": number("B365>2.5"),
            "b365_under_25": number("B365<2.5"),
            "entry_method": "football-data-live",
            "match_status": "NS",
            "raw_data": {},
        })
    return rows


def fetch_current_fixtures() -> list[dict[str, Any]]:
    request = Request(FIXTURES_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=35) as response:
            return parse_fixtures_csv(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"Football-Data HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Football-Data bağlantı hatası: {exc}") from exc

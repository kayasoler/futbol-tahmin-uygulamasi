from __future__ import annotations

from io import BytesIO
from datetime import datetime
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from zoneinfo import ZoneInfo


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def parse_fixtures_csv(content: bytes, now: datetime | None = None) -> list[dict[str, Any]]:
    frame = pd.read_csv(BytesIO(content))
    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Football-Data fikstüründe eksik sütunlar: " + ", ".join(sorted(missing)))
    istanbul_now = now or datetime.now(ZoneInfo("Europe/Istanbul"))
    if istanbul_now.tzinfo is None:
        istanbul_now = istanbul_now.replace(tzinfo=ZoneInfo("Europe/Istanbul"))
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
        parsed_time = pd.to_datetime(kickoff, format="%H:%M", errors="coerce")
        if pd.isna(parsed_time):
            continue
        london_kickoff = datetime.combine(
            parsed_date.date(), parsed_time.time(), tzinfo=ZoneInfo("Europe/London")
        )
        istanbul_kickoff = london_kickoff.astimezone(ZoneInfo("Europe/Istanbul"))
        completed = any(
            not pd.isna(item.get(column)) and str(item.get(column)).strip() not in {"", "nan"}
            for column in ("FTHG", "FTAG", "FTR")
            if column in frame.columns
        )
        if completed or istanbul_kickoff <= istanbul_now:
            continue
        rows.append({
            "id": f"fd-{item['Div']}-{istanbul_kickoff.date()}-{item['HomeTeam']}-{item['AwayTeam']}",
            "division": str(item["Div"]).strip(),
            "match_date": istanbul_kickoff.date().isoformat(),
            "kickoff_time": istanbul_kickoff.strftime("%H:%M:%S"),
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


def parse_uploaded_fixtures(
    content: bytes, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Parse a user-provided fixtures.csv as a separate CSV fallback source."""
    rows = parse_fixtures_csv(content, now=now)
    for row in rows:
        row["id"] = "upload-" + str(row["id"])
        row["entry_method"] = "csv"
    return rows


def fetch_current_fixtures(
    *, attempts: int = 3, retry_delay: float = 1.0
) -> list[dict[str, Any]]:
    """Fetch fixtures with short retries for transient upstream 5xx failures."""
    if attempts < 1:
        raise ValueError("attempts en az 1 olmalıdır")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = Request(FIXTURES_URL, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=35) as response:
                return parse_fixtures_csv(response.read())
        except HTTPError as exc:
            last_error = RuntimeError(f"Football-Data HTTP {exc.code}")
        except (URLError, TimeoutError) as exc:
            last_error = RuntimeError(f"Football-Data bağlantı hatası: {exc}")
        if attempt < attempts and retry_delay > 0:
            time.sleep(retry_delay * attempt)
    raise last_error or RuntimeError("Football-Data fikstürü alınamadı.")

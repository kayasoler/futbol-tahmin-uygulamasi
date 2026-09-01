from __future__ import annotations

import json
from datetime import date, datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pandas as pd
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from supabase import Client


REQUIRED_COLUMNS = [
    "Div",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
]

FIXTURE_REQUIRED_COLUMNS = [
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "B365H",
    "B365D",
    "B365A",
]

FIXTURE_ODDS_MAP = {
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    "B365>2.5": "b365_over_25",
    "B365<2.5": "b365_under_25",
}

COLUMN_MAP = {
    "Div": "division",
    "Date": "match_date",
    "Time": "kickoff_time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "full_time_home_goals",
    "FTAG": "full_time_away_goals",
    "FTR": "full_time_result",
    "HTHG": "half_time_home_goals",
    "HTAG": "half_time_away_goals",
    "HTR": "half_time_result",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    "B365>2.5": "b365_over_25",
    "B365<2.5": "b365_under_25",
    "B365CH": "b365_closing_home",
    "B365CD": "b365_closing_draw",
    "B365CA": "b365_closing_away",
    "B365C>2.5": "b365_closing_over_25",
    "B365C<2.5": "b365_closing_under_25",
}

INTEGER_TARGETS = {
    "full_time_home_goals",
    "full_time_away_goals",
    "half_time_home_goals",
    "half_time_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
}

FLOAT_TARGETS = {
    "b365_home",
    "b365_draw",
    "b365_away",
    "b365_over_25",
    "b365_under_25",
    "b365_closing_home",
    "b365_closing_draw",
    "b365_closing_away",
    "b365_closing_over_25",
    "b365_closing_under_25",
}


def read_football_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read common football-data CSV encodings without altering the source."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(file_bytes), encoding=encoding, low_memory=False)
            frame = frame.dropna(axis=1, how="all")
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV dosyası okunamadı. Dosyanın CSV olduğundan emin olun. " + " | ".join(errors))


def validate_and_prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        raise ValueError("CSV dosyasında maç kaydı bulunamadı.")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("CSV içinde gerekli sütunlar eksik: " + ", ".join(missing))

    prepared = frame.copy()
    for column in ("Div", "HomeTeam", "AwayTeam"):
        prepared[column] = prepared[column].astype("string").str.strip()

    invalid_text = prepared[["Div", "HomeTeam", "AwayTeam"]].isna().any(axis=1)
    invalid_text |= (prepared[["Div", "HomeTeam", "AwayTeam"]] == "").any(axis=1)
    if invalid_text.any():
        rows = [str(index + 2) for index in prepared.index[invalid_text][:10]]
        raise ValueError("Lig veya takım adı eksik olan satırlar var: " + ", ".join(rows))

    parsed_dates = pd.to_datetime(prepared["Date"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        rows = [str(index + 2) for index in prepared.index[parsed_dates.isna()][:10]]
        raise ValueError("Tarihi okunamayan satırlar var: " + ", ".join(rows))

    prepared["_match_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    prepared["_home_key"] = prepared["HomeTeam"].str.lower()
    prepared["_away_key"] = prepared["AwayTeam"].str.lower()

    same_team = prepared["_home_key"] == prepared["_away_key"]
    if same_team.any():
        rows = [str(index + 2) for index in prepared.index[same_team][:10]]
        raise ValueError("Ev sahibi ve deplasman takımı aynı olan satırlar var: " + ", ".join(rows))

    duplicate_mask = prepared.duplicated(
        subset=["_match_date", "_home_key", "_away_key"],
        keep="first",
    )
    duplicate_count = int(duplicate_mask.sum())
    prepared = prepared.loc[~duplicate_mask].reset_index(drop=True)
    return prepared, duplicate_count


def validate_and_prepare_fixtures(
    frame: pd.DataFrame,
    minimum_date: date,
) -> tuple[pd.DataFrame, int, int]:
    """Validate fixture rows and keep only today/future dates."""
    if frame.empty:
        raise ValueError("Bülten CSV dosyasında maç kaydı bulunamadı.")

    missing = [column for column in FIXTURE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Bülten CSV içinde gerekli sütunlar eksik: " + ", ".join(missing))

    prepared = frame.copy()
    for column in ("Div", "HomeTeam", "AwayTeam"):
        prepared[column] = prepared[column].astype("string").str.strip()

    invalid_text = prepared[["Div", "HomeTeam", "AwayTeam"]].isna().any(axis=1)
    invalid_text |= (prepared[["Div", "HomeTeam", "AwayTeam"]] == "").any(axis=1)
    if invalid_text.any():
        rows = [str(index + 2) for index in prepared.index[invalid_text][:10]]
        raise ValueError("Lig veya takım adı eksik olan satırlar var: " + ", ".join(rows))

    parsed_dates = pd.to_datetime(prepared["Date"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        rows = [str(index + 2) for index in prepared.index[parsed_dates.isna()][:10]]
        raise ValueError("Tarihi okunamayan satırlar var: " + ", ".join(rows))

    parsed_times = prepared["Time"].apply(_safe_time)
    if parsed_times.isna().any():
        rows = [str(index + 2) for index in prepared.index[parsed_times.isna()][:10]]
        raise ValueError("Saati okunamayan satırlar var: " + ", ".join(rows))

    for column in ("B365H", "B365D", "B365A"):
        odds = pd.to_numeric(prepared[column], errors="coerce")
        invalid_odds = odds.isna() | (odds <= 1)
        if invalid_odds.any():
            rows = [str(index + 2) for index in prepared.index[invalid_odds][:10]]
            raise ValueError(f"{column} oranı geçersiz olan satırlar var: " + ", ".join(rows))

    istanbul_now = datetime.now(ZoneInfo("Europe/Istanbul"))
    converted_datetimes = [
        datetime.combine(date_value.date(), pd.to_datetime(time_value).time(), tzinfo=ZoneInfo("Europe/London"))
        .astimezone(ZoneInfo("Europe/Istanbul"))
        for date_value, time_value in zip(parsed_dates, parsed_times)
    ]
    prepared["_match_date"] = [value.date().isoformat() for value in converted_datetimes]
    prepared["_match_date_value"] = [value.date() for value in converted_datetimes]
    prepared["_kickoff_time"] = [value.strftime("%H:%M:%S") for value in converted_datetimes]
    prepared["_home_key"] = prepared["HomeTeam"].str.lower()
    prepared["_away_key"] = prepared["AwayTeam"].str.lower()

    same_team = prepared["_home_key"] == prepared["_away_key"]
    if same_team.any():
        rows = [str(index + 2) for index in prepared.index[same_team][:10]]
        raise ValueError("Ev sahibi ve deplasman takımı aynı olan satırlar var: " + ", ".join(rows))

    past_mask = pd.Series(
        [value <= istanbul_now or value.date() < minimum_date for value in converted_datetimes],
        index=prepared.index,
    )
    past_count = int(past_mask.sum())
    prepared = prepared.loc[~past_mask].copy()

    duplicate_mask = prepared.duplicated(
        subset=["_match_date", "_home_key", "_away_key"],
        keep="first",
    )
    duplicate_count = int(duplicate_mask.sum())
    prepared = prepared.loc[~duplicate_mask].reset_index(drop=True)
    return prepared, duplicate_count, past_count


def _safe_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else int(number)


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _safe_time(value: Any) -> str | None:
    text = _safe_text(value)
    if text is None:
        return None
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%H:%M:%S")


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fetch_existing_match_keys(client: Client) -> set[tuple[str, str, str]]:
    """Fetch keys in pages so Supabase's default row limit cannot hide duplicates."""
    existing: set[tuple[str, str, str]] = set()
    page_size = 1000
    start = 0

    while True:
        response = (
            client.table("historical_matches")
            .select("match_date,home_team_key,away_team_key")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            existing.add(
                (
                    str(row["match_date"]),
                    str(row["home_team_key"]),
                    str(row["away_team_key"]),
                )
            )
        if len(rows) < page_size:
            break
        start += page_size

    return existing


def fetch_existing_upcoming_match_keys(client: Client) -> set[tuple[str, str, str]]:
    """Fetch upcoming-match duplicate keys in pages."""
    existing: set[tuple[str, str, str]] = set()
    page_size = 1000
    start = 0

    while True:
        response = (
            client.table("upcoming_matches")
            .select("match_date,home_team_key,away_team_key")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            existing.add(
                (
                    str(row["match_date"]),
                    str(row["home_team_key"]),
                    str(row["away_team_key"]),
                )
            )
        if len(rows) < page_size:
            break
        start += page_size

    return existing


def build_records(
    frame: pd.DataFrame,
    source_file: str,
    existing_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    helper_columns = {"_match_date", "_home_key", "_away_key"}

    for _, row in frame.iterrows():
        key = (str(row["_match_date"]), str(row["_home_key"]), str(row["_away_key"]))
        if key in existing_keys:
            continue

        record: dict[str, Any] = {
            "division": str(row["Div"]).strip(),
            "match_date": str(row["_match_date"]),
            "home_team": str(row["HomeTeam"]).strip(),
            "away_team": str(row["AwayTeam"]).strip(),
            "source_file": source_file,
        }

        if "Time" in frame.columns:
            record["kickoff_time"] = _safe_time(row.get("Time"))

        for source, target in COLUMN_MAP.items():
            if source not in frame.columns or target in record:
                continue
            value = row.get(source)
            if target in INTEGER_TARGETS:
                record[target] = _safe_int(value)
            elif target in FLOAT_TARGETS:
                record[target] = _safe_float(value)
            else:
                record[target] = _safe_text(value)

        raw_data = {
            str(column): _json_safe(value)
            for column, value in row.items()
            if column not in helper_columns and column not in COLUMN_MAP
        }
        record["raw_data"] = json.loads(json.dumps(raw_data, ensure_ascii=False, allow_nan=False))
        records.append(record)
        existing_keys.add(key)

    return records


def build_upcoming_records(
    frame: pd.DataFrame,
    source_file: str,
    existing_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    helper_columns = {
        "_match_date",
        "_match_date_value",
        "_kickoff_time",
        "_home_key",
        "_away_key",
    }

    for _, row in frame.iterrows():
        key = (str(row["_match_date"]), str(row["_home_key"]), str(row["_away_key"]))
        if key in existing_keys:
            continue

        record: dict[str, Any] = {
            "division": str(row["Div"]).strip(),
            "match_date": str(row["_match_date"]),
            "kickoff_time": str(row["_kickoff_time"]),
            "home_team": str(row["HomeTeam"]).strip(),
            "away_team": str(row["AwayTeam"]).strip(),
            "entry_method": "csv",
        }

        for source, target in FIXTURE_ODDS_MAP.items():
            record[target] = _safe_float(row.get(source)) if source in frame.columns else None

        raw_data = {"source_file": source_file}
        raw_data.update(
            {
                str(column): _json_safe(value)
                for column, value in row.items()
                if column not in helper_columns and column not in FIXTURE_ODDS_MAP
            }
        )
        record["raw_data"] = json.loads(json.dumps(raw_data, ensure_ascii=False, allow_nan=False))
        records.append(record)
        existing_keys.add(key)

    return records


def _is_duplicate_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "23505" in message or "duplicate key" in message or "unique constraint" in message


def insert_new_records(client: Client, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert quickly in batches; retry individually only if a race creates a duplicate."""
    inserted = 0
    race_duplicates = 0
    batch_size = 200

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            client.table("historical_matches").insert(batch, returning="minimal").execute()
            inserted += len(batch)
        except Exception as batch_error:
            if len(batch) == 1:
                if _is_duplicate_error(batch_error):
                    race_duplicates += 1
                    continue
                raise

            for record in batch:
                try:
                    client.table("historical_matches").insert(
                        record,
                        returning="minimal",
                    ).execute()
                    inserted += 1
                except Exception as row_error:
                    if _is_duplicate_error(row_error):
                        race_duplicates += 1
                    else:
                        raise

    return inserted, race_duplicates


def insert_upcoming_records(client: Client, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert upcoming fixtures in batches while treating unique matches as duplicates."""
    inserted = 0
    race_duplicates = 0
    batch_size = 200

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            client.table("upcoming_matches").insert(batch, returning="minimal").execute()
            inserted += len(batch)
        except Exception as batch_error:
            if len(batch) == 1:
                if _is_duplicate_error(batch_error):
                    race_duplicates += 1
                    continue
                raise

            for record in batch:
                try:
                    client.table("upcoming_matches").insert(
                        record,
                        returning="minimal",
                    ).execute()
                    inserted += 1
                except Exception as row_error:
                    if _is_duplicate_error(row_error):
                        race_duplicates += 1
                    else:
                        raise

    return inserted, race_duplicates

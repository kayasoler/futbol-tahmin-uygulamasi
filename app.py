from __future__ import annotations

import hmac
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from supabase import Client, create_client

from analysis import (
    build_report,
    decision_summary,
    exact_odds_diagnostic,
    fetch_h2h_rows,
    fetch_league_rows,
    fetch_same_odds_rows,
    fetch_team_form_rows,
    generate_grounded_analysis,
    h2h_summary_tables,
    market_odds_context,
    odds_movement,
    totals_odds_movement,
    rows_to_table,
)
from backtest import aggregate_backtests, run_backtest
from calibration import calibrate_model
from analysis_store import (
    evaluate_analysis,
    load_analysis_history,
    load_latest_analysis,
    load_match_results,
    restore_report_snapshot,
    save_analysis_version,
    save_match_result,
    update_analysis_artifacts,
)
from api_football import fetch_fixtures, normalize_api_keys
from football_data_live import fetch_current_fixtures
from fixture_analysis import (
    analyze_world_fixture,
    build_analysis_evidence,
    world_fixture_table_row,
)
from highlightly import (
    fetch_last_five,
    fetch_lineups,
    fetch_match_day,
    fetch_standings,
    find_match,
    form_rows,
    selected_standings,
)
from results_api import fetch_match_result
from data_import import (
    FIXTURE_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    build_records,
    build_upcoming_records,
    fetch_existing_match_keys,
    fetch_existing_upcoming_match_keys,
    insert_new_records,
    insert_upcoming_records,
    read_football_csv,
    validate_and_prepare,
    validate_and_prepare_fixtures,
)


st.set_page_config(
    page_title="Futbol Tahmin ve Analiz",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.10);
        border: 1px solid rgba(127, 127, 127, 0.28);
        border-radius: 14px;
        padding: 14px;
    }
    .small-note {font-size: 0.9rem; opacity: 0.72;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    service_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, service_key)


def stop_if_secrets_missing() -> None:
    try:
        _ = st.secrets["SUPABASE_URL"]
        _ = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        _ = st.secrets["APP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        st.info(
            "Uygulama hazır. Supabase bağlantısı ve uygulama parolası için "
            "Streamlit Secrets ayarlarının eklenmesi gerekiyor."
        )
        st.stop()


def require_login() -> None:
    if st.session_state.get("authenticated") is True:
        return

    st.subheader("🔐 Güvenli giriş")
    st.write("Veri yönetimi ekranını açmak için uygulama parolanızı girin.")
    password = st.text_input("Uygulama parolası", type="password")

    if st.button("Giriş yap", type="primary", use_container_width=True):
        if password and hmac.compare_digest(password, st.secrets["APP_PASSWORD"]):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Parola hatalı.")
    st.stop()


def get_table_count(client: Client, table_name: str) -> int:
    response = client.table(table_name).select("id", count="exact").limit(1).execute()
    return int(response.count or 0)


def fetch_team_catalog(client: Client) -> tuple[list[str], list[str]]:
    teams: set[str] = set()
    divisions: set[str] = set()
    page_size = 1000
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


def fetch_recent_divisions(client: Client, maximum_rows: int = 5000) -> list[str]:
    """Build a practical league list from the most recent completed matches."""
    divisions: set[str] = set()
    page_size = 1000
    for start in range(0, maximum_rows, page_size):
        rows = (
            client.table("historical_matches")
            .select("division,match_date")
            .order("match_date", desc=True)
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )
        divisions.update(
            str(row["division"]).strip()
            for row in rows
            if row.get("division")
        )
        if len(rows) < page_size:
            break
    return sorted(divisions, key=str.casefold)


def is_duplicate_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "23505" in message or "duplicate key" in message or "unique constraint" in message


def render_historical_page(client: Client) -> None:
    st.caption("Aşama 1 · Geçmiş maç verilerinin güvenli ve tekrarsız yüklenmesi")

    if "last_import_summary" in st.session_state:
        summary = st.session_state.pop("last_import_summary")
        st.success(
            f"{summary['selected_files']} CSV dosyasından "
            f"{summary['inserted']} yeni maç başarıyla eklendi."
        )
        if (
            summary["file_duplicates"]
            or summary["cross_file_duplicates"]
            or summary["existing"]
            or summary["race_duplicates"]
        ):
            st.info(
                f"Atlanan tekrarlar — dosya içinde: {summary['file_duplicates']}, "
                f"seçilen dosyalar arasında: {summary['cross_file_duplicates']}, "
                f"veritabanında zaten bulunan: {summary['existing']}, "
                f"eşzamanlı yakalanan: {summary['race_duplicates']}."
            )

    current_count = get_table_count(client, "historical_matches")
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Veritabanındaki geçmiş maç", f"{current_count:,}".replace(",", "."))
    metric_right.metric("Duplicate koruması", "Aktif")

    st.divider()
    st.subheader("Geçmiş maç CSV dosyalarını toplu yükle")
    st.write(
        "football-data.co.uk üzerinden indirdiğiniz tüm geçmiş CSV dosyalarını aynı anda "
        "seçin. Dosyalar birlikte kontrol edilir; siz düğmeye basmadan veritabanına yazılmaz."
    )

    uploaded_files = st.file_uploader(
        "Geçmiş CSV dosyalarını seçin",
        type=["csv"],
        accept_multiple_files=True,
        help="Windows'ta dosyaları Ctrl+A ile topluca seçebilirsiniz.",
        key="historical_uploader",
    )

    if not uploaded_files:
        return

    valid_files: list[tuple[str, pd.DataFrame]] = []
    file_results: list[dict[str, object]] = []
    total_raw_rows = 0
    file_duplicate_count = 0

    for uploaded_file in uploaded_files:
        try:
            raw_frame = read_football_csv(uploaded_file.getvalue())
            prepared_frame, duplicates_in_file = validate_and_prepare(raw_frame)
            prepared_frame = prepared_frame.copy()
            prepared_frame["__source_file"] = uploaded_file.name

            valid_files.append((uploaded_file.name, prepared_frame))
            total_raw_rows += len(raw_frame)
            file_duplicate_count += duplicates_in_file
            file_results.append(
                {
                    "Dosya": uploaded_file.name,
                    "Durum": "Hazır",
                    "Satır": len(raw_frame),
                    "Benzersiz": len(prepared_frame),
                    "Dosya içi tekrar": duplicates_in_file,
                    "Açıklama": "",
                }
            )
        except ValueError as exc:
            file_results.append(
                {
                    "Dosya": uploaded_file.name,
                    "Durum": "Hata",
                    "Satır": 0,
                    "Benzersiz": 0,
                    "Dosya içi tekrar": 0,
                    "Açıklama": str(exc),
                }
            )

    st.dataframe(file_results, use_container_width=True, hide_index=True)

    invalid_file_count = sum(result["Durum"] == "Hata" for result in file_results)
    if invalid_file_count:
        st.error(
            f"{invalid_file_count} dosya doğrulanamadı. Güvenli toplu işlem için hiçbir dosya "
            "veritabanına yazılmayacak."
        )
        st.caption("Gerekli temel sütunlar: " + ", ".join(REQUIRED_COLUMNS))
        return

    combined_frame = pd.concat(
        [prepared for _, prepared in valid_files],
        ignore_index=True,
        sort=False,
    )
    cross_file_duplicate_mask = combined_frame.duplicated(
        subset=["_match_date", "_home_key", "_away_key"],
        keep="first",
    )
    cross_file_duplicate_count = int(cross_file_duplicate_mask.sum())
    combined_frame = combined_frame.loc[~cross_file_duplicate_mask].reset_index(drop=True)

    preview_cols = [
        "__source_file",
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "B365H",
        "B365D",
        "B365A",
    ]
    preview_cols = [column for column in preview_cols if column in combined_frame.columns]
    preview_frame = combined_frame[preview_cols].head(20).rename(
        columns={"__source_file": "Dosya"}
    )

    st.success(f"{len(uploaded_files)} CSV dosyasının tamamı geçerli.")
    a, b, c, d = st.columns(4)
    a.metric("Seçilen CSV", len(uploaded_files))
    b.metric("Toplam satır", total_raw_rows)
    c.metric("Benzersiz maç", len(combined_frame))
    d.metric("Toplam tekrar", file_duplicate_count + cross_file_duplicate_count)

    st.dataframe(preview_frame, use_container_width=True, hide_index=True)

    if st.button(
        "Tüm geçmiş CSV dosyalarını veritabanına aktar",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Tüm geçmiş kayıtlar kontrol ediliyor ve yükleniyor..."):
            try:
                existing_keys = fetch_existing_match_keys(client)
                records: list[dict[str, object]] = []
                for source_file, source_frame in combined_frame.groupby(
                    "__source_file", sort=False
                ):
                    records.extend(
                        build_records(
                            source_frame.drop(columns=["__source_file"]),
                            source_file=str(source_file),
                            existing_keys=existing_keys,
                        )
                    )
                already_existing = len(combined_frame) - len(records)
                inserted, race_duplicates = insert_new_records(client, records)
            except Exception as exc:
                st.error("Toplu yükleme tamamlanamadı. Gizli bilgileri paylaşmadan hatayı gönderin.")
                st.code(str(exc))
                return

        st.session_state["last_import_summary"] = {
            "inserted": inserted,
            "selected_files": len(uploaded_files),
            "file_duplicates": file_duplicate_count,
            "cross_file_duplicates": cross_file_duplicate_count,
            "existing": already_existing,
            "race_duplicates": race_duplicates,
        }
        st.rerun()


def render_fixture_csv_tab(client: Client, today) -> None:
    st.write(
        "Bir veya birden fazla bülten CSV dosyası seçebilirsiniz. Yalnızca bugün ve "
        "sonraki tarihlerdeki maçlar alınır."
    )

    fixture_files = st.file_uploader(
        "Bülten CSV dosyalarını seçin",
        type=["csv"],
        accept_multiple_files=True,
        key="fixture_uploader",
    )

    if not fixture_files:
        return

    valid_files: list[pd.DataFrame] = []
    file_results: list[dict[str, object]] = []
    total_rows = 0
    past_rows = 0
    within_file_duplicates = 0

    for fixture_file in fixture_files:
        try:
            raw_frame = read_football_csv(fixture_file.getvalue())
            prepared, duplicates, past_count = validate_and_prepare_fixtures(raw_frame, today)
            prepared = prepared.copy()
            prepared["__source_file"] = fixture_file.name
            valid_files.append(prepared)
            total_rows += len(raw_frame)
            past_rows += past_count
            within_file_duplicates += duplicates
            file_results.append(
                {
                    "Dosya": fixture_file.name,
                    "Durum": "Hazır",
                    "Toplam satır": len(raw_frame),
                    "Bugün/gelecek": len(prepared),
                    "Geçmiş tarih": past_count,
                    "Tekrar": duplicates,
                    "Açıklama": "",
                }
            )
        except ValueError as exc:
            file_results.append(
                {
                    "Dosya": fixture_file.name,
                    "Durum": "Hata",
                    "Toplam satır": 0,
                    "Bugün/gelecek": 0,
                    "Geçmiş tarih": 0,
                    "Tekrar": 0,
                    "Açıklama": str(exc),
                }
            )

    st.dataframe(file_results, use_container_width=True, hide_index=True)

    invalid_count = sum(result["Durum"] == "Hata" for result in file_results)
    if invalid_count:
        st.error(
            f"{invalid_count} bülten dosyası hatalı. Güvenli işlem için hiçbir dosya "
            "veritabanına yazılmayacak."
        )
        st.caption("Gerekli bülten sütunları: " + ", ".join(FIXTURE_REQUIRED_COLUMNS))
        return

    non_empty_files = [frame for frame in valid_files if not frame.empty]
    if not non_empty_files:
        st.warning("Seçilen dosyalarda bugün veya sonrasına ait maç bulunamadı.")
        return

    combined = pd.concat(non_empty_files, ignore_index=True, sort=False)
    cross_duplicate_mask = combined.duplicated(
        subset=["_match_date", "_home_key", "_away_key"], keep="first"
    )
    cross_duplicates = int(cross_duplicate_mask.sum())
    combined = combined.loc[~cross_duplicate_mask].reset_index(drop=True)

    preview_columns = [
        "__source_file",
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "B365H",
        "B365D",
        "B365A",
        "B365>2.5",
        "B365<2.5",
    ]
    preview_columns = [column for column in preview_columns if column in combined.columns]
    preview = combined[preview_columns].head(30).rename(columns={"__source_file": "Dosya"})

    a, b, c, d = st.columns(4)
    a.metric("Seçilen bülten", len(fixture_files))
    b.metric("Toplam satır", total_rows)
    c.metric("Aktarılabilir maç", len(combined))
    d.metric("Geçmiş tarih — atlandı", past_rows)

    if within_file_duplicates or cross_duplicates:
        st.info(
            f"Tekrarlar — dosya içinde: {within_file_duplicates}, "
            f"dosyalar arasında: {cross_duplicates}."
        )

    st.dataframe(preview, use_container_width=True, hide_index=True)

    if st.button(
        "Tüm bülten maçlarını veritabanına aktar",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Bülten maçları kontrol ediliyor ve yükleniyor..."):
            try:
                existing_keys = fetch_existing_upcoming_match_keys(client)
                records: list[dict[str, object]] = []
                for source_file, source_frame in combined.groupby("__source_file", sort=False):
                    records.extend(
                        build_upcoming_records(
                            source_frame.drop(columns=["__source_file"]),
                            source_file=str(source_file),
                            existing_keys=existing_keys,
                        )
                    )
                already_existing = len(combined) - len(records)
                inserted, race_duplicates = insert_upcoming_records(client, records)
            except Exception as exc:
                st.error("Bülten yüklenemedi. Gizli bilgileri paylaşmadan hatayı gönderin.")
                st.code(str(exc))
                return

        st.session_state["last_fixture_summary"] = {
            "files": len(fixture_files),
            "inserted": inserted,
            "past": past_rows,
            "within_duplicates": within_file_duplicates,
            "cross_duplicates": cross_duplicates,
            "existing": already_existing,
            "race_duplicates": race_duplicates,
        }
        st.rerun()


def render_manual_fixture_tab(client: Client, today) -> None:
    teams, divisions = fetch_team_catalog(client)
    if len(teams) < 2 or not divisions:
        st.error("Manuel giriş için geçmiş verilerden yeterli takım veya lig bulunamadı.")
        return

    st.write("Takım kutularında yazarak veritabanındaki takım adlarını filtreleyebilirsiniz.")

    with st.form("manual_fixture_form", clear_on_submit=False):
        division = st.selectbox("Lig", divisions)
        date_col, time_col = st.columns(2)
        match_date = date_col.date_input("Maç tarihi", value=today, min_value=today)
        kickoff_time = time_col.time_input("Maç saati", value=time(20, 0))

        team_left, team_right = st.columns(2)
        home_team = team_left.selectbox("Ev sahibi", teams, index=0)
        away_team = team_right.selectbox("Deplasman", teams, index=1)

        st.markdown("##### Analiz anında gördüğünüz oranlar")
        odd_1, odd_x, odd_2 = st.columns(3)
        b365_home = odd_1.number_input("Analiz anı MS 1", min_value=1.01, value=2.00, step=0.01)
        b365_draw = odd_x.number_input("Analiz anı X", min_value=1.01, value=3.00, step=0.01)
        b365_away = odd_2.number_input("Analiz anı MS 2", min_value=1.01, value=3.00, step=0.01)

        over_col, under_col = st.columns(2)
        b365_over = over_col.number_input(
            "Analiz anı 2.5 Üst — yoksa 0", min_value=0.0, value=0.0, step=0.01
        )
        b365_under = under_col.number_input(
            "Analiz anı 2.5 Alt — yoksa 0", min_value=0.0, value=0.0, step=0.01
        )

        st.markdown("##### İsteğe bağlı Bet365 açılış oranları")
        st.caption("Bulamadığınız grubu tamamen boş bırakabilirsiniz.")
        opening_1, opening_x, opening_2 = st.columns(3)
        opening_home = opening_1.number_input(
            "Açılış MS 1", min_value=0.0, value=0.0, step=0.01
        )
        opening_draw = opening_x.number_input(
            "Açılış X", min_value=0.0, value=0.0, step=0.01
        )
        opening_away = opening_2.number_input(
            "Açılış MS 2", min_value=0.0, value=0.0, step=0.01
        )
        opening_over_col, opening_under_col = st.columns(2)
        opening_over = opening_over_col.number_input(
            "Açılış 2.5 Üst", min_value=0.0, value=0.0, step=0.01
        )
        opening_under = opening_under_col.number_input(
            "Açılış 2.5 Alt", min_value=0.0, value=0.0, step=0.01
        )

        submitted = st.form_submit_button(
            "Maçı kaydet", type="primary", use_container_width=True
        )

    if not submitted:
        return

    if home_team == away_team:
        st.error("Ev sahibi ve deplasman takımı aynı olamaz.")
        return
    opening_ms = (opening_home, opening_draw, opening_away)
    if any(value > 0 for value in opening_ms) and not all(value > 1 for value in opening_ms):
        st.error("Açılış MS oranlarının üçünü birlikte girin veya üçünü de boş bırakın.")
        return
    opening_totals = (opening_over, opening_under)
    if any(value > 0 for value in opening_totals) and not all(value > 1 for value in opening_totals):
        st.error("Açılış 2.5 Üst ve Alt oranlarını birlikte girin veya ikisini de boş bırakın.")
        return

    record = {
        "division": division,
        "match_date": match_date.isoformat(),
        "kickoff_time": kickoff_time.strftime("%H:%M:%S"),
        "home_team": home_team,
        "away_team": away_team,
        "b365_home": float(b365_home),
        "b365_draw": float(b365_draw),
        "b365_away": float(b365_away),
        "b365_over_25": float(b365_over) if b365_over > 0 else None,
        "b365_under_25": float(b365_under) if b365_under > 0 else None,
        "entry_method": "manual",
        "raw_data": {
            "opening_odds": {
                "opening_b365_home": float(opening_home) if opening_home > 1 else None,
                "opening_b365_draw": float(opening_draw) if opening_draw > 1 else None,
                "opening_b365_away": float(opening_away) if opening_away > 1 else None,
                "opening_b365_over_25": float(opening_over) if opening_over > 1 else None,
                "opening_b365_under_25": float(opening_under) if opening_under > 1 else None,
            }
        },
    }

    try:
        client.table("upcoming_matches").insert(record, returning="minimal").execute()
    except Exception as exc:
        if is_duplicate_error(exc):
            st.warning("Bu tarih ve takım eşleşmesi yaklaşan maçlarda zaten kayıtlı.")
        else:
            st.error("Maç kaydedilemedi. Gizli bilgileri paylaşmadan hatayı gönderin.")
            st.code(str(exc))
        return

    st.session_state["last_manual_fixture"] = (
        f"{match_date.strftime('%d.%m.%Y')} · {home_team} — {away_team} kaydedildi."
    )
    st.rerun()


def render_match_analysis(client: Client, match: dict[str, object]) -> None:
    requested_match = dict(match)
    latest_analysis, storage_error = load_latest_analysis(client, requested_match)
    refresh_key = f"force_analysis_refresh_{requested_match.get('id')}"
    force_refresh = bool(st.session_state.pop(refresh_key, False))
    if force_refresh:
        st.session_state.pop(f"highlightly_context_{requested_match.get('id')}", None)
        st.session_state.pop(f"tavily_gemini_analysis_v3_{requested_match.get('id')}", None)
    if latest_analysis and not force_refresh:
        stored_match = dict(latest_analysis.get("match_snapshot") or {})
        stored_match["id"] = requested_match.get("id")
        match = stored_match
    else:
        match = requested_match

    analysis_record = latest_analysis if latest_analysis and not force_refresh else None
    stored_report = restore_report_snapshot(analysis_record)

    home = str(match.get("home_team") or "")
    away = str(match.get("away_team") or "")
    division = str(match.get("division") or "")
    st.divider()
    st.subheader(f"📊 Maç analizi · {home} — {away}")
    st.caption(
        f"{division} · {match.get('match_date', '—')} · {match.get('kickoff_time', '—')}"
    )

    if analysis_record and stored_report:
        st.success(
            f"Kayıtlı analiz sürümü {latest_analysis.get('version')} kullanılıyor · "
            f"{latest_analysis.get('analyzed_at', '—')}"
        )
        st.caption("Bu açılışta Highlightly, Tavily ve Gemini kotası harcanmaz.")
    elif analysis_record:
        st.warning("Kayıtlı analiz snapshot'ı eksik veya bozuk; otomatik yeniden hesaplama yapılmadı.")
    if st.button(
        "Yeni oranlarla yeniden analiz et" if latest_analysis else "Analizi yenile",
        key=f"refresh_analysis_{requested_match.get('id')}",
        use_container_width=True,
    ):
        st.session_state[refresh_key] = True
        st.rerun()

    if analysis_record:
        if stored_report is None:
            st.info("Yeni ve geçerli bir sürüm oluşturmak için ‘Yeni oranlarla yeniden analiz et’ düğmesini kullanın.")
            return
        report = stored_report
    else:
        with st.spinner("Geçmiş rekabet ve aynı oran verileri hesaplanıyor..."):
            try:
                h2h_rows = fetch_h2h_rows(client, match)
                league_rows = fetch_league_rows(client, division)
                same_league_rows = fetch_same_odds_rows(client, match, division)
                same_all_rows = fetch_same_odds_rows(client, match)
                home_form_rows = fetch_team_form_rows(client, home, "home")
                away_form_rows = fetch_team_form_rows(client, away, "away")
                report = build_report(
                    match,
                    h2h_rows,
                    same_league_rows,
                    league_rows,
                    home_form_rows=home_form_rows,
                    away_form_rows=away_form_rows,
                    same_odds_all_rows=same_all_rows,
                )
                report["evidence"] = build_analysis_evidence(
                    match,
                    h2h_rows,
                    same_league_rows,
                    same_all_rows,
                    str((report.get("predictions") or {}).get("ms") or ""),
                )
                market_context = market_odds_context(match)
                if market_context:
                    report["external_context"] = {"market_odds": market_context}
            except Exception as exc:
                st.error("Analiz verileri alınamadı.")
                st.code(str(exc))
                return

    if analysis_record:
        stored_context = dict(analysis_record.get("external_context") or {})
        if stored_context:
            report.setdefault("external_context", {}).update(stored_context)
    else:
        analysis_record, save_error = save_analysis_version(client, match, report)
        if save_error:
            st.warning("Analiz veritabanına kaydedilemedi; bu oturumda çalışmaya devam eder.")
            st.caption("Supabase'de supabase_match_analyses.sql dosyasını bir kez çalıştırın. Ayrıntı: " + save_error)
        elif analysis_record:
            st.success(f"Analiz sürümü {analysis_record.get('version', 1)} veritabanına kaydedildi.")
    if storage_error and "match_analyses" in storage_error:
        st.caption("Kalıcı analiz önbelleği henüz kurulmamış olabilir.")

    evidence = dict(report.get("evidence") or {})
    h2h_rows = list(evidence.get("h2h_rows") or [])
    same_league_evidence = dict(evidence.get("same_league") or {})
    same_all_evidence = dict(evidence.get("same_all") or {})
    exact_league_evidence = dict(evidence.get("exact_league") or {})
    exact_all_evidence = dict(evidence.get("exact_all") or {})
    legacy_evidence_missing = bool(analysis_record and not evidence)
    if legacy_evidence_missing:
        st.caption(
            "Bu eski analiz sürümünde ayrıntılı kanıt satırları saklanmamış. "
            "Tahmin snapshot'ı aynen gösteriliyor; geçmiş veri yeniden sorgulanmıyor."
        )

    st.info(
        "İstatistiksel model geçmiş verilerle çalışır. Tavily güncel haberleri arar; "
        "Groq bu kaynakları ve tüm istatistikleri birleştirerek kendi yorumunu üretir."
    )

    movement_rows = odds_movement(
        tuple(match.get(column) for column in ("opening_b365_home", "opening_b365_draw", "opening_b365_away")),
        tuple(match.get(column) for column in ("b365_home", "b365_draw", "b365_away")),
    )
    movement_rows.extend(totals_odds_movement(
        match.get("opening_b365_over_25"), match.get("opening_b365_under_25"),
        match.get("b365_over_25"), match.get("b365_under_25"),
    ))
    predictions = report["predictions"]
    saved_gemini = dict((analysis_record or {}).get("gemini_result") or {}) or None
    final_summary = decision_summary(report, saved_gemini, movement_rows)
    st.markdown("## 🎯 Nihai analiz özeti")
    summary_left, summary_right = st.columns(2)
    summary_left.metric("Ortak seçim", final_summary["Ortak seçim"])
    summary_right.metric("Veri uyumu", final_summary["Uyum"])
    st.dataframe(
        pd.DataFrame([
            {"Kaynak": "İstatistiksel model", "Görüş": final_summary["Model"]},
            {"Kaynak": "Yapay zekâ yorumu", "Görüş": final_summary["Gemini"]},
            {"Kaynak": "Oran hareketi", "Görüş": final_summary["Piyasa"]},
        ]),
        use_container_width=True,
        hide_index=True,
    )
    ms_probabilities = predictions.get("ms_probabilities") or {}
    ms_primary = str(predictions.get("ms") or "—").replace("MS", "").strip()
    ht_options = predictions.get("ht_ms_options") or []
    score_options = predictions.get("score_options") or []
    btts_probability = float(predictions.get("btts_probability") or 0)
    btts_pick_probability = btts_probability if predictions.get("btts_prediction") == "KG Var" else 1 - btts_probability
    total_25 = (predictions.get("totals") or {}).get("2.5") or {}
    probability_rows = [
        {
            "Pazar": "Maç sonucu",
            "Ana tahmin": f"{ms_primary} · %{float(ms_probabilities.get(ms_primary, 0)) * 100:.1f}",
            "Diğer ihtimaller": " · ".join(
                f"{label} %{float(ms_probabilities.get(label, 0)) * 100:.1f}"
                for label in ("1", "X", "2") if label != ms_primary
            ),
        },
        {
            "Pazar": "İY/MS",
            "Ana tahmin": (
                f"{predictions.get('ht_ms')} · %{float(predictions.get('ht_ms_probability') or 0) * 100:.1f}"
            ),
            "Diğer ihtimaller": " · ".join(
                f"{item['selection']} %{float(item['probability']) * 100:.1f}"
                for item in ht_options if item.get("selection") != predictions.get("ht_ms")
            ),
        },
        {
            "Pazar": "Skor",
            "Ana tahmin": f"{predictions.get('score')} · %{float(predictions.get('score_probability') or 0) * 100:.1f}",
            "Diğer ihtimaller": " · ".join(
                f"{item['score']} %{float(item['probability']) * 100:.1f}"
                for item in score_options if item.get("score") != predictions.get("score")
            ),
        },
        {
            "Pazar": "Karşılıklı gol",
            "Ana tahmin": f"{predictions.get('btts_prediction')} · %{btts_pick_probability * 100:.1f}",
            "Diğer ihtimaller": (
                f"{'KG Yok' if predictions.get('btts_prediction') == 'KG Var' else 'KG Var'} · %{(1 - btts_pick_probability) * 100:.1f}"
            ),
        },
        {
            "Pazar": "2.5 Alt/Üst",
            "Ana tahmin": f"{total_25.get('prediction', '—')} · %{float(total_25.get('prediction_probability') or 0) * 100:.1f}",
            "Diğer ihtimaller": (
                f"{'Alt' if total_25.get('prediction') == 'Üst' else 'Üst'} · "
                f"%{(1 - float(total_25.get('prediction_probability') or 0)) * 100:.1f}"
            ),
        },
    ]
    st.markdown("#### Olasılıklı tahmin görünümü")
    st.dataframe(pd.DataFrame(probability_rows), use_container_width=True, hide_index=True)
    st.caption("Bu kart kaynakların ortak yönünü özetler; kesin sonuç veya kazanç garantisi değildir.")

    with st.expander("1. Geçmiş rekabet · Son 10 maç", expanded=False):
        if h2h_rows:
            outcome_summary, goal_summary = h2h_summary_tables(h2h_rows, home, away)
            st.dataframe(outcome_summary, use_container_width=True, hide_index=True)
            st.dataframe(goal_summary, use_container_width=True, hide_index=True)
            st.dataframe(rows_to_table(h2h_rows), use_container_width=True, hide_index=True)
        elif legacy_evidence_missing:
            st.info("H2H kanıt satırları bu eski snapshot sürümünde saklanmamış.")
        else:
            st.info("Bu iki takım arasında veritabanında geçmiş karşılaşma bulunamadı.")

    with st.expander("2. Açılış–analiz anı oran hareketi", expanded=False):
        if movement_rows:
            movement_frame = pd.DataFrame(movement_rows).rename(
                columns={"Güncel olasılık": "Analiz anı olasılığı"}
            )
            for column in ("Açılış olasılığı", "Analiz anı olasılığı", "Hareket"):
                movement_frame[column] = movement_frame[column].map(lambda value: f"{float(value) * 100:+.1f}%")
            st.dataframe(movement_frame, use_container_width=True, hide_index=True)
        else:
            st.caption("Açılış oranları boşsa hareket hesaplanmaz; analiz mevcut referans oranıyla devam eder.")

    with st.expander("3. Benzer Bet365 piyasa analizi", expanded=False):
        st.caption(
            "Marjı temizlenmiş olasılık toleransları kullanılır. Özet tüm eşleşmeleri, ayrıntı yalnızca en yakın 20 maçı gösterir."
        )
        st.markdown("##### A) Aynı ligde benzer oranlar")
        if same_league_evidence.get("sample_count"):
            st.dataframe(
                pd.DataFrame(same_league_evidence.get("summary") or []),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                rows_to_table(list(same_league_evidence.get("rows") or [])),
                use_container_width=True,
                hide_index=True,
            )
        elif legacy_evidence_missing:
            st.info("Benzer oran kanıtları bu eski snapshot sürümünde saklanmamış.")
        else:
            st.info("Bu ligde tolerans aralığına giren benzer piyasa bulunamadı.")
        st.markdown("##### B) Tüm liglerde benzer oranlar")
        if same_all_evidence.get("sample_count"):
            st.dataframe(
                pd.DataFrame(same_all_evidence.get("summary") or []),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                rows_to_table(list(same_all_evidence.get("rows") or [])),
                use_container_width=True,
                hide_index=True,
            )
        elif legacy_evidence_missing:
            st.info("Benzer oran kanıtları bu eski snapshot sürümünde saklanmamış.")
        else:
            st.info("Tüm geçmiş veriler içinde benzer piyasa bulunamadı.")

    with st.expander("3B. Birebir Bet365 oran doğrulaması · ana modeli etkilemez", expanded=False):
        st.caption(
            "Analiz anındaki MS 1/X/2 oranlarının iki ondalıkta tamamen aynı olduğu geçmiş "
            "maçlar gösterilir. Bu bölüm yalnızca ek fikir verir; tahmin ağırlıklarını değiştirmez."
        )
        for title, exact_evidence in (
            ("Aynı ligde birebir oranlar", exact_league_evidence),
            ("Tüm liglerde birebir oranlar", exact_all_evidence),
        ):
            st.markdown(f"##### {title}")
            if legacy_evidence_missing:
                st.info("Birebir oran kanıtları bu eski snapshot sürümünde saklanmamış.")
                continue
            exact_rows = list(exact_evidence.get("rows") or [])
            diagnostic = dict(exact_evidence.get("diagnostic") or {})
            if not diagnostic:
                diagnostic = exact_odds_diagnostic([], predictions.get("ms") or "")
            exact_metrics = st.columns(4)
            exact_metrics[0].metric("Örneklem", diagnostic["count"])
            exact_metrics[1].metric("Güven", diagnostic["confidence"])
            exact_metrics[2].metric("En sık MS", diagnostic["leader"])
            exact_metrics[3].metric("Ana model ilişkisi", diagnostic["relation"])
            if exact_evidence.get("sample_count"):
                st.dataframe(
                    pd.DataFrame(exact_evidence.get("summary") or []),
                    use_container_width=True,
                    hide_index=True,
                )
                st.dataframe(
                    rows_to_table(exact_rows[:20]), use_container_width=True, hide_index=True
                )
            else:
                st.info("Bu kapsamda birebir üçlü oran eşleşmesi bulunamadı.")

    with st.expander("4. Ayrıntılı model tahminleri", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("MS", predictions["ms"])
        p2.metric("İY/MS", predictions["ht_ms"])
        p3.metric("Skor", predictions["score"])
        p4.metric("KG", predictions["btts_prediction"])
        st.dataframe(pd.DataFrame([
            {"Sonuç": label, "Birleşik model olasılığı": f"{float(ms_probabilities.get(label, 0)) * 100:.1f}%"}
            for label in ("1", "X", "2")
        ]), use_container_width=True, hide_index=True)
        st.caption(
            f"Beklenen goller: {home} {float(predictions.get('expected_home_goals', 0)):.2f} — "
            f"{away} {float(predictions.get('expected_away_goals', 0)):.2f} · Model güveni: {predictions.get('confidence', '—')}"
        )
        totals = predictions["totals"]
        st.dataframe(pd.DataFrame([
            {"Baremi": f"{threshold} üst/alt", "Tahmin": data["prediction"],
             "Tahmin olasılığı": f"{data['prediction_probability'] * 100:.1f}%" if data.get("prediction_probability") is not None else "—"}
            for threshold, data in totals.items()
        ]), use_container_width=True, hide_index=True)
        components = report.get("components") or []
        if components:
            st.markdown("##### Tahmini etkileyen veri kaynakları")
            component_frame = pd.DataFrame(components)
            for column in ("Ağırlık", "1", "X", "2"):
                component_frame[column] = component_frame[column].map(lambda value: f"{float(value) * 100:.1f}%")
            st.dataframe(component_frame, use_container_width=True, hide_index=True)

    with st.expander("İstatistiksel yorum ve riskler", expanded=False):
        for warning in report.get("warnings") or []:
            st.warning(str(warning))
        st.write(report["comment"])
        st.info(f"İstatistiksel ön kupon: {report['coupon']}")
        st.caption(f"Lig örneklemi: {predictions['sample_size']} tamamlanmış maç.")

    st.markdown("#### 5. Güncel takım ve oyuncu bağlamı")
    try:
        highlightly_api_key = str(st.secrets["HIGHLIGHTLY_API_KEY"]).strip()
    except (KeyError, FileNotFoundError):
        highlightly_api_key = ""
    context_state_key = f"highlightly_context_{match.get('id')}"
    context_changed = False
    stored_external = dict((analysis_record or {}).get("external_context") or {})
    has_stored_team_context = any(
        key in stored_external for key in ("standings", "home_last_five", "away_last_five")
    )
    if context_state_key not in st.session_state and any(
        key in stored_external for key in ("standings", "home_last_five", "away_last_five")
    ):
        st.session_state[context_state_key] = {
            "match": {"id": stored_external.get("highlightly_match_id") or 0},
            "standings": stored_external.get("standings") or [],
            "home_form": stored_external.get("home_last_five") or [],
            "away_form": stored_external.get("away_last_five") or [],
        }
    if has_stored_team_context:
        st.info("Kayıtlı takım bağlamı kullanılıyor; Highlightly çağrısı yapılmadı.")
    elif not highlightly_api_key:
        st.info("Highlightly verileri için HIGHLIGHTLY_API_KEY tanımlanmalıdır.")
    elif st.button("Güncel form ve puan durumunu getir", key=f"highlightly_context_button_{match.get('id')}", use_container_width=True):
        with st.spinner("Highlightly takım eşleşmesi ve güncel veriler alınıyor..."):
            try:
                st.session_state[context_state_key] = get_highlightly_context(
                    highlightly_api_key, str(match.get("match_date") or ""), home, away
                )
                context_changed = True
            except Exception as exc:
                st.session_state[context_state_key] = {"error": str(exc)}
    live_context = st.session_state.get(context_state_key)
    if isinstance(live_context, dict) and live_context.get("error"):
        st.warning("Highlightly verileri alınamadı.")
        st.caption(str(live_context["error"]))
    elif isinstance(live_context, dict) and live_context.get("match"):
        st.success("Highlightly maçı ve takımları güvenilir biçimde eşleştirdi.")
        standings_rows = live_context.get("standings") or []
        if standings_rows:
            st.markdown("##### Güncel puan durumu")
            st.dataframe(pd.DataFrame(standings_rows), use_container_width=True, hide_index=True)
        form_left, form_right = st.columns(2)
        with form_left:
            st.markdown(f"##### {home} · son 5")
            st.dataframe(pd.DataFrame(live_context.get("home_form") or []), use_container_width=True, hide_index=True)
        with form_right:
            st.markdown(f"##### {away} · son 5")
            st.dataframe(pd.DataFrame(live_context.get("away_form") or []), use_container_width=True, hide_index=True)
        report.setdefault("external_context", {}).update({
            "standings": standings_rows,
            "home_last_five": live_context.get("home_form") or [],
            "away_last_five": live_context.get("away_form") or [],
            "highlightly_match_id": live_context.get("match", {}).get("id"),
        })
        match_id = int(live_context["match"]["id"])
        if match_id and st.button("Kesin kadroları kontrol et", key=f"highlightly_lineups_{match_id}", use_container_width=True):
            try:
                st.session_state[f"highlightly_lineup_result_{match_id}"] = get_highlightly_lineups(highlightly_api_key, match_id)
                context_changed = True
            except Exception as exc:
                st.session_state[f"highlightly_lineup_result_{match_id}"] = {"error": str(exc)}
        lineup_result = st.session_state.get(f"highlightly_lineup_result_{match_id}")
        if isinstance(lineup_result, dict) and lineup_result.get("error"):
            st.caption("Kadrolar henüz yayımlanmamış olabilir: " + str(lineup_result["error"]))
        elif lineup_result:
            for side_key, side_label in (("homeTeam", "Ev sahibi kadrosu"), ("awayTeam", "Deplasman kadrosu")):
                side = lineup_result.get(side_key) or {}
                players = side.get("initialLineup") or []
                if players:
                    st.markdown(f"##### {side_label} · {side.get('formation') or '—'}")
                    st.write(", ".join(str(player.get("name") or "") for player in players))
            report["external_context"]["lineups"] = lineup_result
        if context_changed:
            artifact_error = update_analysis_artifacts(
                client,
                (analysis_record or {}).get("id"),
                external_context=report.get("external_context") or {},
            )
            if artifact_error:
                st.caption("Güncel takım bağlamı kalıcı önbelleğe yazılamadı: " + artifact_error)
    elif isinstance(live_context, dict):
        st.info("Highlightly seçilen maçı güvenilir biçimde eşleştiremedi; yanlış takım verisi kullanılmadı.")

    st.markdown("#### 6. Canlı araştırma ve yapay zekâ yorumu")
    st.write(
        "Tavily yalnızca seçili maç ve takımlar için güncel kaynakları arar. Groq, "
        "bu haberleri veritabanındaki istatistiklerle birleştirerek kendi tahminini üretir."
    )
    try:
        groq_api_key = str(st.secrets["GROQ_API_KEY"]).strip()
    except KeyError:
        groq_api_key = ""
    try:
        gemini_api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except KeyError:
        gemini_api_key = ""
    try:
        tavily_api_key = str(st.secrets["TAVILY_API_KEY"]).strip()
    except KeyError:
        tavily_api_key = ""

    gemini_state_key = f"tavily_gemini_analysis_v3_{match.get('id')}"
    if gemini_state_key not in st.session_state and (analysis_record or {}).get("gemini_result"):
        st.session_state[gemini_state_key] = dict(analysis_record["gemini_result"])
    has_stored_gemini = bool((analysis_record or {}).get("gemini_result"))
    if has_stored_gemini:
        st.info("Kayıtlı canlı araştırma ve yapay zekâ yorumu kullanılıyor; yeni kota harcanmadı.")

    if ((not groq_api_key and not gemini_api_key) or not tavily_api_key) and not has_stored_gemini:
        missing_keys = []
        if not groq_api_key and not gemini_api_key:
            missing_keys.append("GROQ_API_KEY")
        if not tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
        st.warning(
            "Canlı araştırma için eksik Streamlit Secrets anahtarı: "
            + ", ".join(missing_keys)
        )
    elif not has_stored_gemini and st.button(
                "Güncel haberleri araştır ve yapay zekâ yorumunu oluştur",
                key=f"tavily_gemini_button_{match.get('id')}",
                use_container_width=True,
            ):
            with st.spinner(
                "Tavily güncel kaynakları arıyor; Groq tüm verileri yorumluyor..."
            ):
                try:
                    generated_result = generate_grounded_analysis(
                            groq_api_key,
                            tavily_api_key,
                            match,
                            report,
                            gemini_api_key,
                        )
                    st.session_state[gemini_state_key] = generated_result
                    artifact_error = update_analysis_artifacts(
                        client,
                        (analysis_record or {}).get("id"),
                        external_context=report.get("external_context") or {},
                        gemini_result=generated_result,
                    )
                    if artifact_error:
                        st.session_state[gemini_state_key]["storage_warning"] = artifact_error
                except Exception as exc:
                    st.session_state[gemini_state_key] = {"error": str(exc)}
            st.rerun()

    gemini_result = st.session_state.get(gemini_state_key)
    if isinstance(gemini_result, dict) and "error" in gemini_result:
        st.error(
            "Canlı araştırma tamamlanamadı. Tavily/Groq anahtarını ve "
            "ücretsiz kullanım kotasını kontrol edin."
        )
        st.caption(str(gemini_result["error"]))
    elif gemini_result:
        with st.expander("Yapay zekâ raporunun tamamını göster", expanded=False):
            st.markdown(str(gemini_result["text"]))
            if gemini_result.get("storage_warning"):
                st.caption("Yapay zekâ sonucu kalıcı önbelleğe yazılamadı: " + str(gemini_result["storage_warning"]))
            if gemini_result.get("model"):
                provider = gemini_result.get("provider") or "Gemini"
                st.caption(f"Kullanılan yorum modeli: {provider} · {gemini_result['model']}")
            sources = gemini_result.get("sources") or []
            if sources:
                st.markdown("##### Tavily tarafından bulunan kaynaklar")
                for number, source in enumerate(sources, start=1):
                    title = str(source["title"]).replace("[", "(").replace("]", ")")
                    category = str(source.get("category") or "Kaynak")
                    st.markdown(f"{number}. **{category}:** [{title}]({source['url']})")
            search_warnings = gemini_result.get("search_warnings") or []
            if search_warnings:
                st.warning("Bazı haber aramaları tamamlanamadı; mevcut kaynaklarla analiz yapıldı.")
            st.caption(
                "Yapay zekâ tahminleri kesin sonuç veya kazanç garantisi değildir. "
                "Kadro ve haberleri maç öncesinde kaynaklardan doğrulayın."
            )


def render_upcoming_list_tab(client: Client, today) -> None:
    try:
        response = (
            client.table("upcoming_matches")
            .select(
                "id,division,match_date,kickoff_time,home_team,away_team,"
                "b365_home,b365_draw,b365_away,b365_over_25,b365_under_25,"
                "entry_method,match_status,raw_data"
            )
            .gte("match_date", today.isoformat())
            .order("match_date")
            .order("kickoff_time")
            .limit(1000)
            .execute()
        )
    except Exception as exc:
        st.error("Yaklaşan maç listesi alınamadı.")
        st.code(str(exc))
        return

    rows = response.data or []
    if not rows:
        st.info("Henüz bugün veya sonrasına ait yaklaşan maç kaydı yok.")
        return

    frame = pd.DataFrame(rows).rename(
        columns={
            "id": "Kayıt ID",
            "division": "Lig",
            "match_date": "Tarih",
            "kickoff_time": "Saat",
            "home_team": "Ev sahibi",
            "away_team": "Deplasman",
            "b365_home": "MS 1",
            "b365_draw": "X",
            "b365_away": "MS 2",
            "b365_over_25": "2.5 Üst",
            "b365_under_25": "2.5 Alt",
            "entry_method": "Giriş",
            "match_status": "Durum",
        }
    )
    frame = frame.drop(columns=["Kayıt ID", "raw_data"], errors="ignore")
    st.caption("Analiz başlatmak için aşağıdaki tabloda bir maç satırına tıklayın.")
    selection_event = st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="upcoming_match_selector",
    )
    selected_rows = getattr(getattr(selection_event, "selection", None), "rows", [])
    if selected_rows:
        selected_index = int(selected_rows[0])
        if 0 <= selected_index < len(rows):
            selected_match = dict(rows[selected_index])
            raw_data = selected_match.get("raw_data") or {}
            opening_odds = raw_data.get("opening_odds") if isinstance(raw_data, dict) else {}
            if isinstance(opening_odds, dict):
                selected_match.update(opening_odds)
            selected_match["analysis_odds_source"] = "Manuel analiz anı oranı"
            render_match_analysis(client, selected_match)


def render_upcoming_page(client: Client) -> None:
    today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
    st.caption("Aşama 3 · Geçmiş istatistik ve oran analizi")

    if "last_fixture_summary" in st.session_state:
        summary = st.session_state.pop("last_fixture_summary")
        st.success(
            f"{summary['files']} bülten dosyasından {summary['inserted']} yeni yaklaşan maç eklendi."
        )
        st.info(
            f"Atlananlar — geçmiş tarih: {summary['past']}, dosya içi tekrar: "
            f"{summary['within_duplicates']}, dosyalar arası tekrar: "
            f"{summary['cross_duplicates']}, veritabanında bulunan: {summary['existing']}, "
            f"eşzamanlı tekrar: {summary['race_duplicates']}."
        )

    if "last_manual_fixture" in st.session_state:
        st.success(st.session_state.pop("last_manual_fixture"))

    total_upcoming = get_table_count(client, "upcoming_matches")
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Kayıtlı yaklaşan maç", total_upcoming)
    metric_right.metric("Tarih filtresi", f"{today.strftime('%d.%m.%Y')} ve sonrası")

    csv_tab, manual_tab, list_tab = st.tabs(
        ["Toplu Bülten CSV", "Manuel Maç Girişi", "Yaklaşan Maç Listesi"]
    )
    with csv_tab:
        render_fixture_csv_tab(client, today)
    with manual_tab:
        render_manual_fixture_tab(client, today)
    with list_tab:
        render_upcoming_list_tab(client, today)


@st.cache_data(ttl=3600, show_spinner=False)
def get_api_football_fixtures(api_keys: tuple[str, ...], fixture_date: str) -> dict[str, object]:
    return fetch_fixtures(api_keys, fixture_date)


def get_api_football_keys() -> tuple[str, ...]:
    values: list[str] = []
    try:
        values.extend(normalize_api_keys(st.secrets["API_FOOTBALL_KEYS"]))
    except (KeyError, FileNotFoundError, TypeError):
        pass
    try:
        values.extend(normalize_api_keys(st.secrets["API_FOOTBALL_KEY"]))
    except (KeyError, FileNotFoundError, TypeError):
        pass
    return tuple(normalize_api_keys(values))


@st.cache_data(ttl=21600, show_spinner=False)
def get_football_data_fixtures() -> list[dict[str, object]]:
    return fetch_current_fixtures()


@st.cache_data(ttl=21600, show_spinner=False)
def get_supported_divisions() -> list[str]:
    """Use a fresh Supabase HTTP client so a stale HTTP/2 stream cannot break the fixture page."""
    last_error: Exception | None = None
    for _ in range(2):
        try:
            fresh_client = create_client(
                st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
            )
            return fetch_recent_divisions(fresh_client)
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Desteklenen lig listesi iki denemede alınamadı.") from last_error


@st.cache_data(ttl=21600, show_spinner=False)
def get_highlightly_context(api_key: str, match_date: str, home: str, away: str) -> dict[str, object]:
    match = find_match(fetch_match_day(api_key, match_date), home, away)
    if not match:
        match = find_match(
            fetch_match_day(api_key, match_date, home, away), home, away
        )
    if not match:
        return {"match": None}
    home_id = int((match.get("homeTeam") or {}).get("id"))
    away_id = int((match.get("awayTeam") or {}).get("id"))
    league = match.get("league") or {}
    league_id = league.get("id")
    season = league.get("season")
    standings = []
    if league_id is not None and season is not None:
        standings = selected_standings(
            fetch_standings(api_key, int(league_id), int(season)), {home_id, away_id}
        )
    return {
        "match": match,
        "home_form": form_rows(fetch_last_five(api_key, home_id), home_id),
        "away_form": form_rows(fetch_last_five(api_key, away_id), away_id),
        "standings": standings,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_highlightly_lineups(api_key: str, match_id: int) -> object:
    return fetch_lineups(api_key, match_id)


@st.cache_data(ttl=21600, show_spinner=False)
def get_thesportsdb_result(
    api_key: str, match_date: str, home_team: str, away_team: str
) -> dict[str, object] | None:
    return fetch_match_result(api_key, match_date, home_team, away_team)


def render_football_data_fixtures_page(client: Client) -> None:
    st.caption("Aşama 23 · Football-Data güncel fikstürü ve Highlightly canlı bağlamı")
    st.subheader("🌍 Güncel Fikstür")
    st.caption("Fikstür ve oranlar Football-Data.co.uk kaynağından alınır; API kotası harcanmaz.")
    today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
    if "last_manual_fixture" in st.session_state:
        st.success(st.session_state.pop("last_manual_fixture"))
    with st.expander("➕ Manuel maç ekle", expanded=False):
        render_manual_fixture_tab(client, today)
    with st.expander("📋 Manuel ve dosyadan eklenen maçları göster", expanded=False):
        render_upcoming_list_tab(client, today)
    if st.button(
        "🔄 Fikstürü yenile",
        key="refresh_football_data_fixtures",
        type="primary",
        use_container_width=True,
    ):
        get_football_data_fixtures.clear()
        st.session_state.pop("football_data_fixture_selector", None)
        st.session_state["football_data_fixture_refreshed"] = True
        st.rerun()
    if st.session_state.pop("football_data_fixture_refreshed", False):
        st.success("Fikstür Football-Data sitesinden yeniden alındı.")
    try:
        fixtures = get_football_data_fixtures()
    except Exception as exc:
        st.error("Football-Data güncel fikstürü alınamadı.")
        st.caption(str(exc))
        return
    try:
        available_divisions = set(get_supported_divisions())
    except Exception as exc:
        st.error("Geçmiş verisi bulunan ligler geçici bağlantı hatası nedeniyle alınamadı.")
        st.caption("Sayfayı yenileyip tekrar deneyin. Ayrıntı: " + str(exc))
        return
    supported = [row for row in fixtures if str(row.get("division")) in available_divisions]
    if not supported:
        st.warning("Güncel dosyada geçmiş verimizle eşleşen fikstür bulunamadı.")
        return
    dates = sorted({str(row["match_date"]) for row in supported})
    selected_date = st.selectbox("Fikstür tarihi", dates, index=0)
    date_rows = [row for row in supported if row["match_date"] == selected_date]
    divisions = sorted({str(row["division"]) for row in date_rows})
    selected_divisions = st.multiselect("Lig filtresi", divisions)
    visible = [row for row in date_rows if not selected_divisions or row["division"] in selected_divisions]
    st.metric("Gösterilen maç", len(visible))
    table = pd.DataFrame([{
        "Saat": row.get("kickoff_time") or "—", "Lig": row["division"],
        "Ev sahibi": row["home_team"], "Deplasman": row["away_team"],
        "FD referans 1": row.get("b365_home") or "—", "FD referans X": row.get("b365_draw") or "—",
        "FD referans 2": row.get("b365_away") or "—",
    } for row in visible])
    event = st.dataframe(table, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="football_data_fixture_selector")
    selected_rows = getattr(getattr(event, "selection", None), "rows", [])
    if selected_rows:
        index = int(selected_rows[0])
        if 0 <= index < len(visible):
            match = dict(visible[index])
            csv_odds = {
                "b365_home": match.get("b365_home"),
                "b365_draw": match.get("b365_draw"),
                "b365_away": match.get("b365_away"),
                "b365_over_25": match.get("b365_over_25"),
                "b365_under_25": match.get("b365_under_25"),
            }
            with st.expander("İsteğe bağlı yaklaşık Bet365 açılış oranı"):
                columns = st.columns(3)
                match["opening_b365_home"] = columns[0].number_input("Açılış 1", min_value=0.0, value=0.0, step=0.01, key=f"fd_open_h_{match['id']}") or None
                match["opening_b365_draw"] = columns[1].number_input("Açılış X", min_value=0.0, value=0.0, step=0.01, key=f"fd_open_d_{match['id']}") or None
                match["opening_b365_away"] = columns[2].number_input("Açılış 2", min_value=0.0, value=0.0, step=0.01, key=f"fd_open_a_{match['id']}") or None
                total_columns = st.columns(2)
                match["opening_b365_over_25"] = total_columns[0].number_input("Açılış 2.5 Üst", min_value=0.0, value=0.0, step=0.01, key=f"fd_open_o25_{match['id']}") or None
                match["opening_b365_under_25"] = total_columns[1].number_input("Açılış 2.5 Alt", min_value=0.0, value=0.0, step=0.01, key=f"fd_open_u25_{match['id']}") or None
            with st.expander("Analiz anında gördüğünüz Bet365 oranları"):
                st.caption(
                    "Bir kez girmeniz yeterlidir. Üçlü MS veya ikili 2.5 grubunu eksiksiz girin; "
                    "boş grup için Football-Data referans oranı kullanılır."
                )
                current_columns = st.columns(3)
                current_home = current_columns[0].number_input("Analiz anı 1", min_value=0.0, value=0.0, step=0.01, key=f"fd_now_h_{match['id']}")
                current_draw = current_columns[1].number_input("Analiz anı X", min_value=0.0, value=0.0, step=0.01, key=f"fd_now_d_{match['id']}")
                current_away = current_columns[2].number_input("Analiz anı 2", min_value=0.0, value=0.0, step=0.01, key=f"fd_now_a_{match['id']}")
                current_total_columns = st.columns(2)
                current_over = current_total_columns[0].number_input("Analiz anı 2.5 Üst", min_value=0.0, value=0.0, step=0.01, key=f"fd_now_o25_{match['id']}")
                current_under = current_total_columns[1].number_input("Analiz anı 2.5 Alt", min_value=0.0, value=0.0, step=0.01, key=f"fd_now_u25_{match['id']}")
            for name, value in csv_odds.items():
                match[f"csv_{name}"] = value
            if all(value > 1 for value in (current_home, current_draw, current_away)):
                match.update({"b365_home": current_home, "b365_draw": current_draw, "b365_away": current_away})
                match["analysis_odds_source"] = "Manuel analiz anı oranı"
            else:
                match["analysis_odds_source"] = "Football-Data referans oranı"
            if all(value > 1 for value in (current_over, current_under)):
                match.update({"b365_over_25": current_over, "b365_under_25": current_under})
            render_match_analysis(client, match)


def render_world_fixtures_page(client: Client) -> None:
    st.caption("Aşama 24 · API-Football dünya fikstürü, toplu ön analiz ve kayıtlı detaylar")
    st.subheader("🌍 Dünya Fikstürü")
    st.write(
        "Seçilen tarihteki tüm API maçları listelenir. Geçmiş lig ve takım verisiyle "
        "güvenilir eşleşen başlamamış maçlar tablo gösterilmeden önce analiz edilip kaydedilir."
    )
    api_keys = get_api_football_keys()
    if not api_keys:
        st.info(
            "Dünya fikstürünü açmak için Streamlit Secrets alanına "
            "API_FOOTBALL_KEY veya API_FOOTBALL_KEYS ekleyin."
        )
        return

    today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
    fixture_date = st.date_input("Fikstür tarihi", value=today, key="api_fixture_date")
    requested_date = fixture_date.isoformat()
    if st.button(
        "Fikstürü getir, tüm uygun maçları değerlendir ve kaydet",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Dünya genelindeki fikstür alınıyor..."):
                api_result = get_api_football_fixtures(api_keys, requested_date)
                all_fixtures = list(api_result.get("fixtures") or [])
                available_divisions = set(fetch_recent_divisions(client))
            outcomes: list[dict[str, object]] = []
            league_cache: dict[str, list[dict[str, object]]] = {}
            progress = st.progress(0, text="Maçlar değerlendiriliyor...")
            total = len(all_fixtures)
            processing_fixtures = sorted(
                all_fixtures,
                key=lambda fixture: (
                    str(fixture.get("league_id") or ""),
                    str(fixture.get("timestamp") or ""),
                ),
            )
            active_league_id = None
            for index, fixture in enumerate(processing_fixtures, start=1):
                league_id = fixture.get("league_id")
                if league_id != active_league_id:
                    league_cache.clear()
                    active_league_id = league_id
                outcomes.append(
                    analyze_world_fixture(client, fixture, available_divisions, league_cache)
                )
                progress.progress(
                    index / max(total, 1),
                    text=f"Değerlendirilen maç: {index}/{total}",
                )
            progress.empty()
            outcomes.sort(
                key=lambda outcome: (
                    str((outcome.get("fixture") or {}).get("match_date") or ""),
                    str((outcome.get("fixture") or {}).get("kickoff_time") or ""),
                    str((outcome.get("fixture") or {}).get("country") or ""),
                )
            )
            st.session_state["world_fixture_analysis"] = {
                "date": requested_date,
                "outcomes": outcomes,
                "quota": api_result.get("quota") or {},
            }
            st.session_state.pop("world_fixture_selector", None)
        except Exception as exc:
            st.session_state["world_fixture_analysis"] = {
                "date": requested_date,
                "error": str(exc),
            }

    result = st.session_state.get("world_fixture_analysis")
    if not isinstance(result, dict) or result.get("date") != requested_date:
        st.caption(
            "Tarihi seçip toplu değerlendirmeyi başlatın. Aynı tarih için mevcut kayıtlı "
            "snapshotlar yeniden hesaplanmadan kullanılır."
        )
        return
    if result.get("error"):
        st.error("Dünya fikstürü değerlendirilemedi.")
        st.caption(str(result["error"]))
        return

    outcomes = list(result.get("outcomes") or [])
    if not outcomes:
        st.info("Seçilen tarihte API kapsamında maç bulunamadı.")
        return
    quota = dict(result.get("quota") or {})
    if quota.get("daily_remaining") is not None:
        st.caption(f"API-Football günlük kalan istek: {quota['daily_remaining']}")

    analyzed_statuses = {"Kayıtlı analiz", "Yeni kaydedildi"}
    metric_columns = st.columns(4)
    metric_columns[0].metric("Toplam fikstür", len(outcomes))
    metric_columns[1].metric(
        "Analizi hazır", sum(outcome.get("status") in analyzed_statuses for outcome in outcomes)
    )
    metric_columns[2].metric(
        "Yeni kaydedilen", sum(outcome.get("status") == "Yeni kaydedildi" for outcome in outcomes)
    )
    metric_columns[3].metric(
        "Destek dışı/hatalı", sum(outcome.get("status") not in analyzed_statuses for outcome in outcomes)
    )

    countries = sorted(
        {str((outcome.get("fixture") or {}).get("country") or "—") for outcome in outcomes},
        key=str.casefold,
    )
    selected_countries = st.multiselect("Ülke filtresi", countries, key="world_countries")
    country_rows = [
        outcome for outcome in outcomes
        if not selected_countries
        or str((outcome.get("fixture") or {}).get("country") or "—") in selected_countries
    ]
    leagues = sorted(
        {str((outcome.get("fixture") or {}).get("league") or "—") for outcome in country_rows},
        key=str.casefold,
    )
    selected_leagues = st.multiselect("Lig filtresi", leagues, key="world_leagues")
    visible = [
        outcome for outcome in country_rows
        if not selected_leagues
        or str((outcome.get("fixture") or {}).get("league") or "—") in selected_leagues
    ]

    st.caption(
        "Bir satıra tıklayarak kaydedilmiş analiz detayını açın. Desteklenmeyen maçların "
        "nedeni Açıklama sütununda gösterilir."
    )
    event = st.dataframe(
        pd.DataFrame([world_fixture_table_row(outcome) for outcome in visible]),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="world_fixture_selector",
    )
    selected_rows = getattr(getattr(event, "selection", None), "rows", [])
    if not selected_rows:
        return
    selected_index = int(selected_rows[0])
    if not 0 <= selected_index < len(visible):
        return
    selected = visible[selected_index]
    analysis_match = selected.get("analysis_match")
    if selected.get("status") not in analyzed_statuses or not isinstance(analysis_match, dict):
        st.warning(str(selected.get("reason") or "Bu maç için kayıtlı analiz bulunmuyor."))
        return
    render_match_analysis(client, analysis_match)


def render_analysis_history_page(client: Client) -> None:
    st.subheader("📚 Analiz Geçmişi ve Sonuçlar")
    st.caption(
        "Her maçın son analiz sürümü gösterilir. Gerçek skor bir kez saklanır ve "
        "tahmin pazarları otomatik değerlendirilir."
    )
    analyses, analysis_error = load_analysis_history(client)
    if analysis_error:
        st.error("Kayıtlı analizler alınamadı.")
        st.caption(analysis_error)
        return
    if not analyses:
        st.info("Henüz kaydedilmiş maç analizi bulunmuyor.")
        return

    latest_by_match: dict[str, dict[str, object]] = {}
    for row in analyses:
        key = str(row.get("match_key") or "")
        if key not in latest_by_match:
            latest_by_match[key] = row
    latest = list(latest_by_match.values())
    results, result_error = load_match_results(client)
    if result_error:
        st.warning("Sonuç kayıt tablosu henüz kurulmamış olabilir.")
        with st.expander("Supabase için sonuç tablosu SQL kodunu göster"):
            st.code(
                """create table if not exists public.match_results (
  match_key text primary key,
  division text not null,
  match_date date not null,
  home_team text not null,
  away_team text not null,
  full_time_home smallint not null check (full_time_home >= 0),
  full_time_away smallint not null check (full_time_away >= 0),
  half_time_home smallint check (half_time_home >= 0),
  half_time_away smallint check (half_time_away >= 0),
  source text not null default 'manual',
  updated_at timestamptz not null default now()
);
create index if not exists match_results_date_idx
  on public.match_results (match_date desc);
alter table public.match_results enable row level security;""",
                language="sql",
            )

    now_istanbul = datetime.now(ZoneInfo("Europe/Istanbul"))
    result_candidates: list[dict[str, object]] = []
    for analysis in latest:
        if str(analysis.get("match_key") or "") in results:
            continue
        try:
            candidate_date = datetime.fromisoformat(str(analysis.get("match_date"))).date()
            candidate_time = time.fromisoformat(
                str(analysis.get("kickoff_time") or "23:59:59")[:8]
            )
            expected_finish = datetime.combine(
                candidate_date, candidate_time, tzinfo=ZoneInfo("Europe/Istanbul")
            ) + timedelta(hours=2, minutes=30)
        except ValueError:
            continue
        if expected_finish <= now_istanbul:
            result_candidates.append(analysis)

    if st.session_state.pop("automatic_result_sync_message", None):
        sync_message = st.session_state["automatic_result_sync_last"]
        st.success(
            f"TheSportsDB kontrolü tamamlandı: {sync_message['checked']} maç kontrol edildi, "
            f"{sync_message['saved']} sonuç kaydedildi."
        )
        if sync_message["errors"]:
            st.caption(f"{sync_message['errors']} sorgu geçici hata nedeniyle tamamlanamadı.")
    if result_candidates and not result_error:
        st.caption(
            f"Sonucu bekleyen ve tahmini bitiş saati geçen maç: {len(result_candidates)}"
        )
        if st.button(
            "🔄 Eksik sonuçları API'den otomatik getir",
            type="primary",
            use_container_width=True,
        ):
            try:
                thesportsdb_key = str(st.secrets["THESPORTSDB_API_KEY"]).strip()
            except (KeyError, FileNotFoundError):
                thesportsdb_key = "123"
            checked = saved = errors = 0
            with st.spinner("TheSportsDB üzerinden tamamlanan maçlar kontrol ediliyor..."):
                for analysis in result_candidates[:20]:
                    checked += 1
                    try:
                        api_result = get_thesportsdb_result(
                            thesportsdb_key,
                            str(analysis.get("match_date") or ""),
                            str(analysis.get("home_team") or ""),
                            str(analysis.get("away_team") or ""),
                        )
                        if not api_result:
                            continue
                        save_error = save_match_result(
                            client,
                            analysis,
                            full_time_home=int(api_result["full_time_home"]),
                            full_time_away=int(api_result["full_time_away"]),
                            source="thesportsdb",
                        )
                        if save_error:
                            errors += 1
                        else:
                            saved += 1
                    except Exception:
                        errors += 1
            st.session_state["automatic_result_sync_last"] = {
                "checked": checked, "saved": saved, "errors": errors,
            }
            st.session_state["automatic_result_sync_message"] = True
            st.rerun()

    evaluations: list[dict[str, object]] = []
    for analysis in latest:
        result = results.get(str(analysis.get("match_key") or ""))
        if result:
            evaluations.extend(evaluate_analysis(analysis, result))
    ms_rows = [row for row in evaluations if row["Pazar"] == "Maç sonucu"]
    total_rows = [row for row in evaluations if row["Pazar"] == "2.5 Alt/Üst"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Analiz edilen maç", len(latest))
    metric_columns[1].metric("Sonucu girilen", len(results))
    metric_columns[2].metric(
        "MS başarısı",
        f"%{sum(bool(row['Doğru']) for row in ms_rows) / len(ms_rows) * 100:.1f}"
        if ms_rows else "—",
    )
    metric_columns[3].metric(
        "2.5 başarısı",
        f"%{sum(bool(row['Doğru']) for row in total_rows) / len(total_rows) * 100:.1f}"
        if total_rows else "—",
    )

    history_rows: list[dict[str, object]] = []
    for row in latest:
        predictions = dict((row.get("report_snapshot") or {}).get("predictions") or {})
        result = results.get(str(row.get("match_key") or ""))
        history_rows.append({
            "Tarih": row.get("match_date"), "Lig": row.get("division"),
            "Maç": f"{row.get('home_team')} — {row.get('away_team')}",
            "Sürüm": row.get("version"), "MS": predictions.get("ms") or "—",
            "Skor": predictions.get("score") or "—",
            "Gerçek": f"{result['full_time_home']}-{result['full_time_away']}" if result else "Bekleniyor",
        })
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

    labels = {
        str(row["match_key"]): f"{row.get('match_date')} · {row.get('home_team')} — {row.get('away_team')}"
        for row in latest
    }
    selected_key = st.selectbox(
        "Sonucunu görüntüle veya gir", list(labels), format_func=lambda key: labels[key]
    )
    selected = latest_by_match[selected_key]
    stored_result = results.get(selected_key)
    match_date = datetime.fromisoformat(str(selected.get("match_date"))).date()
    kickoff_text = str(selected.get("kickoff_time") or "23:59:59")[:8]
    try:
        kickoff_clock = time.fromisoformat(kickoff_text)
    except ValueError:
        kickoff_clock = time(23, 59, 59)
    kickoff = datetime.combine(match_date, kickoff_clock, tzinfo=ZoneInfo("Europe/Istanbul"))
    match_started = datetime.now(ZoneInfo("Europe/Istanbul")) >= kickoff
    if not match_started and not stored_result:
        st.info("Bu maç henüz başlamadığı için gerçek sonuç kaydı kapalıdır.")

    st.markdown("#### Gerçek maç sonucu")
    with st.form(f"result_form_{selected_key}"):
        full_columns = st.columns(2)
        full_home = full_columns[0].number_input(
            f"MS · {selected.get('home_team')}", min_value=0, max_value=30,
            value=int((stored_result or {}).get("full_time_home") or 0), step=1,
        )
        full_away = full_columns[1].number_input(
            f"MS · {selected.get('away_team')}", min_value=0, max_value=30,
            value=int((stored_result or {}).get("full_time_away") or 0), step=1,
        )
        has_half_time = st.checkbox(
            "İlk yarı skoru da mevcut",
            value=(stored_result or {}).get("half_time_home") is not None,
        )
        half_home = half_away = 0
        if has_half_time:
            half_columns = st.columns(2)
            half_home = half_columns[0].number_input(
                f"İY · {selected.get('home_team')}", min_value=0, max_value=20,
                value=int((stored_result or {}).get("half_time_home") or 0), step=1,
            )
            half_away = half_columns[1].number_input(
                f"İY · {selected.get('away_team')}", min_value=0, max_value=20,
                value=int((stored_result or {}).get("half_time_away") or 0), step=1,
            )
        submitted = st.form_submit_button(
            "Sonucu kaydet veya güncelle", use_container_width=True,
            disabled=bool(result_error) or (not match_started and not stored_result),
        )
    if submitted:
        save_error = save_match_result(
            client, selected, full_time_home=int(full_home), full_time_away=int(full_away),
            half_time_home=int(half_home) if has_half_time else None,
            half_time_away=int(half_away) if has_half_time else None,
        )
        if save_error:
            st.error("Maç sonucu kaydedilemedi.")
            st.caption(save_error)
        else:
            st.session_state["result_saved_message"] = True
            st.rerun()
    if st.session_state.pop("result_saved_message", False):
        st.success("Maç sonucu kaydedildi ve tahminler değerlendirildi.")

    if stored_result:
        comparison_frame = pd.DataFrame(evaluate_analysis(selected, stored_result))
        comparison_frame["Durum"] = comparison_frame["Doğru"].map(
            lambda correct: "✅ Doğru" if correct else "❌ Yanlış"
        )
        st.dataframe(
            comparison_frame.drop(columns=["Doğru"]), use_container_width=True, hide_index=True
        )


def render_backtest_page(client: Client) -> None:
    st.caption("Model doğrulama · Geçmiş maçlarda ileri yürüyen tarafsız test")
    st.subheader("🧪 İstatistiksel model backtesti")
    st.write(
        "Her maç, yalnızca o maçtan önce oynanmış karşılaşmalar kullanılarak yeniden tahmin edilir. "
        "Gerçek sonuç daha sonra tahminle karşılaştırılır."
    )
    st.info(
        "Bu test istatistiksel modeli ölçer. Geçmiş tarihteki canlı haberleri güvenilir biçimde "
        "yeniden oluşturamadığımız için Gemini ve Tavily teste dahil edilmez."
    )

    try:
        divisions = fetch_recent_divisions(client)
    except Exception as exc:
        st.error("Lig listesi alınamadı.")
        st.code(str(exc))
        return
    if not divisions:
        st.warning("Backtest yapılabilecek lig bulunamadı.")
        return

    mode = st.radio(
        "Test biçimi",
        ["Tek lig", "Toplu lig"],
        horizontal=True,
        key="backtest_mode",
    )
    left, right = st.columns(2)
    if mode == "Tek lig":
        with left:
            selected_divisions = [
                st.selectbox("Test edilecek lig", divisions, key="backtest_division")
            ]
    else:
        preferred = [
            division for division in ("EC", "E0", "SP1", "D1", "I1", "F1")
            if division in divisions
        ]
        with left:
            selected_divisions = st.multiselect(
                "Test edilecek ligler",
                divisions,
                default=preferred or divisions[: min(6, len(divisions))],
                key="backtest_divisions",
            )
    with right:
        test_size = st.selectbox(
            "Her ligde test edilecek son maç",
            [50, 100, 200, 300],
            index=3,
            key="backtest_size",
        )

    button_label = "Toplu testi başlat" if mode == "Toplu lig" else "Backtesti başlat"
    if st.button(button_label, type="primary", use_container_width=True):
        if not selected_divisions:
            st.warning("En az bir lig seçin.")
            return
        progress = st.progress(0, text="Lig verileri hazırlanıyor...")
        league_results: list[tuple[str, dict[str, object]]] = []
        league_errors: list[str] = []
        league_total = len(selected_divisions)
        for league_index, division in enumerate(selected_divisions):
            try:
                league_rows = fetch_league_rows(client, division)

                def update_progress(done: int, total: int) -> None:
                    overall = (league_index + done / max(1, total)) / league_total
                    progress.progress(
                        min(1.0, overall),
                        text=f"{division}: {done}/{total} · Lig {league_index + 1}/{league_total}",
                    )

                league_result = run_backtest(
                    league_rows,
                    test_size=int(test_size),
                    progress_callback=update_progress,
                )
                league_results.append((division, league_result))
            except Exception as exc:
                league_errors.append(f"{division}: {exc}")
        progress.empty()
        if not league_results:
            st.error("Backtest tamamlanamadı.")
            for error in league_errors:
                st.warning(error)
            return
        if mode == "Toplu lig":
            result = aggregate_backtests(league_results)
            try:
                result["calibration"] = calibrate_model(result["calibration_leagues"])
            except Exception as exc:
                result["calibration_error"] = str(exc)
            result["league_errors"] = league_errors
            title = f"{result['league_count']} lig · toplam {result['tested']} maç"
        else:
            result = league_results[0][1]
            result["league_errors"] = league_errors
            title = f"{selected_divisions[0]} · son {result['tested']} maç"
        st.session_state["backtest_result"] = result
        st.session_state["backtest_result_title"] = title

    result = st.session_state.get("backtest_result")
    if not result:
        st.caption("Lig ve maç sayısını seçtikten sonra Backtesti başlat düğmesine basın.")
        return

    st.divider()
    st.subheader(f"📋 Sonuçlar · {st.session_state.get('backtest_result_title', '')}")
    metric_lookup = {row["Ölçüm"]: row for row in result["metrics"]}
    value_five = next(
        (row for row in result.get("value_metrics") or [] if float(row.get("Eşik", -1)) == 0.05),
        None,
    )
    c1, c2, c3, c4 = st.columns(4)
    if result.get("league_summary"):
        c1.metric("Test edilen lig", result.get("league_count", 0))
        c2.metric("Test edilen maç", result["tested"])
        c3.metric("MS başarısı", f"{metric_lookup['Maç sonucu']['Başarı'] * 100:.1f}%")
        c4.metric(
            "+5 değer ROI",
            f"{float(value_five['ROI']) * 100:+.1f}%"
            if value_five and value_five.get("ROI") is not None else "—",
        )
    else:
        c1.metric("Test edilen maç", result["tested"])
        c2.metric("MS başarısı", f"{metric_lookup['Maç sonucu']['Başarı'] * 100:.1f}%")
        c3.metric("2.5 Alt/Üst", f"{metric_lookup['2.5 Alt/Üst']['Başarı'] * 100:.1f}%")
        c4.metric("KG başarısı", f"{metric_lookup['Karşılıklı gol']['Başarı'] * 100:.1f}%")

    metrics_frame = pd.DataFrame(result["metrics"])
    metrics_frame["Başarı"] = metrics_frame["Başarı"].map(lambda value: f"{value * 100:.1f}%")
    st.markdown("#### Pazar bazında başarı")
    st.dataframe(metrics_frame, use_container_width=True, hide_index=True)

    league_frame = pd.DataFrame(result.get("league_summary") or [])
    if not league_frame.empty:
        for column in (
            "MS başarısı",
            "Yüksek güven MS",
            "Tüm seçimler ROI",
            "Yüksek güven ROI",
            "Bet365 favorisi ROI",
            "+5 değer ROI",
        ):
            league_frame[column] = league_frame[column].map(
                lambda value: (
                    "—"
                    if value is None or pd.isna(value)
                    else f"{value * 100:+.1f}%" if column.endswith("ROI")
                    else f"{value * 100:.1f}%"
                )
            )
        st.markdown("#### Lig bazında toplu sonuç")
        st.dataframe(league_frame, use_container_width=True, hide_index=True)

    confidence_frame = pd.DataFrame(result.get("confidence_metrics") or [])
    if not confidence_frame.empty:
        confidence_frame["MS başarısı"] = confidence_frame["MS başarısı"].map(
            lambda value: f"{value * 100:.1f}%"
        )
        st.markdown("#### Güven seviyesine göre maç sonucu başarısı")
        st.dataframe(confidence_frame, use_container_width=True, hide_index=True)

    comparison_frame = pd.DataFrame(result.get("comparisons") or [])
    if not comparison_frame.empty:
        for column in ("Model başarısı", "Referans başarısı", "Model farkı"):
            comparison_frame[column] = comparison_frame[column].map(
                lambda value: f"{value * 100:+.1f}%" if column == "Model farkı" else f"{value * 100:.1f}%"
            )
        st.markdown("#### Model ve basit referans karşılaştırması")
        st.caption(
            "Model farkının artı olması modelin referanstan daha başarılı, eksi olması daha zayıf olduğunu gösterir."
        )
        st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

    value_frame = pd.DataFrame(result.get("value_metrics") or [])
    if not value_frame.empty:
        value_frame = value_frame.drop(columns=["Eşik"], errors="ignore")
        value_frame["Ortalama oran"] = value_frame["Ortalama oran"].map(
            lambda value: "—" if value is None or pd.isna(value) else f"{value:.2f}"
        )
        value_frame["Yatırılan"] = value_frame["Yatırılan"].map(
            lambda value: f"{value:,.0f} birim"
        )
        value_frame["Net sonuç"] = value_frame["Net sonuç"].map(
            lambda value: f"{value:+,.1f} birim"
        )
        value_frame["ROI"] = value_frame["ROI"].map(
            lambda value: "—" if value is None or pd.isna(value) else f"{value * 100:+.1f}%"
        )
        st.markdown("#### Değer farkı eşiğine göre sanal MS testi")
        st.caption(
            "Değer farkı = model olasılığı − seçilen oranın başabaş olasılığı. Daha yüksek eşik daha seçici davranır."
        )
        st.dataframe(value_frame, use_container_width=True, hide_index=True)

    profit_frame = pd.DataFrame(result.get("profit_metrics") or [])
    if not profit_frame.empty:
        profit_frame["Ortalama oran"] = profit_frame["Ortalama oran"].map(
            lambda value: f"{value:.2f}"
        )
        profit_frame["Yatırılan"] = profit_frame["Yatırılan"].map(
            lambda value: f"{value:,.0f} birim"
        )
        profit_frame["Net sonuç"] = profit_frame["Net sonuç"].map(
            lambda value: f"{value:+,.1f} birim"
        )
        profit_frame["ROI"] = profit_frame["ROI"].map(
            lambda value: f"{value * 100:+.1f}%"
        )
        st.markdown("#### 100 birimlik sanal MS bahis testi")
        st.warning(
            "Bu tablo geçmiş performans simülasyonudur; gelecekte kâr veya kazanç garantisi değildir."
        )
        st.dataframe(profit_frame, use_container_width=True, hide_index=True)

    calibration = result.get("calibration")
    if isinstance(calibration, dict):
        st.divider()
        st.markdown("### 🎯 70/30 olasılık kalibrasyonu")
        st.info(
            "Ağırlıklar yalnızca eski %70 bölümünde öğrenildi. Aşağıdaki ana karşılaştırma, "
            "ayar sırasında hiç kullanılmayan en yeni %30 maç üzerinde yapıldı."
        )
        split_frame = pd.DataFrame(calibration.get("split_summary") or [])
        if not split_frame.empty:
            st.markdown("#### Liglere göre eğitim ve sınav ayrımı")
            st.dataframe(split_frame, use_container_width=True, hide_index=True)

        weights_frame = pd.DataFrame(calibration.get("weights") or [])
        if not weights_frame.empty:
            weights_frame["Öğrenilen ağırlık"] = weights_frame["Öğrenilen ağırlık"].map(
                lambda value: f"{value * 100:.1f}%"
            )
            st.markdown("#### Eski %70 bölümünde öğrenilen ağırlıklar")
            st.dataframe(weights_frame, use_container_width=True, hide_index=True)
            st.caption(
                f"Olasılık yumuşatma sıcaklığı: {float(calibration.get('temperature', 1)):.2f} · "
                "1'in üzerindeki değer, aşırı güveni azaltır."
            )

        def format_evaluation(rows: list[dict[str, object]]) -> pd.DataFrame:
            frame = pd.DataFrame(rows)
            if frame.empty:
                return frame
            for column in ("Doğruluk", "Ortalama güven", "ROI"):
                frame[column] = frame[column].map(
                    lambda value: (
                        f"{float(value) * 100:+.1f}%"
                        if column == "ROI"
                        else f"{float(value) * 100:.1f}%"
                    )
                )
            for column in ("Brier", "Log loss"):
                frame[column] = frame[column].map(lambda value: f"{float(value):.4f}")
            frame["Ortalama oran"] = frame["Ortalama oran"].map(
                lambda value: f"{float(value):.2f}"
            )
            frame["Net sonuç"] = frame["Net sonuç"].map(
                lambda value: f"{float(value):+,.1f} birim"
            )
            return frame

        st.markdown("#### Dokunulmamış yeni %30 sınav sonucu")
        st.caption("Brier ve Log loss değerlerinde daha düşük sonuç daha iyidir.")
        holdout_frame = format_evaluation(calibration.get("holdout_comparison") or [])
        st.dataframe(holdout_frame, use_container_width=True, hide_index=True)

        with st.expander("Eski %70 eğitim bölümünün sonucunu göster"):
            training_frame = format_evaluation(calibration.get("training_comparison") or [])
            st.dataframe(training_frame, use_container_width=True, hide_index=True)

        calibrated_value = pd.DataFrame(calibration.get("value_comparison") or [])
        if not calibrated_value.empty:
            calibrated_value["Ortalama oran"] = calibrated_value["Ortalama oran"].map(
                lambda value: "—" if value is None or pd.isna(value) else f"{value:.2f}"
            )
            calibrated_value["Net sonuç"] = calibrated_value["Net sonuç"].map(
                lambda value: f"{value:+,.1f} birim"
            )
            calibrated_value["ROI"] = calibrated_value["ROI"].map(
                lambda value: "—" if value is None or pd.isna(value) else f"{value * 100:+.1f}%"
            )
            st.markdown("#### Yeni %30 bölümünde değer filtresi")
            st.dataframe(calibrated_value, use_container_width=True, hide_index=True)

        value_band_frame = pd.DataFrame(
            calibration.get("value_band_diagnostics") or []
        )
        if not value_band_frame.empty:
            value_band_frame["Ortalama oran"] = value_band_frame["Ortalama oran"].map(
                lambda value: "—" if value is None or pd.isna(value) else f"{float(value):.2f}"
            )
            value_band_frame["Net sonuç"] = value_band_frame["Net sonuç"].map(
                lambda value: f"{float(value):+,.1f} birim"
            )
            value_band_frame["ROI"] = value_band_frame["ROI"].map(
                lambda value: "—" if value is None or pd.isna(value) else f"{float(value) * 100:+.1f}%"
            )
            st.markdown("#### +3 değer seçimlerinin oran aralığı teşhisi")
            st.caption(
                "Yüksek oranlı sürpriz seçimlerin küçük olasılık hataları nedeniyle sonucu bozup bozmadığını gösterir."
            )
            st.dataframe(value_band_frame, use_container_width=True, hide_index=True)

        bins_frame = pd.DataFrame(calibration.get("calibration_bins") or [])
        if not bins_frame.empty:
            for column in ("Ortalama tahmin", "Gerçek başarı", "Kalibrasyon farkı"):
                bins_frame[column] = bins_frame[column].map(
                    lambda value: f"{value * 100:+.1f}%" if column == "Kalibrasyon farkı" else f"{value * 100:.1f}%"
                )
            st.markdown("#### Kalibre modelin güven kontrolü")
            st.dataframe(bins_frame, use_container_width=True, hide_index=True)

        robustness_frame = pd.DataFrame(calibration.get("robustness_checks") or [])
        if not robustness_frame.empty:
            robustness_frame["Durum"] = robustness_frame["Geçti"].map(
                lambda passed: "✅ Geçti" if passed else "❌ Geçemedi"
            )
            robustness_frame = robustness_frame.drop(columns=["Geçti"])
            st.markdown("#### Canlı modele geçiş sağlamlık kontrolleri")
            st.caption(
                "Küçük metrik farklarının tesadüf olma ihtimali ve değer filtresinin örneklem büyüklüğü ayrıca denetlenir."
            )
            st.dataframe(robustness_frame, use_container_width=True, hide_index=True)

        calibration_league_frame = pd.DataFrame(
            calibration.get("league_diagnostics") or []
        )
        if not calibration_league_frame.empty:
            for column in ("Mevcut Brier", "Kalibre Brier"):
                calibration_league_frame[column] = calibration_league_frame[column].map(
                    lambda value: f"{float(value):.4f}"
                )
            for column in (
                "Brier iyileşmesi",
                "Doğruluk farkı",
                "Kalibre ROI",
                "+3 değer ROI",
            ):
                calibration_league_frame[column] = calibration_league_frame[column].map(
                    lambda value: (
                        "—"
                        if value is None or pd.isna(value)
                        else f"{float(value) * 100:+.1f}%"
                    )
                )
            calibration_league_frame["Durum"] = calibration_league_frame["İyileşti"].map(
                lambda improved: "✅ İyileşti" if improved else "❌ Geriledi"
            )
            calibration_league_frame = calibration_league_frame.drop(
                columns=["İyileşti"]
            )
            st.markdown("#### Dokunulmamış %30 sınavın lig bazında sağlamlığı")
            st.caption(
                "Toplu ortalamanın tek bir güçlü lig tarafından yanıltılmaması için her lig ayrıca değerlendirilir."
            )
            st.dataframe(
                calibration_league_frame,
                use_container_width=True,
                hide_index=True,
            )

        rolling_frame = pd.DataFrame(calibration.get("rolling_diagnostics") or [])
        if not rolling_frame.empty:
            rolling_frame["Eğitim oranı"] = rolling_frame["Eğitim oranı"].map(
                lambda value: f"%{float(value) * 100:.0f}"
            )
            for column in ("Mevcut Brier", "Kalibre Brier"):
                rolling_frame[column] = rolling_frame[column].map(
                    lambda value: f"{float(value):.4f}"
                )
            for column in (
                "Brier iyileşmesi",
                "Doğruluk farkı",
                "Kalibre ROI",
                "+3 değer ROI",
            ):
                rolling_frame[column] = rolling_frame[column].map(
                    lambda value: (
                        "—"
                        if value is None or pd.isna(value)
                        else f"{float(value) * 100:+.1f}%"
                    )
                )
            rolling_frame["Durum"] = rolling_frame["İyileşti"].map(
                lambda improved: "✅ İyileşti" if improved else "❌ Geriledi"
            )
            rolling_frame = rolling_frame.drop(columns=["İyileşti"])
            st.markdown("#### Genişleyen eğitimle üç dönemlik ileri sınav")
            st.caption(
                "Her satırda yalnızca daha eski maçlarda öğrenilen yeni ağırlıklar, hemen sonraki ve birbirini tekrar etmeyen dönemde sınanır."
            )
            st.dataframe(rolling_frame, use_container_width=True, hide_index=True)

        boundary_weights = calibration.get("boundary_weights") or []
        if boundary_weights:
            st.warning(
                "Optimizasyon bazı ağırlıklarda arama sınırına dayandı: "
                + ", ".join(map(str, boundary_weights))
                + ". Bu durum kararsız veya aşırı piyasa ağırlıklı bir çözüme işaret edebilir."
            )

        if calibration.get("recommended"):
            st.success(str(calibration.get("decision") or "Kalibrasyon sınavı başarılı."))
        else:
            st.warning(str(calibration.get("decision") or "Kalibrasyon sınavı yeterli değil."))
        st.caption("Bu aşamada öğrenilen ağırlıklar canlı tahminlere otomatik uygulanmaz.")
    elif result.get("calibration_error"):
        st.warning(f"Kalibrasyon tamamlanamadı: {result['calibration_error']}")

    if result.get("details"):
        with st.expander("Test edilen maçların ayrıntılarını göster"):
            st.dataframe(pd.DataFrame(result["details"]), use_container_width=True, hide_index=True)
    for error in result.get("league_errors") or []:
        st.warning(f"Atlanan lig · {error}")
    st.caption(result["note"])


st.title("⚽ Futbol Tahmin ve Analiz")

with st.sidebar:
    st.subheader("Proje durumu")
    st.success("Veritabanı tabloları hazır")
    if st.session_state.get("authenticated") is True:
        if st.button("Güvenli çıkış"):
            st.session_state["authenticated"] = False
            st.rerun()

stop_if_secrets_missing()
require_login()

with st.sidebar:
    section = st.radio(
        "Bölüm",
        [
            "Geçmiş Veri", "Yaklaşan Maçlar", "Güncel Fikstür", "Dünya Fikstürü",
            "Analiz Geçmişi", "Model Testi",
        ],
        key="app_section",
    )
    st.caption("Maç analizi yapabilir veya modelin geçmiş performansını ölçebilirsiniz.")

try:
    supabase = get_supabase_client()
except Exception as exc:
    st.error("Supabase bağlantısı kurulamadı.")
    st.code(str(exc))
    st.stop()

st.success("Supabase bağlantısı başarılı.")

try:
    if section == "Geçmiş Veri":
        render_historical_page(supabase)
    elif section == "Yaklaşan Maçlar":
        render_upcoming_page(supabase)
    elif section == "Güncel Fikstür":
        render_football_data_fixtures_page(supabase)
    elif section == "Dünya Fikstürü":
        render_world_fixtures_page(supabase)
    elif section == "Analiz Geçmişi":
        render_analysis_history_page(supabase)
    else:
        render_backtest_page(supabase)
except Exception as exc:
    st.error("Ekran hazırlanırken beklenmeyen bir hata oluştu.")
    st.code(str(exc))

st.divider()
st.markdown(
    '<p class="small-note">Bu uygulama istatistiksel analiz sağlar; kesin kazanç vaat etmez. '
    "Bahis kararlarında bütçe sınırı belirleyin ve sorumlu oynayın.</p>",
    unsafe_allow_html=True,
)

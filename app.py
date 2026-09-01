from __future__ import annotations

import hmac
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from supabase import Client, create_client

from analysis import (
    build_report,
    fetch_h2h_rows,
    fetch_league_rows,
    fetch_same_odds_rows,
    fetch_team_form_rows,
    generate_gemini_grounded_analysis,
    h2h_summary_tables,
    odds_summary_table,
    rows_to_table,
)
from backtest import aggregate_backtests, run_backtest
from calibration import calibrate_model
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

        odd_1, odd_x, odd_2 = st.columns(3)
        b365_home = odd_1.number_input("Bet365 MS 1", min_value=1.01, value=2.00, step=0.01)
        b365_draw = odd_x.number_input("Bet365 X", min_value=1.01, value=3.00, step=0.01)
        b365_away = odd_2.number_input("Bet365 MS 2", min_value=1.01, value=3.00, step=0.01)

        over_col, under_col = st.columns(2)
        b365_over = over_col.number_input(
            "Bet365 2.5 Üst — yoksa 0", min_value=0.0, value=0.0, step=0.01
        )
        b365_under = under_col.number_input(
            "Bet365 2.5 Alt — yoksa 0", min_value=0.0, value=0.0, step=0.01
        )

        submitted = st.form_submit_button(
            "Maçı kaydet", type="primary", use_container_width=True
        )

    if not submitted:
        return

    if home_team == away_team:
        st.error("Ev sahibi ve deplasman takımı aynı olamaz.")
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
        "raw_data": {},
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
    home = str(match.get("home_team") or "")
    away = str(match.get("away_team") or "")
    division = str(match.get("division") or "")
    st.divider()
    st.subheader(f"📊 Maç analizi · {home} — {away}")
    st.caption(
        f"{division} · {match.get('match_date', '—')} · {match.get('kickoff_time', '—')}"
    )

    if st.button("Analizi yenile", key=f"refresh_analysis_{match.get('id')}"):
        st.rerun()

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
            report["same_odds_all"] = same_all_rows
        except Exception as exc:
            st.error("Analiz verileri alınamadı.")
            st.code(str(exc))
            return

    st.info(
        "İstatistiksel model geçmiş verilerle çalışır. Tavily güncel haberleri arar; "
        "Gemini bu kaynakları ve tüm istatistikleri birleştirerek kendi tahminini üretir."
    )

    st.markdown("#### 1. Geçmiş rekabet · Son 10 maç")
    if h2h_rows:
        outcome_summary, goal_summary = h2h_summary_tables(h2h_rows, home, away)
        st.dataframe(outcome_summary, use_container_width=True, hide_index=True)
        st.dataframe(goal_summary, use_container_width=True, hide_index=True)
        st.dataframe(rows_to_table(h2h_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Bu iki takım arasında veritabanında geçmiş karşılaşma bulunamadı.")

    st.markdown("#### 2. Aynı Bet365 oran analizi")
    st.caption(
        "Önce aynı lig, ardından tüm ligler içindeki birebir MS oranı eşleşmeleri gösterilir."
    )
    st.markdown("##### A) Aynı ligde aynı oranlar")
    if same_league_rows:
        st.dataframe(
            odds_summary_table(same_league_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            rows_to_table(same_league_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Bu ligde aynı üçlü Bet365 MS oranına sahip geçmiş maç bulunamadı.")

    st.markdown("##### B) Tüm liglerde aynı oranlar")
    if same_all_rows:
        st.dataframe(
            odds_summary_table(same_all_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            rows_to_table(same_all_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Tüm geçmiş veriler içinde aynı üçlü Bet365 MS oranına sahip maç bulunamadı.")

    predictions = report["predictions"]
    st.markdown("#### 3. Tahmin özeti")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("MS", predictions["ms"])
    p2.metric("İY/MS", predictions["ht_ms"])
    p3.metric("Skor", predictions["score"])
    p4.metric("KG", predictions["btts_prediction"])

    ms_probabilities = predictions.get("ms_probabilities") or {}
    probability_frame = pd.DataFrame(
        [
            {
                "Sonuç": label,
                "Birleşik model olasılığı": (
                    f"{float(ms_probabilities.get(label, 0)) * 100:.1f}%"
                ),
            }
            for label in ("1", "X", "2")
        ]
    )
    st.dataframe(probability_frame, use_container_width=True, hide_index=True)
    st.caption(
        f"Beklenen goller: {home} {float(predictions.get('expected_home_goals', 0)):.2f} — "
        f"{away} {float(predictions.get('expected_away_goals', 0)):.2f} · "
        f"Model güveni: {predictions.get('confidence', '—')}"
    )

    totals = predictions["totals"]
    totals_frame = pd.DataFrame(
        [
            {
                "Baremi": f"{threshold} üst/alt",
                "Tahmin": data["prediction"],
                "Tahmin olasılığı": (
                    f"{data['prediction_probability'] * 100:.1f}%"
                    if data.get("prediction_probability") is not None
                    else "—"
                ),
            }
            for threshold, data in totals.items()
        ]
    )
    st.dataframe(totals_frame, use_container_width=True, hide_index=True)

    components = report.get("components") or []
    if components:
        st.markdown("##### Tahmini etkileyen veri kaynakları")
        component_frame = pd.DataFrame(components)
        for column in ("Ağırlık", "1", "X", "2"):
            component_frame[column] = component_frame[column].map(
                lambda value: f"{float(value) * 100:.1f}%"
            )
        st.dataframe(component_frame, use_container_width=True, hide_index=True)

    for warning in report.get("warnings") or []:
        st.warning(str(warning))

    st.markdown("#### İstatistiksel yorum")
    st.write(report["comment"])
    st.info(f"İstatistiksel ön kupon: {report['coupon']}")
    st.caption(
        f"Lig örneklemi: {predictions['sample_size']} tamamlanmış maç. "
        "Bu çıktı kesin sonuç veya kazanç garantisi değildir."
    )

    st.markdown("#### 4. Canlı araştırma ve Gemini tahmini")
    st.write(
        "Tavily yalnızca seçili maç ve takımlar için güncel kaynakları arar. Gemini, "
        "bu haberleri veritabanındaki istatistiklerle birleştirerek kendi tahminini üretir."
    )
    try:
        gemini_api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except KeyError:
        gemini_api_key = ""
    try:
        tavily_api_key = str(st.secrets["TAVILY_API_KEY"]).strip()
    except KeyError:
        tavily_api_key = ""

    if not gemini_api_key or not tavily_api_key:
        missing_keys = []
        if not gemini_api_key:
            missing_keys.append("GEMINI_API_KEY")
        if not tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
        st.warning(
            "Canlı araştırma için eksik Streamlit Secrets anahtarı: "
            + ", ".join(missing_keys)
        )
    else:
        gemini_state_key = f"tavily_gemini_analysis_v3_{match.get('id')}"
        if st.button(
            "Güncel haberleri araştır ve Gemini tahmini oluştur",
            key=f"tavily_gemini_button_{match.get('id')}",
            use_container_width=True,
        ):
            with st.spinner(
                "Tavily güncel kaynakları arıyor; Gemini tüm verileri yorumluyor..."
            ):
                try:
                    st.session_state[gemini_state_key] = (
                        generate_gemini_grounded_analysis(
                            gemini_api_key,
                            tavily_api_key,
                            match,
                            report,
                        )
                    )
                except Exception as exc:
                    st.session_state[gemini_state_key] = {"error": str(exc)}
            st.rerun()

        gemini_result = st.session_state.get(gemini_state_key)
        if isinstance(gemini_result, dict) and "error" in gemini_result:
            st.error(
                "Canlı araştırma tamamlanamadı. Tavily veya Gemini anahtarını ve "
                "ücretsiz kullanım kotasını kontrol edin."
            )
            st.caption(str(gemini_result["error"]))
        elif gemini_result:
            st.markdown(str(gemini_result["text"]))
            if gemini_result.get("model"):
                st.caption(f"Kullanılan Gemini modeli: {gemini_result['model']}")
            sources = gemini_result.get("sources") or []
            if sources:
                st.markdown("##### Tavily tarafından bulunan kaynaklar")
                for number, source in enumerate(sources, start=1):
                    title = str(source["title"]).replace("[", "(").replace("]", ")")
                    category = str(source.get("category") or "Kaynak")
                    st.markdown(
                        f"{number}. **{category}:** [{title}]({source['url']})"
                    )
            search_warnings = gemini_result.get("search_warnings") or []
            if search_warnings:
                st.warning(
                    "Bazı haber aramaları tamamlanamadı; mevcut kaynaklarla analiz yapıldı."
                )
            st.caption(
                "Gemini tahminleri kesin sonuç veya kazanç garantisi değildir. "
                "Kadro ve haberleri maç öncesinde kaynaklardan doğrulayın."
            )


def render_upcoming_list_tab(client: Client, today) -> None:
    try:
        response = (
            client.table("upcoming_matches")
            .select(
                "id,division,match_date,kickoff_time,home_team,away_team,"
                "b365_home,b365_draw,b365_away,b365_over_25,b365_under_25,"
                "entry_method,match_status"
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
    frame = frame.drop(columns=["Kayıt ID"], errors="ignore")
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
            render_match_analysis(client, rows[selected_index])


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
        "Bölüm", ["Geçmiş Veri", "Yaklaşan Maçlar", "Model Testi"], key="app_section"
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

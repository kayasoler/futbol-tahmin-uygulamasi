from __future__ import annotations

import hmac

import pandas as pd
import streamlit as st
from supabase import Client, create_client

from data_import import (
    REQUIRED_COLUMNS,
    build_records,
    fetch_existing_match_keys,
    insert_new_records,
    read_football_csv,
    validate_and_prepare,
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
    """Create one server-side Supabase client for the Streamlit session."""
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
    """Protect the server-side data tools with a shared app password."""
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


def get_historical_count(client: Client) -> int:
    response = (
        client.table("historical_matches")
        .select("id", count="exact")
        .limit(1)
        .execute()
    )
    return int(response.count or 0)


st.title("⚽ Futbol Tahmin ve Analiz")
st.caption("Aşama 1 · Geçmiş maç verilerinin güvenli ve tekrarsız yüklenmesi")

with st.sidebar:
    st.subheader("Proje durumu")
    st.success("Veritabanı tabloları hazır")
    st.info("Şu anda: Geçmiş CSV yükleme")
    st.caption("Tahmin ve canlı araştırma sonraki aşamalarda eklenecek.")

    if st.session_state.get("authenticated") is True:
        if st.button("Güvenli çıkış"):
            st.session_state["authenticated"] = False
            st.rerun()

stop_if_secrets_missing()
require_login()

try:
    supabase = get_supabase_client()
    current_count = get_historical_count(supabase)
except Exception as exc:
    st.error("Supabase bağlantısı kurulamadı.")
    st.code(str(exc))
    st.stop()

st.success("Supabase bağlantısı başarılı.")

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

metric_left, metric_right = st.columns(2)
metric_left.metric("Veritabanındaki geçmiş maç", f"{current_count:,}".replace(",", "."))
metric_right.metric("Duplicate koruması", "Aktif")

st.divider()
st.subheader("Geçmiş maç CSV dosyalarını toplu yükle")
st.write(
    "football-data.co.uk üzerinden indirdiğiniz tüm CSV dosyalarını aynı anda seçin. "
    "Dosyalar birlikte kontrol edilir; siz düğmeye basmadan veritabanına yazılmaz."
)

uploaded_files = st.file_uploader(
    "CSV dosyalarını seçin",
    type=["csv"],
    accept_multiple_files=True,
    help="Windows'ta dosyaları Ctrl+A ile topluca seçebilirsiniz.",
)

if uploaded_files:
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
            "henüz veritabanına yazılmayacak. Hatalı dosyaları seçimden çıkarın veya düzeltin."
        )
        st.caption("Gerekli temel sütunlar: " + ", ".join(REQUIRED_COLUMNS))
        st.stop()

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

    if file_duplicate_count or cross_file_duplicate_count:
        st.info(
            f"Tekrar ayrıntısı — dosyaların kendi içinde: {file_duplicate_count}, "
            f"farklı dosyalar arasında: {cross_file_duplicate_count}."
        )

    st.dataframe(
        preview_frame,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Tüm CSV dosyalarını veritabanına aktar",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Tüm kayıtlar kontrol ediliyor ve yükleniyor..."):
            try:
                existing_keys = fetch_existing_match_keys(supabase)
                records: list[dict[str, object]] = []

                for source_file, source_frame in combined_frame.groupby(
                    "__source_file",
                    sort=False,
                ):
                    clean_source_frame = source_frame.drop(columns=["__source_file"])
                    records.extend(
                        build_records(
                            clean_source_frame,
                            source_file=str(source_file),
                            existing_keys=existing_keys,
                        )
                    )

                already_existing = len(combined_frame) - len(records)
                inserted, race_duplicates = insert_new_records(supabase, records)
            except Exception as exc:
                st.error(
                    "Toplu yükleme tamamlanamadı. Hiçbir şifreyi paylaşmadan "
                    "hata metnini gönderin."
                )
                st.code(str(exc))
                st.stop()

        st.session_state["last_import_summary"] = {
            "inserted": inserted,
            "selected_files": len(uploaded_files),
            "file_duplicates": file_duplicate_count,
            "cross_file_duplicates": cross_file_duplicate_count,
            "existing": already_existing,
            "race_duplicates": race_duplicates,
        }
        st.rerun()

st.divider()
st.markdown(
    '<p class="small-note">Bu uygulama istatistiksel analiz sağlar; kesin kazanç vaat etmez. '
    "Bahis kararlarında bütçe sınırı belirleyin ve sorumlu oynayın.</p>",
    unsafe_allow_html=True,
)

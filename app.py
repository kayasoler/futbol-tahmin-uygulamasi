from __future__ import annotations

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
        background: #f7f8fa;
        border: 1px solid #e7e9ee;
        border-radius: 14px;
        padding: 14px;
    }
    .small-note {color: #667085; font-size: 0.9rem;}
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
    except (KeyError, FileNotFoundError):
        st.info(
            "Uygulama hazır. Supabase bağlantısı için Streamlit Secrets "
            "ayarlarının eklenmesi gerekiyor."
        )
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

stop_if_secrets_missing()

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
    st.success(f"{summary['inserted']} yeni maç başarıyla eklendi.")
    if summary["file_duplicates"] or summary["existing"] or summary["race_duplicates"]:
        st.info(
            f"Atlanan tekrarlar — dosya içinde: {summary['file_duplicates']}, "
            f"veritabanında zaten bulunan: {summary['existing']}, "
            f"eşzamanlı yakalanan: {summary['race_duplicates']}."
        )

metric_left, metric_right = st.columns(2)
metric_left.metric("Veritabanındaki geçmiş maç", f"{current_count:,}".replace(",", "."))
metric_right.metric("Duplicate koruması", "Aktif")

st.divider()
st.subheader("Geçmiş maç CSV dosyası yükle")
st.write(
    "football-data.co.uk üzerinden indirdiğiniz CSV dosyasını seçin. "
    "Dosya önce kontrol edilir; siz düğmeye basmadan veritabanına yazılmaz."
)

uploaded_file = st.file_uploader(
    "CSV dosyasını seçin",
    type=["csv"],
    help="Örnek: EC.csv",
)

if uploaded_file is not None:
    try:
        raw_frame = read_football_csv(uploaded_file.getvalue())
        prepared_frame, file_duplicate_count = validate_and_prepare(raw_frame)
    except ValueError as exc:
        st.error(str(exc))
        st.caption("Gerekli temel sütunlar: " + ", ".join(REQUIRED_COLUMNS))
        st.stop()

    preview_cols = [
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
    preview_cols = [column for column in preview_cols if column in prepared_frame.columns]

    st.success("CSV biçimi geçerli.")
    a, b, c = st.columns(3)
    a.metric("Dosyadaki satır", len(raw_frame))
    b.metric("Geçerli benzersiz maç", len(prepared_frame))
    c.metric("Dosya içi tekrar", file_duplicate_count)

    st.dataframe(
        prepared_frame[preview_cols].head(10),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Veritabanına aktar", type="primary", use_container_width=True):
        with st.spinner("Kayıtlar kontrol ediliyor ve yükleniyor..."):
            try:
                existing_keys = fetch_existing_match_keys(supabase)
                records = build_records(
                    prepared_frame,
                    source_file=uploaded_file.name,
                    existing_keys=existing_keys,
                )
                already_existing = len(prepared_frame) - len(records)
                inserted, race_duplicates = insert_new_records(supabase, records)
            except Exception as exc:
                st.error("Yükleme tamamlanamadı. Hiçbir şifreyi paylaşmadan hata metnini gönderin.")
                st.code(str(exc))
                st.stop()

        st.session_state["last_import_summary"] = {
            "inserted": inserted,
            "file_duplicates": file_duplicate_count,
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

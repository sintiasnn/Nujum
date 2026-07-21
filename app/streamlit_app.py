"""
app/streamlit_app.py — UI Streamlit untuk Nujum
Prediksi penjualan & rekomendasi stok untuk UMKM Indonesia
"""

import sys
from pathlib import Path
import os

# Tambahkan src/ ke path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from google import genai
from google.genai import types

from preprocess import standardize_columns, clean_data, aggregate_daily, get_product_list
from train import train_prophet, evaluate_model, save_model, split_train_test
from predict import forecast, recommend_stock, get_available_models, get_historical_trend

load_dotenv()


# ── Konfigurasi Halaman ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nujum — Prediksi Penjualan UMKM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Custom ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Palet warna profesional - biru dan abu-abu */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .tagline {
        font-size: 1.3rem;
        color: #64748B;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: #1E40AF;
        color: white;
        border-radius: 8px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    /* Ukuran font lebih besar untuk keterbacaan */
    .stMarkdown {
        font-size: 1.05rem;
    }
    /* Button lebih besar dan jelas */
    .stButton > button {
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ────────────────────────────────────────────────────────────

def get_gemini_insight(recommendation: dict, product_name: str, periods: int) -> str:
    """Generate insight narasi dari Gemini AI."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
Kamu adalah asisten bisnis untuk UMKM Indonesia. Berikan insight singkat dan praktis (3-4 kalimat) 
dalam Bahasa Indonesia yang mudah dipahami pemilik toko kecil.

Data prediksi penjualan:
- Produk: {product_name}
- Periode prediksi: {periods} hari ke depan
- Total perkiraan terjual: {recommendation['total_predicted_sales']} unit
- Rata-rata penjualan harian: {recommendation['avg_daily_sales']} unit/hari
- Penjualan tertinggi diperkirakan pada: {recommendation['peak_date']}
- Rekomendasi stok yang perlu disiapkan: {recommendation['recommended_stock']} unit

Berikan insight yang:
1. Menjelaskan tren penjualan dengan bahasa sederhana
2. Menekankan kapan perlu siapkan stok ekstra
3. Saran praktis untuk pemilik UMKM
Jangan gunakan bahasa teknis atau istilah statistik.
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return None


def plot_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, product_name: str) -> go.Figure:
    """Buat grafik gabungan historis + prediksi."""
    fig = go.Figure()

    # Data historis
    fig.add_trace(go.Scatter(
        x=history_df["ds"],
        y=history_df["y"],
        mode="lines",
        name="Penjualan Aktual",
        line=dict(color="#6C63FF", width=2),
    ))

    # Area prediksi (confidence interval)
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Rentang Prediksi",
        showlegend=True,
    ))

    # Garis prediksi utama
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines",
        name="Prediksi Penjualan",
        line=dict(color="#FF6B35", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=f"Tren & Prediksi Penjualan — {product_name.replace('_', ' ').title()}",
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Terjual",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

    return fig


def plot_daily_forecast_bar(forecast_df: pd.DataFrame) -> go.Figure:
    """Bar chart prediksi penjualan harian."""
    fig = px.bar(
        forecast_df,
        x="ds",
        y="yhat",
        color="yhat",
        color_continuous_scale="Purpor",
        labels={"ds": "Tanggal", "yhat": "Prediksi Penjualan"},
        title="Detail Prediksi Penjualan Harian",
    )
    fig.update_layout(
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
    )
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Nujum")
    st.markdown("**Prediksi Penjualan & Stok untuk UMKM**")
    st.markdown("*Ramal penjualanmu, kelola stok lebih cerdas.*")
    st.divider()

    st.markdown("### Pengaturan Prediksi")
    periods = st.selectbox(
        "Prediksi untuk berapa hari ke depan?",
        options=[7, 14, 30, 60, 90],
        index=2,
        help="Pilih periode waktu untuk prediksi penjualan"
    )

    safety_factor = st.slider(
        "Buffer Stok Keamanan (%)",
        min_value=100,
        max_value=200,
        value=120,
        step=5,
        help="Tambahan stok cadangan di atas prediksi dasar. 120% = tambah 20% dari prediksi."
    )
    
    # Konversi % ke float untuk kalkulasi
    safety_factor = safety_factor / 100.0

    st.divider()
    st.markdown("### Panduan Penggunaan")
    st.markdown("""
    **Langkah 1:** Upload file data penjualan (format CSV)
    
    **Langkah 2:** Pilih kolom tanggal, jumlah, dan produk
    
    **Langkah 3:** Klik tombol Mulai Training
    
    **Langkah 4:** Lihat hasil prediksi dan rekomendasi stok
    """)
    st.divider()
    st.caption("Dibuat untuk IDCamp Developer Challenge 2026")


# ── Main Content ────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Nujum</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Sistem Prediksi Penjualan dan Rekomendasi Stok untuk UMKM Indonesia</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Upload & Training Model", "Prediksi & Rekomendasi Stok", "Analisis & Perbandingan"])


# ── TAB 1: UPLOAD & TRAINING ────────────────────────────────────────────────────

with tab1:
    st.markdown("### Upload File Data Penjualan")
    st.markdown("Silakan upload file CSV yang berisi data penjualan toko Anda. Minimal 30 hari data untuk hasil yang akurat.")

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        help="Format: CSV dengan kolom tanggal, jumlah terjual, dan nama produk (opsional)"
    )

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {str(e)}")
            df_raw = None

        if df_raw is not None:
            st.success(f"File berhasil diupload: **{len(df_raw):,} baris**, **{len(df_raw.columns)} kolom**")

            with st.expander("Preview Data (5 baris pertama)"):
                st.dataframe(df_raw.head(), use_container_width=True)

            st.markdown("### Pilih Kolom")
            col1, col2, col3 = st.columns(3)
            cols = df_raw.columns.tolist()

            with col1:
                date_col = st.selectbox("Kolom Tanggal", options=cols, help="Kolom yang berisi tanggal transaksi")
            with col2:
                qty_col = st.selectbox("Kolom Jumlah Terjual", options=cols, index=min(1, len(cols)-1), help="Kolom yang berisi jumlah unit terjual")
            with col3:
                product_options = ["(Tidak ada kolom produk)"] + cols
                product_col_selected = st.selectbox("Kolom Produk (opsional)", options=product_options, help="Kolom nama produk, kosongkan jika tidak ada")

            product_col = None if product_col_selected == "(Tidak ada kolom produk)" else product_col_selected

            st.divider()

            if st.button("MULAI TRAINING MODEL", type="primary", use_container_width=True):
                with st.spinner("Sedang melatih model... Ini mungkin membutuhkan beberapa menit ⏳"):
                    try:
                        df_clean = standardize_columns(df_raw.copy(), date_col, qty_col, product_col)
                        df_clean = clean_data(df_clean)

                        if len(df_clean) < 10:
                            st.error("Data terlalu sedikit. Minimal 10 hari data diperlukan.")
                        else:
                            # Training model total
                            daily_all = aggregate_daily(df_clean)
                            train_df, test_df = split_train_test(daily_all)
                            model = train_prophet(train_df)
                            metrics = evaluate_model(model, test_df)
                            save_model(model, "all")

                            # Training per produk
                            trained_products = ["all"]
                            products = get_product_list(df_clean)
                            for product in products[:10]:
                                try:
                                    daily_p = aggregate_daily(df_clean, product=product)
                                    if len(daily_p) < 30:
                                        continue
                                    train_p, test_p = split_train_test(daily_p)
                                    model_p = train_prophet(train_p)
                                    save_model(model_p, product)
                                    trained_products.append(product)
                                except Exception:
                                    continue

                            st.success(f"Training selesai! Model tersedia untuk: **{len(trained_products)}** produk/kategori")

                            # Tampilkan metrik evaluasi
                            st.markdown("#### Performa Model (Total Penjualan)")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("MAE", f"{float(metrics['MAE']):,.1f}", help="Mean Absolute Error — rata-rata selisih prediksi vs aktual")
                            m2.metric("RMSE", f"{float(metrics['RMSE']):,.1f}", help="Root Mean Square Error")
                            m3.metric("MAPE", f"{float(metrics['MAPE']):.1f}%", help="Mean Absolute Percentage Error — semakin kecil semakin akurat")

                            st.session_state["trained"] = True
                            st.session_state["trained_products"] = trained_products
                            st.info("Sekarang buka tab **Prediksi & Stok** untuk melihat hasil prediksi!")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        # Tampilkan contoh format CSV
        st.info("**Format CSV yang diharapkan:**")
        sample_data = pd.DataFrame({
            "Tanggal": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Produk": ["Baju Batik", "Kerudung", "Baju Batik"],
            "Jumlah_Terjual": [15, 8, 12],
        })
        st.dataframe(sample_data, use_container_width=True)
        st.caption("Kolom tanggal, produk, dan jumlah terjual bisa memiliki nama apapun — kamu akan memilihnya setelah upload.")


# ── TAB 2: PREDIKSI & STOK ──────────────────────────────────────────────────────

with tab2:
    available_models = get_available_models()

    if not available_models:
        st.warning("Belum ada model yang ditraining. Silakan upload data dan lakukan training di tab **Upload & Training** terlebih dahulu.")
    else:
        st.markdown("### Prediksi Penjualan")

        col_prod, col_btn = st.columns([3, 1])
        with col_prod:
            selected_product = st.selectbox(
                "Pilih Produk",
                options=available_models,
                format_func=lambda x: "🛍 Semua Produk (Total)" if x == "all" else f"{x.replace('_', ' ').title()}",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("TAMPILKAN PREDIKSI", type="primary", use_container_width=True)

        if predict_btn:
            with st.spinner(f"Sedang meramal penjualan untuk {periods} hari ke depan... 🔮"):
                try:
                    # Jalankan prediksi
                    forecast_df = forecast(product_name=selected_product, periods=periods)
                    recommendation = recommend_stock(forecast_df, safety_factor=safety_factor)

                    # Ambil data historis
                    try:
                        history_df = get_historical_trend(product_name=selected_product, last_n_days=90)
                    except Exception:
                        history_df = pd.DataFrame(columns=["ds", "y"])

                    # Simpan ke session state
                    st.session_state["forecast_df"] = forecast_df
                    st.session_state["recommendation"] = recommendation
                    st.session_state["history_df"] = history_df
                    st.session_state["selected_product"] = selected_product

                except Exception as e:
                    st.error(f"Gagal melakukan prediksi: {str(e)}")

        # Tampilkan hasil jika sudah ada di session state
        if "recommendation" in st.session_state:
            rec = st.session_state["recommendation"]
            f_df = st.session_state["forecast_df"]
            h_df = st.session_state["history_df"]
            prod = st.session_state["selected_product"]

            st.divider()

            # Kotak rekomendasi stok utama
            st.markdown(f"""
            <div class="recommendation-box">
                <h3 style="margin-top:0;">REKOMENDASI STOK UNTUK {periods} HARI KE DEPAN</h3>
                <h1 style="font-size: 3.5rem; margin: 1rem 0; font-weight: 700;">{rec['recommended_stock']:,} UNIT</h1>
                <p style="font-size: 1.1rem; opacity: 0.9;">Sudah termasuk buffer keamanan {int((safety_factor-1)*100)}% dari prediksi dasar</p>
            </div>
            """, unsafe_allow_html=True)

            # Metrik detail
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Prediksi Terjual", f"{rec['total_predicted_sales']:,} unit")
            c2.metric("Rata-rata/Hari", f"{rec['avg_daily_sales']} unit")
            c3.metric("🔝 Penjualan Tertinggi", f"{rec['max_daily_sales']:,} unit")
            c4.metric("📆 Puncak Penjualan", rec['peak_date'])

            st.divider()

            # Grafik
            if len(h_df) > 0:
                st.plotly_chart(plot_forecast(h_df, f_df, prod), use_container_width=True)

            st.plotly_chart(plot_daily_forecast_bar(f_df), use_container_width=True)

            # Insight dari Gemini AI
            st.markdown("### Insight AI")
            with st.spinner("Menganalisis tren dan membuat insight..."):
                insight = get_gemini_insight(rec, prod, periods)

            if insight:
                st.markdown(f"""
                <div class="insight-box">
                    <b>Analisis dari AI:</b><br><br>
                    {insight}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Fallback insight tanpa Gemini
                st.markdown(f"""
                <div class="insight-box">
                    <b>Ringkasan Prediksi:</b><br><br>
                    Dalam <b>{periods} hari</b> ke depan, diperkirakan terjual sekitar 
                    <b>{rec['total_predicted_sales']:,} unit</b> dengan rata-rata <b>{rec['avg_daily_sales']} unit per hari</b>. 
                    Puncak penjualan diperkirakan terjadi pada <b>{rec['peak_date']}</b>. 
                    Siapkan minimal <b>{rec['recommended_stock']:,} unit</b> stok untuk mengantisipasi permintaan 
                    (sudah termasuk buffer {int((safety_factor-1)*100)}%).
                    <br><br>
                    <small><i>Tambahkan GEMINI_API_KEY di file .env untuk mendapatkan insight AI yang lebih detail.</i></small>
                </div>
                """, unsafe_allow_html=True)

            # Tabel detail prediksi
            with st.expander("Lihat Detail Prediksi Per Hari"):
                display_df = f_df.copy()
                display_df.columns = ["Tanggal", "Prediksi", "Batas Bawah", "Batas Atas"]
                st.dataframe(display_df, use_container_width=True)

                csv = display_df.to_csv(index=False)
                st.download_button(
                    "Download Prediksi (CSV)",
                    data=csv,
                    file_name=f"prediksi_{prod}_{periods}hari.csv",
                    mime="text/csv",
                )


# ── TAB 3: ANALITIK ─────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### Analitik Model")

    available_models = get_available_models()
    if not available_models:
        st.warning("Belum ada model yang tersedia. Lakukan training terlebih dahulu.")
    else:
        st.markdown(f"**Total model tersedia:** {len(available_models)}")

        # Daftar model
        model_data = []
        for m in available_models:
            model_data.append({
                "Produk/Kategori": "Semua Produk (Total)" if m == "all" else m.replace("_", " ").title(),
                "Status": "Siap",
                "Kode Model": m,
            })

        st.dataframe(pd.DataFrame(model_data), use_container_width=True)

        st.divider()
        st.markdown("### Perbandingan Tren Antar Produk")
        st.info("Pilih beberapa produk untuk membandingkan tren penjualannya.")

        compare_products = st.multiselect(
            "Pilih produk untuk dibandingkan",
            options=available_models,
            default=available_models[:min(3, len(available_models))],
            format_func=lambda x: "Semua Produk" if x == "all" else x.replace("_", " ").title(),
        )

        if compare_products and st.button("Tampilkan Perbandingan"):
            fig_compare = go.Figure()
            colors = ["#6C63FF", "#FF6B35", "#4CAF50", "#FF9800", "#E91E63"]

            for i, product in enumerate(compare_products):
                try:
                    h_df = get_historical_trend(product_name=product, last_n_days=60)
                    if len(h_df) > 0:
                        fig_compare.add_trace(go.Scatter(
                            x=h_df["ds"],
                            y=h_df["y"],
                            mode="lines",
                            name="Semua Produk" if product == "all" else product.replace("_", " ").title(),
                            line=dict(color=colors[i % len(colors)], width=2),
                        ))
                except Exception:
                    continue

            fig_compare.update_layout(
                title="Perbandingan Tren Penjualan (60 Hari Terakhir)",
                xaxis_title="Tanggal",
                yaxis_title="Jumlah Terjual",
                hovermode="x unified",
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_compare, use_container_width=True)

# 🔮 Nujum

> *Ramal penjualanmu, kelola stok lebih cerdas.*

Nujum adalah aplikasi prediksi penjualan dan rekomendasi stok berbasis AI untuk pelaku UMKM Indonesia. Dibangun sebagai MVP dalam rangka IDCamp Developer Challenge 2026.

---

## 🎯 Fitur Utama

- 📊 **Prediksi Penjualan** — Forecast penjualan 7 atau 30 hari ke depan berbasis data historis
- 📦 **Rekomendasi Stok** — Saran jumlah stok yang perlu disiapkan berdasarkan prediksi
- 💡 **Insight AI** — Narasi otomatis dari Gemini AI yang mudah dipahami pemilik UMKM
- 📈 **Visualisasi Tren** — Grafik interaktif tren penjualan historis dan prediksi

---

## 🛠 Tech Stack

| Komponen | Teknologi |
|---|---|
| Model Forecasting | Prophet / XGBoost |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| AI Narasi | Google Gemini API |
| Visualisasi | Plotly |

---

## 🚀 Cara Menjalankan

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/username/nujum.git
cd nujum
pip install -r requirements.txt
```

### 2. Konfigurasi Environment

```bash
cp .env.example .env
# Edit .env dan isi GEMINI_API_KEY dengan API key kamu
```

### 3. Jalankan Backend API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Jalankan Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Struktur Project

```
Nujum/
├── data/
│   ├── raw/           # Dataset mentah
│   └── processed/     # Dataset hasil preprocessing
├── notebooks/
│   └── eda.ipynb      # Exploratory Data Analysis
├── src/
│   ├── preprocess.py  # Preprocessing pipeline
│   ├── train.py       # Training model
│   └── predict.py     # Inference / prediksi
├── api/
│   └── main.py        # FastAPI backend
├── app/
│   └── streamlit_app.py  # Streamlit UI
├── models/            # Saved model artifacts
├── .env.example
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

Menggunakan [Anonymous Transactional Dataset](https://data.mendeley.com/datasets/kcgf45y24m/2) — data transaksi nyata dari **UMKM Kopi & F&B Indonesia** yang beroperasi di beberapa outlet (Januari–September 2025).

Dataset mencakup 53.820 transaksi dari sistem Point-of-Sale (POS) dengan 96 produk kopi, minuman, dan makanan. Dataset dipublikasikan di Mendeley Data dengan lisensi CC BY 4.0.

---

## 👩‍💻 Developer

Dibuat oleh **Ni Putu Sintia Wati** — IDCamp 2026, jalur MLOps.

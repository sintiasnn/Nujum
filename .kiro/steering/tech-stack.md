# Tech Stack & Architecture

## Stack Utama

| Layer | Teknologi | Keterangan |
|---|---|---|
| ML Model | Prophet (Meta) | Time series forecasting utama |
| ML Alternatif | XGBoost | Fallback jika data tidak cukup untuk Prophet |
| Backend API | FastAPI | REST API untuk serve model |
| Frontend UI | Streamlit | UI interaktif, satu file |
| AI Narasi | Google Gemini 1.5 Flash | Generate insight bahasa Indonesia |
| Visualisasi | Plotly | Grafik interaktif |
| Serialisasi Model | joblib | Simpan/load model .pkl |
| Env Management | python-dotenv | Load .env file |

## Struktur Direktori

```
Nujum/
├── .kiro/steering/     ← konteks untuk Kiro AI
├── data/
│   ├── raw/            ← dataset mentah (tidak di-commit)
│   └── processed/      ← hasil preprocessing
├── notebooks/          ← EDA dan eksplorasi
├── src/
│   ├── preprocess.py   ← pipeline preprocessing
│   ├── train.py        ← training & evaluasi model
│   └── predict.py      ← inference & rekomendasi stok
├── api/
│   └── main.py         ← FastAPI app
├── app/
│   └── streamlit_app.py ← Streamlit UI
├── models/             ← saved model .pkl (tidak di-commit)
├── .env.example
├── requirements.txt
└── README.md
```

## Flow Data

```
CSV Upload → preprocess.py → aggregate_daily()
                                    ↓
                             train.py → Prophet model → models/*.pkl
                                    ↓
                             predict.py → forecast() → recommend_stock()
                                    ↓
                        Streamlit UI ← FastAPI ← Gemini Insight
```

## API Endpoints (FastAPI)

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/` | Info app |
| GET | `/health` | Health check |
| POST | `/upload` | Upload CSV, preview kolom |
| POST | `/train` | Training model dari data yang diupload |
| POST | `/predict` | Prediksi + rekomendasi stok |
| GET | `/models` | Daftar model tersedia |
| GET | `/history/{product}` | Data historis + fitted values |

## Konvensi Penamaan Model

Model disimpan di `models/prophet_{product_name}.pkl`.
- Semua produk digabung: `prophet_all.pkl`
- Per produk: `prophet_{nama_produk_lowercase_underscore}.pkl`

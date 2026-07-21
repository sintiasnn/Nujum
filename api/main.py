"""
api/main.py — FastAPI backend untuk Nujum
Endpoints: upload data, training, prediksi, rekomendasi stok
"""

import sys
import os
from pathlib import Path

# Tambahkan src/ ke path agar bisa import modul
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import io

from preprocess import clean_data, standardize_columns, aggregate_daily, get_product_list
from train import train_prophet, evaluate_model, save_model, split_train_test
from predict import forecast, recommend_stock, get_available_models, get_historical_trend


# ── App Init ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nujum API",
    description="API prediksi penjualan & rekomendasi stok untuk UMKM Indonesia 🔮",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State sederhana untuk menyimpan dataframe yang sudah diupload (in-memory)
uploaded_data: dict = {}


# ── Schema ─────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    date_col: str
    qty_col: str
    product_col: Optional[str] = None
    session_id: str


class PredictRequest(BaseModel):
    product_name: str = "all"
    periods: int = 30
    safety_factor: float = 1.2


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "Nujum",
        "tagline": "Ramal penjualanmu, kelola stok lebih cerdas 🔮",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload file CSV data penjualan.
    Mengembalikan preview kolom dan beberapa baris pertama.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Hanya file CSV yang didukung.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {str(e)}")

    # Simpan ke state in-memory dengan session_id = filename
    session_id = file.filename
    uploaded_data[session_id] = df

    return {
        "session_id": session_id,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Training model Prophet berdasarkan data yang sudah diupload.
    """
    if req.session_id not in uploaded_data:
        raise HTTPException(
            status_code=404,
            detail="Data tidak ditemukan. Silakan upload file CSV terlebih dahulu."
        )

    df = uploaded_data[req.session_id].copy()

    # Validasi kolom
    required_cols = [req.date_col, req.qty_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Kolom tidak ditemukan: {missing}")

    try:
        # Preprocessing
        df = standardize_columns(df, req.date_col, req.qty_col, req.product_col)
        df = clean_data(df)

        results = {}

        # Train model total
        daily_all = aggregate_daily(df)
        if len(daily_all) < 10:
            raise HTTPException(status_code=400, detail="Data terlalu sedikit (minimal 10 hari).")

        train_df, test_df = split_train_test(daily_all)
        model = train_prophet(train_df)
        metrics = evaluate_model(model, test_df)
        save_model(model, "all")
        results["all"] = metrics

        # Train per produk jika ada
        products = get_product_list(df)
        for product in products[:10]:  # Batasi 10 produk untuk MVP
            try:
                daily = aggregate_daily(df, product=product)
                if len(daily) < 30:
                    continue
                train_p, test_p = split_train_test(daily)
                model_p = train_prophet(train_p)
                metrics_p = evaluate_model(model_p, test_p)
                save_model(model_p, product)
                results[product] = metrics_p
            except Exception:
                continue

        return {
            "status": "success",
            "message": f"Model berhasil ditraining untuk {len(results)} produk/kategori.",
            "metrics": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat training: {str(e)}")


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Prediksi penjualan dan rekomendasi stok.
    """
    available = get_available_models()
    if not available:
        raise HTTPException(
            status_code=404,
            detail="Belum ada model yang ditraining. Silakan upload data dan lakukan training terlebih dahulu."
        )

    if req.product_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model untuk '{req.product_name}' tidak ditemukan. Model tersedia: {available}"
        )

    if req.periods < 1 or req.periods > 365:
        raise HTTPException(status_code=400, detail="Periode prediksi harus antara 1-365 hari.")

    try:
        forecast_df = forecast(product_name=req.product_name, periods=req.periods)
        recommendation = recommend_stock(forecast_df, safety_factor=req.safety_factor)

        return {
            "product": req.product_name,
            "periods": req.periods,
            "forecast": forecast_df.to_dict(orient="records"),
            "recommendation": recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat prediksi: {str(e)}")


@app.get("/models")
def list_models():
    """Daftar model yang tersedia."""
    available = get_available_models()
    return {"models": available, "count": len(available)}


@app.get("/history/{product_name}")
def get_history(product_name: str, last_n_days: int = 90):
    """Ambil data historis + fitted values untuk grafik."""
    available = get_available_models()
    if product_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{product_name}' tidak ditemukan. Model tersedia: {available}"
        )

    try:
        history_df = get_historical_trend(product_name=product_name, last_n_days=last_n_days)
        return {
            "product": product_name,
            "history": history_df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

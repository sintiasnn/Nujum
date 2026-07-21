"""
predict.py — Inference & rekomendasi stok dari model yang sudah ditraining
"""

import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet

from train import load_model


def forecast(product_name: str = "all", periods: int = 30) -> pd.DataFrame:
    """
    Prediksi penjualan ke depan.

    Args:
        product_name: nama produk atau 'all' untuk semua produk
        periods: jumlah hari ke depan yang ingin diprediksi

    Returns:
        DataFrame dengan kolom: ds, yhat, yhat_lower, yhat_upper
    """
    model = load_model(product_name)
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast_df = model.predict(future)

    # Ambil hanya periode prediksi ke depan (bukan historis)
    result = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()

    # Pastikan tidak ada nilai negatif
    result["yhat"] = result["yhat"].clip(lower=0).round(0).astype(int)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0).round(0).astype(int)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0).round(0).astype(int)

    return result


def recommend_stock(forecast_df: pd.DataFrame, safety_factor: float = 1.2) -> dict:
    """
    Hitung rekomendasi stok berdasarkan hasil prediksi.

    Args:
        forecast_df: hasil dari fungsi forecast()
        safety_factor: faktor keamanan stok (default 1.2 = tambah 20% buffer)

    Returns:
        Dictionary berisi ringkasan rekomendasi stok
    """
    total_predicted = int(forecast_df["yhat"].sum())
    max_daily = int(forecast_df["yhat"].max())
    avg_daily = float(forecast_df["yhat"].mean())
    peak_date = forecast_df.loc[forecast_df["yhat"].idxmax(), "ds"].strftime("%d %B %Y")

    recommended_stock = int(total_predicted * safety_factor)
    period_days = len(forecast_df)

    return {
        "period_days": period_days,
        "total_predicted_sales": total_predicted,
        "recommended_stock": recommended_stock,
        "avg_daily_sales": round(avg_daily, 1),
        "max_daily_sales": max_daily,
        "peak_date": peak_date,
        "safety_factor": safety_factor,
    }


def get_historical_trend(product_name: str = "all", last_n_days: int = 90) -> pd.DataFrame:
    """
    Ambil data historis dari model (in-sample fit) untuk ditampilkan di grafik.

    Returns:
        DataFrame dengan kolom: ds, y (aktual), yhat (fitted)
    """
    model = load_model(product_name)

    # Ambil data yang sudah di-fit oleh model
    history = model.history.copy()
    future_hist = model.make_future_dataframe(periods=0)
    fitted = model.predict(future_hist)

    merged = history.merge(fitted[["ds", "yhat"]], on="ds", how="left")
    merged["yhat"] = merged["yhat"].clip(lower=0).round(0).astype(int)

    return merged.tail(last_n_days)


def get_available_models() -> list:
    """Ambil daftar model yang sudah tersedia di folder models/."""
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        return []
    models = [f.stem.replace("prophet_", "") for f in models_dir.glob("prophet_*.pkl")]
    return sorted(models)


if __name__ == "__main__":
    # Contoh penggunaan
    available = get_available_models()
    print(f"Model tersedia: {available}")

    if "all" in available:
        print("\n🔮 Prediksi 30 hari ke depan (semua produk):")
        result = forecast(product_name="all", periods=30)
        print(result.head(10))

        print("\n📦 Rekomendasi Stok:")
        rec = recommend_stock(result)
        for key, val in rec.items():
            print(f"  {key}: {val}")

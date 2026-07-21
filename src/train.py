"""
train.py — Training model forecasting penjualan dengan Prophet
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import run_pipeline, aggregate_daily, get_product_list


MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def split_train_test(df: pd.DataFrame, test_days: int = 30) -> tuple:
    """Split data menjadi train dan test set."""
    cutoff = df["ds"].max() - pd.Timedelta(days=test_days)
    train = df[df["ds"] <= cutoff].copy()
    test = df[df["ds"] > cutoff].copy()
    return train, test


def train_prophet(df: pd.DataFrame) -> Prophet:
    """
    Training model Prophet untuk forecasting time series.
    
    Args:
        df: DataFrame dengan kolom 'ds' (tanggal) dan 'y' (penjualan)
    
    Returns:
        Model Prophet yang sudah di-fit
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(df[["ds", "y"]])
    print("✅ Model Prophet berhasil ditraining")
    return model


def evaluate_model(model: Prophet, test_df: pd.DataFrame) -> dict:
    """
    Evaluasi model pada test set.
    
    Returns:
        Dictionary berisi MAE, RMSE, dan MAPE
    """
    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)

    # Ambil prediksi untuk periode test saja
    pred = forecast.tail(len(test_df))["yhat"].values
    actual = test_df["y"].values

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    # Hanya hitung MAPE untuk hari dengan penjualan > 0 (hindari div-by-zero pada hari tanpa transaksi)
    nonzero_mask = actual > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero_mask] - pred[nonzero_mask]) / actual[nonzero_mask])) * 100
    else:
        mape = 0.0

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}
    print(f"📊 Evaluasi Model: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    return metrics


def save_model(model: Prophet, product_name: str = "all"):
    """Simpan model ke folder models/."""
    safe_name = product_name.replace(" ", "_").replace("/", "-").lower()
    filepath = MODELS_DIR / f"prophet_{safe_name}.pkl"
    joblib.dump(model, filepath)
    print(f"✅ Model tersimpan: {filepath}")
    return filepath


def load_model(product_name: str = "all") -> Prophet:
    """Load model dari folder models/."""
    safe_name = product_name.replace(" ", "_").replace("/", "-").lower()
    filepath = MODELS_DIR / f"prophet_{safe_name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {filepath}")
    model = joblib.load(filepath)
    print(f"✅ Model dimuat: {filepath}")
    return model


def train_all_products(df: pd.DataFrame):
    """
    Training model untuk semua produk yang ada di dataset.
    Juga training satu model 'all' untuk total keseluruhan.
    """
    results = {}

    # Train model total (semua produk digabung)
    print("\n🔄 Training model total penjualan...")
    daily_all = aggregate_daily(df)
    train, test = split_train_test(daily_all)
    model_all = train_prophet(train)
    metrics_all = evaluate_model(model_all, test)
    save_model(model_all, "all")
    results["all"] = metrics_all

    # Train model per produk jika ada kolom product
    products = get_product_list(df)
    if products:
        for product in products:
            print(f"\n🔄 Training model untuk produk: {product}")
            try:
                daily = aggregate_daily(df, product=product)
                if len(daily) < 30:  # Skip jika data terlalu sedikit
                    print(f"⚠️  Skip {product}: data kurang dari 30 hari")
                    continue
                train, test = split_train_test(daily)
                model = train_prophet(train)
                metrics = evaluate_model(model, test)
                save_model(model, product)
                results[product] = metrics
            except Exception as e:
                print(f"❌ Gagal training {product}: {e}")

    return results


if __name__ == "__main__":
    # Sesuaikan parameter dengan kolom dataset yang digunakan
    df = run_pipeline(
        filename="retail_sales.csv",
        date_col="Date",
        qty_col="Quantity",
        product_col="Product"
    )
    results = train_all_products(df)
    print("\n📊 Ringkasan hasil training:")
    for product, metrics in results.items():
        print(f"  {product}: {metrics}")

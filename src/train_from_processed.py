"""
train_from_processed.py — Latih ulang semua model Prophet dari data/processed/

Script ini reproducible: membaca semua file daily_*.csv di data/processed/,
melatih satu model Prophet per file, mengevaluasinya, lalu menyimpan model ke
models/ beserta ringkasan metrik ke models/metrics.json.

Cara pakai:
    python src/train_from_processed.py
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from train import train_prophet, evaluate_model, save_model, split_train_test


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MIN_DAYS = 30  # minimal hari data untuk melatih model per produk


def product_name_from_file(filepath: Path) -> str:
    """Ambil nama produk dari nama file (daily_<produk>.csv -> <produk>)."""
    return filepath.stem.replace("daily_", "", 1)


def train_all_from_processed() -> dict:
    """
    Latih model untuk semua file daily_*.csv di data/processed/.

    Returns:
        Dictionary metrik per produk: {produk: {MAE, RMSE, MAPE}}.
    """
    files = sorted(PROCESSED_DIR.glob("daily_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file daily_*.csv di {PROCESSED_DIR}. "
            "Jalankan preprocessing dulu (src/preprocess_umkm_fb.py)."
        )

    results: dict = {}
    print(f"🔄 Menemukan {len(files)} file untuk ditraining...\n")

    for filepath in files:
        product = product_name_from_file(filepath)
        df = pd.read_csv(filepath, parse_dates=["ds"])

        if len(df) < MIN_DAYS:
            print(f"⚠️  Skip {product}: data {len(df)} hari (< {MIN_DAYS} hari)")
            continue

        print(f"🔄 Training: {product} ({len(df)} hari)")
        try:
            train_df, test_df = split_train_test(df)
            model = train_prophet(train_df)
            metrics = evaluate_model(model, test_df)
            save_model(model, product)
            results[product] = metrics
        except Exception as e:
            print(f"❌ Gagal training {product}: {e}")

    # Simpan ringkasan metrik + metadata agar hasil training terdokumentasi
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": "prophet",
        "test_days": 30,
        "total_models": len(results),
        "metrics": results,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Metrik tersimpan: {metrics_path}")

    return results


if __name__ == "__main__":
    results = train_all_from_processed()
    print(f"\n📊 Ringkasan hasil training ({len(results)} model):")
    for product, metrics in results.items():
        print(f"  {product}: {metrics}")

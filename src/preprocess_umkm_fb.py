"""
preprocess_umkm_fb.py — Preprocessing untuk dataset UMKM F&B Indonesia
(Transactions.csv + Products.csv dari Mendeley)
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_transactions() -> pd.DataFrame:
    """Load dan preprocessing Transactions.csv."""
    df = pd.read_csv(RAW_DIR / "Transactions.csv")
    
    # Parse tanggal
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    
    # Rename kolom ke format internal Nujum
    df = df.rename(columns={
        "Date": "ds",
        "TotalItem": "y",
        "Items": "product",
    })
    
    # Bersihkan data
    df = df[df["y"] > 0]  # Hapus transaksi dengan qty 0 atau negatif
    df = df.dropna(subset=["ds", "y"])
    
    # Ambil kolom yang relevan saja
    df = df[["ds", "y", "product", "Outlet"]].copy()
    
    print(f"✅ Transactions loaded: {len(df):,} transaksi")
    return df


def aggregate_daily_total(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi total penjualan harian (semua produk)."""
    daily = df.groupby("ds")["y"].sum().reset_index()
    daily = daily.sort_values("ds").reset_index(drop=True)
    
    # Fill tanggal yang kosong dengan 0
    date_range = pd.date_range(start=daily["ds"].min(), end=daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]
    
    print(f"✅ Agregasi harian: {len(daily)} hari ({daily['ds'].min()} s/d {daily['ds'].max()})")
    return daily


def get_top_products(df: pd.DataFrame, top_n: int = 10) -> list:
    """Ambil top N produk dengan penjualan terbanyak."""
    # Split kolom product jika berisi multiple items (dipisah koma)
    products_exploded = []
    
    for idx, row in df.iterrows():
        items = str(row["product"]).split(",")
        for item in items:
            item = item.strip()
            if item and item != "nan":
                products_exploded.append({"product": item, "y": row["y"]})
    
    df_exploded = pd.DataFrame(products_exploded)
    top_products = df_exploded.groupby("product")["y"].sum().sort_values(ascending=False).head(top_n)
    
    print(f"\n📊 Top {top_n} Produk Terlaris:")
    for i, (prod, qty) in enumerate(top_products.items(), 1):
        print(f"  {i}. {prod}: {qty:,} unit")
    
    return top_products.index.tolist()


def aggregate_by_product(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """Agregasi penjualan harian untuk satu produk spesifik."""
    # Filter transaksi yang mengandung produk ini (exact match, non-regex)
    df_filtered = df[df["product"].str.contains(product_name, case=False, na=False, regex=False)].copy()
    
    if len(df_filtered) == 0:
        print(f"⚠️  Tidak ada transaksi untuk produk: {product_name}")
        return pd.DataFrame(columns=["ds", "y"])
    
    daily = df_filtered.groupby("ds")["y"].sum().reset_index()
    daily = daily.sort_values("ds").reset_index(drop=True)
    
    # Fill tanggal yang kosong dengan 0
    date_range = pd.date_range(start=daily["ds"].min(), end=daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]
    
    return daily


def run_preprocessing():
    """Jalankan full preprocessing pipeline untuk dataset UMKM F&B."""
    print("🔄 Memproses dataset UMKM F&B Indonesia...\n")
    
    # Load data
    df = load_transactions()
    
    # Agregasi total penjualan harian (semua produk)
    daily_all = aggregate_daily_total(df)
    daily_all.to_csv(PROCESSED_DIR / "daily_all.csv", index=False)
    print(f"✅ Tersimpan: {PROCESSED_DIR / 'daily_all.csv'}")
    
    # Identifikasi top products
    top_products = get_top_products(df, top_n=10)
    
    # Agregasi per produk top
    for product in top_products:
        try:
            daily_product = aggregate_by_product(df, product)
            if len(daily_product) >= 30:  # Minimal 30 hari data
                safe_name = product.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").lower()
                daily_product.to_csv(PROCESSED_DIR / f"daily_{safe_name}.csv", index=False)
                print(f"✅ Tersimpan: daily_{safe_name}.csv ({len(daily_product)} hari)")
        except Exception as e:
            print(f"❌ Gagal proses {product}: {e}")
    
    print(f"\n✨ Preprocessing selesai! File tersimpan di: {PROCESSED_DIR}")
    return df, daily_all, top_products


if __name__ == "__main__":
    df, daily_all, top_products = run_preprocessing()
    
    print("\n📊 Ringkasan Dataset:")
    print(f"  Total transaksi: {len(df):,}")
    print(f"  Rentang tanggal: {df['ds'].min()} s/d {df['ds'].max()}")
    print(f"  Total hari: {len(daily_all)}")
    print(f"  Rata-rata penjualan/hari: {daily_all['y'].mean():.1f} item")

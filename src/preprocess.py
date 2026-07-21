"""
preprocess.py — Pipeline preprocessing data penjualan UMKM
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_data(filename: str) -> pd.DataFrame:
    """Load CSV dari folder data/raw."""
    filepath = RAW_DIR / filename
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def standardize_columns(df: pd.DataFrame, date_col: str, qty_col: str, product_col: str = None) -> pd.DataFrame:
    """
    Standarisasi nama kolom ke format internal Nujum:
    - ds: tanggal
    - y: jumlah terjual
    - product: nama produk (opsional)
    """
    rename_map = {
        date_col: "ds",
        qty_col: "y",
    }
    if product_col:
        rename_map[product_col] = "product"

    df = df.rename(columns=rename_map)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data: parse tanggal, hapus duplikat, handle missing values."""
    # Parse tanggal
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])

    # Pastikan kolom y numerik
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])

    # Hapus nilai negatif
    df = df[df["y"] >= 0]

    # Hapus duplikat
    df = df.drop_duplicates()

    # Sort by tanggal
    df = df.sort_values("ds").reset_index(drop=True)

    print(f"✅ Data bersih: {df.shape[0]} baris tersisa")
    return df


def aggregate_daily(df: pd.DataFrame, product: str = None) -> pd.DataFrame:
    """
    Agregasi ke level harian.
    Jika product dipilih, filter dulu berdasarkan produk.
    """
    if product and "product" in df.columns:
        df = df[df["product"] == product].copy()

    daily = df.groupby("ds")["y"].sum().reset_index()
    daily = daily.sort_values("ds").reset_index(drop=True)

    # Fill tanggal yang kosong dengan 0
    date_range = pd.date_range(start=daily["ds"].min(), end=daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]

    return daily


def get_product_list(df: pd.DataFrame) -> list:
    """Ambil daftar produk unik dari dataset."""
    if "product" not in df.columns:
        return []
    return sorted(df["product"].dropna().unique().tolist())


def save_processed(df: pd.DataFrame, filename: str):
    """Simpan data yang sudah diproses ke data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"✅ Data tersimpan: {filepath}")


def run_pipeline(filename: str, date_col: str, qty_col: str, product_col: str = None) -> pd.DataFrame:
    """
    Jalankan full preprocessing pipeline.
    
    Args:
        filename: nama file CSV di data/raw/
        date_col: nama kolom tanggal di dataset asli
        qty_col: nama kolom jumlah penjualan di dataset asli
        product_col: nama kolom produk (opsional)
    
    Returns:
        DataFrame yang sudah bersih dan siap untuk training
    """
    df = load_data(filename)
    df = standardize_columns(df, date_col, qty_col, product_col)
    df = clean_data(df)
    save_processed(df, f"processed_{filename}")
    return df


if __name__ == "__main__":
    # Contoh penggunaan — sesuaikan dengan kolom dataset yang dipakai
    df = run_pipeline(
        filename="retail_sales.csv",
        date_col="Date",
        qty_col="Quantity",
        product_col="Product"
    )
    print(df.head())
    print(f"\nProduk tersedia: {get_product_list(df)}")

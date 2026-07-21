# Dataset & ML Pipeline

## Dataset yang Digunakan

**Sumber:** [Anonymous Transactional Dataset](https://data.mendeley.com/datasets/kcgf45y24m/2) — Mendeley Data

Dataset transaksi nyata dari UMKM Food & Beverage Indonesia dengan multi-outlet (Jan–Sep 2025).

File yang tersedia:
- `Transactions.csv` — 53.820 transaksi
- `Products.csv` — 96 produk

Simpan file di: `data/raw/`

## Preprocessing Pipeline

Urutan yang harus diikuti:
1. `load_data(filename)` — baca CSV dari `data/raw/`
2. `standardize_columns(df, date_col, qty_col, product_col)` — rename kolom ke format internal (`ds`, `y`, `product`)
3. `clean_data(df)` — parse tanggal, buang null/negatif, deduplicate
4. `aggregate_daily(df, product)` — agregasi ke level harian, fill missing date dengan 0
5. `save_processed(df, filename)` — simpan ke `data/processed/`

## Format Internal DataFrame

Setelah preprocessing, semua DataFrame harus mengikuti format ini:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `ds` | datetime64 | Tanggal (Prophet convention) |
| `y` | float/int | Jumlah terjual |
| `product` | str (opsional) | Nama produk |

## Model: Prophet

- `yearly_seasonality=True`
- `weekly_seasonality=True`
- `daily_seasonality=False`
- `seasonality_mode="multiplicative"` — cocok untuk data dengan fluktuasi proporsional
- `changepoint_prior_scale=0.05` — konservatif, hindari overfitting

## Evaluasi Model

Metrik yang digunakan:
- **MAE** (Mean Absolute Error) — utama, mudah diinterpretasi
- **RMSE** (Root Mean Squared Error) — sensitif terhadap outlier
- **MAPE** (Mean Absolute Percentage Error) — dalam persen, target < 20%

Test set: 30 hari terakhir dari data historis.

## Rekomendasi Stok

```
recommended_stock = total_predicted_sales × safety_factor
```

- Default `safety_factor = 1.2` (buffer 20%)
- User bisa adjust via slider di UI (range 1.0 — 2.0)

## Batasan MVP

- Maksimal 10 produk yang ditraining per sesi (untuk performa)
- Minimal 10 hari data untuk training model total
- Minimal 30 hari data untuk training model per produk
- Prediksi maksimal 365 hari ke depan

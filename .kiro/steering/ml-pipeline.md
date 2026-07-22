# Dataset & ML Pipeline

## Dataset yang Digunakan

**Sumber:** [Anonymous Transactional Dataset](https://data.mendeley.com/datasets/kcgf45y24m/2) — Mendeley Data

Dataset transaksi nyata dari **UMKM Kopi & F&B Indonesia** dengan multi-outlet (Jan–Sep 2025).

File yang tersedia:
- `Transactions.csv` — 53.820 transaksi
- `Products.csv` — 96 produk kopi, minuman, dan makanan

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

- `yearly_seasonality` — **adaptif**: aktif hanya jika rentang data >= ~1.5 tahun (540 hari).
  Dataset UMKM saat ini < 1 tahun (272 hari), jadi otomatis `False` agar Prophet tidak
  memaksakan estimasi pola tahunan dari data yang tidak cukup (menghindari overfitting + warning).
- `weekly_seasonality=True`
- `daily_seasonality=False`
- `seasonality_mode="additive"` — lebih stabil untuk produk bervolume rendah yang banyak
  hari bernilai 0. (Multiplicative kurang stabil pada data jarang/sparse.)
- `changepoint_prior_scale=0.05` — konservatif, hindari overfitting

## Evaluasi Model

Metrik yang digunakan:
- **MAE** (Mean Absolute Error) — metrik **utama**, mudah diinterpretasi (meleset berapa unit).
- **RMSE** (Root Mean Squared Error) — sensitif terhadap outlier.
- **MAPE** (Mean Absolute Percentage Error) — target < 20%, **hanya andal untuk model total toko**.

> ⚠️ Catatan penting: untuk produk bervolume rendah (1–4 unit/hari), MAPE secara matematis
> selalu tinggi (meleset 2 unit dari basis 2 unit = 100%) walau MAE-nya kecil. Jadi MAPE
> menyesatkan di level per-produk. **Gunakan MAE untuk per-produk, dan MAPE untuk model total.**

Test set: 30 hari terakhir dari data historis. Hasil evaluasi disimpan ke `models/metrics.json`.

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

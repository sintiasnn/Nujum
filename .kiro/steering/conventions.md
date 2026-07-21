# Konvensi Kode

## Bahasa & Gaya

- **Bahasa komentar & docstring:** Bahasa Indonesia (karena target audiens adalah developer lokal dan juri IDCamp)
- **Bahasa variabel/fungsi:** Bahasa Inggris (standar Python)
- **Bahasa UI (Streamlit):** Bahasa Indonesia penuh — semua label, pesan error, dan tooltip harus dalam BI
- **Bahasa API response:** Bahasa Indonesia untuk field `message` dan `detail`, key tetap Inggris

## Python Style

- Ikuti PEP 8
- Gunakan type hints pada semua fungsi publik
- Docstring format: satu baris deskripsi + Args + Returns jika perlu
- Maksimal 120 karakter per baris

## Contoh Docstring

```python
def forecast(product_name: str = "all", periods: int = 30) -> pd.DataFrame:
    """
    Prediksi penjualan ke depan.

    Args:
        product_name: nama produk atau 'all' untuk semua produk
        periods: jumlah hari ke depan yang ingin diprediksi

    Returns:
        DataFrame dengan kolom: ds, yhat, yhat_lower, yhat_upper
    """
```

## Error Handling

- FastAPI: selalu raise `HTTPException` dengan `detail` berbahasa Indonesia
- Streamlit: tampilkan `st.error()` dengan pesan yang actionable, bukan stack trace mentah
- Semua fungsi di `src/` boleh raise exception — handling dilakukan di layer API dan UI

## Struktur Fungsi

- `preprocess.py` — fungsi murni, tidak ada side effect selain `save_processed()`
- `train.py` — boleh menyimpan model ke disk via `save_model()`
- `predict.py` — hanya membaca model, tidak memodifikasi apapun
- `api/main.py` — orchestrate pemanggilan src/, jangan taruh logika ML di sini
- `app/streamlit_app.py` — orchestrate UI, panggil src/ langsung (tanpa lewat API untuk MVP)

## Dependency Management

- Gunakan versi **exact** (pinned) di `requirements.txt`
- Jangan tambah dependency baru tanpa alasan yang jelas
- Preferensi library yang sudah ada di requirements daripada tambah baru

## Hal yang Harus Dihindari

- Jangan hardcode path absolut — selalu gunakan `Path(__file__).parent`
- Jangan commit file `.env`, data mentah, atau model `.pkl`
- Jangan taruh logika bisnis di layer UI (streamlit_app.py)
- Jangan gunakan `print()` di production — tapi untuk MVP ini masih OK di `src/`

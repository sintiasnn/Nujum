# Panduan Deployment

## Opsi Deploy (Gratis)

### 1. Streamlit Community Cloud ⭐ (Rekomendasi untuk submission)
- URL publik otomatis, gratis selamanya
- Langsung dari GitHub repo
- Steps:
  1. Push ke GitHub (pastikan `models/*.pkl` di-commit atau di-generate saat startup)
  2. Buka [share.streamlit.io](https://share.streamlit.io)
  3. Connect repo → pilih `app/streamlit_app.py` sebagai entry point
  4. Tambahkan `GEMINI_API_KEY` di **Secrets** (Settings → Secrets)

### 2. Hugging Face Spaces
- Mendukung Streamlit natively
- Steps:
  1. Buat Space baru di [huggingface.co/spaces](https://huggingface.co/spaces)
  2. Pilih SDK: Streamlit
  3. Upload file atau connect GitHub repo
  4. Tambahkan `GEMINI_API_KEY` di Space Secrets

### 3. Railway / Render (Untuk FastAPI backend)
- Cocok jika mau deploy backend terpisah
- Hubungkan `API_BASE_URL` di `.env` Streamlit ke URL backend

## Persiapan Sebelum Deploy

### 1. Pastikan model sudah ada di repo
Karena `models/*.pkl` di-gitignore, ada dua opsi:
- **Opsi A:** Commit model secara eksplisit: `git add -f models/prophet_all.pkl`
- **Opsi B:** Tambahkan script `startup.py` yang auto-training dari sample data saat pertama kali dijalankan

### 2. File yang wajib ada di repo
```
✅ requirements.txt
✅ app/streamlit_app.py
✅ src/preprocess.py
✅ src/train.py
✅ src/predict.py
✅ models/prophet_all.pkl  (atau auto-generated)
✅ data/raw/sample_data.csv (data contoh untuk demo)
```

### 3. Environment Variables
```
GEMINI_API_KEY=...   # wajib untuk fitur insight AI
API_BASE_URL=...     # opsional, hanya jika deploy FastAPI terpisah
```

## Checklist Submission IDCamp

- [ ] Aplikasi bisa diakses publik via URL
- [ ] Tidak memerlukan login untuk mencoba fitur utama
- [ ] Ada sample data / data contoh untuk demo
- [ ] Gemini API key dikonfigurasi di Secrets (bukan di-commit)
- [ ] README.md lengkap dengan cara penggunaan
- [ ] Project brief sudah diisi dan link-nya dibagikan ke "Anyone"

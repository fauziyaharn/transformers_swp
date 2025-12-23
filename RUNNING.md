# Menjalankan Backend (transformers_swp/app.py)

Panduan singkat untuk menjalankan backend Flask lokal dan menguji endpoint `/api/process`.

Persyaratan:
- Python (direkomendasikan 3.10+)
- Dependensi: lihat `transformers_swp/requirements.txt`

Langkah cepat (PowerShell):

```powershell

# Install dependensi
python -m pip install -r transformers_swp/requirements.txt

# (opsional) install flask-cors jika belum terpasang
python -m pip install flask-cors

# Jalankan server (di terminal baru):
cd D:\Nurul\Prokon\transformers_swp
python app.py

# Cek endpoint (PowerShell):
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/process -Method Post -ContentType 'application/json' -Body '{"query":"cari catering di bandung budget 20 juta"}'
```

Catatan:
- Aplikasi mencoba terhubung ke MySQL; jika tidak tersedia, otomatis menggunakan fallback CSV.
- Frontend (Vite) harus diarahkan ke `VITE_API_URL` yang sesuai, mis. `http://localhost:5000` saat testing lokal.
 - Jika ingin mengaktifkan seq2seq (Hugging Face) untuk reply generator, set environment variable `ENABLE_SEQ2SEQ=1` sebelum menjalankan `app.py`. Perlu menginstal paket `transformers` dan `sentencepiece` (sudah ditambahkan ke `requirements.txt`).

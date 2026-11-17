# Model Intent Chatbot Himar Coffee

Klasifikasi intent percakapan berbahasa Indonesia (santai/slang) untuk chatbot **Himar Coffee**.
Pendekatan klasik NLP: **TF-IDF (unigram + bigram) → Logistic Regression**, dengan
**Calibrated LinearSVC** sebagai pembanding. Bukan LLM, tidak butuh GPU.

- Dataset: `intent_dataset.csv` — 372 kalimat, 11 kelas, seimbang (rasio 1.30)
- Split stratified: 297 latih / 75 uji
- Model terbaik: `LogReg (Tuned)`, `C=10`, `ngram_range=(1, 2)`
- Hasil: CV macro-F1 **0.8879**, akurasi uji **0.84**, macro-F1 **0.8427**
- Confidence threshold **0.50** → di bawahnya diarahkan ke `fallback`

Kelas intent: `sapaan`, `pamitan`, `keluhan`, `tanya_menu`, `tanya_menu_favorit`,
`tanya_harga`, `tanya_jam_buka`, `tanya_lokasi`, `tanya_event`, `tanya_pemesanan`, `fallback`.

---

## 1. Kebutuhan perangkat

Sudah dicek di mesin ini:

| Komponen | Spesifikasi | Kebutuhan model | Status |
|---|---|---|---|
| CPU | Intel i5-13420H (8C/12T, s/d 4.6 GHz) | CPU 2 core apa saja | ✅ jauh lebih dari cukup |
| RAM | 15 GiB (≈8 GiB tersedia) | < 500 MB saat training | ✅ |
| Disk | 39 GB bebas di `/` | ±500 MB (venv + dependency) | ✅ |
| GPU | RTX 3050 4 GB (opsional) | **tidak dipakai** | ✅ tidak relevan |
| OS | Arch Linux, kernel 7.2, x86_64 | Linux/macOS/Windows | ✅ |
| Python | 3.14.10 + `uv` | 3.10+ | ✅ |

Waktu training penuh (CV + GridSearch + 4 model + evaluasi): **± 15 detik**.
Inference: **< 10 ms** per kalimat. Kesimpulan: **device sangat layak**, tidak ada
kebutuhan khusus. GPU tidak akan terpakai dan memang tidak diperlukan.

## 2. Struktur folder

```
model-himar/
├── intent_dataset.csv        # data latih (372 kalimat, 11 intent)
├── preprocessing.py          # clean_text(): lowercase, hapus tanda baca, normalisasi slang
├── train_intent.ipynb        # notebook eksperimen + visualisasi (sudah diperbaiki, bisa jalan lokal)
├── train.py                  # training end-to-end via CLI → simpan model
├── predict.py               # inference via CLI
├── app.py                    # REST API lokal (FastAPI)
├── requirements.txt          # dependency
├── models/                   # hasil training (dibuat otomatis)
│   ├── intent_logreg.pkl
│   └── metadata.json         # metrik, params, versi library
└── .venv/                    # virtual environment lokal
```

## 3. Setup

Paling cepat dengan [`uv`](https://docs.astral.sh/uv/) (sudah terpasang):

```zsh
cd ~/Learning/asah-linux/model-himar
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Alternatif tanpa `uv`:

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Cara menjalankan

**a. Latih model** (menghasilkan `models/intent_logreg.pkl` + `metadata.json`):

```zsh
python train.py
python train.py --cv 10        # ubah fold cross-validation
python train.py --no-save      # hanya evaluasi
```

**b. Prediksi via CLI**:

```zsh
python predict.py "brp harga kopi disini?"
python predict.py --top-k 3 "halo min, mau pesan tiket event"
python predict.py --json "apa itu machine learning"
echo "jam buka kapan?" | python predict.py
```

Contoh output:

```
Teks       : brp harga kopi disini?
Bersih     : berapa harga kopi disini
Intent     : tanya_harga  (confidence 0.90)
Kandidat   : tanya_harga=0.90, fallback=0.03, keluhan=0.01
```

**c. REST API lokal**:

```zsh
uvicorn app:app --reload --port 8000
# dokumentasi interaktif: http://127.0.0.1:8000/docs
```

```zsh
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "brp harga kopi disini?", "top_k": 3}'
```

Response: `intent`, `raw_intent`, `confidence`, `used_fallback`, `clean_text`, dan `top` (kandidat).

**d. Notebook** (grafik distribusi, confusion matrix, error analysis):

```zsh
jupyter lab train_intent.ipynb
```

Jalankan **Run All** dari folder ini. Sudah diverifikasi berjalan penuh tanpa error
di Python 3.14 + scikit-learn 1.9.1, hasilnya identik dengan versi Colab.

**e. Tes cepat bahwa semuanya sehat**:

```zsh
python preprocessing.py
python train.py --no-save
python predict.py "cara pesen tiket event gimana"
```

## 5. Hasil evaluasi (data uji, 75 kalimat)

| Model | Accuracy | Macro F1 |
|---|---|---|
| LogReg (Default) | 0.8133 | 0.8143 |
| LinearSVC (Default) | 0.8267 | 0.8270 |
| **LogReg (Tuned)** | **0.8400** | **0.8427** |
| LinearSVC (Tuned) | 0.8267 | 0.8270 |

Rata-rata confidence: jawaban benar **0.70**, jawaban salah **0.41** — karena itu
ambang **0.50** memisahkan cukup baik untuk diarahkan ke *fallback* (RAG/jawaban umum).

## 6. Keterbatasan

- Kalimat **campuran** bisa salah: `"halo min, kopi signature brp harganya?"` → `sapaan` (0.59),
  bukan `tanya_harga`. Model single-label, klausa pertama cenderung dominan.
- `"Apa ibu kota negara Jepang?"` sempat diprediksi `tanya_menu_favorit` (0.80) — di luar
  domain, tetapi confidence tinggi. Untuk produksi, tambahkan guard/daftar topik out-of-domain.
- Dataset kecil (372) dan sintetis; tambah data nyata sebelum dipakai serius.
- Perbaikan lanjutan: multi-label, indeks TF-IDF yang dibatasi ke kosakata domain,
  atau fine-tune model IndoBERT jika akurasi 0.84 tidak cukup.

## 7. Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError: preprocessing` | Jupyter dijalankan dari folder lain | Jalankan dari folder project; notebook sudah menyisipkan `sys.path` |
| `Model tidak ditemukan` | Belum training | `python train.py` |
| `No module named 'google.colab'` | Notebook versi lama | Sudah dihapus pada versi ini |
| `503 model_missing` di API | Path model salah | Set `INTENT_MODEL_PATH=/path/model.pkl` |
| Warning `palette` dari seaborn | API seaborn baru | Tidak berpengaruh pada hasil |

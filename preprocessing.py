"""
Modul: preprocessing.py
Fungsi: Membersihkan teks input dan menormalisasi kata-kata singkatan/slang
        agar seragam saat pelatihan model maupun saat inference di API.
"""

import re
import string

# Kamus normalisasi singkatan & slang chat sehari-hari
SLANG_DICTIONARY = {
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "bgmn": "bagaimana",
    "dmn": "dimana",
    "dmana": "dimana",
    "dimanakah": "dimana",
    "brp": "berapa",
    "brapa": "berapa",
    "psn": "pesan",
    "pesen": "pesan",
    "mesen": "pesan",
    "order": "pesan",
    "boking": "booking",
    "reserv": "reservasi",
    "bgt": "banget",
    "bgtu": "begitu",
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "gk": "tidak",
    "bisa": "bisa",
    "bs": "bisa",
    "dpt": "dapat",
    "km": "kamu",
    "sy": "saya",
    "gw": "saya",
    "gue": "saya",
    "aku": "saya",
    "ka": "kak",
    "kakak": "kak",
    "min": "admin",
    "mimin": "admin",
    "dftr": "daftar",
    "hrg": "harga",
    "klo": "kalau",
    "kl": "kalau",
    "kalo": "kalau",
    "makasih": "terima kasih",
    "thanks": "terima kasih",
    "thx": "terima kasih",
    "tengkyu": "terima kasih",
    "nuhun": "terima kasih",
    "hatur nuhun": "terima kasih",
    "rekomended": "rekomendasi",
    "fav": "favorit",
    "bestseller": "best seller"
}

def clean_text(text: str) -> str:
    """
    Membersihkan dan menormalisasi teks:
    1. Mengubah ke huruf kecil (lowercase).
    2. Menghapus tanda baca dan karakter khusus.
    3. Menormalisasi kata singkatan/slang berdasarkan kamus.
    4. Menghapus spasi berlebih.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower().strip()

    # 2. Hapus tanda baca (ganti dengan spasi agar kata tidak menempel)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 3. Hapus angka berlebih atau karakter aneh jika ada
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Normalisasi kata menggunakan kamus slang
    words = text.split()
    normalized_words = [SLANG_DICTIONARY.get(w, w) for w in words]
    clean_result = " ".join(normalized_words)

    return clean_result

if __name__ == "__main__":
    # Pengujian mandiri (Unit Test Sederhana)
    sample_text = "Halo min, gmn cara psn kopi? brp hrg signaturenya bgt?"
    print("Teks Asli   :", sample_text)
    print("Hasil Bersih:", clean_text(sample_text))
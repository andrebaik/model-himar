"""
Modul: predict.py
Fungsi: Uji model intent yang sudah dilatih, dari command line.

Contoh:
    python predict.py "brp harga kopi disini?"
    python predict.py --top-k 3 "halo min, mau pesan tiket event"
    echo "jam buka kapan?" | python predict.py
    python predict.py --json "apa itu machine learning"

Aturan bisnis: jika confidence < threshold (default 0.50), intent diubah
menjadi "fallback" supaya chatbot mengarahkan ke jawaban umum/RAG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "intent_logreg.pkl"


def load_model(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Model tidak ditemukan: {path}\nJalankan dulu: python train.py"
        )
    return joblib.load(path)


def predict(model, text: str, threshold: float, top_k: int = 3) -> dict:
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)

    order = probs.argsort()[::-1]
    ranking = [
        {"intent": classes[i], "confidence": round(float(probs[i]), 4)}
        for i in order[: max(1, top_k)]
    ]
    raw_intent = classes[int(order[0])]
    confidence = float(probs[order[0]])
    used_fallback = confidence < threshold

    return {
        "text": text,
        "clean_text": cleaned,
        "intent": "fallback" if used_fallback else raw_intent,
        "raw_intent": raw_intent,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "used_fallback": used_fallback,
        "top": ranking,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi intent Himar Coffee.")
    parser.add_argument("text", nargs="*", help="kalimat yang ingin diprediksi (kosong = baca stdin)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path model .pkl")
    parser.add_argument("--threshold", type=float, default=0.50, help="batas confidence sebelum fallback")
    parser.add_argument("--top-k", type=int, default=3, help="jumlah kandidat intent yang ditampilkan")
    parser.add_argument("--json", action="store_true", help="cetak hasil sebagai JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = " ".join(args.text).strip()
    if not text:
        if sys.stdin.isatty():
            print("Masukkan kalimat (Ctrl+D untuk selesai), contoh: brp harga kopi?")
        text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("Tidak ada teks untuk diprediksi.")

    model = load_model(args.model)
    result = predict(model, text, threshold=args.threshold, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Teks       : {result['text']}")
    print(f"Bersih     : {result['clean_text']}")
    print(f"Intent     : {result['intent']}  (confidence {result['confidence']:.2f})")
    if result["used_fallback"]:
        needle = result["raw_intent"]
        print(f"             ↳ di bawah threshold {result['threshold']:.2f}; kandidat asli: {needle}")
    print("Kandidat   : " + ", ".join(f"{item['intent']}={item['confidence']:.2f}" for item in result["top"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

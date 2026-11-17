"""
Modul: app.py
Fungsi: Menyajikan model intent Himar Coffee sebagai REST API lokal (FastAPI).

Jalankan:
    uvicorn app:app --reload --port 8000

Contoh:
    curl -s http://127.0.0.1:8000/health
    curl -s -X POST http://127.0.0.1:8000/predict \
         -H 'Content-Type: application/json' \
         -d '{"text": "brp harga kopi disini?"}'

Dokumentasi interaktif: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict
from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("INTENT_MODEL_PATH", BASE_DIR / "models" / "intent_logreg.pkl"))
DEFAULT_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.50"))

app = FastAPI(
    title="Himar Coffee Intent API",
    description="Klasifikasi intent percakapan chatbot Himar Coffee (TF-IDF + Logistic Regression).",
    version="1.0.0",
)

_model = None


def get_model():
    """Muat model sekali saja (lazy load) agar startup tetap cepat."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model belum tersedia di {MODEL_PATH}. Jalankan: python train.py",
            )
        import joblib

        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["brp harga kopi disini?"])
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Override batas confidence")
    top_k: int = Field(3, ge=1, le=11, description="Jumlah kandidat intent")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_missing",
        "model_path": str(MODEL_PATH),
        "threshold": DEFAULT_THRESHOLD,
    }


@app.post("/predict")
def predict_endpoint(payload: PredictRequest) -> dict:
    model = get_model()
    threshold = DEFAULT_THRESHOLD if payload.threshold is None else payload.threshold
    result = predict(model, payload.text, threshold=threshold, top_k=payload.top_k)
    result["clean_text"] = clean_text(payload.text)
    return result

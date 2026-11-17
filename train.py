"""
Modul: train.py
Fungsi: Melatih ulang model klasifikasi intent Himar Coffee dari dataset CSV,
        mengevaluasi performa, lalu menyimpan model siap-pakai ke folder models/.

Jalankan:
    python train.py                 # latih + evaluasi + simpan model
    python train.py --no-save       # hanya latih + evaluasi
    python train.py --cv 10         # ganti jumlah fold cross-validation

Perbedaan dengan train_intent.ipynb:
- Tidak butuh Google Colab / matplotlib (output berupa teks).
- Model & metadata hasil training disimpan otomatis, jadi tidak perlu cell manual.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "intent_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_logreg.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CONFIDENCE_THRESHOLD = 0.50


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Baca dataset dan terapkan preprocessing yang sama dengan saat inference."""
    df = pd.read_csv(path)
    missing = {"text", "intent"} - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di {path.name}: {sorted(missing)}")

    df = df[["text", "intent"]].dropna()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def describe_dataset(df: pd.DataFrame) -> float:
    """Cetak distribusi kelas dan kembalikan rasio kelas mayoritas/minoritas."""
    counts = df["intent"].value_counts().sort_values(ascending=False)
    ratio = counts.max() / counts.min()

    print("=== DISTRIBUSI KELAS INTENT ===")
    for intent, jumlah in counts.items():
        print(f"{jumlah:5d}  {intent}")
    print(f"\nTotal sampel : {len(df)}")
    print(f"Jumlah kelas : {counts.size}")
    print(f"Rasio kelas : {ratio:.2f} ", end="")
    print("(seimbang)" if ratio < 1.5 else "(tidak seimbang, pertimbangkan resampling)")
    return ratio


def build_models() -> dict[str, Pipeline]:
    """Model pembanding dengan konfigurasi default (sama seperti notebook)."""
    return {
        "LogReg (Default)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "LinearSVC (Default)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("clf", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, dual=False), cv=3)),
            ]
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    """Hitung metrik utama (macro average) untuk satu set prediksi."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def train(cv_folds: int = 5, save: bool = True) -> dict:
    df = load_dataset()
    ratio = describe_dataset(df)

    X, y = df["clean_text"], df["intent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nData latih : {len(X_train)} kalimat")
    print(f"Data uji   : {len(X_test)} kalimat")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()

    print(f"\n=== {cv_folds}-FOLD CROSS-VALIDATION (macro F1, data latih) ===")
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_scores[name] = scores
        print(f"{name:22}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        model.fit(X_train, y_train)

    print("\n=== HYPERPARAMETER TUNING (LogReg) ===")
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 5.0, 10.0],
    }
    grid = GridSearchCV(
        build_models()["LogReg (Default)"], param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Parameter terbaik : {grid.best_params_}")
    print(f"Skor CV terbaik   : {grid.best_score_:.4f}")

    best_logreg = grid.best_estimator_
    svm_tuned = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_STATE, dual=False), cv=3)),
        ]
    ).fit(X_train, y_train)

    models.update({"LogReg (Tuned)": best_logreg, "LinearSVC (Tuned)": svm_tuned})

    print("\n=== PERBANDINGAN PERFORMA DI DATA UJI ===")
    rows = []
    for name, model in models.items():
        metrics = evaluate(y_test, model.predict(X_test))
        rows.append({"model": name, **metrics})
    comparison = pd.DataFrame(rows)
    print(comparison.round(4).to_string(index=False))

    best_name = comparison.loc[comparison["f1_macro"].idxmax(), "model"]
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print(f"\n=== CLASSIFICATION REPORT MODEL TERBAIK ({best_name}) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== CONFUSION MATRIX (baris = aktual, kolom = prediksi) ===")
    labels = list(best_model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = " ".join(f"{label[:6]:>6}" for label in labels)
    print(f"{'':22}{header}")
    for label, row in zip(labels, cm):
        print(f"{label:22}" + " ".join(f"{value:>6d}" for value in row))

    # --- Analisis confidence: dasar pemilihan threshold 0.50 ---
    probs = best_model.predict_proba(X_test)
    confidence = np.max(probs, axis=1)
    correct = y_test.to_numpy() == y_pred
    print("\n=== ANALISIS CONFIDENCE ===")
    print(f"Rata-rata confidence benar : {confidence[correct].mean():.4f}")
    if (~correct).any():
        print(f"Rata-rata confidence salah : {confidence[~correct].mean():.4f}")
        errors = pd.DataFrame(
            {
                "kalimat": df.loc[X_test.index, "text"].to_numpy(),
                "aktual": y_test.to_numpy(),
                "prediksi": y_pred,
                "confidence": confidence,
            }
        )
        print("\nContoh salah prediksi:")
        print(errors[~correct].sort_values("confidence", ascending=False).head(10).to_string(index=False))
    else:
        print("Rata-rata confidence salah : - (tidak ada kesalahan prediksi)")

    result = {
        "best_model": best_name,
        "best_params": grid.best_params_,
        "cv_f1_macro": float(grid.best_score_),
        "test_metrics": metrics,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_ratio": round(float(ratio), 4),
        "classes": labels,
        "cv_scores": {name: float(scores.mean()) for name, scores in cv_scores.items()},
        "comparison": comparison.round(6).to_dict(orient="records"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH, compress=3)
        METADATA_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"\n✅ Model disimpan : {MODEL_PATH.relative_to(BASE_DIR)}")
        print(f"✅ Metadata       : {METADATA_PATH.relative_to(BASE_DIR)}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latih model intent Himar Coffee.")
    parser.add_argument("--cv", type=int, default=5, help="jumlah fold cross-validation (default: 5)")
    parser.add_argument("--no-save", action="store_true", help="jangan simpan model ke folder models/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(cv_folds=args.cv, save=not args.no_save)

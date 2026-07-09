"""
run_full_submission_v2.py
=========================
G.G.A Takımı — v2 Modeli ile Kaggle Submission Üretimi (9 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script, run_train_full_v2.py ile kaydedilen 5 fold modelini yükleyerek
submission_pairs.csv (3.36M satır) için Kaggle formatında tahmin üretir.

Çalıştırma sırası:
  1. python run_train_full_v2.py       → Model eğit ve kaydet
  2. python run_full_submission_v2.py  → Submission CSV üret
  3. python run_submission_qa.py outputs/submission_v2.csv  → QA kontrol
  4. Kaggle'a yükle!

Çıktı:
  outputs/submission_v2.csv  → Kaggle formatında tahmin dosyası
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import lightgbm as lgb

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.tfidf_features    import add_tfidf_features, load_vectorizer
from src.validate_submission import validate_submission

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# v2 modeli için dosya yolları
MODEL_PATHS = [os.path.join(OUTPUT_DIR, f"lgbm_v2_fold_{i}.txt") for i in range(1, 6)]
VEC_PATH    = os.path.join(OUTPUT_DIR, "tfidf_vectorizer_v2.pkl")
THRESH_PATH = os.path.join(OUTPUT_DIR, "best_threshold_v2.txt")
SUB_OUTPUT  = os.path.join(OUTPUT_DIR, "submission_v2.csv")

# Bellek tasarrufu için submission'ı batch'ler halinde işle
BATCH_SIZE  = 100_000


def main():
    print("=" * 65)
    print("  G.G.A v2 — Kaggle Submission Uretimi")
    print("  9 Temmuz 2026 — Omer Faruk Kara")
    print("=" * 65)

    # ─── 0. Dosya Kontrolü ────────────────────────────────────────────────────
    print("\n[0/5] Gerekli dosyalar kontrol ediliyor...")
    missing = []
    for p in MODEL_PATHS:
        if not os.path.exists(p):
            missing.append(p)
    if not os.path.exists(VEC_PATH):
        missing.append(VEC_PATH)

    if missing:
        print("  [HATA] Asagidaki dosyalar bulunamadi:")
        for m in missing:
            print(f"    - {m}")
        print("\n  Once 'python run_train_full_v2.py' calistirin!")
        sys.exit(1)

    # Threshold oku
    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH) as f:
            threshold = float(f.read().strip())
        print(f"  Threshold yuklendi: {threshold}  (outputs/best_threshold_v2.txt)")
    else:
        threshold = 0.45   # EXP-001'den bilinen iyi değer
        print(f"  [UYARI] Threshold dosyasi bulunamadi. Varsayilan kullaniliyor: {threshold}")

    # ─── 1. Veri + Model Yükle ────────────────────────────────────────────────
    print("\n[1/5] Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))

    print(f"  {len(MODEL_PATHS)} fold modeli yukleniyor...")
    models = [lgb.Booster(model_file=p) for p in MODEL_PATHS]

    print("  TF-IDF vectorizer yukleniyor...")
    vectorizer = load_vectorizer(VEC_PATH)

    feature_cols = FEATURE_COLS + ["tfidf_cosine"]
    print(f"  Feature sayisi: {len(feature_cols)}")

    # ─── 2. Submission Çiftlerini Yükle ───────────────────────────────────────
    sub_path = os.path.join(DATA_DIR, "submission_pairs.csv")
    print(f"\n[2/5] Submission ciftleri yukleniyor: {sub_path}")
    sub_df = pd.read_csv(
        sub_path,
        dtype={"id": "string", "term_id": "string", "item_id": "string"}
    )
    print(f"  Tahmin edilecek cift sayisi: {len(sub_df):,}")

    # ─── 3. Batch Halinde Feature + Tahmin Üret ───────────────────────────────
    print(f"\n[3/5] {len(sub_df):,} satir icin batch feature + tahmin uretiliyor")
    print(f"  Batch boyutu: {BATCH_SIZE:,} satir / batch")

    all_ids     = []
    all_probas  = []

    n_batches = (len(sub_df) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, start in enumerate(range(0, len(sub_df), BATCH_SIZE), start=1):
        end   = min(start + BATCH_SIZE, len(sub_df))
        batch = sub_df.iloc[start:end].copy()

        # Merge
        batch = batch.merge(terms_df, on="term_id", how="left")
        batch = batch.merge(items_df,  on="item_id",  how="left")

        # Temel feature'lar
        batch = build_features(batch)

        # TF-IDF feature
        batch = add_tfidf_features(batch, vectorizer, batch_size=BATCH_SIZE)

        # 5-fold ensemble tahmin
        X_batch = batch[feature_cols]
        proba_list = [m.predict(X_batch) for m in models]
        avg_proba  = np.mean(proba_list, axis=0)

        all_ids.extend(batch["id"].tolist())
        all_probas.extend(avg_proba.tolist())

        if batch_idx % 5 == 0 or batch_idx == n_batches:
            print(f"  Batch {batch_idx:>3}/{n_batches}  "
                  f"({end:,}/{len(sub_df):,} satir islendi)")

    # ─── 4. Binary Tahminler + Kaydet ─────────────────────────────────────────
    print(f"\n[4/5] Binary tahminler olusturuluyor (threshold={threshold})...")
    all_probas  = np.array(all_probas)
    predictions = (all_probas >= threshold).astype(int)

    pos_count = predictions.sum()
    pos_rate  = pos_count / len(predictions)
    print(f"  Pozitif (1): {pos_count:,}  ({pos_rate:.2%})")
    print(f"  Negatif (0): {len(predictions)-pos_count:,}  ({1-pos_rate:.2%})")

    submission = pd.DataFrame({"id": all_ids, "prediction": predictions})
    submission.to_csv(SUB_OUTPUT, index=False)
    print(f"\n  Submission kaydedildi: {SUB_OUTPUT}")
    print(f"  Dosya boyutu: {os.path.getsize(SUB_OUTPUT)/1024/1024:.1f} MB")

    # ─── 5. Format Doğrulaması ────────────────────────────────────────────────
    print("\n[5/5] Format dogrulamasi yapiliyor...")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    ok = validate_submission(SUB_OUTPUT, sample_submission_path=sample_sub_path, verbose=True)

    print("\n" + "=" * 65)
    if ok:
        print("  [HAZIR] Submission v2 Kaggle'a yuklenmeye hazir!")
        print(f"  Dosya: {SUB_OUTPUT}")
        print(f"\n  QA icin:")
        print(f"    python run_submission_qa.py {SUB_OUTPUT}")
    else:
        print("  [HATA] Format dogrulamasinda sorun var!")
        print("  Lutfen hatalari giderip tekrar calistirin.")
    print("=" * 65)


if __name__ == "__main__":
    main()

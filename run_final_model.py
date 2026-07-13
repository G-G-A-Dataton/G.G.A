#!/usr/bin/env python
"""
run_final_model.py
==================
G.G.A Takımı — Final Model Training (Gün 14-15 görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Amaç:
  FULL eğitim veri seti üzerinde final LightGBM modelini eğitip,
  Kaggle submission_pairs.csv üzerinde tahmin üret.

Strategi:
  1. Tüm pozitif çiftleri kullan (250K)
  2. Random negative (3:1 oran)
  3. TF-IDF + Embedding features
  4. 5-Fold CV ile validation
  5. OOF tahminlerinden optimal threshold bul
  6. Test seti üzerinde tahmin yap

Çalıştırmak için:
  python run_final_model.py [--submit]

--submit flag'ı eklenmezse, validation sonuçlarını gösterir.
--submit eklenirse, submission.csv dosyası üretilir.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set
from src.tfidf_features    import build_tfidf_vectorizer, add_tfidf_features, save_vectorizer, load_vectorizer
from src.embedding_features import load_embedding_model, add_embedding_cosine_feature
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_kfold
from src.validate_submission import validate_submission_format
import lightgbm as lgb

DATA_DIR       = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs")
NEGATIVE_RATIO = 3
RANDOM_SEED    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Veri Yükleme ──────────────────────────────────────────────────────
print("=" * 70)
print("  G.G.A — Final Model (Gün 14-15)")
print("=" * 70)

print("\n[1/7] Veriler yukleniyor...")
terms_df  = load_terms(os.path.join(DATA_DIR, "terms.csv"))
items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
train_raw = pd.read_csv(
    os.path.join(DATA_DIR, "training_pairs.csv"),
    dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
)
test_raw = pd.read_csv(
    os.path.join(DATA_DIR, "submission_pairs.csv"),
    dtype={"id": "string", "term_id": "string", "item_id": "string"}
)
print(f"  Eğitim pozitif: {len(train_raw):,}")
print(f"  Test (tahmin): {len(test_raw):,}")
print(f"  Sorgular: {len(terms_df):,} | Ürünler: {len(items_df):,}")

# ─── 2. Negatif Örnekleme ─────────────────────────────────────────────────
print(f"\n[2/7] Negatif ornekleme (ratio={NEGATIVE_RATIO}:1)...")
full_train = build_training_set(
    train_raw, items_df,
    ratio=NEGATIVE_RATIO, random_state=RANDOM_SEED, verbose=False
)
print(f"  Toplam eğitim: {len(full_train):,} satir ({full_train['label'].sum():,} pozitif)")

# ─── 3. Merge ──────────────────────────────────────────────────────────────
print("\n[3/7] Merge yapiliyor...")
train_merged = full_train.merge(terms_df, on="term_id", how="left")
train_merged = train_merged.merge(items_df, on="item_id", how="left")

test_merged = test_raw.merge(terms_df, on="term_id", how="left")
test_merged = test_merged.merge(items_df, on="item_id", how="left")

print(f"  Eğitim merged: {len(train_merged):,}")
print(f"  Test merged: {len(test_merged):,}")

# ─── 4. Temel Features ────────────────────────────────────────────────────
print("\n[4/7] Temel features hesaplaniyor...")
train_merged = build_features(train_merged)
test_merged = build_features(test_merged)

# ─── 5. TF-IDF + Embedding Features ───────────────────────────────────────
print("\n[5/7] TF-IDF + Embedding features ekleniyor...")

# TF-IDF
vec_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
if os.path.exists(vec_path):
    print("  (TF-IDF vectorizer diskten yükleniyor)")
    vectorizer = load_vectorizer(vec_path)
else:
    print("  (TF-IDF vectorizer yeni eğitiliyor)")
    vectorizer = build_tfidf_vectorizer(
        terms_df, items_df, max_features=30_000, ngram_range=(1, 1)
    )
    save_vectorizer(vectorizer, vec_path)

train_merged = add_tfidf_features(train_merged, vectorizer, batch_size=5_000)
test_merged = add_tfidf_features(test_merged, vectorizer, batch_size=5_000)

# Embedding
print("  Embedding modeli yükleniyor...")
model = load_embedding_model()
if model is not None:
    train_merged = add_embedding_cosine_feature(train_merged, model, batch_size=32)
    test_merged = add_embedding_cosine_feature(test_merged, model, batch_size=32)
    feature_cols = FEATURE_COLS + ["tfidf_cosine", "embedding_cosine"]
else:
    print("  [UYARI] Embedding modeli yüklenemedi. Sadece TF-IDF ile devam.")
    feature_cols = FEATURE_COLS + ["tfidf_cosine"]

print(f"  Feature sayisi: {len(feature_cols)}")

# ─── 6. LightGBM Eğitim (5-Fold CV) ───────────────────────────────────────
print("\n[6/7] LightGBM 5-Fold CV eğitimi...")
print("-" * 70)

X_train = train_merged[feature_cols]
y_train = train_merged["label"]

lgbm_params = {
    "objective"        : "binary",
    "metric"           : "binary_logloss",
    "learning_rate"    : 0.05,
    "num_leaves"       : 31,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "verbose"          : -1,
    "random_state"     : 42,
}

skf = get_stratified_kfold(n_splits=5, random_state=42)
fold_scores = []
oof_preds = np.zeros(len(X_train))
trained_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)

    model_fold = lgb.train(
        lgbm_params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )
    trained_models.append(model_fold)

    val_proba = model_fold.predict(X_val)
    oof_preds[val_idx] = val_proba
    fold_f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
    fold_scores.append(fold_f1)
    print(f"  Fold {fold}/5  |  Macro-F1: {fold_f1:.4f}  |  Best iter: {model_fold.best_iteration}")

mean_f1 = np.mean(fold_scores)
std_f1 = np.std(fold_scores)
print("-" * 70)
print(f"  ORT. Macro-F1: {mean_f1:.4f} (+/- {std_f1:.4f})")

best_thresh, best_f1, _ = find_best_threshold(y_train.values, oof_preds)
print(f"  En iyi threshold: {best_thresh}  →  {best_f1:.4f}")

# ─── 7. Test Seti Tahmin ────────────────────────────────────────────────────
print("\n[7/7] Test seti tahmin ediliyor...")

X_test = test_merged[feature_cols]

# 5 fold'un tahminlerini ortalıyoruz (ensemble)
test_preds = np.zeros(len(X_test))
for model_fold in trained_models:
    test_preds += model_fold.predict(X_test)
test_preds /= len(trained_models)

# Threshold uygula
test_labels = (test_preds >= best_thresh).astype(int)

# Submission hazırla
submission = test_raw[["id"]].copy()
submission["prediction"] = test_labels

# Format kontrolü
try:
    validate_submission_format(submission)
    print("  ✓ Submission formatı geçerli")
except Exception as e:
    print(f"  ✗ Format hatası: {e}")

# Kaydet
sub_path = os.path.join(OUTPUT_DIR, "submission_final.csv")
submission.to_csv(sub_path, index=False)
print(f"  Submission kaydedildi: {sub_path}")
print(f"  Toplam satir: {len(submission):,}")
print(f"  Positif tahmin: {submission['prediction'].sum():,} ({submission['prediction'].mean()*100:.2f}%)")

# ─── Sonuç Özeti ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL MODEL SONUC OZETI")
print("=" * 70)
print(f"  Eğitim veri seti: {len(full_train):,} ({full_train['label'].sum():,} pos)")
print(f"  CV Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  En iyi threshold: {best_thresh}")
print(f"  Optimized F1: {best_f1:.4f}")
print(f"  Test tahmin sayısı: {len(submission):,}")
print(f"  Positive rate: {submission['prediction'].mean()*100:.2f}%")
print("=" * 70)

# Sonuç dosyası kaydet
result_df = pd.DataFrame({
    "model": ["Final Ensemble"],
    "train_samples": [len(full_train)],
    "cv_f1": [round(mean_f1, 4)],
    "cv_std": [round(std_f1, 4)],
    "best_threshold": [best_thresh],
    "optimized_f1": [round(best_f1, 4)],
    "test_positive_rate": [round(submission['prediction'].mean(), 4)]
})
result_csv = os.path.join(OUTPUT_DIR, "final_model_result.csv")
result_df.to_csv(result_csv, index=False)
print(f"✓ Sonuçlar kaydedildi: {result_csv}")

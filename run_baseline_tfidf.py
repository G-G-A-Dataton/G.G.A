"""
run_baseline_tfidf.py
=====================
G.G.A Takımı — TF-IDF Eklenmiş LightGBM Baseline (4 Temmuz Görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Bu script, src/tfidf_features.py modülündeki TF-IDF cosine similarity
feature'ını baseline pipeline'a bağlar ve modele ek olarak verir.

Çalıştırmak için:
  python run_baseline_tfidf.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set
from src.tfidf_features    import build_tfidf_vectorizer, add_tfidf_features, save_vectorizer
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_kfold

import lightgbm as lgb

DATA_DIR       = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs")
SAMPLE_SIZE    = 5_000
NEGATIVE_RATIO = 3
RANDOM_SEED    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Veri Yükleme ──────────────────────────────────────────────────────
print("=" * 60)
print("  G.G.A — LightGBM + TF-IDF Baseline (4 Temmuz)")
print("=" * 60)

print("\n[1/7] Veriler yukleniyor...")
terms_df  = load_terms(os.path.join(DATA_DIR, "terms.csv"))
items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
train_raw = pd.read_csv(
    os.path.join(DATA_DIR, "training_pairs.csv"),
    dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
)
print(f"  Sorgular: {len(terms_df):,} | Urunler: {len(items_df):,} | Pozitif: {len(train_raw):,}")

# ─── 2. Negatif Örnekleme ─────────────────────────────────────────────────
print(f"\n[2/7] Negatif ornekleme ({SAMPLE_SIZE:,} poz, ratio={NEGATIVE_RATIO}:1)...")
pos_sample = train_raw.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
full_train = build_training_set(
    pos_sample, items_df,
    ratio=NEGATIVE_RATIO, random_state=RANDOM_SEED, verbose=False
)
print(f"  Toplam: {len(full_train):,} satir")

# ─── 3. Merge ──────────────────────────────────────────────────────────────
print("\n[3/7] Merge yapiliyor...")
merged = full_train.merge(terms_df, on="term_id", how="left")
merged = merged.merge(items_df, on="item_id", how="left")

# ─── 4. Temel Feature'lar ─────────────────────────────────────────────────
print("\n[4/7] Temel feature'lar (overlap + demografik) hesaplaniyor...")
merged = build_features(merged)

# ─── 5. TF-IDF Cosine Feature Ekleme ─────────────────────────────────────
print("\n[5/7] TF-IDF Cosine feature ekleniyor...")
# Vectorizer eğit (tüm sorgu + ürün metinleri üzerinde)
vectorizer = build_tfidf_vectorizer(
    terms_df, items_df, max_features=30_000, ngram_range=(1, 2)
)
# Kaydedelim — bir daha eğitmek zorunda kalmayalım
vec_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
save_vectorizer(vectorizer, vec_path)

# Eğitim verisine TF-IDF cosine feature'ı ekle
merged = add_tfidf_features(merged, vectorizer)

# TF-IDF'i feature listesine ekle
feature_cols_tfidf = FEATURE_COLS + ["tfidf_cosine"]
print(f"  Toplam feature sayisi: {len(feature_cols_tfidf)} ({len(FEATURE_COLS)} temel + 1 TF-IDF)")

# ─── 6. LightGBM 5-Fold CV ────────────────────────────────────────────────
print("\n[6/7] LightGBM 5-Fold Stratified CV...")
print("-" * 50)

X = merged[feature_cols_tfidf]
y = merged["label"]

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

skf          = get_stratified_kfold(n_splits=5, random_state=42)
fold_scores  = []
oof_preds    = np.zeros(len(X))
trained_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        lgbm_params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
    )
    trained_models.append(model)

    val_proba = model.predict(X_val)
    oof_preds[val_idx] = val_proba
    fold_f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
    fold_scores.append(fold_f1)
    print(f"  Fold {fold}/5  |  Macro-F1: {fold_f1:.4f}  |  Best iter: {model.best_iteration}")

mean_f1 = np.mean(fold_scores)
std_f1  = np.std(fold_scores)
print("-" * 50)
print(f"  ORT. Macro-F1 (TF-IDF ile): {mean_f1:.4f} (+/- {std_f1:.4f})")

# Threshold opt
best_thresh, best_score, _ = find_best_threshold(y.values, oof_preds)
print(f"  En iyi threshold          : {best_thresh}  ->  {best_score:.4f}")

# ─── 7. Feature Importance ────────────────────────────────────────────────
print("\n[7/7] Feature importance:")
print("-" * 50)
importance_arr = np.zeros(len(feature_cols_tfidf))
for m in trained_models:
    importance_arr += m.feature_importance(importance_type="gain")
importance_arr /= len(trained_models)

feat_imp = pd.DataFrame({"feature": feature_cols_tfidf, "importance": importance_arr})
feat_imp = feat_imp.sort_values("importance", ascending=False)

max_imp = feat_imp["importance"].max()
for _, row in feat_imp.iterrows():
    bar = "#" * int(row["importance"] / max_imp * 30)
    print(f"  {row['feature']:<28} {bar} ({row['importance']:.1f})")

# ─── Sonuç Özeti ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TFIDF BASELINE SONUC OZETI (4 Temmuz)")
print("=" * 60)
print(f"  Feature sayisi    : {len(feature_cols_tfidf)} (onceki: {len(FEATURE_COLS)})")
print(f"  Ort. Macro-F1     : {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"  En iyi threshold  : {best_thresh}")
print(f"  Optimized F1      : {best_score:.4f}")
print(f"  Vectorizer kaydi  : {vec_path}")
print("=" * 60)

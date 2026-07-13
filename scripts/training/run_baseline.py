"""
run_baseline.py
===============
G.G.A Takımı — LightGBM Baseline v0 Çalıştırıcı

Bu script notebooks/04_baseline_lgbm_omerfaruk.ipynb ile aynı pipeline'ı
terminal üzerinden çalıştırmak için yazılmıştır.

Çalıştırmak için:
  python run_baseline.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Proje kök dizinini Python yoluna ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_kfold

import lightgbm as lgb

DATA_DIR       = os.path.join(PROJECT_ROOT, "datasets")
SAMPLE_SIZE    = 5_000   # Baseline için kaç pozitif kullanılacak
NEGATIVE_RATIO = 3       # Her pozitife kaç negatif
RANDOM_SEED    = 42

# ─── 1. Veri Yükleme ───────────────────────────────────────────────────────
print("=" * 55)
print("  G.G.A — LightGBM Baseline v0")
print("=" * 55)

print("\n[1/6] Veriler yukleniyor...")
terms_df  = load_terms(os.path.join(DATA_DIR, "terms.csv"))
items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
train_raw = pd.read_csv(
    os.path.join(DATA_DIR, "training_pairs.csv"),
    dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
)
print(f"  Sorgular : {len(terms_df):,}")
print(f"  Urunler  : {len(items_df):,}")
print(f"  Pozitif  : {len(train_raw):,}")

# ─── 2. Negatif Örnekleme ──────────────────────────────────────────────────
print(f"\n[2/6] Negatif ornekleme ({SAMPLE_SIZE:,} poz, ratio={NEGATIVE_RATIO}:1)...")
pos_sample = train_raw.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
full_train = build_training_set(
    pos_sample, items_df,
    ratio=NEGATIVE_RATIO,
    random_state=RANDOM_SEED,
    verbose=False
)
counts = full_train["label"].value_counts()
print(f"  Pozitif (1): {counts.get(1, 0):,}")
print(f"  Negatif (0): {counts.get(0, 0):,}")
print(f"  Toplam     : {len(full_train):,}")

# ─── 3. Merge ve Feature Üretimi ───────────────────────────────────────────
print("\n[3/6] Merge ve feature hesaplama...")
merged = full_train.merge(terms_df, on="term_id", how="left")
merged = merged.merge(items_df,  on="item_id",  how="left")
merged = build_features(merged)
print(f"  Feature sayisi: {len(FEATURE_COLS)}")
print(f"  Feature listesi: {FEATURE_COLS}")

# ─── 4. Model Eğitimi — 5-Fold CV ─────────────────────────────────────────
print("\n[4/6] LightGBM 5-Fold Stratified CV basliyor...")
print("-" * 45)

X = merged[FEATURE_COLS]
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
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )
    trained_models.append(model)

    val_proba = model.predict(X_val)
    oof_preds[val_idx] = val_proba

    fold_f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
    fold_scores.append(fold_f1)
    print(f"  Fold {fold}/5  |  Macro-F1: {fold_f1:.4f}  |  Best iter: {model.best_iteration}")

mean_f1 = np.mean(fold_scores)
std_f1  = np.std(fold_scores)
print("-" * 45)
print(f"  ORT. Macro-F1: {mean_f1:.4f} (+/- {std_f1:.4f})")

# ─── 5. Threshold Optimizasyonu ───────────────────────────────────────────
print("\n[5/6] Threshold optimizasyonu...")
default_f1 = macro_f1_from_proba(y, oof_preds, threshold=0.5)
best_thresh, best_score, _ = find_best_threshold(y.values, oof_preds)
print(f"  Varsayilan (t=0.50): {default_f1:.4f}")
print(f"  En iyi    (t={best_thresh:.2f}): {best_score:.4f}")
print(f"  Kazanim  : +{best_score - default_f1:.4f}")

# ─── 6. Feature Importance ────────────────────────────────────────────────
print("\n[6/6] Feature importance:")
print("-" * 45)
importance_arr = np.zeros(len(FEATURE_COLS))
for m in trained_models:
    importance_arr += m.feature_importance(importance_type="gain")
importance_arr /= len(trained_models)

feat_imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": importance_arr})
feat_imp = feat_imp.sort_values("importance", ascending=False)

max_imp = feat_imp["importance"].max()
for _, row in feat_imp.iterrows():
    bar = "#" * int(row["importance"] / max_imp * 30)
    print(f"  {row['feature']:<28} {bar} ({row['importance']:.1f})")

# ─── Sonuç Özeti ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  BASELINE v0 SONUC OZETI")
print("=" * 55)
print(f"  Egitim seti    : {SAMPLE_SIZE:,} poz + {SAMPLE_SIZE*NEGATIVE_RATIO:,} neg")
print(f"  Feature sayisi : {len(FEATURE_COLS)}")
print(f"  Ort. Macro-F1  : {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"  En iyi thresh  : {best_thresh}")
print(f"  Optimized F1   : {best_score:.4f}")
print("=" * 55)
print()
print("Sonraki adimlar:")
print("  [4 Temmuz] Tum 250K pozitif ile tam egitim")
print("  [4 Temmuz] TF-IDF cosine feature ekleme")
print("  [5 Temmuz] Ilk Kaggle submission")

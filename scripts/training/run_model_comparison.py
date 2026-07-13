"""
run_model_comparison.py
=======================
G.G.A Takımı — XGBoost vs LightGBM Model Kıyası (9 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script aynı feature seti ve aynı veri üzerinde XGBoost ile LightGBM'i
yan yana karşılaştırır. Amacımız en uygun gradient boosting çerçevesini
belirlemek.

XGBoost vs LightGBM:
  LightGBM : Leaf-wise ağaç büyümesi → daha hızlı, az bellek
  XGBoost  : Level-wise büyüme → daha konservatif, bazen daha stabil

Çalıştırmak için:
  python run_model_comparison.py

Not: xgboost kurulu değilse: pip install xgboost
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_group_kfold
import lightgbm as lgb

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_POS  = 3_000
NEG_RATIO   = 3
RANDOM_SEED = 42


def run_lightgbm(X, y, groups):
    """
    LightGBM'i 5-Fold CV ile eğitir ve sonuçları döndürür.

    Parametreler — EXP-005'teki en iyi değerler baz alındı:
      num_leaves=31, lr=0.05, min_child_samples=20
    """
    params = {
        "objective"        : "binary",
        "metric"           : "binary_logloss",
        "num_leaves"       : 31,
        "learning_rate"    : 0.05,
        "min_child_samples": 20,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "verbose"          : -1,
        "random_state"     : RANDOM_SEED,
    }

    skf       = get_stratified_group_kfold(n_splits=5, random_state=RANDOM_SEED)
    scores    = []
    oof_preds = np.zeros(len(X))
    t0        = time.time()

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X, y, groups=groups), start=1
    ):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )
        val_proba = model.predict(X_val)
        oof_preds[val_idx] = val_proba
        f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
        scores.append(f1)
        print(f"    Fold {fold}/5  F1={f1:.4f}  iter={model.best_iteration}")

    elapsed = time.time() - t0
    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    best_thresh, best_f1, _ = find_best_threshold(y.values, oof_preds)

    return {
        "model"          : "LightGBM",
        "mean_f1"        : round(mean_f1, 4),
        "std_f1"         : round(std_f1, 4),
        "best_threshold" : best_thresh,
        "best_f1"        : round(best_f1, 4),
        "train_sec"      : round(elapsed, 1),
    }


def run_xgboost(X, y, groups):
    """
    XGBoost'u 5-Fold CV ile eğitir ve sonuçları döndürür.

    LightGBM ile adil karşılaştırma için:
    - Aynı early stopping (30 tur)
    - Aynı 5-Fold CV şeması
    - Benzer hiperparametreler (learning_rate=0.05)
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("  [UYARI] xgboost kurulu degil. pip install xgboost")
        return None

    params = {
        "objective"       : "binary:logistic",
        "eval_metric"     : "logloss",
        "max_depth"       : 6,           # LightGBM num_leaves=31 yaklaşık karşılığı
        "learning_rate"   : 0.05,
        "min_child_weight": 20,          # LightGBM min_child_samples karşılığı
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "verbosity"       : 0,
        "seed"            : RANDOM_SEED,
    }

    skf       = get_stratified_group_kfold(n_splits=5, random_state=RANDOM_SEED)
    scores    = []
    oof_preds = np.zeros(len(X))
    t0        = time.time()

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X, y, groups=groups), start=1
    ):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        val_proba = model.predict(dval)
        oof_preds[val_idx] = val_proba
        f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
        scores.append(f1)
        print(f"    Fold {fold}/5  F1={f1:.4f}  iter={model.best_iteration}")

    elapsed = time.time() - t0
    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    best_thresh, best_f1, _ = find_best_threshold(y.values, oof_preds)

    return {
        "model"          : "XGBoost",
        "mean_f1"        : round(mean_f1, 4),
        "std_f1"         : round(std_f1, 4),
        "best_threshold" : best_thresh,
        "best_f1"        : round(best_f1, 4),
        "train_sec"      : round(elapsed, 1),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Model Kiyasi: LightGBM vs XGBoost (9 Temmuz)")
    print("=" * 60)

    # 1. Veri hazırlama
    print("\n[1/4] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    print(f"[2/4] {SAMPLE_POS} poz ornek ile egitim seti hazirlaniyor...")
    pos_sample = train_raw.sample(SAMPLE_POS, random_state=RANDOM_SEED)
    full_train = build_training_set(
        pos_sample, items_df, ratio=NEG_RATIO,
        random_state=RANDOM_SEED, verbose=False,
        positive_reference_df=train_raw,
    )
    merged = full_train.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    merged = build_features(merged)

    X = merged[FEATURE_COLS]
    y = merged["label"]
    groups = merged["term_id"]
    print(f"  {len(merged):,} satir, {len(FEATURE_COLS)} feature")

    results = []

    # 3. LightGBM
    print("\n[3/4] LightGBM 5-Fold CV...")
    print("-" * 40)
    lgbm_result = run_lightgbm(X, y, groups)
    results.append(lgbm_result)
    print(f"  LightGBM ort. F1: {lgbm_result['mean_f1']:.4f}")

    # 4. XGBoost
    print("\n[4/4] XGBoost 5-Fold CV...")
    print("-" * 40)
    xgb_result = run_xgboost(X, y, groups)
    if xgb_result:
        results.append(xgb_result)
        print(f"  XGBoost ort. F1: {xgb_result['mean_f1']:.4f}")

    # 5. Karşılaştırma
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("  KARSILASTIRMA TABLOSU")
    print("=" * 60)
    print(df[["model", "mean_f1", "std_f1", "best_threshold", "best_f1", "train_sec"]].to_string(index=False))

    if len(results) == 2:
        diff = results[1]["best_f1"] - results[0]["best_f1"]
        kazanan = results[0]["model"] if diff < 0 else results[1]["model"]
        print(f"\n  Fark (XGB - LGBM): {diff:+.4f}")
        print(f"  Kazanan: {kazanan}")
        if abs(diff) < 0.005:
            print("  Fark kucuk — LightGBM tercih edilmeli (hiz avantaji)")

    out_path = os.path.join(OUTPUT_DIR, "model_kiyasi_v1.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Sonuclar kaydedildi: {out_path}")
    print("=" * 60)

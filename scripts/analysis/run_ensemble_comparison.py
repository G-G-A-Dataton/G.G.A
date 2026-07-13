"""
run_ensemble_comparison.py
===========================
G.G.A Takimi — Ensemble Aday Karsilastirmasi (13 Temmuz Gorevi)

Omer Faruk Kara tarafından hazırlanmıştır.

Bu script 4 model adayını 5-Fold CV ile karsilastirir:

  1. LGBM_BASE   : LightGBM, temel parametreler
  2. XGB_BASE    : XGBoost, temel parametreler
  3. LGBM_TUNED  : LightGBM, 8 Temmuz tuning sonucu
  4. ENS_LGBM_XGB: LGBM + XGBoost soft voting ensemble (ortalama proba)

Her model icin:
  - 5-Fold CV Macro-F1
  - Optimal threshold
  - Egitim suresi

Calistirmak icin:
  python run_ensemble_comparison.py
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
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_kfold
import lightgbm as lgb
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[ensemble] XGBoost bulunamadi, XGB deneyler atlanacak.")

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR   = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_POS  = 3_000
NEG_RATIO   = 2       # 11 Temmuz deney matrisinden en iyi oran
RANDOM_SEED = 42

LGBM_BASE_PARAMS = {
    "objective"   : "binary",
    "metric"      : "binary_logloss",
    "verbose"     : -1,
    "random_state": RANDOM_SEED,
}

LGBM_TUNED_PARAMS = {
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

XGB_BASE_PARAMS = {
    "objective"       : "binary:logistic",
    "eval_metric"     : "logloss",
    "eta"             : 0.1,
    "max_depth"       : 6,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "verbosity"       : 0,
    "seed"            : RANDOM_SEED,
}


def run_lgbm_cv(X, y, params, label):
    """
    LightGBM 5-Fold CV yapar ve OOF tahminleri dondurur.

    LightGBM'in kendi Dataset/train API'si kullanilir.
    Early stopping ile gerekneden fazla round egitilmez.

    Parametreler
    ----------
    X : pd.DataFrame
    y : pd.Series
    params : dict   — LightGBM parametreleri
    label : str     — Cikti icin model etiketi

    Dondurur
    -------
    (np.ndarray, float)  — (oof_preds, elapsed_sec)
    """
    skf       = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    t0        = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [{label}] Fold {fold}/5 ...", end="\r")
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval   = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
        model  = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])

    print(f"  [{label}] 5 fold tamamlandi.          ")
    return oof_preds, time.time() - t0


def run_xgb_cv(X, y, params, label):
    """
    XGBoost 5-Fold CV yapar.

    XGBoost'un kendi DMatrix ve train API'si kullanilir.
    Verbose suppress edildi — sadece fold ilerleme yazilir.

    Parametreler
    ----------
    X : pd.DataFrame
    y : pd.Series
    params : dict   — XGBoost parametreleri
    label : str

    Dondurur
    -------
    (np.ndarray, float)
    """
    if not XGB_AVAILABLE:
        return np.zeros(len(y)), 0.0

    skf       = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    t0        = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [{label}] Fold {fold}/5 ...", end="\r")
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval   = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        model  = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        oof_preds[val_idx] = model.predict(xgb.DMatrix(X.iloc[val_idx]))

    print(f"  [{label}] 5 fold tamamlandi.          ")
    return oof_preds, time.time() - t0


def score(y, oof_preds, label, elapsed):
    """
    OOF tahminleri uzerinden F1 ve optimal threshold hesaplar.

    Hem 0.5 sabit threshold hem de taramali optimal threshold hesaplanir.
    Raporlama icin dict dondurur.
    """
    f1_default              = macro_f1_from_proba(y, oof_preds, threshold=0.5)
    best_thresh, best_f1, _ = find_best_threshold(y.values, oof_preds)
    return {
        "model"          : label,
        "f1_default"     : round(f1_default, 4),
        "best_f1"        : round(best_f1, 4),
        "best_threshold" : best_thresh,
        "train_sec"      : round(elapsed, 1),
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  G.G.A - Ensemble Aday Karsilastirmasi (13 Temmuz)")
    print("=" * 65)

    # 1. Veri
    print("\n[1/3] Veri hazirlaniyor...")
    terms_df  = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )
    pos_sample = train_raw.sample(SAMPLE_POS, random_state=RANDOM_SEED)
    full       = build_training_set(pos_sample, items_df, ratio=NEG_RATIO,
                                     random_state=RANDOM_SEED, verbose=False)
    merged     = full.merge(terms_df, on="term_id", how="left").merge(items_df, on="item_id", how="left")
    merged     = build_features(merged)
    X          = merged[FEATURE_COLS]
    y          = merged["label"]
    print(f"  {len(merged):,} satir, {len(FEATURE_COLS)} feature")

    # 2. Modelleri calistir
    print("\n[2/3] Modeller egitiliyor...")
    results = []
    oof_store = {}

    print("\n  -- LGBM_BASE --")
    oof_lgbm_base, t = run_lgbm_cv(X, y, LGBM_BASE_PARAMS, "LGBM_BASE")
    results.append(score(y, oof_lgbm_base, "LGBM_BASE", t))
    oof_store["lgbm_base"] = oof_lgbm_base

    print("\n  -- LGBM_TUNED --")
    oof_lgbm_tuned, t = run_lgbm_cv(X, y, LGBM_TUNED_PARAMS, "LGBM_TUNED")
    results.append(score(y, oof_lgbm_tuned, "LGBM_TUNED", t))
    oof_store["lgbm_tuned"] = oof_lgbm_tuned

    if XGB_AVAILABLE:
        print("\n  -- XGB_BASE --")
        oof_xgb, t = run_xgb_cv(X, y, XGB_BASE_PARAMS, "XGB_BASE")
        results.append(score(y, oof_xgb, "XGB_BASE", t))
        oof_store["xgb_base"] = oof_xgb

        print("\n  -- ENS_LGBM_XGB (Soft Voting) --")
        # Ensemble: tuned LGBM + XGB ortalaması
        oof_ens = (oof_lgbm_tuned + oof_xgb) / 2.0
        results.append(score(y, oof_ens, "ENS_LGBM_XGB", t))  # sure XGB'den

    # 3. Sonuclar
    df = pd.DataFrame(results)
    best = df.loc[df["best_f1"].idxmax()]

    print("\n" + "=" * 65)
    print("  ENSEMBLE KARSILASTIRMA TABLOSU")
    print("=" * 65)
    print(f"  {'Model':<20} {'F1 (0.5)':>10} {'Best F1':>10} {'Threshold':>11} {'Sure':>8}")
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        marker = " <<" if row["model"] == best["model"] else ""
        print(f"  {row['model']:<20} {row['f1_default']:>10.4f} {row['best_f1']:>10.4f} "
              f"{row['best_threshold']:>11} {row['train_sec']:>7.1f}s{marker}")

    print(f"\n  En iyi model: {best['model']}  (best_F1={best['best_f1']:.4f})")

    # Kaydet
    out_csv = os.path.join(OUTPUT_DIR, "ensemble_karsilastirma.csv")
    df.to_csv(out_csv, index=False)

    out_md = os.path.join(DOCS_DIR, "ensemble_karsilastirma.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Ensemble Aday Karsilastirma Tablosu (13 Temmuz)\n\n")
        f.write("**Hazırlayan:** Ömer Faruk Kara  \n**Tarih:** 13 Temmuz 2026\n\n---\n\n")
        f.write("| Model | F1 (0.5) | Best F1 | Threshold | Sure |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df.iterrows():
            bold = "**" if row["model"] == best["model"] else ""
            f.write(f"| {bold}{row['model']}{bold} | {row['f1_default']} | "
                    f"{bold}{row['best_f1']}{bold} | {row['best_threshold']} | {row['train_sec']}s |\n")
        f.write(f"\n**En iyi aday:** `{best['model']}`  \n")
        f.write(f"*CSV: `outputs/ensemble_karsilastirma.csv`*\n")

    print(f"\n  Rapor: {out_md}")
    print("=" * 65)

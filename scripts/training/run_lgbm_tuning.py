"""
run_lgbm_tuning.py
==================
G.G.A Takımı — LightGBM Hiperparametre Ayarlama (8 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script LightGBM modelinin temel parametrelerini sistematik olarak
dener ve Macro-F1 üzerindeki etkisini ölçer.

Denenen parametreler:
  - num_leaves      : 15, 31, 63
  - learning_rate   : 0.01, 0.05, 0.1
  - min_child_samples: 10, 20, 50

Çalıştırmak için:
  python run_lgbm_tuning.py
"""

import os
import sys
import itertools
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

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sabit parametreler — sadece denenen parametreler değişecek
BASE_PARAMS = {
    "objective"    : "binary",
    "metric"       : "binary_logloss",
    "subsample"    : 0.8,
    "colsample_bytree": 0.8,
    "verbose"      : -1,
    "random_state" : 42,
}

# Denenen parametre değerleri
PARAM_GRID = {
    "num_leaves"       : [15, 31, 63],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "min_child_samples": [10, 20, 50],
}

SAMPLE_POS  = 2_000
NEG_RATIO   = 3
RANDOM_SEED = 42


def run_single_experiment(X, y, num_leaves, learning_rate, min_child_samples):
    """
    Tek bir parametre kombinasyonu için 5-Fold CV ile Macro-F1 skorunu ölçer.

    Parametreler
    ----------
    X : pd.DataFrame
        Feature matrisi.
    y : pd.Series
        Etiketler.
    num_leaves, learning_rate, min_child_samples : scalar
        Denenecek LightGBM parametreleri.

    Döndürür
    -------
    dict
        Deney sonuçları.
    """
    params = {
        **BASE_PARAMS,
        "num_leaves"       : num_leaves,
        "learning_rate"    : learning_rate,
        "min_child_samples": min_child_samples,
    }

    skf       = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    scores    = []
    oof_preds = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
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
        scores.append(macro_f1_from_proba(y_val, val_proba, threshold=0.5))

    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    best_thresh, best_f1, _ = find_best_threshold(y.values, oof_preds)

    return {
        "num_leaves"       : num_leaves,
        "learning_rate"    : learning_rate,
        "min_child_samples": min_child_samples,
        "mean_f1"          : round(mean_f1, 4),
        "std_f1"           : round(std_f1, 4),
        "best_threshold"   : best_thresh,
        "best_f1"          : round(best_f1, 4),
    }


if __name__ == "__main__":
    total = (
        len(PARAM_GRID["num_leaves"])
        * len(PARAM_GRID["learning_rate"])
        * len(PARAM_GRID["min_child_samples"])
    )
    print("=" * 60)
    print("  G.G.A - LightGBM Hiperparametre Tuning (8 Temmuz)")
    print(f"  Toplam deney: {total}")
    print("=" * 60)

    # --- Veri hazırlama ---
    print("\n[1/3] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    print(f"[2/3] {SAMPLE_POS} poz ornek ile egitim seti hazirlaniyor...")
    pos_sample = train_raw.sample(SAMPLE_POS, random_state=RANDOM_SEED)
    full_train = build_training_set(
        pos_sample, items_df, ratio=NEG_RATIO,
        random_state=RANDOM_SEED, verbose=False
    )
    merged = full_train.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    merged = build_features(merged)

    X = merged[FEATURE_COLS]
    y = merged["label"]
    print(f"  Egitim seti: {len(merged):,} satir, {len(FEATURE_COLS)} feature")

    # --- Grid search ---
    print("\n[3/3] Grid search basliyor...")
    results = []
    combos = list(itertools.product(
        PARAM_GRID["num_leaves"],
        PARAM_GRID["learning_rate"],
        PARAM_GRID["min_child_samples"],
    ))

    for i, (nl, lr, mcs) in enumerate(combos, start=1):
        print(
            f"  [{i:02d}/{total}] "
            f"num_leaves={nl}, lr={lr}, min_child={mcs} ...",
            end=" ", flush=True
        )
        result = run_single_experiment(X, y, nl, lr, mcs)
        results.append(result)
        print(f"mean_F1={result['mean_f1']:.4f}  best={result['best_f1']:.4f}")

    # --- Sonuçlar ---
    df = pd.DataFrame(results).sort_values("best_f1", ascending=False)

    print("\n" + "=" * 60)
    print("  SONUCLAR (Best F1'e gore sirali)")
    print("=" * 60)
    print(df.to_string(index=False))

    # En iyi 3
    print("\n  En iyi 3 kombinasyon:")
    for rank, (_, row) in enumerate(df.head(3).iterrows(), start=1):
        print(
            f"  {rank}. num_leaves={row['num_leaves']}, "
            f"lr={row['learning_rate']}, min_child={row['min_child_samples']} "
            f"-> best_F1={row['best_f1']:.4f}"
        )

    best = df.iloc[0]
    print(f"\n  ONERI: num_leaves={best['num_leaves']}, "
          f"learning_rate={best['learning_rate']}, "
          f"min_child_samples={best['min_child_samples']}")

    out_path = os.path.join(OUTPUT_DIR, "lgbm_tuning_sonuclari.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Sonuclar kaydedildi: {out_path}")
    print("=" * 60)

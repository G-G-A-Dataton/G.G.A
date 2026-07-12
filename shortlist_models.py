"""
scripts/training/run_model_shortlist.py
=======================================
G.G.A Takımı — Model Shortlist & Validation Tahminleri (14 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script:
  1. En iyi iki modeli (LightGBM Tuned ve XGBoost Tuned) eğitir.
  2. 5-Fold Stratified CV kullanarak her iki model için Out-of-Fold (OOF)
     tahmin olasılıklarını üretir.
  3. Test/Submission veri seti üzerinde tahmin olasılıklarını hesaplar.
  4. Daha sonra ensemble ağırlıklandırması ve ortak optimizasyon yapmak üzere
     OOF ve test tahminlerini diske kaydeder (npy formatında).

Çalıştırmak için (tüm veri):
  python scripts/training/run_model_shortlist.py

Hızlı test için (örn. 5K pozitif, 10K test):
  python scripts/training/run_model_shortlist.py --sample 5000 --test-sample 10000
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.tfidf_features    import build_tfidf_vectorizer, add_tfidf_features, save_vectorizer
from src.train_mix_v2      import build_mixed_training_set
from src.metrics           import get_stratified_kfold, macro_f1_from_proba, find_best_threshold

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
NEGATIVE_RATIO = 2  # Deney matrisinde en iyi sonuç veren oran

# LightGBM Tuned Parametreleri
LGBM_PARAMS = {
    "objective"        : "binary",
    "metric"           : "binary_logloss",
    "learning_rate"    : 0.05,
    "num_leaves"       : 31,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "verbose"          : -1,
    "random_state"     : RANDOM_SEED,
    "n_jobs"           : -1,
}

# XGBoost Tuned Parametreleri
XGB_PARAMS = {
    "objective"        : "binary:logistic",
    "eval_metric"      : "logloss",
    "learning_rate"    : 0.05,
    "max_depth"        : 6,
    "min_child_weight" : 1,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "seed"             : RANDOM_SEED,
    "n_jobs"           : -1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Model Shortlist & OOF Tahmin Export")
    parser.add_argument("--sample", type=int, default=None, help="Hizli test icin pozitif ornek sayisi")
    parser.add_argument("--test-sample", type=int, default=None, help="Hizli test icin test ornek sayisi")
    return parser.parse_args()


def train_and_predict_oof(X, y, X_test, model_type="lgbm"):
    """
    5-Fold Stratified CV ile OOF ve Test tahminlerini üretir.
    """
    skf = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [{model_type.upper()}] Fold {fold}/5 eğitiliyor...", flush=True)

        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if model_type == "lgbm":
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                LGBM_PARAMS, dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / 5.0

        elif model_type == "xgb":
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(
                XGB_PARAMS, dtrain,
                num_boost_round=1000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            oof_preds[val_idx] = model.predict(dval)
            test_preds += model.predict(dtest) / 5.0

    return oof_preds, test_preds


def main():
    args = parse_args()

    print("=" * 65)
    print("  G.G.A — Model Shortlist & Validation Tahminleri")
    print("=" * 65)

    if not XGB_AVAILABLE:
        print("[HATA] XGBoost kütüphanesi yüklü değil!")
        sys.exit(1)

    # 1. Verileri Yükle
    print("\n[1/5] Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    if args.sample:
        print(f"  Eğitim seti pozitif örnekleniyor: {args.sample:,} satır")
        train_raw = train_raw.sample(args.sample, random_state=RANDOM_SEED).reset_index(drop=True)

    # 2. Negatif Örnekleme ve Feature Pipeline
    print("\n[2/5] Karisik negatif egitim seti hazirlaniyor...")
    full_train = build_mixed_training_set(
        train_raw, terms_df, items_df,
        ratio=NEGATIVE_RATIO,
        bm25_top_n=50,
        bm25_max_df_ratio=0.15,
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    merged = full_train.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df, on="item_id", how="left")
    merged = build_features(merged)

    # TF-IDF fit & transform
    print("  TF-IDF vectorizer egitiliyor...")
    vectorizer = build_tfidf_vectorizer(terms_df, items_df, max_features=10000, ngram_range=(1, 1))
    merged = add_tfidf_features(merged, vectorizer)
    save_vectorizer(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer_v2.pkl"))

    # Test/Submission setini yükle ve hazırla
    sub_path = os.path.join(DATA_DIR, "submission_pairs.csv")
    print(f"\n[3/5] Test seti yukleniyor: {sub_path}")
    sub_df = pd.read_csv(sub_path, dtype={"id": "string", "term_id": "string", "item_id": "string"})

    if args.test_sample:
        print(f"  Test seti örnekleniyor: {args.test_sample:,} satır")
        sub_df = sub_df.sample(args.test_sample, random_state=RANDOM_SEED).reset_index(drop=True)

    sub_merged = sub_df.merge(terms_df, on="term_id", how="left")
    sub_merged = sub_merged.merge(items_df, on="item_id", how="left")
    sub_merged = build_features(sub_merged)
    sub_merged = add_tfidf_features(sub_merged, vectorizer)

    feature_cols = FEATURE_COLS + ["tfidf_cosine"]
    X = merged[feature_cols]
    y = merged["label"]
    X_test = sub_merged[feature_cols]

    print(f"  Tren boyutu : {X.shape}")
    print(f"  Test boyutu : {X_test.shape}")
    print(f"  Öznitelikler: {feature_cols}")

    # 3. Model Adaylarını Eğit & OOF Tahminleri Çıkar
    print("\n[4/5] Shortlist model adaylarinin egitimi basliyor...")
    
    # Model A: LightGBM
    oof_lgbm, test_lgbm = train_and_predict_oof(X, y, X_test, model_type="lgbm")
    f1_lgbm = macro_f1_from_proba(y, oof_lgbm, threshold=0.5)
    best_th_lgb, best_f1_lgb, _ = find_best_threshold(y.values, oof_lgbm)
    print(f"  -> LightGBM Default F1 (0.50): {f1_lgbm:.4f} | Best F1 ({best_th_lgb:.2f}): {best_f1_lgb:.4f}")

    # Model B: XGBoost
    oof_xgb, test_xgb = train_and_predict_oof(X, y, X_test, model_type="xgb")
    f1_xgb = macro_f1_from_proba(y, oof_xgb, threshold=0.5)
    best_th_xgb, best_f1_xgb, _ = find_best_threshold(y.values, oof_xgb)
    print(f"  -> XGBoost  Default F1 (0.50): {f1_xgb:.4f} | Best F1 ({best_th_xgb:.2f}): {best_f1_xgb:.4f}")

    # 4. Tahminleri Dışa Aktar (Export)
    print("\n[5/5] OOF ve test tahminleri kaydediliyor...")
    np.save(os.path.join(OUTPUT_DIR, "oof_lgbm.npy"), oof_lgbm)
    np.save(os.path.join(OUTPUT_DIR, "test_lgbm.npy"), test_lgbm)
    np.save(os.path.join(OUTPUT_DIR, "oof_xgb.npy"), oof_xgb)
    np.save(os.path.join(OUTPUT_DIR, "test_xgb.npy"), test_xgb)
    np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y.values)
    
    # Test setindeki ID'leri de kaydet ki eşleştirebilelim
    sub_df[["id", "term_id", "item_id"]].to_csv(os.path.join(OUTPUT_DIR, "test_metadata.csv"), index=False)

    print("\n[OK] Tüm adımlar başarıyla tamamlandı!")
    print(f"  OOF LGBM  : {os.path.join(OUTPUT_DIR, 'oof_lgbm.npy')}")
    print(f"  Test LGBM : {os.path.join(OUTPUT_DIR, 'test_lgbm.npy')}")
    print(f"  OOF XGB   : {os.path.join(OUTPUT_DIR, 'oof_xgb.npy')}")
    print(f"  Test XGB  : {os.path.join(OUTPUT_DIR, 'test_xgb.npy')}")
    print("=" * 65)


if __name__ == "__main__":
    main()

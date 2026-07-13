"""
run_hard_neg_comparison.py
==========================
G.G.A Takımı — Hard Negative vs Random Negative Skor Kıyası (7 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

AMAÇ:
  Negatif örnekleme stratejisinin model kalitesine etkisini ölçmek.

  Random Negative: Katalogdan tamamen rastgele seçilen ürünler.
    → Kolay negatifleri içerir (çok belirgin alakasız örnekler).
    → Model çok kolay öğrenir ama gerçek dünyada yanılır.

  BM25 Hard Negative: Sorguya benzer ama pozitif olmayan ürünler.
    → Zor negatifleri içerir ("spor ayakkabı" sorgusunda başka marka sneaker).
    → Model daha dikkatli olmak zorunda kalır → genellikle daha iyi F1.

KULLANIM:
  1. Sadece Random Negative (şu an):
       python run_hard_neg_comparison.py

  2. BM25 negatifleri hazır olduğunda (Mert'in negative_bm25_v1 dosyasıyla):
       python run_hard_neg_comparison.py --bm25 outputs/negative_bm25_v1.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set, verify_no_leakage
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_kfold
import lightgbm as lgb

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Karşılaştırma için sabit parametreler
SAMPLE_POS  = 3_000   # Pozitif örnek sayısı — ikisi için de aynı, adil karşılaştırma
NEG_RATIO   = 3       # Negatif oranı — ikisi için de aynı
RANDOM_SEED = 42      # Tekrar üretilebilirlik


def train_and_eval(pos_df, neg_df, terms_df, items_df, label):
    """
    Verilen pozitif + negatif örneklerle bir LightGBM modeli eğitir ve
    5-Fold CV ile Macro-F1 skorunu ölçer.

    Parametreler
    ----------
    pos_df : pd.DataFrame
        Pozitif çiftler (label=1). training_pairs.csv'den gelir.
    neg_df : pd.DataFrame
        Negatif çiftler (label=0). Random veya BM25 negatifler.
    terms_df, items_df : pd.DataFrame
        Sorgu ve ürün verileri (merge için).
    label : str
        Bu deneyin adı — "Random" veya "BM25 Hard" gibi.

    Döndürür
    -------
    dict
        Deney sonuçları: mean_f1, std_f1, best_threshold, best_f1
    """
    print(f"\n" + "-"*55)
    print(f"  Strateji: {label}")
    print("-"*55)

    # Pozitif ve negatif örnekleri birleştir, karıştır
    pos_df = pos_df[["term_id", "item_id"]].copy()
    pos_df["label"] = 1
    neg_df = neg_df[["term_id", "item_id"]].copy()
    neg_df["label"] = 0

    full_df = pd.concat([pos_df, neg_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Pozitif: {(full_df['label']==1).sum():,}  |  Negatif: {(full_df['label']==0).sum():,}  |  Toplam: {len(full_df):,}")

    # Sorgu ve ürün bilgilerini merge et
    print("  Merge ve feature hesaplaniyor...")
    merged = full_df.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    merged = build_features(merged)

    X = merged[FEATURE_COLS]
    y = merged["label"]

    # LightGBM hiperparametreleri — her iki strateji için aynı (adil karşılaştırma)
    lgbm_params = {
        "objective"        : "binary",
        "metric"           : "binary_logloss",
        "learning_rate"    : 0.05,
        "num_leaves"       : 31,
        "min_child_samples": 20,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "verbose"          : -1,
        "random_state"     : RANDOM_SEED,
    }

    # 5-Fold Stratified CV
    skf         = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    fold_scores = []
    oof_preds   = np.zeros(len(X))

    print("  5-Fold CV basliyor...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

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

        val_proba = model.predict(X_val)
        oof_preds[val_idx] = val_proba

        fold_f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
        fold_scores.append(fold_f1)
        print(f"    Fold {fold}/5  Macro-F1: {fold_f1:.4f}  best_iter: {model.best_iteration}")

    mean_f1 = np.mean(fold_scores)
    std_f1  = np.std(fold_scores)

    # Threshold optimizasyonu
    best_thresh, best_f1, _ = find_best_threshold(y.values, oof_preds)

    print(f"\n  Ort. Macro-F1 : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  En iyi thresh : {best_thresh} -> {best_f1:.4f}")

    return {
        "strateji"      : label,
        "n_pos"         : int((full_df["label"] == 1).sum()),
        "n_neg"         : int((full_df["label"] == 0).sum()),
        "mean_f1"       : round(mean_f1, 4),
        "std_f1"        : round(std_f1, 4),
        "best_threshold": best_thresh,
        "best_f1"       : round(best_f1, 4),
    }


def main(bm25_path=None):
    print("=" * 55)
    print("  G.G.A - Hard Negative vs Random Negative Kiyasi")
    print("  7 Temmuz 2026 - Omer Faruk Kara")
    print("=" * 55)

    # ─── 1. Veri yükle ───────────────────────────────────────────────────────
    print("\n[1/4] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    # Adil karşılaştırma için: her iki deneyde de aynı pozitif örnekler kullan
    pos_sample = train_raw.sample(n=SAMPLE_POS, random_state=RANDOM_SEED)
    print(f"  {SAMPLE_POS:,} pozitif ornek secildi (seed={RANDOM_SEED})")

    results = []

    # ─── 2. Random Negative Deneyi ───────────────────────────────────────────
    print("\n[2/4] Random Negative uretiliyor...")
    random_full = build_training_set(
        pos_sample, items_df,
        ratio=NEG_RATIO, random_state=RANDOM_SEED, verbose=False
    )
    # Sadece negatif örnekleri al
    random_neg = random_full[random_full["label"] == 0].copy()
    print(f"  {len(random_neg):,} random negatif ornek hazirlandi.")

    result_random = train_and_eval(pos_sample, random_neg, terms_df, items_df, "Random Negative")
    results.append(result_random)

    # ─── 3. BM25 Hard Negative Deneyi (varsa) ────────────────────────────────
    if bm25_path and os.path.exists(bm25_path):
        print(f"\n[3/4] BM25 Hard Negative yukleniyor: {bm25_path}")
        bm25_neg = pd.read_csv(
            bm25_path,
            dtype={"term_id": "string", "item_id": "string"}
        )

        # Sızıntı kontrolü — BM25 negatifleri asla pozitif olmamalı
        print("  Sizinti kontrolu yapiliyor...")
        ok = verify_no_leakage(bm25_neg, train_raw)
        if not ok:
            print("  [UYARI] Sizinti var! BM25 negatifleri pozitiflerle cakisiyor.")
            print("  Cakisan cifter filtreleniyor...")
            pos_pairs = set(zip(train_raw["term_id"], train_raw["item_id"]))
            mask = ~bm25_neg.apply(lambda r: (r["term_id"], r["item_id"]) in pos_pairs, axis=1)
            bm25_neg = bm25_neg[mask].copy()

        # BM25 negatiflerini de aynı orana getir
        needed = len(pos_sample) * NEG_RATIO
        if len(bm25_neg) > needed:
            bm25_neg = bm25_neg.sample(n=needed, random_state=RANDOM_SEED)
        print(f"  {len(bm25_neg):,} BM25 hard negatif ornek hazirlandi.")

        result_bm25 = train_and_eval(pos_sample, bm25_neg, terms_df, items_df, "BM25 Hard Negative")
        results.append(result_bm25)
    else:
        print("\n[3/4] BM25 Hard Negative henuz hazir degil.")
        if bm25_path:
            print(f"  Dosya bulunamadi: {bm25_path}")
        print("  Mert'in negative_bm25_v1 dosyasini bekle,")
        print("  sonra: python run_hard_neg_comparison.py --bm25 <dosya_yolu>")

    # ─── 4. Karşılaştırma Tablosu ────────────────────────────────────────────
    print("\n[4/4] Karsilastirma Sonuclari:")
    print("=" * 55)
    results_df = pd.DataFrame(results)
    print(results_df[[
        "strateji", "n_pos", "n_neg", "mean_f1", "std_f1", "best_threshold", "best_f1"
    ]].to_string(index=False))

    if len(results) >= 2:
        # Farkı hesapla ve yorumla
        r_f1    = results[0]["best_f1"]
        bm25_f1 = results[1]["best_f1"]
        diff    = bm25_f1 - r_f1
        etki    = 'Pozitif etki!' if diff > 0 else 'Negatif etki - random daha iyi?'
        sonuc   = 'BM25 kullanilmali' if diff > 0.002 else 'Fark kucuk, daha fazla veri gerekebilir'
        print(f"\n  BM25 kazanimi: {diff:+.4f}  ({etki})")
        print(f"  Sonuc: {sonuc}")

    # Sonuçları kaydet
    out_csv = os.path.join(OUTPUT_DIR, "hard_neg_comparison.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Sonuclar kaydedildi: {out_csv}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard Negative vs Random Negative karsilastirmasi")
    parser.add_argument(
        "--bm25",
        type=str,
        default=None,
        help="BM25 hard negative CSV dosyasinin yolu (Mert'in ciktisi)"
    )
    args = parser.parse_args()
    main(bm25_path=args.bm25)

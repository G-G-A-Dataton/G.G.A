"""
run_embedding_score_comparison.py
==================================
G.G.A Takimi -- Embedding Cosine Feature Skora Etkisi (12 Temmuz Gorevi)

Omer Faruk Kara tarafından hazırlanmıştır.

Bu script, embedding cosine similarity feature'inin model performansina
katkisini olcer. Iki ayar karsilastirilir:

  LGBM_BASE  : 15 temel feature (tfidf dahil, embedding yok)
  LGBM_EMB   : 15 temel feature + embedding_cosine

Yalnizca manifest ile doğrulanmış gerçek embedding artifactleri kullanılır.

Calistirmak icin:
  python run_embedding_score_comparison.py

Not: Gercek embedding elde etmek icin once:
  python run_term_embeddings.py
  python src/embedding_batch.py --target items
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
from src.candidate_sampling import build_test_shaped_training_set, sample_complete_terms
from src.context_features import add_context_features
from src.features          import build_features
from src.metrics           import macro_f1_from_proba, get_stratified_group_kfold
from src.embedding_cosine  import add_embedding_cosine_feature, load_embedding_indexes
from src.modeling import MODEL_FEATURE_COLS, build_group_fold_ids, cross_fitted_threshold_evaluation
from src.tfidf_features import build_tfidf_vectorizer, add_tfidf_features
import lightgbm as lgb

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR   = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_TERMS = 300
RANDOM_SEED = 42

LGBM_PARAMS = {
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

EMBEDDING_FEATURE = "embedding_cosine"


def run_cv(X, y, groups, feature_cols, label=""):
    """
    5-Fold CV ile Macro-F1 olcer.

    Parametreler
    ----------
    X : pd.DataFrame
    y : pd.Series
    feature_cols : list of str
    label : str  — cikti icin etiket

    Dondurur
    -------
    dict
    """
    missing = sorted(set(feature_cols) - set(X.columns))
    if missing:
        raise ValueError(f"Embedding experiment is missing features: {missing}")
    X_sub = X[feature_cols]

    skf       = get_stratified_group_kfold(n_splits=5, random_state=RANDOM_SEED)
    scores    = []
    oof_preds = np.zeros(len(X_sub))
    t0        = time.time()

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X_sub, y, groups=groups), start=1
    ):
        print(f"  [{label}] Fold {fold}/5 ...", end="\r")
        dtrain = lgb.Dataset(X_sub.iloc[tr_idx], label=y.iloc[tr_idx])
        dval   = lgb.Dataset(X_sub.iloc[val_idx], label=y.iloc[val_idx])

        model = lgb.train(
            LGBM_PARAMS, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )
        val_proba = model.predict(X_sub.iloc[val_idx])
        oof_preds[val_idx] = val_proba
        scores.append(macro_f1_from_proba(y.iloc[val_idx], val_proba, threshold=0.5))

    elapsed = time.time() - t0
    print(f"  [{label}] 5 fold tamamlandi.          ")

    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    fold_ids = build_group_fold_ids(y.values, groups.values, n_splits=5)
    report = cross_fitted_threshold_evaluation(y.values, oof_preds, fold_ids)

    return {
        "label"          : label,
        "n_features"     : len(feature_cols),
        "mean_f1"        : round(mean_f1, 4),
        "std_f1"         : round(std_f1, 4),
        "best_threshold" : report["deploy_threshold"],
        "best_f1"        : round(report["cross_fitted_macro_f1"], 4),
        "train_sec"      : round(elapsed, 1),
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  G.G.A - Embedding Cosine Feature Skora Etkisi (12 Temmuz)")
    print("=" * 65)

    # 1. Veri hazirla
    print("\n[1/4] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    pos_sample = sample_complete_terms(train_raw, SAMPLE_TERMS, RANDOM_SEED)
    full_train = build_test_shaped_training_set(
        pos_sample, items_df,
        positive_reference_df=train_raw,
        random_state=RANDOM_SEED, verbose=False,
    )
    merged = full_train.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    merged = build_features(merged, copy=False)
    vectorizer = build_tfidf_vectorizer(terms_df, items_df)
    merged = add_tfidf_features(merged, vectorizer, copy=False)
    merged = add_context_features(merged, copy=False)
    groups = merged["term_id"]
    print(f"  {len(merged):,} satir, {len(MODEL_FEATURE_COLS)} temel feature hazir")

    # 2. Embedding cosine ekle (gercek veya sentetik)
    print("\n[2/4] Embedding cosine feature hazirlaniyor...")
    term_index, item_index = load_embedding_indexes(PROJECT_ROOT)

    print("  Doğrulanmış gerçek embedding lookup yapılıyor...")
    merged = add_embedding_cosine_feature(merged, term_index, item_index)
    emb_source = "manifest_verified"

    sep = merged.loc[merged["label"] == 1, EMBEDDING_FEATURE].mean() - \
          merged.loc[merged["label"] == 0, EMBEDDING_FEATURE].mean()
    print(f"  Cosine separation: {sep:.4f}  (kaynak: {emb_source})")

    X = merged[MODEL_FEATURE_COLS + [EMBEDDING_FEATURE]]
    y = merged["label"]

    # 3. CV karsilastirma: baseline vs embedding ekli
    print("\n[3/4] Karsilastirma deneyleri basliyor...")
    results = []

    print("\n  -- LGBM_BASE (embedding yok) --")
    r_base = run_cv(X, y, groups, MODEL_FEATURE_COLS, label="LGBM_BASE")
    results.append(r_base)

    print("\n  -- LGBM_EMB (embedding cosine ekli) --")
    r_emb = run_cv(
        X, y, groups, MODEL_FEATURE_COLS + [EMBEDDING_FEATURE], label="LGBM_EMB"
    )
    results.append(r_emb)

    # 4. Sonuclar
    diff = r_emb["best_f1"] - r_base["best_f1"]
    print("\n" + "=" * 65)
    print("  KARSILASTIRMA TABLOSU")
    print("=" * 65)
    print(f"  {'Model':<15} {'N Feature':>10} {'mean_F1':>9} {'best_F1':>9} {'Threshold':>10} {'Sure':>7}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['label']:<15} {r['n_features']:>10} {r['mean_f1']:>9.4f} "
              f"{r['best_f1']:>9.4f} {r['best_threshold']:>10} {r['train_sec']:>6.1f}s")

    print(f"\n  Embedding cosine etkisi: {diff:+.4f}  (kaynak: {emb_source})")
    if diff > 0.001:
        print("  >> Embedding cosine feature kalici olarak eklenmeli!")
    elif diff > 0:
        print("  >> Kucuk pozitif etki. Gercek embedding ile tekrar test edilmeli.")
    else:
        print("  >> Negatif etki! Sentetik cosine gurultulu olabilir.")
        print("     Gercek embedding ile tekrar test edilmeli.")

    # CSV kaydet
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, "embedding_skor_kiyasi.csv")
    df.to_csv(out_csv, index=False)

    # Markdown raporu
    out_md = os.path.join(DOCS_DIR, "embedding_skor_kiyasi.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Embedding Cosine Feature Skor Kiyasi (12 Temmuz)\n\n")
        f.write("**Hazırlayan:** Ömer Faruk Kara  \n")
        f.write("**Tarih:** 12 Temmuz 2026  \n")
        f.write(f"**Cosine kaynagi:** {emb_source}  \n\n---\n\n")
        f.write("## Sonuc\n\n")
        f.write("| Model | N Feature | mean_F1 | best_F1 | Threshold |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['n_features']} | {r['mean_f1']} | **{r['best_f1']}** | {r['best_threshold']} |\n")
        f.write(f"\n**Embedding cosine etkisi:** `{diff:+.4f}`  \n")
        f.write(f"**Cosine separation:** `{sep:.4f}`  \n\n")
        f.write(f"\n*CSV: `outputs/embedding_skor_kiyasi.csv`*\n")

    print(f"\n  CSV  : {out_csv}")
    print(f"  Rapor: {out_md}")
    print("=" * 65)

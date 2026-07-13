"""
run_train_full_v2.py
====================
G.G.A Takımı — v2 Model Eğitimi: Tam Veri + Karışık Negatif (9 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script, Sprint 1'deki 5K örneklik baseline'ın aksine:
  ─ TÜM 250K pozitif çifti kullanır
  ─ BM25 hard negative + random fallback (train_mix_v2) ile negatif üretir
  ─ 16 feature (15 temel/attribute + TF-IDF) ile modeli eğitir
  ─ Eğitilmiş modeli kaydeder → canonical pipeline ile submission üretilir

Çalıştırmak için (uzun sürer — tüm veri):
  python scripts/training/run_train_full_v2.py

Hızlı test için (sample_size ayarla):
  python scripts/training/run_train_full_v2.py --sample 10000

Çıktılar:
  outputs/lgbm_v2_fold_{i}.txt   → 5 fold modeli
  outputs/tfidf_vectorizer_v2.pkl → TF-IDF vectorizer
  outputs/oof_preds_v2.npy        → OOF tahminleri (threshold opt. için)
"""

import os
import sys
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS, FEATURE_SCHEMA_VERSION
from src.tfidf_features    import build_tfidf_vectorizer, add_tfidf_features, save_vectorizer
from src.train_mix_v2      import build_mixed_training_set, verify_mix_no_leakage
from src.metrics           import macro_f1_from_proba, find_best_threshold, get_stratified_group_kfold
from src.error_analysis    import generate_error_report
import lightgbm as lgb

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NEGATIVE_RATIO = 3
RANDOM_SEED    = 42
EXPECTED_POSITIVE_ROWS = 250_000

# TF-IDF: 6 Temmuz deneyi sonucu (docs/tfidf_deney_tablosu.md)
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM        = (1, 1)

# LightGBM hiperparametreleri — EXP-001 baseline'ından
LGBM_PARAMS = {
    "objective"        : "binary",
    "metric"           : "binary_logloss",
    "learning_rate"    : 0.05,
    "num_leaves"       : 63,          # v2: 31'den 63'e çıkarıldı (daha derin ağaçlar — daha fazla veri var)
    "min_child_samples": 50,          # v2: 20'den 50'ye (1M satırda overfitting'i önlemek için)
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,         # v2: L1 regularization eklendi
    "reg_lambda"       : 1.0,         # v2: L2 regularization eklendi
    "verbose"          : -1,
    "random_state"     : RANDOM_SEED,
    "n_jobs"           : -1,          # Tüm CPU'ları kullan
}


def parse_args():
    parser = argparse.ArgumentParser(description="G.G.A v2 tam model egitimi")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Hizli test icin pozitif ornek sayisi (None = tum 250K)"
    )
    parser.add_argument(
        "--bm25-top-n", type=int, default=50,
        help="BM25 hard negative icin top-N aday (varsayilan: 50)"
    )
    parser.add_argument(
        "--no-error-analysis", action="store_true",
        help="Hata analizini atla (daha hizli calisir)"
    )
    parser.add_argument(
        "--artifact-dir", default=None,
        help="Artifact cikti dizini (sample varsayilani: outputs/sample_artifacts_v2)",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(
    artifact_dir,
    feature_cols,
    model_paths,
    vectorizer_path,
    threshold_path,
    threshold,
    mean_f1,
    std_f1,
    best_f1,
    training_mode,
    positive_rows,
    negative_rows,
    total_rows,
    positive_reference_rows,
):
    artifact_paths = model_paths + [vectorizer_path, threshold_path]
    manifest = {
        "artifact_schema_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_mode": training_mode,
        "feature_columns": feature_cols,
        "validation": {
            "splitter": "StratifiedGroupKFold",
            "group_column": "term_id",
            "n_splits": 5,
            "random_state": RANDOM_SEED,
        },
        "negative_sampling": {
            "strategy": "bm25_random_fallback",
            "ratio": NEGATIVE_RATIO,
            "positive_reference_rows": int(positive_reference_rows),
        },
        "training": {
            "positive_rows": int(positive_rows),
            "negative_rows": int(negative_rows),
            "total_rows": int(total_rows),
        },
        "metrics": {
            "mean_macro_f1": float(mean_f1),
            "std_macro_f1": float(std_f1),
            "optimized_macro_f1": float(best_f1),
        },
        "threshold": float(threshold),
        "artifacts": {
            "models": [os.path.basename(path) for path in model_paths],
            "vectorizer": os.path.basename(vectorizer_path),
            "threshold": os.path.basename(threshold_path),
        },
        "sha256": {
            os.path.basename(path): sha256_file(path) for path in artifact_paths
        },
    }
    manifest_path = os.path.join(artifact_dir, "model_manifest_v2.json")
    temporary_path = manifest_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    os.replace(temporary_path, manifest_path)
    return manifest_path


def main():
    args = parse_args()
    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be positive")
    if args.bm25_top_n <= 0:
        raise ValueError("--bm25-top-n must be positive")
    artifact_dir = os.path.abspath(
        args.artifact_dir
        or (
            os.path.join(OUTPUT_DIR, "sample_artifacts_v2")
            if args.sample
            else OUTPUT_DIR
        )
    )
    if args.sample and os.path.realpath(artifact_dir) == os.path.realpath(OUTPUT_DIR):
        raise ValueError("Sample training cannot write to the production artifact directory")
    os.makedirs(artifact_dir, exist_ok=True)

    print("=" * 65)
    print("  G.G.A v2 — Tam Veri + Karisik Negatif Model Egitimi")
    print("  9 Temmuz 2026 — Omer Faruk Kara")
    print("=" * 65)

    # ─── 1. Veri Yükle ────────────────────────────────────────────────────────
    print("\n[1/7] Veriler yukleniyor...")
    terms_df  = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )
    if len(train_raw) != EXPECTED_POSITIVE_ROWS or set(train_raw["label"].unique()) != {1}:
        raise ValueError(
            "training_pairs.csv must contain exactly 250,000 positive rows"
        )
    print(f"  Sorgular : {len(terms_df):,}  |  Urunler: {len(items_df):,}")
    print(f"  Pozitif  : {len(train_raw):,}")

    # Örnek modunda küçük veri
    if args.sample:
        train_pos = train_raw.sample(n=args.sample, random_state=RANDOM_SEED)
        print(f"  [SAMPLE MODU] {args.sample:,} pozitif ile calisiliyor.")
    else:
        train_pos = train_raw.copy()
        print(f"  [TAM VERİ] Tum {len(train_raw):,} pozitif kullaniliyor.")

    # ─── 2. Karışık Negatif Üret ──────────────────────────────────────────────
    print("\n[2/7] Karisik negatif (BM25 + random fallback) uretiliyor...")
    full_train = build_mixed_training_set(
        train_df=train_pos,
        terms_df=terms_df,
        items_df=items_df,
        ratio=NEGATIVE_RATIO,
        bm25_top_n=args.bm25_top_n,
        random_state=RANDOM_SEED,
        verbose=True,
        positive_reference_df=train_raw,
    )

    print("\n  Sizinti kontrolu...")
    if not verify_mix_no_leakage(full_train, train_raw):
        raise RuntimeError("Negative sampling leakage check failed")

    # ─── 3. Merge + Feature'lar ───────────────────────────────────────────────
    print("\n[3/7] Merge ve temel feature hesaplaniyor...")
    merged = full_train.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    merged = build_features(merged)

    # ─── 4. TF-IDF Feature ────────────────────────────────────────────────────
    print(f"\n[4/7] TF-IDF feature ekleniyor (max={TFIDF_MAX_FEATURES}, ngram={TFIDF_NGRAM})...")
    vec_path = os.path.join(artifact_dir, "tfidf_vectorizer_v2.pkl")
    vectorizer = build_tfidf_vectorizer(
        terms_df, items_df,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
    )
    save_vectorizer(vectorizer, vec_path)
    merged = add_tfidf_features(merged, vectorizer)

    feature_cols = FEATURE_COLS + ["tfidf_cosine"]
    print(f"  Toplam feature: {len(feature_cols)} ({len(FEATURE_COLS)} temel + 1 TF-IDF)")

    X = merged[feature_cols]
    y = merged["label"]

    print(f"\n  Egitim seti: {len(X):,} satir  |  Pozitif: {(y==1).sum():,}  |  Negatif: {(y==0).sum():,}")

    # ─── 5. 5-Fold CV ─────────────────────────────────────────────────────────
    print("\n[5/7] LightGBM 5-Fold Stratified Group CV basliyor...")
    print("-" * 60)

    skf         = get_stratified_group_kfold(n_splits=5, random_state=RANDOM_SEED)
    fold_scores = []
    oof_preds   = np.zeros(len(X))
    trained_models = []
    model_paths = []

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X, y, groups=merged["term_id"]), start=1
    ):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            LGBM_PARAMS, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )
        trained_models.append(model)

        # Modeli diske kaydet (submission için tekrar yüklenir)
        model_path = os.path.join(artifact_dir, f"lgbm_v2_fold_{fold}.txt")
        model.save_model(model_path)
        model_paths.append(model_path)

        val_proba = model.predict(X_val)
        oof_preds[val_idx] = val_proba
        fold_f1 = macro_f1_from_proba(y_val, val_proba, threshold=0.5)
        fold_scores.append(fold_f1)
        print(f"  Fold {fold}/5  |  Macro-F1: {fold_f1:.4f}  |  Best iter: {model.best_iteration}")

    mean_f1 = np.mean(fold_scores)
    std_f1  = np.std(fold_scores)
    print("-" * 60)
    print(f"  ORT. Macro-F1 (v2): {mean_f1:.4f} (+/- {std_f1:.4f})")

    # OOF tahminlerini kaydet
    oof_path = os.path.join(artifact_dir, "oof_preds_v2.npy")
    np.save(oof_path, oof_preds)

    # ─── 6. Threshold Optimizasyonu ───────────────────────────────────────────
    print("\n[6/7] Threshold optimizasyonu...")
    best_thresh, best_f1, all_results = find_best_threshold(y.values, oof_preds)
    print(f"  En iyi threshold : {best_thresh}  ->  {best_f1:.4f}")
    print(f"  Varsayilan (0.50): {macro_f1_from_proba(y.values, oof_preds, 0.5):.4f}")

    # Threshold'u kaydet
    thresh_path = os.path.join(artifact_dir, "best_threshold_v2.txt")
    with open(thresh_path, "w", encoding="utf-8") as f:
        f.write(str(best_thresh))
    print(f"  Threshold kaydedildi: {thresh_path}")

    manifest_path = write_artifact_manifest(
        artifact_dir=artifact_dir,
        feature_cols=feature_cols,
        model_paths=model_paths,
        vectorizer_path=vec_path,
        threshold_path=thresh_path,
        threshold=best_thresh,
        mean_f1=mean_f1,
        std_f1=std_f1,
        best_f1=best_f1,
        training_mode="sample" if args.sample else "full",
        positive_rows=(full_train["label"] == 1).sum(),
        negative_rows=(full_train["label"] == 0).sum(),
        total_rows=len(full_train),
        positive_reference_rows=len(train_raw),
    )
    print(f"  Artifact manifest kaydedildi: {manifest_path}")

    # ─── 7. Feature Importance ────────────────────────────────────────────────
    print("\n[7/7] Feature importance (5-fold ortalama):")
    print("-" * 60)
    importance_arr = np.zeros(len(feature_cols))
    for m in trained_models:
        importance_arr += m.feature_importance(importance_type="gain")
    importance_arr /= len(trained_models)

    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importance_arr})
    feat_imp = feat_imp.sort_values("importance", ascending=False)
    feat_imp.to_csv(os.path.join(artifact_dir, "feature_importance_v2.csv"), index=False)

    max_imp = feat_imp["importance"].max()
    for _, row in feat_imp.iterrows():
        bar = "#" * int(row["importance"] / max_imp * 30)
        print(f"  {row['feature']:<28} {bar} ({row['importance']:.1f})")

    # ─── Hata Analizi (opsiyonel) ─────────────────────────────────────────────
    if not args.no_error_analysis:
        print("\n  Hata analizi yapiliyor (--no-error-analysis ile atlabilir)...")
        error_report_path = os.path.join(artifact_dir, "error_report_v2.md")
        generate_error_report(
            merged, oof_preds,
            threshold=best_thresh,
            output_path=error_report_path,
        )

    # ─── Sonuç Özeti ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  V2 EGITIM SONUC OZETI")
    print("=" * 65)
    neg_sources = full_train[full_train["label"] == 0]["neg_source"].value_counts()
    print(f"  Veri boyutu       : {len(full_train):,} satir")
    print(f"  BM25 negatif      : {neg_sources.get('bm25', 0):,}")
    print(f"  Random negatif    : {neg_sources.get('random', 0):,}")
    print(f"  Feature sayisi    : {len(feature_cols)}")
    print(f"  Ort. Macro-F1     : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  En iyi threshold  : {best_thresh}")
    print(f"  Optimized F1      : {best_f1:.4f}")
    print(f"  Artifact dizini   : {artifact_dir}")
    print(f"  Modeller kaydi    : lgbm_v2_fold_{{1-5}}.txt")
    print(f"  Vectorizer kaydi  : {vec_path}")
    print(f"  OOF preds kaydi   : {oof_path}")
    print(f"\n  Submission icin:")
    print(f"    python scripts/submission/run_pipeline.py --mode predict")
    print("=" * 65)

    # EXP satırı
    n_pos_used = (full_train["label"] == 1).sum()
    print(f"\n  EXP-007 icin experiment_log.md satiri:")
    print(f"  | EXP-007 | 9 Tem | Omer Faruk | LightGBM | Mix(BM25+Rand) 3:1 / "
          f"{n_pos_used:,} poz | {len(feature_cols)} | "
          f"{mean_f1:.4f} +/- {std_f1:.4f} | — | "
          f"v2 full model, thresh={best_thresh}, F1={best_f1:.4f} |")


if __name__ == "__main__":
    main()

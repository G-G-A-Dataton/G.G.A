"""
run_deney_matrisi_v2.py
=======================
G.G.A Takımı — Negatif Oran + Feature Kombinasyon Deney Matrisi (11 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script iki eksende kontrollü deneyler yapar:

  1. NEGATİF ORAN ekseni: 1:1, 2:1, 3:1, 5:1
     Soru: Kaç negatif örnek en iyi F1'i veriyor?

  2. FEATURE KOMBİNASYONU ekseni:
     A) Temel 7 feature (3 Temmuz)
     B) Temel + TF-IDF (EXP-003 konfigürasyonu)
     C) Temel + L2/L3/depth (6 Temmuz)
     D) Tüm 15 feature (mevcut tam set)

  Toplam: 4 oran × 4 feature seti = 16 deney

Çalıştırmak için:
  python run_deney_matrisi_v2.py
"""

import os
import sys
import itertools
import warnings
import time
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
DOCS_DIR   = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_POS  = 2_000   # Hız için küçük tutuldu — 4×4=16 deney
RANDOM_SEED = 42

# Denenecek negatif oranlar
NEG_RATIOS = [1, 2, 3, 5]

# Denenecek feature setleri
FEATURE_SETS = {
    "A_temel7": [
        "query_title_overlap", "query_category_overlap", "query_brand_match",
        "query_cat_l1_overlap", "title_len", "query_len", "gender_match",
    ],
    "B_tfidf": [
        "query_title_overlap", "query_category_overlap", "query_brand_match",
        "query_cat_l1_overlap", "title_len", "query_len", "gender_match",
        "tfidf_cosine",
    ],
    "C_kategori": [
        "query_title_overlap", "query_category_overlap", "query_brand_match",
        "query_cat_l1_overlap", "title_len", "query_len", "gender_match",
        "age_group_match", "demographic_conflict",
        "query_cat_l2_overlap", "query_cat_l3_overlap", "cat_depth",
    ],
    "D_tam15": FEATURE_COLS,  # Tüm 15 feature
}

# LightGBM parametreleri — 8 Temmuz tuning'den en iyi kombinasyon
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


def run_single(X, y, groups, feature_names):
    """
    Verilen feature seti üzerinde 5-Fold CV yaparak Macro-F1 döndürür.

    Parametreler
    ----------
    X : pd.DataFrame  — tüm feature matrisi (FEATURE_COLS sırasında)
    y : pd.Series     — etiketler
    feature_names : list of str  — bu deneyde kullanılacak feature'lar

    Döndürür
    -------
    dict  — mean_f1, std_f1, best_threshold, best_f1, train_sec
    """
    # Sadece bu deneyin feature'larını seç
    # tfidf_cosine gibi bazı feature'lar X'te olmayabilir — eksikleri filtrele
    available = [f for f in feature_names if f in X.columns]
    X_sub = X[available]

    skf       = get_stratified_group_kfold(n_splits=5, random_state=RANDOM_SEED)
    scores    = []
    oof_preds = np.zeros(len(X_sub))
    t0        = time.time()

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X_sub, y, groups=groups), start=1
    ):
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
    mean_f1 = float(np.mean(scores))
    std_f1  = float(np.std(scores))
    best_thresh, best_f1, _ = find_best_threshold(y.values, oof_preds)

    return {
        "mean_f1"        : round(mean_f1, 4),
        "std_f1"         : round(std_f1, 4),
        "best_threshold" : best_thresh,
        "best_f1"        : round(best_f1, 4),
        "n_features"     : len(available),
        "train_sec"      : round(elapsed, 1),
    }


def write_report(results_df, path):
    """Deney matrisini Markdown tablosu olarak kaydeder."""
    lines = [
        "# Deney Matrisi v2 (11 Temmuz)",
        "",
        "**Hazırlayan:** Ömer Faruk Kara  ",
        "**Tarih:** 11 Temmuz 2026  ",
        "**Yöntem:** 5-Fold StratifiedGroupKFold, group=term_id, seed=42  ",
        f"**Pozitif örnek:** {SAMPLE_POS:,}",
        "",
        "## Eksenler",
        "",
        "- **Negatif Oran:** 1:1, 2:1, 3:1, 5:1",
        "- **Feature Seti:**",
        "  - A: Temel 7 feature (3 Temmuz)",
        "  - B: Temel + TF-IDF",
        "  - C: Temel + Kategori L2/L3/depth",
        "  - D: Tam 15 feature (mevcut)",
        "",
        "---",
        "",
        "## Sonuç Matrisi (Best F1)",
        "",
    ]

    # Pivot tablo: satır=oran, sütun=feature seti
    pivot = results_df.pivot(index="neg_ratio", columns="feature_set", values="best_f1")
    lines.append("| Neg. Oran | " + " | ".join(pivot.columns) + " |")
    lines.append("|" + "---|" * (len(pivot.columns) + 1))
    for ratio, row in pivot.iterrows():
        vals = " | ".join(f"**{v:.4f}**" if v == pivot.max().max() else f"{v:.4f}" for v in row)
        lines.append(f"| {ratio}:1 | {vals} |")

    # En iyi kombinasyon
    best_row = results_df.loc[results_df["best_f1"].idxmax()]
    lines += [
        "",
        "---",
        "",
        "## En İyi Kombinasyon",
        "",
        f"| Parametre | Değer |",
        f"|---|---|",
        f"| Feature seti | {best_row['feature_set']} |",
        f"| Negatif oran | {best_row['neg_ratio']}:1 |",
        f"| Best F1 | **{best_row['best_f1']:.4f}** |",
        f"| Best Threshold | {best_row['best_threshold']} |",
        f"| Feature sayısı | {int(best_row['n_features'])} |",
        "",
        "---",
        "",
        "## Ham Sonuclar",
        "",
    ]
    # Manuel markdown tablosu (tabulate gerektirmez)
    cols = ["neg_ratio", "feature_set", "mean_f1", "std_f1", "best_threshold", "best_f1", "n_features", "train_sec"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in results_df[cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines += [
        "",
        f"*CSV: `outputs/deney_matrisi_v2.csv`*"
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    total_exp = len(NEG_RATIOS) * len(FEATURE_SETS)
    print("=" * 65)
    print("  G.G.A - Deney Matrisi v2 (11 Temmuz)")
    print(f"  {len(NEG_RATIOS)} oran x {len(FEATURE_SETS)} feature seti = {total_exp} deney")
    print("=" * 65)

    # 1. Veri yükle
    print("\n[1/3] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )
    pos_sample = train_raw.sample(SAMPLE_POS, random_state=RANDOM_SEED)

    # 2. Her oran için birleşik feature seti önceden üret (tekrar hesaplamayı önle)
    print("\n[2/3] Her negatif oran icin egitim seti hazirlaniyor...")
    datasets = {}
    for ratio in NEG_RATIOS:
        print(f"  Oran {ratio}:1 ...", end=" ", flush=True)
        full = build_training_set(
            pos_sample, items_df,
            ratio=ratio, random_state=RANDOM_SEED, verbose=False,
            positive_reference_df=train_raw,
        )
        merged = full.merge(terms_df, on="term_id", how="left")
        merged = merged.merge(items_df,  on="item_id",  how="left")
        merged = build_features(merged)
        datasets[ratio] = merged
        print(f"{len(merged):,} satir hazir")

    # 3. Deneyleri çalıştır
    print("\n[3/3] Deneyler basliyor...\n")
    results = []
    exp_num = 0

    for ratio, fs_name in itertools.product(NEG_RATIOS, FEATURE_SETS.keys()):
        exp_num += 1
        fs_cols = FEATURE_SETS[fs_name]
        merged  = datasets[ratio]
        X = merged[[c for c in FEATURE_COLS if c in merged.columns]]
        y = merged["label"]

        print(f"  [{exp_num:02d}/{total_exp}] oran={ratio}:1, features={fs_name} ...", end=" ", flush=True)
        result = run_single(X, y, merged["term_id"], fs_cols)
        result.update({"neg_ratio": ratio, "feature_set": fs_name})
        results.append(result)
        print(f"mean_F1={result['mean_f1']:.4f}  best={result['best_f1']:.4f}")

    # 4. Sonuçlar
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("  DENEY MATRİSİ (Best F1)")
    print("=" * 65)
    pivot = results_df.pivot(index="neg_ratio", columns="feature_set", values="best_f1")
    print(pivot.to_string())

    best = results_df.loc[results_df["best_f1"].idxmax()]
    print(f"\n  En iyi: oran={best['neg_ratio']}:1, features={best['feature_set']}, F1={best['best_f1']:.4f}")

    out_csv = os.path.join(OUTPUT_DIR, "deney_matrisi_v2.csv")
    results_df.to_csv(out_csv, index=False)

    out_md = os.path.join(DOCS_DIR, "deney_matrisi_v2.md")
    write_report(results_df, out_md)

    print(f"\n  CSV  : {out_csv}")
    print(f"  Rapor: {out_md}")
    print("=" * 65)

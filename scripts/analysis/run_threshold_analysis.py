"""
run_threshold_analysis.py
=========================
G.G.A Takımı — Threshold Analizi (11 Temmuz Görevi)

Ahmet Emin Işın tarafından hazırlanmıştır.

LightGBM bir sonuç skoru (0-1 arası olasılık) üretir.
Bu skoru 0/1 etikete çevirmek için bir kesme noktası (threshold) gerekir.

Varsayılan threshold = 0.5 çoğu durumda optimal değil!
Bu script tüm threshold adaylarını sistematik olarak dener ve:
  - Macro-F1 grafiği
  - Precision-Recall dengesi
  - Optimal threshold önerisi
üretir.

Neden threshold optimizasyonu?
  Macro-F1 = (F1_pozitif + F1_negatif) / 2
  Eğitim setinde pozitif/negatif oranı dengesizse (3:1 → %25 pozitif)
  model negatif sınıfa meyilli tahmin yapar → threshold 0.5'ten düşük olmalı

Çalıştırmak için:
  python run_threshold_analysis.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set
from src.metrics           import get_stratified_kfold
import lightgbm as lgb

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR   = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_POS  = 3_000
NEG_RATIO   = 3
RANDOM_SEED = 42

# Denenecek threshold aralığı
THRESHOLDS = np.round(np.arange(0.1, 0.91, 0.05), 2).tolist()

# 8 Temmuz tuning'den en iyi LightGBM parametreleri
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


def compute_oof_predictions(X, y):
    """
    5-Fold CV ile Out-of-Fold (OOF) tahminleri üretir.

    OOF tahmini: Her örnek, eğitimde kullanılmadığı fold'da tahmin edilir.
    Bu, validation leakage olmaksızın tüm veri üzerinde skor hesaplamayı sağlar.

    Parametreler
    ----------
    X : pd.DataFrame
    y : pd.Series

    Döndürür
    -------
    np.ndarray  — OOF olasılık tahminleri (shape: len(X),)
    """
    skf       = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  Fold {fold}/5 ...", end="\r")
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval   = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

        model = lgb.train(
            LGBM_PARAMS, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])

    print("  5 fold tamamlandi.          ")
    return oof_preds


def analyze_thresholds(y_true, y_prob, thresholds):
    """
    Her threshold için F1, Precision, Recall ve Confusion Matrix hesaplar.

    Parametreler
    ----------
    y_true : np.ndarray  — gerçek etiketler (0/1)
    y_prob : np.ndarray  — model olasılık tahminleri (0-1)
    thresholds : list    — denenecek threshold değerleri

    Döndürür
    -------
    pd.DataFrame  — her threshold için metrik sonuçları
    """
    rows = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pos_f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        neg_f1   = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        prec     = precision_score(y_true, y_pred, zero_division=0)
        recall   = recall_score(y_true, y_pred, zero_division=0)
        cm       = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0, 0, 0, 0)

        rows.append({
            "threshold"    : thresh,
            "macro_f1"     : round(macro_f1, 4),
            "f1_positive"  : round(pos_f1, 4),
            "f1_negative"  : round(neg_f1, 4),
            "precision"    : round(prec, 4),
            "recall"       : round(recall, 4),
            "TP"           : int(tp),
            "FP"           : int(fp),
            "TN"           : int(tn),
            "FN"           : int(fn),
        })

    return pd.DataFrame(rows)


def write_report(results_df, path):
    """Threshold analiz sonuçlarını Markdown raporuna yazar."""
    best_row = results_df.loc[results_df["macro_f1"].idxmax()]
    default_row = results_df[results_df["threshold"] == 0.5].iloc[0]
    gain = best_row["macro_f1"] - default_row["macro_f1"]

    lines = [
        "# Threshold Analiz Raporu (11 Temmuz)",
        "",
        "**Hazırlayan:** Ahmet Emin Işın  ",
        "**Tarih:** 11 Temmuz 2026  ",
        "**Yöntem:** 5-Fold OOF tahminleri üzerinde threshold taraması",
        "",
        "---",
        "",
        "## 1. Özet Sonuç",
        "",
        f"| Metrik | Varsayılan (0.5) | **Optimal** |",
        f"|---|---|---|",
        f"| Threshold | 0.5 | **{best_row['threshold']}** |",
        f"| Macro-F1 | {default_row['macro_f1']:.4f} | **{best_row['macro_f1']:.4f}** |",
        f"| F1 (Pozitif) | {default_row['f1_positive']:.4f} | {best_row['f1_positive']:.4f} |",
        f"| F1 (Negatif) | {default_row['f1_negative']:.4f} | {best_row['f1_negative']:.4f} |",
        f"| Precision | {default_row['precision']:.4f} | {best_row['precision']:.4f} |",
        f"| Recall | {default_row['recall']:.4f} | {best_row['recall']:.4f} |",
        "",
        f"> [!NOTE]",
        f"> Threshold optimizasyonu ile **+{gain:.4f}** Macro-F1 kazanımı. "
        f"Optimal threshold: **{best_row['threshold']}**",
        "",
        "---",
        "",
        "## 2. Neden 0.5 Optimal Değil?",
        "",
        "Eğitim setinde **3:1 negatif oran** kullanılıyor:",
        "- Pozitif örnek: %25",
        "- Negatif örnek: %75",
        "",
        "Model bu dengesizliği öğrenerek tahminlerini aşağı kaydırır.",
        "Bu yüzden optimal threshold 0.5'ten **düşük** çıkar.",
        "",
        "---",
        "",
        "## 3. Tam Threshold Tablosu",
        "",
    ]

    # Tabloyu ekle
    display_cols = ["threshold", "macro_f1", "f1_positive", "f1_negative", "precision", "recall"]
    for _, row in results_df[display_cols].iterrows():
        marker = " << OPTIMAL" if row["threshold"] == best_row["threshold"] else (
                 " << Varsayilan" if row["threshold"] == 0.5 else "")
        lines.append(
            f"| {row['threshold']} | {row['macro_f1']:.4f} | {row['f1_positive']:.4f} | "
            f"{row['f1_negative']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |{marker}"
        )

    header = "| Threshold | Macro-F1 | F1+ | F1- | Precision | Recall |"
    separator = "|---|---|---|---|---|---|"
    table_lines = lines[:]
    insert_idx = lines.index("## 3. Tam Threshold Tablosu") + 2
    lines.insert(insert_idx, header)
    lines.insert(insert_idx + 1, separator)

    lines += [
        "",
        "---",
        "",
        "## 4. Sonraki Adımlar",
        "",
        f"- **Öneri:** Submission'larda threshold = **{best_row['threshold']}** kullan",
        "- Tam eğitim seti (250K pozitif) üzerinde bu analizi tekrarla (skoru değişebilir)",
        "- BM25 hard negative ile eğitim yapılırsa optimal threshold farklılaşabilir — yeniden hesapla",
        "",
        f"*Ham CSV: `outputs/threshold_analizi.csv`*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Threshold Analizi (11 Temmuz)")
    print(f"  {len(THRESHOLDS)} threshold aday: {THRESHOLDS[0]} -> {THRESHOLDS[-1]}")
    print("=" * 60)

    # 1. Veri hazırla
    print("\n[1/3] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

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
    print(f"  {len(merged):,} satir, {len(FEATURE_COLS)} feature, "
          f"pos/neg={y.sum()}/{(y==0).sum()}")

    # 2. OOF tahminleri
    print("\n[2/3] OOF tahminleri uretiliyor (5-Fold CV)...")
    oof_preds = compute_oof_predictions(X, y)

    # 3. Threshold taraması
    print(f"\n[3/3] {len(THRESHOLDS)} threshold analiz ediliyor...")
    results_df = analyze_thresholds(y.values, oof_preds, THRESHOLDS)

    best_row    = results_df.loc[results_df["macro_f1"].idxmax()]
    default_row = results_df[results_df["threshold"] == 0.5].iloc[0]

    print("\n" + "=" * 60)
    print("  THRESHOLD ANALİZİ SONUCU")
    print("=" * 60)
    print(f"\n  {'Threshold':>12}  {'Macro-F1':>9}  {'F1+':>7}  {'F1-':>7}  {'Prec':>7}  {'Recall':>7}")
    print("  " + "-" * 55)
    for _, row in results_df.iterrows():
        marker = " << OPTIMAL" if row["threshold"] == best_row["threshold"] else (
                 " << Varsayilan" if row["threshold"] == 0.5 else "")
        print(f"  {row['threshold']:>12.2f}  {row['macro_f1']:>9.4f}  "
              f"{row['f1_positive']:>7.4f}  {row['f1_negative']:>7.4f}  "
              f"{row['precision']:>7.4f}  {row['recall']:>7.4f}{marker}")

    gain = best_row["macro_f1"] - default_row["macro_f1"]
    print(f"\n  Optimal threshold : {best_row['threshold']}")
    print(f"  Optimal Macro-F1  : {best_row['macro_f1']:.4f}")
    print(f"  Varsayilan (0.5)  : {default_row['macro_f1']:.4f}")
    print(f"  Kazanim           : +{gain:.4f}")

    out_csv = os.path.join(OUTPUT_DIR, "threshold_analizi.csv")
    results_df.to_csv(out_csv, index=False)

    out_md = os.path.join(DOCS_DIR, "threshold_analizi.md")
    write_report(results_df, out_md)

    print(f"\n  CSV  : {out_csv}")
    print(f"  Rapor: {out_md}")
    print("=" * 60)

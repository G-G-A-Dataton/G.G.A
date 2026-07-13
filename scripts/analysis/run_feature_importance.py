"""
run_feature_importance.py
=========================
G.G.A Takımı — Feature Importance Analizi (10 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script LightGBM'in öğrendiği feature önemlerini analiz eder ve şu
soruları yanıtlar:
  1. Hangi feature'lar modele en çok katkı sağlıyor?
  2. Hangi feature'lar sıfıra yakın önem taşıyor (gereksiz)?
  3. Herhangi bir feature veri sızıntısı (data leakage) riski taşıyor mu?

LightGBM iki tür feature importance verir:
  - split   : Feature kaç kez ağaç dalı için kullanıldı?
  - gain    : Feature'ın kullanıldığı dallarda toplam bilgi kazancı ne kadar?
  Gain genellikle daha güvenilirdir.

Çalıştırmak için:
  python run_feature_importance.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# 8 Temmuz tuning sonucuna göre belirlenen en iyi parametreler
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

# Sızıntı riski taşıyabilecek feature'lar için uyarı eşiği
# Bir feature'ın importance oranı bu değerin üzerindeyse dikkat et
LEAKAGE_WARNING_THRESHOLD = 0.50


def compute_importance(X, y):
    """
    5-Fold CV yaparak tüm fold'lardaki feature importance'ları toplar.

    Tek bir fold'un sonucuna güvenmek yerine 5 fold ortalaması alınır.
    Bu, daha stabil ve güvenilir bir önem sıralaması verir.

    Parametreler
    ----------
    X : pd.DataFrame
        Feature matrisi.
    y : pd.Series
        Etiketler (0/1).

    Döndürür
    -------
    pd.DataFrame
        Her feature için gain ve split ortalamaları.
    """
    skf = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)

    # Her fold'un importance'ını biriktir
    gain_list  = []
    split_list = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  Fold {fold}/5 egitiliyor...", end="\r")
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
        gain_list.append(model.feature_importance(importance_type="gain"))
        split_list.append(model.feature_importance(importance_type="split"))

    print("  5 fold tamamlandi.          ")

    # 5 fold ortalaması
    gain_mean  = np.mean(gain_list,  axis=0)
    split_mean = np.mean(split_list, axis=0)

    df = pd.DataFrame({
        "feature"    : FEATURE_COLS,
        "gain_mean"  : gain_mean,
        "split_mean" : split_mean,
    })

    # Normalize edilmiş oran (toplamı 1.0)
    df["gain_ratio"]  = df["gain_mean"]  / df["gain_mean"].sum()
    df["split_ratio"] = df["split_mean"] / df["split_mean"].sum()

    return df.sort_values("gain_mean", ascending=False).reset_index(drop=True)


def check_leakage_risk(importance_df):
    """
    Tek bir feature'ın çok yüksek importance taşıması veri sızıntısına
    işaret edebilir. Bu fonksiyon olası sızıntı risklerini işaretler.

    Mantık:
      Gerçek bir ML probleminde hiçbir feature %50+ gain'e sahip olmamalı.
      Eğer bir feature o kadar dominantsa, muhtemelen label bilgisini içeriyor
      (sızıntı) ya da çok güçlü bir proxy özellik.

    Parametreler
    ----------
    importance_df : pd.DataFrame
        compute_importance() çıktısı.

    Döndürür
    -------
    list of str
        Uyarı mesajları.
    """
    warnings_list = []
    top = importance_df.iloc[0]

    if top["gain_ratio"] > LEAKAGE_WARNING_THRESHOLD:
        warnings_list.append(
            f"[UYARI] '{top['feature']}' tek basina gain'in "
            f"{top['gain_ratio']:.1%}'ini olusturuyor! "
            f"Veri sizintisi riski kontrol edilmeli."
        )

    # Sıfır importance'lı feature'lar
    zero_features = importance_df[importance_df["gain_mean"] == 0]["feature"].tolist()
    if zero_features:
        warnings_list.append(
            f"[BİLGİ] Sifir importance'li feature'lar (cikartiabilir): "
            f"{', '.join(zero_features)}"
        )

    # Çok düşük importance'lı feature'lar (alt %20'si, üst %80'in 1/10'undan az)
    threshold = importance_df["gain_mean"].quantile(0.80) * 0.1
    low_features = importance_df[
        (importance_df["gain_mean"] > 0) &
        (importance_df["gain_mean"] < threshold)
    ]["feature"].tolist()
    if low_features:
        warnings_list.append(
            f"[BİLGİ] Cok dusuk importance'li feature'lar: "
            f"{', '.join(low_features)}"
        )

    return warnings_list


def write_markdown_report(importance_df, warnings_list, path):
    """
    Feature importance sonuçlarını Markdown formatında kaydeder.
    """
    lines = [
        "# Feature Importance Raporu (10 Temmuz)",
        "",
        "**Hazırlayan:** Ömer Faruk Kara  ",
        "**Tarih:** 10 Temmuz 2026  ",
        f"**Feature sayısı:** {len(importance_df)}  ",
        "**Yöntem:** 5-Fold CV ortalama Gain Importance",
        "",
        "---",
        "",
        "## 1. Feature Önem Sıralaması (Gain'e Göre)",
        "",
        "| Sıra | Feature | Gain Ort. | Gain % | Split Ort. | Split % |",
        "|---|---|---|---|---|---|",
    ]

    for i, row in importance_df.iterrows():
        bar = "█" * int(row["gain_ratio"] * 30)
        lines.append(
            f"| {i+1} | `{row['feature']}` | {row['gain_mean']:.0f} | "
            f"{row['gain_ratio']:.1%} {bar} | "
            f"{row['split_mean']:.0f} | {row['split_ratio']:.1%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Bulgular ve Uyarılar",
        "",
    ]

    if warnings_list:
        for w in warnings_list:
            if "UYARI" in w:
                lines.append(f"> [!WARNING]  \n> {w}")
            else:
                lines.append(f"> [!NOTE]  \n> {w}")
            lines.append("")
    else:
        lines.append("> [!NOTE]  \n> Dikkat çekici bir bulgu yok. Tum feature'lar saglikli gorunuyor.")

    lines += [
        "",
        "---",
        "",
        "## 3. Öneri",
        "",
        "### Tutulması Gereken (Top feature'lar)",
    ]

    top5 = importance_df.head(5)["feature"].tolist()
    for f in top5:
        lines.append(f"- `{f}`")

    # Çıkarılabilir feature'lar
    bottom = importance_df[importance_df["gain_ratio"] < 0.01]["feature"].tolist()
    if bottom:
        lines += [
            "",
            "### Çıkarılabilir veya İzlenecek Düşük Önemli Feature'lar",
        ]
        for f in bottom:
            lines.append(f"- `{f}` (gain < %1)")

    lines += [
        "",
        "---",
        "",
        "## 4. Sonraki Adımlar",
        "",
        "- Düşük önemli feature'lar Sprint 2'de çıkarılarak model yeniden eğitilebilir",
        "- `tfidf_cosine` — TF-IDF'in 10K unigram konfigürasyonuyla yeniden ölçülmeli",
        "- Embedding cosine feature eklendikten sonra bu analiz tekrarlanmalı (12 Temmuz)",
        "",
        f"*Ham CSV: `outputs/feature_importance.csv`*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Feature Importance Analizi (10 Temmuz)")
    print("=" * 60)

    # 1. Veri hazırlama
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
    print(f"  {len(merged):,} satir, {len(FEATURE_COLS)} feature hazir")

    # 2. Feature importance hesapla
    print("\n[3/3] 5-Fold importance hesaplaniyor...")
    importance_df = compute_importance(X, y)

    # 3. Sızıntı kontrolü
    warnings_list = check_leakage_risk(importance_df)

    # 4. Sonuçları yazdır
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE SIRASI (Gain)")
    print("=" * 60)
    for i, row in importance_df.iterrows():
        bar = "#" * int(row["gain_ratio"] * 25)
        print(f"  {i+1:2}. {row['feature']:<28} {row['gain_mean']:8.0f}  {row['gain_ratio']:.1%}  {bar}")

    if warnings_list:
        print("\n  UYARILAR:")
        for w in warnings_list:
            print(f"  {w}")

    # 5. Kaydet
    out_csv = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    importance_df.to_csv(out_csv, index=False)

    out_md = os.path.join(DOCS_DIR, "feature_importance_raporu.md")
    write_markdown_report(importance_df, warnings_list, out_md)

    print(f"\n  CSV    : {out_csv}")
    print(f"  Rapor  : {out_md}")
    print("=" * 60)

"""
run_hata_taksonomisi.py
========================
G.G.A Takimi -- Hatali Tahmin Taksonomisi (12 Temmuz Gorevi)

Ahmet Emin Isın tarafından hazırlanmıştır.

Bu script modelin yanilttigi ornekleri sistematik olarak siniflandirir:

  Hata Turleri:
  1. MARKA HATASI     -- Sorgu bir marka icerir ama tahmin edilen urun farkli markaya ait
  2. KATEGORI HATASI  -- Sorgu ve urun farkli L1 kategorisinde
  3. RENK HATASI      -- Sorguda renk var, urunde farkli renk var
  4. SEMANTIK HATASI  -- Hicbir acik kural ihlali yok ama model yanildi (zor ornekler)

Bu siniflandirma sayesinde:
  - Hangi tur hatalarin ne kadar sik oldugu gorulur
  - Her hata turune ozel feature/strateji onerileri yapilir
  - Rapor icin somut ornekler sunulur

Calistirmak icin:
  python run_hata_taksonomisi.py
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
NEG_RATIO   = 2
RANDOM_SEED = 42
THRESHOLD   = 0.35  # 11 Temmuz threshold analizinden optimal deger

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. OOF Tahmin Uretimi
# ─────────────────────────────────────────────────────────────────────────────

def get_oof_predictions(X, y):
    """5-Fold OOF tahminleri uretir."""
    skf       = get_stratified_kfold(n_splits=5, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  Fold {fold}/5 ...", end="\r")
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval   = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
        model  = lgb.train(
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hata Taksonomisi
# ─────────────────────────────────────────────────────────────────────────────

def classify_error(row):
    """
    Tek bir hatali tahmini siniflandirir.

    Oncelik sirasi: marka > kategori > renk > semantik
    Bir ornek birden fazla hata turune girebilir ama en spesifik sinif atanir.

    Parametreler
    ----------
    row : pd.Series
        Sorgu ve urun bilgilerini iceren satir.

    Dondurur
    -------
    str
        Hata sinifi etiketi.
    """
    query = str(row.get("query", "") or "").lower()
    title = str(row.get("title", "") or "").lower()

    # Marka kontrolu: sorgu markasini iceriyor mu, urun farkli markali mi?
    brand = str(row.get("brand", "") or "").lower().strip()
    if brand and brand in query:
        if brand not in title:
            return "MARKA_HATASI"

    # Kategori kontrolu: L1 kategorisi cakisiyor mu?
    category = str(row.get("category", "") or "")
    cat_l1 = category.split("/")[0].strip().lower() if "/" in category else category.lower()
    query_words = set(query.split())
    # Basit L1 kategori kelimesi sorguda var mi?
    cat_words = set(cat_l1.split())
    # Sorgu kelimelerinde hic kategori ipucu yok ve query_cat_l1_overlap sifirsa
    if row.get("query_cat_l1_overlap", 1) == 0 and len(query_words) > 0:
        # Baska bir kategorinin kelimelerini icerip icermedigi cok karmasik
        # Basit heuristik: L1 overlap tamamen sifir ve query uzunsa muhtemelen kategori hatasi
        if len(query_words) >= 2:
            return "KATEGORI_HATASI"

    # Renk kontrolu: sorguda renk var ama urun farkli renk mi?
    RENKLER = [
        "siyah", "beyaz", "kirmizi", "mavi", "yesil", "sari", "turuncu",
        "mor", "pembe", "gri", "kahverengi", "bej", "lacivert", "haki",
        "black", "white", "red", "blue", "green", "yellow", "pink", "grey",
    ]
    query_color = next((r for r in RENKLER if r in query), None)
    if query_color:
        attrs = str(row.get("attributes", "") or "").lower()
        if attrs and query_color not in attrs and query_color not in title:
            return "RENK_HATASI"

    # Hic kural ihlali yok -> semantik hata (zor ornek)
    return "SEMANTIK_HATASI"


def analyze_errors(merged, oof_preds, threshold):
    """
    FP ve FN orneklerini tespit eder ve siniflandirir.

    Parametreler
    ----------
    merged : pd.DataFrame
        Feature'lar ve ham veri bir arada.
    oof_preds : np.ndarray
        OOF tahmin olasılıkları.
    threshold : float
        Karar esigi.

    Dondurur
    -------
    pd.DataFrame, pd.DataFrame
        (fp_df, fn_df)
    """
    y_true = merged["label"].values
    y_pred = (oof_preds >= threshold).astype(int)

    # FP: model 1 dedi, gercek 0
    fp_mask = (y_pred == 1) & (y_true == 0)
    # FN: model 0 dedi, gercek 1
    fn_mask = (y_pred == 0) & (y_true == 1)

    def enrich(mask, error_type_prefix):
        subset = merged[mask].copy()
        subset["pred_prob"]    = oof_preds[mask]
        subset["error_prefix"] = error_type_prefix
        subset["hata_turu"]    = subset.apply(classify_error, axis=1)
        return subset

    fp_df = enrich(fp_mask, "FP")
    fn_df = enrich(fn_mask, "FN")
    return fp_df, fn_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rapor Uretimi
# ─────────────────────────────────────────────────────────────────────────────

def write_report(fp_df, fn_df, path, threshold):
    """Hata taksonomisi Markdown raporunu kaydeder."""
    fp_counts = fp_df["hata_turu"].value_counts()
    fn_counts = fn_df["hata_turu"].value_counts()
    all_types = ["MARKA_HATASI", "KATEGORI_HATASI", "RENK_HATASI", "SEMANTIK_HATASI"]

    lines = [
        "# Hata Taksonomisi Raporu (12 Temmuz)",
        "",
        "**Hazırlayan:** Ahmet Emin Işın  ",
        "**Tarih:** 12 Temmuz 2026  ",
        f"**Threshold:** {threshold}  ",
        f"**Toplam FP:** {len(fp_df)}  ",
        f"**Toplam FN:** {len(fn_df)}  ",
        "",
        "---",
        "",
        "## 1. Hata Dagilimi",
        "",
        "| Hata Turu | FP Sayisi | FP % | FN Sayisi | FN % |",
        "|---|---|---|---|---|",
    ]

    for ht in all_types:
        fp_n = fp_counts.get(ht, 0)
        fn_n = fn_counts.get(ht, 0)
        fp_p = 100 * fp_n / max(len(fp_df), 1)
        fn_p = 100 * fn_n / max(len(fn_df), 1)
        lines.append(f"| {ht} | {fp_n} | {fp_p:.1f}% | {fn_n} | {fn_p:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 2. Hata Tur Aciklamalari",
        "",
        "### MARKA_HATASI",
        "Sorguda marka gecmesine ragmen model farkli markalı urunu secti.",
        "- **Cozum:** `query_brand_match` feature'i zaten var. Marka odakli hard negative ornekleri artirmak yardimci olabilir.",
        "",
        "### KATEGORI_HATASI",
        "Sorgu ile urun farkli L1 kategorisinde bulunuyor.",
        "- **Cozum:** Kategori L2/L3 feature'lari eklendi (6 Temmuz). Bu hatalarin azalmasi bekleniyor.",
        "",
        "### RENK_HATASI",
        "Sorguda renk bilgisi var ama urun farkli renkte.",
        "- **Cozum:** `query_color_match` feature'i eklendi (8 Temmuz) ancak importance sifir cikti. BM25 hard negative ile renk catismali ornekler uretilirse bu feature aktive olabilir.",
        "",
        "### SEMANTIK_HATASI",
        "Acik kural ihlali yok. Model anlam olarak benzer ama alakasiz urunleri secti.",
        "- **Cozum:** Embedding cosine feature eklenmesi (12 Temmuz, Omer Faruk) bu hatalari azaltabilir.",
        "",
        "---",
        "",
        "## 3. Ornek FP Hatalar (Ilk 5)",
        "",
        "| Sorgu | Urun Basligi | Pred Prob | Hata Turu |",
        "|---|---|---|---|",
    ]

    for _, row in fp_df.nlargest(5, "pred_prob").iterrows():
        q = str(row.get("query", ""))[:40]
        t = str(row.get("title", ""))[:40]
        lines.append(f"| {q} | {t} | {row['pred_prob']:.3f} | {row['hata_turu']} |")

    lines += [
        "",
        "## 4. Ornek FN Hatalar (Ilk 5 — En Dusuk Olasilik)",
        "",
        "| Sorgu | Urun Basligi | Pred Prob | Hata Turu |",
        "|---|---|---|---|",
    ]

    for _, row in fn_df.nsmallest(5, "pred_prob").iterrows():
        q = str(row.get("query", ""))[:40]
        t = str(row.get("title", ""))[:40]
        lines.append(f"| {q} | {t} | {row['pred_prob']:.3f} | {row['hata_turu']} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Oneriler",
        "",
        "1. **Semantik hatalar** en yaygin tur → Embedding cosine feature sprint 3'te kritik",
        "2. **Renk hatalari** icin BM25 hard negative renk catismasi olusturan ornekler uretilmeli",
        "3. **Marka hatalari** azsa model marka sinyalini iyi ogrenmiş demek",
        "",
        f"*Ham CSV: `outputs/hata_taksonomisi_fp.csv`, `outputs/hata_taksonomisi_fn.csv`*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Hata Taksonomisi (12 Temmuz)")
    print(f"  Threshold: {THRESHOLD}")
    print("=" * 60)

    # 1. Veri
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
    print(f"  {len(merged):,} satir hazir")

    X = merged[FEATURE_COLS]
    y = merged["label"]

    # 2. OOF tahminleri
    print("\n[2/3] OOF tahminleri (5-Fold CV)...")
    oof_preds = get_oof_predictions(X, y)

    # 3. Hata analizi
    print(f"\n[3/3] Hata taksonomisi yapiliyor (threshold={THRESHOLD})...")
    fp_df, fn_df = analyze_errors(merged, oof_preds, THRESHOLD)

    fp_counts = fp_df["hata_turu"].value_counts()
    fn_counts = fn_df["hata_turu"].value_counts()

    print("\n" + "=" * 60)
    print("  HATA TAKSONOMISI")
    print("=" * 60)
    print(f"\n  Toplam FP: {len(fp_df)}, Toplam FN: {len(fn_df)}")
    print("\n  Hata Turu              FP     FN")
    print("  " + "-" * 40)
    for ht in ["MARKA_HATASI", "KATEGORI_HATASI", "RENK_HATASI", "SEMANTIK_HATASI"]:
        fp_n = fp_counts.get(ht, 0)
        fn_n = fn_counts.get(ht, 0)
        print(f"  {ht:<22} {fp_n:5}  {fn_n:5}")

    # Kaydet
    fp_df[["query", "title", "brand", "category", "pred_prob", "hata_turu"]].to_csv(
        os.path.join(OUTPUT_DIR, "hata_taksonomisi_fp.csv"), index=False
    )
    fn_df[["query", "title", "brand", "category", "pred_prob", "hata_turu"]].to_csv(
        os.path.join(OUTPUT_DIR, "hata_taksonomisi_fn.csv"), index=False
    )
    out_md = os.path.join(DOCS_DIR, "hata_taksonomisi.md")
    write_report(fp_df, fn_df, out_md, THRESHOLD)

    print(f"\n  Rapor: {out_md}")
    print("=" * 60)

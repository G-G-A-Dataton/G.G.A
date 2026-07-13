"""
src/error_analysis.py
=====================
G.G.A Takımı — Hata Analizi Modülü (6 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu modül, eğitilmiş modelin yaptığı hataları (false positive ve false
negative) analiz ederek modelin nerede başarısız olduğunu anlamaya yarar.

Tanımlar:
  - False Positive (FP): Model "alakalı" dedi, gerçekte "alakasız"
    → Model çok geniş düşünüyor, alakasız ürünleri ilgili sanıyor
  - False Negative (FN): Model "alakasız" dedi, gerçekte "alakalı"
    → Model çok dar düşünüyor, gerçek eşleşmeleri kaçırıyor

Bu analiz sonucunda:
  - En sık hata yapılan kategoriler/markalar belirlenir
  - Feature'ların nerede yetersiz kaldığı görülür
  - Sprint 2'de hangi feature'ların eklenmesi gerektiği planlanır
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. OOF Tahminlerinden Hataları Ayır
# ─────────────────────────────────────────────────────────────────────────────

def split_errors(df, oof_preds, threshold=0.5, predictions=None):
    """
    OOF (Out-Of-Fold) tahminlerini kullanarak doğru ve yanlış sınıflandırmaları ayırır.

    Parametre olarak verilen threshold değerinin üzerindeki tahminler
    "pozitif" (alakalı), altındakiler "negatif" (alakasız) kabul edilir.

    Parametreler
    ----------
    df : pd.DataFrame
        Merge edilmiş ve feature'ları hesaplanmış veri seti.
        'label' kolonu içermeli.
    oof_preds : np.ndarray
        5-Fold CV'den elde edilen out-of-fold olasılık tahminleri (0.0-1.0).
    threshold : float, default=0.5
        Tahminleri 0/1'e çevirmek için eşik değeri.

    Döndürür
    -------
    tuple of pd.DataFrame
        (false_positives, false_negatives, true_positives, true_negatives)
    """
    if "label" not in df.columns or df.empty:
        raise ValueError("df must contain a non-empty label column")
    oof_preds = np.asarray(oof_preds, dtype=np.float64)
    if (
        oof_preds.ndim != 1
        or len(oof_preds) != len(df)
        or not np.isfinite(oof_preds).all()
        or ((oof_preds < 0) | (oof_preds > 1)).any()
    ):
        raise ValueError("oof_preds must be aligned finite probabilities in [0, 1]")
    if predictions is None:
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            threshold_value = np.nan
        if isinstance(threshold, bool) or not np.isfinite(threshold_value) or not (
            0 <= threshold_value <= 1
        ):
            raise ValueError("threshold must be a finite scalar in [0, 1]")
        predictions = (oof_preds >= threshold_value).astype(np.int8)
    else:
        predictions = np.asarray(predictions)
        if predictions.shape != oof_preds.shape or not np.isin(
            predictions, [0, 1]
        ).all():
            raise ValueError("predictions must be aligned binary values")
        predictions = predictions.astype(np.int8, copy=False)
    y = df["label"].to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("label must be binary")

    # Olasılık tahminini binary'e çevir
    preds = predictions

    # Her kategoriyi bir boolean mask olarak tanımla
    # Gerçekte 0, model 1 dedi → False Positive (aşırı geniş tahmin)
    fp_mask = (preds == 1) & (y == 0)
    # Gerçekte 1, model 0 dedi → False Negative (kaçırılan eşleşme)
    fn_mask = (preds == 0) & (y == 1)
    # Gerçekte 1, model 1 dedi → True Positive (doğru eşleşme)
    tp_mask = (preds == 1) & (y == 1)
    # Gerçekte 0, model 0 dedi → True Negative (doğru reddediş)
    tn_mask = (preds == 0) & (y == 0)

    # Her maskeye ek bilgi ekle: tahmin olasılığı ve hata türü
    fp_df = df[fp_mask].copy()
    fn_df = df[fn_mask].copy()
    tp_df = df[tp_mask].copy()
    tn_df = df[tn_mask].copy()
    for subset, mask, error_type in (
        (fp_df, fp_mask, "FP"),
        (fn_df, fn_mask, "FN"),
        (tp_df, tp_mask, "TP"),
        (tn_df, tn_mask, "TN"),
    ):
        subset["proba"] = oof_preds[mask]
        subset["error_type"] = error_type

    return fp_df, fn_df, tp_df, tn_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hata Örüntüsü Analizleri
# ─────────────────────────────────────────────────────────────────────────────

def analyze_error_patterns(fp_df, fn_df, n_top=10):
    """
    False Positive ve False Negative'lerde hangi kategorilerin ve markaların
    en sık hata yaptığını analiz eder.

    Bu analiz şu soruyu yanıtlar:
      "Model hangi tür ürünlerde / hangi kategorilerde sürekli hata yapıyor?"

    Parametreler
    ----------
    fp_df : pd.DataFrame
        False Positive örnekler (model 1 dedi, gerçek 0).
    fn_df : pd.DataFrame
        False Negative örnekler (model 0 dedi, gerçek 1).
    n_top : int
        Her kategori için gösterilecek en sık hata sayısı.

    Döndürür
    -------
    dict
        Analiz sonuçlarını içeren sözlük.
    """
    results = {}

    # Kategorileri L1 seviyesine böl (genel kategori analizi için)
    def get_l1(cat):
        if not isinstance(cat, str):
            return "unknown"
        return cat.split("/")[0].strip()

    if "category" in fp_df.columns and "category" in fn_df.columns:
        fp_df["cat_l1"] = fp_df["category"].apply(get_l1)
        fn_df["cat_l1"] = fn_df["category"].apply(get_l1)

        # FP hataları hangi kategoride yoğunlaşıyor?
        results["fp_by_category"] = (
            fp_df["cat_l1"].value_counts().head(n_top)
        )

        # FN hataları hangi kategoride yoğunlaşıyor?
        results["fn_by_category"] = (
            fn_df["cat_l1"].value_counts().head(n_top)
        )

    # Marka bazlı analiz (marka karışıklığı önemli bir hata kaynağı)
    if "brand" in fp_df.columns and "brand" in fn_df.columns:
        results["fp_by_brand"] = (
            fp_df["brand"].value_counts().head(n_top)
        )
        results["fn_by_brand"] = (
            fn_df["brand"].value_counts().head(n_top)
        )

    # Feature değerlerine göre hata analizi
    # FP'lerin feature dağılımı: hangi feature değerlerinde FP oluşuyor?
    feature_cols = [
        "query_title_overlap", "query_category_overlap",
        "query_brand_match", "tfidf_cosine"
    ]
    available = [
        c for c in feature_cols if c in fp_df.columns and c in fn_df.columns
    ]
    if available:
        results["fp_feature_stats"] = fp_df[available].describe().round(3)
        results["fn_feature_stats"] = fn_df[available].describe().round(3)

    return results


def find_hard_cases(fp_df, fn_df, n=10):
    """
    En "güvenerek yapılan" hataları bulur — modelin çok emin olmasına rağmen
    yanlış yaptığı örnekler. Bunlar en öğretici hatalardır.

    - Güvenli FP: Yüksek olasılıkla 1 dedi ama gerçek 0 (model çok iyimser)
    - Güvenli FN: Düşük olasılıkla 0 dedi ama gerçek 1 (model çok kötümser)

    Parametreler
    ----------
    fp_df, fn_df : pd.DataFrame
        Hata örnekleri (proba kolonu içermeli).
    n : int
        Her kategoriden gösterilecek örnek sayısı.
    """
    cols = ["query", "title", "category", "brand", "proba", "error_type"]
    fp_columns = [c for c in cols if c in fp_df.columns]
    fn_columns = [c for c in cols if c in fn_df.columns]

    # Güvenli FP: olasılık > 0.8 iken gerçek etiket 0
    confident_fp = (
        fp_df.nlargest(n, "proba")[fp_columns]
        if len(fp_df) > 0
        else pd.DataFrame(columns=fp_columns)
    )

    # Güvenli FN: olasılık < 0.2 iken gerçek etiket 1
    confident_fn = (
        fn_df.nsmallest(n, "proba")[fn_columns]
        if len(fn_df) > 0
        else pd.DataFrame(columns=fn_columns)
    )

    return confident_fp, confident_fn


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ana Rapor Üreticisi
# ─────────────────────────────────────────────────────────────────────────────

def generate_error_report(
    df, oof_preds, threshold=0.5, output_path=None, predictions=None
):
    """
    Tam hata analizi raporu üretir ve opsiyonel olarak dosyaya kaydeder.

    Parametreler
    ----------
    df : pd.DataFrame
        Merge + feature edilmiş veri seti.
    oof_preds : np.ndarray
        OOF tahminleri.
    threshold : float
        Karar eşiği.
    output_path : str or None
        Rapor kaydedilecek Markdown dosyası yolu.
    """
    print("=" * 60)
    print("  HATA ANALIZI RAPORU")
    print("=" * 60)

    # Hataları ayır
    fp_df, fn_df, tp_df, tn_df = split_errors(
        df, oof_preds, threshold, predictions=predictions
    )

    n = len(df)
    print(f"\n  Toplam ornek       : {n:,}")
    print(f"  True  Positive (TP): {len(tp_df):,}  ({len(tp_df)/n:.1%})")
    print(f"  True  Negative (TN): {len(tn_df):,}  ({len(tn_df)/n:.1%})")
    print(f"  False Positive (FP): {len(fp_df):,}  ({len(fp_df)/n:.1%})  <- Model 1 dedi, gercek 0")
    print(f"  False Negative (FN): {len(fn_df):,}  ({len(fn_df)/n:.1%})  <- Model 0 dedi, gercek 1")

    # Örüntü analizi
    patterns = analyze_error_patterns(fp_df, fn_df)

    if "fp_by_category" in patterns:
        print("\n  En cok FP yapilan kategoriler (L1):")
        for cat, cnt in patterns["fp_by_category"].items():
            print(f"    {cat:<30} {cnt:>4} FP")

        print("\n  En cok FN yapilan kategoriler (L1):")
        for cat, cnt in patterns["fn_by_category"].items():
            print(f"    {cat:<30} {cnt:>4} FN")

    # En güvenli hatalar
    conf_fp, conf_fn = find_hard_cases(fp_df, fn_df, n=5)

    print("\n  En guvenli 5 False Positive (model cok emin, ama yanlis):")
    if len(conf_fp) > 0:
        for _, row in conf_fp.iterrows():
            q = str(row.get("query", ""))[:35]
            t = str(row.get("title", ""))[:35]
            print(f"    proba={row['proba']:.3f}  query='{q}'  title='{t}'")
    else:
        print("    (Yeterli FP yok)")

    print("\n  En guvenli 5 False Negative (model cok emin, ama yanlis):")
    if len(conf_fn) > 0:
        for _, row in conf_fn.iterrows():
            q = str(row.get("query", ""))[:35]
            t = str(row.get("title", ""))[:35]
            print(f"    proba={row['proba']:.3f}  query='{q}'  title='{t}'")
    else:
        print("    (Yeterli FN yok)")

    # Markdown raporu kaydet
    if output_path:
        _write_markdown_report(
            fp_df, fn_df, tp_df, tn_df, patterns, output_path, threshold
        )
        print(f"\n  Rapor kaydedildi: {output_path}")

    print("=" * 60)
    return {"fp": fp_df, "fn": fn_df, "tp": tp_df, "tn": tn_df, "patterns": patterns}


def _write_markdown_report(fp_df, fn_df, tp_df, tn_df, patterns, path, threshold):
    """Hata analizi sonuçlarını Markdown formatında dosyaya yazar."""
    n = len(fp_df) + len(fn_df) + len(tp_df) + len(tn_df)
    lines = [
        "# Grouped OOF Error Analysis",
        "",
        f"**Threshold:** {threshold}",
        "",
        "---",
        "",
        "## 1. Genel İstatistikler",
        "",
        f"| Kategori | Sayı | Oran |",
        f"|---|---|---|",
        f"| True Positive (TP) | {len(tp_df):,} | {len(tp_df)/n:.1%} |",
        f"| True Negative (TN) | {len(tn_df):,} | {len(tn_df)/n:.1%} |",
        f"| **False Positive (FP)** | **{len(fp_df):,}** | **{len(fp_df)/n:.1%}** |",
        f"| **False Negative (FN)** | **{len(fn_df):,}** | **{len(fn_df)/n:.1%}** |",
        "",
        "> **FP**: Model \"alakalı\" dedi, gerçekte \"alakasız\" → Aşırı geniş tahmin  ",
        "> **FN**: Model \"alakasız\" dedi, gerçekte \"alakalı\" → Kaçırılan eşleşme",
        "",
        "---",
        "",
        "## 2. Kategori Bazlı Hata Analizi",
        "",
    ]

    if "fp_by_category" in patterns:
        lines += ["### En Çok FP Yapılan Kategoriler (L1)", ""]
        lines += ["| Kategori | FP Sayısı |", "|---|---|"]
        for cat, cnt in patterns["fp_by_category"].items():
            lines.append(f"| {cat} | {cnt} |")

        lines += ["", "### En Çok FN Yapılan Kategoriler (L1)", ""]
        lines += ["| Kategori | FN Sayısı |", "|---|---|"]
        for cat, cnt in patterns["fn_by_category"].items():
            lines.append(f"| {cat} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Interpretation Guardrails",
        "",
        "Counts show where errors concentrate, but do not establish causality. Compare normalized error rates against category and brand support before changing features or sampling.",
    ]

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # Rapor üretmek için grouped OOF tahminleri elde edilmiş olmalıdır.
    # Bu script doğrudan çalıştırılırsa kısa bir demo gösterir.
    print("[error_analysis] Bu modul dogrudan import edilerek kullanilir.")
    print("Ornek kullanim:")
    print("  from src.error_analysis import generate_error_report")
    print("  generate_error_report(merged_df, oof_preds, threshold=0.45,")
    print("      output_path='docs/hata_analizi_notlari.md')")

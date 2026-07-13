"""
src/train_mix_v2.py
===================
G.G.A Takımı — Karışık Negatif Eğitim Pipeline'ı (9 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Neden karışık negatif (mix)?
  BM25 hard negative güçlü ama bir eksiği var:
  Sorgunun BM25 top-N adayları arasında yeterli pozitif-olmayan bulunamazsa
  (nadir sorgu, çok kısa metin gibi) o sorgu için ratio'dan az hard negative
  üretilir — veri dengesizliği oluşur.

  train_mix_v2 bunu çözer:
  ─ Önce BM25 hard negative üret (top_n=50, ratio=3)
  ─ Eksik kalan (ratio'ya ulaşamayan) sorguların kotasını random ile doldur
  ─ Sonuç: her pozitif çift için tam 3 negatif, toplamda %0 veri kaybı

  Bu "hibrit" strateji:
  ─ Modeli zor örneklerle eğitiyor (BM25 katkısı)
  ─ Veri dengesini koruyor (random dolgu katkısı)
  ─ 7 Temmuz'daki run_hard_neg_comparison.py'de görulen BM25 kısıtını aşıyor

Referans:
  bm25_hard_negative.py docstring: "Random ve hard negative'i karıştırıp
  eksikleri random ile tamamlamak Gün 9 görevi (train_mix_v2) kapsamına giriyor"
"""

import numpy as np
import pandas as pd

from src.bm25_hard_negative import generate_bm25_hard_negatives
from src.negative_sampling  import generate_random_negatives, verify_no_leakage


def build_mixed_training_set(
    train_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    ratio: int = 3,
    bm25_top_n: int = 50,
    bm25_max_df_ratio: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
    positive_reference_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    BM25 hard negative + random fallback karışık eğitim seti üretir.

    Strateji:
      1. BM25 ile her pozitif çift için ratio kadar hard negative üret
      2. Bazı sorguların kotası dolmayabilir (BM25 yeterli aday bulamazsa)
      3. Eksik olan kısımlar için random negative üret (aynı term_id'ler için)
      4. Pozitifler + (BM25 hard neg + random dolgu) birleştirilerek tam set döner

    Parametreler
    ----------
    train_df : pd.DataFrame
        Pozitif çiftler (term_id, item_id). training_pairs.csv'den.
    terms_df : pd.DataFrame
        Sorgular (term_id, query).
    items_df : pd.DataFrame
        Ürün kataloğu.
    ratio : int, default=3
        Her pozitif çift için hedef negatif sayısı.
    bm25_top_n : int, default=50
        BM25'in her sorgu için değerlendireceği aday sayısı.
    bm25_max_df_ratio : float, default=0.15
        BM25 indeksinde görmezden gelinecek aşırı yaygın kelimelerin oranı.
    random_state : int, default=42
        Tekrar üretilebilirlik için seed.
    verbose : bool, default=True
        İlerleme bilgisi yazdır.
    positive_reference_df : pd.DataFrame, optional
        Negatiflerden dışlanacak tüm bilinen pozitif çiftler.

    Döndürür
    -------
    pd.DataFrame
        Karıştırılmış pozitif + negatif çiftler.
        Kolonlar: term_id, item_id, label, neg_source (bm25 / random / positive)
    """
    if not isinstance(ratio, int) or ratio <= 0:
        raise ValueError(f"ratio must be a positive integer, got {ratio}")
    if train_df.empty:
        raise ValueError("train_df must contain at least one positive pair")
    positive_reference_df = (
        train_df if positive_reference_df is None else positive_reference_df
    )

    if verbose:
        print("=" * 60)
        print("  train_mix_v2 — Karma Negatif Eğitim Seti Üretimi")
        print("=" * 60)
        print(f"  Pozitif çift: {len(train_df):,} | Ratio: {ratio}:1")
        print(f"  Hedef toplam: {len(train_df) * (ratio + 1):,}")

    # ─── 1. BM25 Hard Negative Üret ──────────────────────────────────────────
    if verbose:
        print("\n[1/4] BM25 hard negative üretiliyor...")
    bm25_neg = generate_bm25_hard_negatives(
        train_df=train_df,
        terms_df=terms_df,
        items_df=items_df,
        top_n=bm25_top_n,
        ratio=ratio,
        max_df_ratio=bm25_max_df_ratio,
        verbose=verbose,
        positive_reference_df=positive_reference_df,
    )

    # ─── 2. Eksikleri Hesapla ─────────────────────────────────────────────────
    # Her term_id için kaç BM25 negatif üretilebildi?
    bm25_counts = bm25_neg.groupby("term_id").size()

    # Her term'in hedefi, o term'e ait pozitif çift sayısı x ratio'dur.
    target_by_term = train_df.groupby("term_id").size().mul(ratio)
    all_terms = target_by_term.index

    # Eksik: hedef - gerçekleşen (negatif olursa 0'a sabitlenir)
    gerceklesen = bm25_counts.reindex(all_terms, fill_value=0)
    eksik = (target_by_term - gerceklesen).clip(lower=0).astype(int)

    n_eksik_term = (eksik > 0).sum()
    n_eksik_neg  = int(eksik.sum())

    if verbose:
        print(f"\n[2/4] Eksik analizi:")
        print(f"  BM25 üretilen  : {len(bm25_neg):,} negatif")
        print(f"  Hedef          : {int(target_by_term.sum()):,} negatif")
        print(f"  Eksik          : {n_eksik_neg:,} negatif ({n_eksik_term:,} sorgu için)")

    # ─── 3. Eksik Kısımları Random ile Doldur ────────────────────────────────
    if n_eksik_neg > 0:
        if verbose:
            print(f"\n[3/4] {n_eksik_neg:,} eksik negatifi random ile dolduruluyor...")

        missing_terms = np.repeat(
            eksik[eksik > 0].index.to_numpy(),
            eksik[eksik > 0].to_numpy(),
        )
        quota_df = pd.DataFrame(
            {"term_id": missing_terms, "item_id": pd.NA}, dtype="string"
        )
        random_neg = generate_random_negatives(
            train_df=quota_df,
            items_df=items_df,
            ratio=1,
            random_state=random_state,
            verbose=False,
            positive_reference_df=positive_reference_df,
            excluded_pairs_df=bm25_neg,
        )
        random_neg["neg_source"] = "random"
    else:
        if verbose:
            print("\n[3/4] Eksik yok — tum negativler BM25'ten geldi.")
        random_neg = pd.DataFrame(columns=["term_id", "item_id", "label", "neg_source"])

    # ─── 4. Birleştir ve Karıştır ────────────────────────────────────────────
    if verbose:
        print("\n[4/4] Pozitif + BM25 neg + random dolgu birleştiriliyor...")

    # Kaynak etiketleri ekle
    bm25_neg["neg_source"] = "bm25"
    positives = train_df[["term_id", "item_id"]].copy()
    positives["label"]      = 1
    positives["neg_source"] = "positive"

    all_parts = [positives, bm25_neg]
    if len(random_neg) > 0:
        all_parts.append(random_neg[["term_id", "item_id", "label", "neg_source"]])

    full_df = pd.concat(all_parts, ignore_index=True)
    full_df  = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    negatives = full_df[full_df["label"] == 0]
    expected_negative_count = len(train_df) * ratio
    if len(negatives) != expected_negative_count:
        raise RuntimeError(
            f"Negative quota mismatch: expected {expected_negative_count}, "
            f"got {len(negatives)}"
        )
    if negatives.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("Mixed negative set contains duplicate term-item pairs")
    expected_by_term = target_by_term.sort_index()
    actual_by_term = negatives.groupby("term_id").size().reindex(
        expected_by_term.index, fill_value=0
    )
    if not actual_by_term.equals(expected_by_term):
        raise RuntimeError("Per-term negative quotas were not satisfied")
    positive_pairs = set(
        zip(positive_reference_df["term_id"], positive_reference_df["item_id"])
    )
    negative_pairs = set(zip(negatives["term_id"], negatives["item_id"]))
    if positive_pairs & negative_pairs:
        raise RuntimeError("Mixed negative set overlaps known positive pairs")

    # ─── Özet ────────────────────────────────────────────────────────────────
    if verbose:
        n_pos    = (full_df["label"] == 1).sum()
        n_neg    = (full_df["label"] == 0).sum()
        n_bm25   = (full_df["neg_source"] == "bm25").sum()
        n_rand   = (full_df["neg_source"] == "random").sum()
        bm25_pct = n_bm25 / n_neg * 100 if n_neg else 0
        rand_pct = n_rand / n_neg * 100 if n_neg else 0

        print(f"\n  {'-'*50}")
        print(f"  TRAIN_MIX_V2 SONUCU:")
        print(f"  Pozitif          : {n_pos:>8,}")
        print(f"  BM25 negatif     : {n_bm25:>8,}  ({bm25_pct:.1f}%)")
        print(f"  Random negatif   : {n_rand:>8,}  ({rand_pct:.1f}%)")
        print(f"  TOPLAM           : {len(full_df):>8,}")
        print(f"  Pozitif oran     : {n_pos/len(full_df):.2%}")
        print(f"  {'-'*50}")

    return full_df


def verify_mix_no_leakage(full_df: pd.DataFrame, train_df: pd.DataFrame) -> bool:
    """
    Karışık veri setindeki negatif örneklerin pozitiflerle çakışmadığını doğrular.

    Parametreler
    ----------
    full_df : pd.DataFrame
        build_mixed_training_set() çıktısı.
    train_df : pd.DataFrame
        Orijinal pozitif çiftler.

    Döndürür
    -------
    bool
        True: Sızıntı yok.
    """
    negatives = full_df[full_df["label"] == 0]
    return verify_no_leakage(negatives, train_df)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data import load_terms, load_items

    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    print("Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(data_dir, "terms.csv"))
    items_df  = load_items(os.path.join(data_dir, "items.csv"))
    train_raw = __import__("pandas").read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    # Hızlı test: 500 benzersiz sorgu ile çalış
    sample_terms = train_raw["term_id"].drop_duplicates().sample(n=500, random_state=42)
    sample_train = train_raw[train_raw["term_id"].isin(sample_terms)]
    print(f"Test seti: {len(sample_train):,} pozitif / {len(sample_terms):,} benzersiz sorgu")

    mixed = build_mixed_training_set(
        sample_train, terms_df, items_df,
        ratio=3, bm25_top_n=50, verbose=True,
        positive_reference_df=train_raw,
    )

    print("\nSizinti kontrolu...")
    verify_mix_no_leakage(mixed, train_raw)
    print(f"\nNeg kaynak dagilimi:\n{mixed[mixed['label']==0]['neg_source'].value_counts()}")

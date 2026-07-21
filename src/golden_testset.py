"""
src/golden_testset.py
======================
G.G.A Takımı — Golden Test Set Builder

İnsan doğrulaması için hazır, güvenilir offline evaluation test seti üretir.

Neden gerekli?
Mevcut değerlendirme kör: tüm 3.36M çift için F1 hesaplanıyor.
Golden test set; sorgu bazlı Recall@K, NDCG@K gibi retrieval metriklerini
güvenilir şekilde ölçmek için insan doğrulamalı pozitif/negatif çiftler seti.

Strateji:
  1. training_pairs.csv'den stratified sorgu örneklemesi
  2. Her sorgu için BM25 hard negatifler (zorlu, yanlış etiketleme riski var)
  3. İnsan annotator için okunabilir format: query_text + item_title + bm25_rank
  4. Parquet formatında + SHA-256 manifest

Kullanım:
  >>> from src.golden_testset import build_golden_testset
  >>> df = build_golden_testset(training_pairs, items, terms, n_queries=500)
  >>> df.to_parquet("datasets/golden_testset_v1.parquet", index=False)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# İç Yardımcı: SHA-256
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# İç Yardımcı: BM25 Hard Negatives
# ---------------------------------------------------------------------------

def _get_bm25_hard_negatives(
    term_id: str,
    query_text: str,
    items_df: pd.DataFrame,
    known_positives: set,
    n: int,
    bm25_index,
) -> list[dict]:
    """
    Tek bir sorgu için BM25 hard negatifler üret.
    Bilinen pozitifler dışlanır.
    """
    try:
        top_items = bm25_index.top_n(query_text, n=n * 5)  # Geniş havuz al
    except Exception:
        return []

    results = []
    for rank, item_id in enumerate(top_items, start=1):
        item_id_str = str(item_id)
        if item_id_str in known_positives:
            continue
        results.append({
            "term_id": term_id,
            "item_id": item_id_str,
            "label": 0,
            "bm25_rank": rank,
            "source": "bm25_hard_negative",
        })
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Ana Fonksiyon
# ---------------------------------------------------------------------------

def build_golden_testset(
    training_pairs: pd.DataFrame,
    items_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    n_queries: int = 500,
    negatives_per_query: int = 20,
    seed: int = 42,
    min_positives_per_query: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Human-validation-ready golden test set oluşturur.

    Çıktı her satır için şunları içerir:
      - term_id, item_id, label (1=pozitif, 0=negatif-aday)
      - bm25_rank (BM25 sıralamasındaki konumu; pozitifler için NaN)
      - query_text, item_title, item_category, item_brand
      - source ("training_positive" veya "bm25_hard_negative")

    UYARI: label=0 satırlar insan doğrulaması gerektiriyor!
    BM25 hard negativeler bazen aslında pozitif olabilir (false negative riski).
    Parquet çıktısını annotation tool'una aktararak doğrulayın.

    Parameters
    ----------
    training_pairs : pd.DataFrame
        term_id, item_id, label (sadece pozitifler) içermeli.
    items_df : pd.DataFrame
        items.csv içeriği.
    terms_df : pd.DataFrame
        terms.csv içeriği (term_id, query).
    n_queries : int
        Kaç benzersiz sorgu seçileceği. Varsayılan: 500.
    negatives_per_query : int
        Her sorgu için üretilecek BM25 hard negatif sayısı. Varsayılan: 20.
    seed : int
        Tekrar üretilebilirlik için seed. Varsayılan: 42.
    min_positives_per_query : int
        Bu kadar pozitifi olmayan sorgular atlanır.
    verbose : bool
        İlerleme bilgisi yazdır.

    Returns
    -------
    pd.DataFrame
        Kolonlar: term_id, item_id, label, bm25_rank, source,
                  query_text, item_title, item_category, item_brand
    """
    # ── Giriş doğrulama ──────────────────────────────────────────────────
    for df_name, df, required in [
        ("training_pairs", training_pairs, {"term_id", "item_id"}),
        ("items_df", items_df, {"item_id", "title"}),
        ("terms_df", terms_df, {"term_id", "query"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{df_name}: eksik sütunlar {sorted(missing)}")

    if n_queries <= 0:
        raise ValueError("n_queries pozitif tam sayı olmalı")
    if negatives_per_query <= 0:
        raise ValueError("negatives_per_query pozitif tam sayı olmalı")

    # ── BM25 index oluştur ────────────────────────────────────────────────
    if verbose:
        print("[golden_testset] BM25 index oluşturuluyor...")
    try:
        from src.bm25_hard_negative import BM25Index, standardize_item_text
        item_texts = standardize_item_text(items_df)
        # item_id → index mapping
        item_ids_list = items_df["item_id"].astype(str).tolist()
        bm25_index = BM25Index(item_texts.tolist(), item_ids_list)
        bm25_available = True
        if verbose:
            print(f"  BM25 index hazır: {len(item_ids_list):,} ürün")
    except Exception as exc:
        if verbose:
            print(f"  BM25 index oluşturulamadı: {exc}. Sadece pozitifler kullanılacak.")
        bm25_available = False
        bm25_index = None

    # ── Sorgu seçimi ──────────────────────────────────────────────────────
    # Her term_id için pozitif sayısını hesapla
    pos_counts = (
        training_pairs.groupby("term_id").size()
        .rename("pos_count")
        .reset_index()
    )
    eligible = pos_counts[pos_counts["pos_count"] >= min_positives_per_query]

    if len(eligible) == 0:
        raise ValueError(
            f"min_positives_per_query={min_positives_per_query} koşulunu "
            f"sağlayan sorgu yok"
        )

    rng = np.random.default_rng(seed)
    actual_n = min(n_queries, len(eligible))
    selected_ids = rng.choice(eligible["term_id"].values, size=actual_n, replace=False)

    if verbose:
        print(f"  Seçilen sorgu sayısı: {actual_n:,} / {len(eligible):,} uygun sorgu")

    # ── Yardımcı lookup'lar ───────────────────────────────────────────────
    terms_lookup = terms_df.set_index("term_id")["query"].to_dict()

    items_lookup = items_df.set_index("item_id").apply(
        lambda row: {
            "title": str(row.get("title", "")),
            "category": str(row.get("category", "")),
            "brand": str(row.get("brand", "")),
        },
        axis=1,
    ).to_dict()

    # term_id → set(positive item_ids)
    positive_index: dict = (
        training_pairs.groupby("term_id")["item_id"]
        .apply(lambda x: set(x.astype(str).tolist()))
        .to_dict()
    )

    # ── Satır toplama ─────────────────────────────────────────────────────
    all_rows: list[dict] = []

    for i, term_id in enumerate(selected_ids):
        if verbose and (i + 1) % 50 == 0:
            print(f"  İşlenen sorgu: {i+1:,}/{actual_n:,}")

        term_id_str = str(term_id)
        query_text = str(terms_lookup.get(term_id_str, ""))
        known_positives = positive_index.get(term_id_str, set())

        # Pozitif satırlar
        for item_id in known_positives:
            item_info = items_lookup.get(str(item_id), {})
            all_rows.append({
                "term_id": term_id_str,
                "item_id": str(item_id),
                "label": 1,
                "label_status": "known_positive",
                "bm25_rank": float("nan"),
                "source": "training_positive",
                "query_text": query_text,
                "item_title": item_info.get("title", ""),
                "item_category": item_info.get("category", ""),
                "item_brand": item_info.get("brand", ""),
            })

        # BM25 hard negativeler
        if bm25_available and bm25_index is not None:
            hard_negs = _get_bm25_hard_negatives(
                term_id_str,
                query_text,
                items_df,
                known_positives,
                negatives_per_query,
                bm25_index,
            )
            for neg in hard_negs:
                item_id_str = neg["item_id"]
                item_info = items_lookup.get(item_id_str, {})
                all_rows.append({
                    "term_id": term_id_str,
                    "item_id": item_id_str,
                    "label": 0,
                    "label_status": "unverified_candidate",
                    "bm25_rank": float(neg["bm25_rank"]),
                    "source": "bm25_hard_negative",
                    "query_text": query_text,
                    "item_title": item_info.get("title", ""),
                    "item_category": item_info.get("category", ""),
                    "item_brand": item_info.get("brand", ""),
                })

    result_df = pd.DataFrame(all_rows)
    if len(result_df) == 0:
        raise RuntimeError("Golden test set boş üretildi — veri yükleme sorununu kontrol edin")

    if verbose:
        n_pos = int((result_df["label"] == 1).sum())
        n_neg = int((result_df["label"] == 0).sum())
        n_q = result_df["term_id"].nunique()
        print(f"\n  Golden test set özeti:")
        print(f"    Sorgu sayısı     : {n_q:,}")
        print(f"    Pozitif satır    : {n_pos:,}")
        print(f"    Negatif aday     : {n_neg:,}")
        print(f"    Toplam satır     : {len(result_df):,}")
        print("  [!] label=0 satırlar insan doğrulaması gerektirir!")

    return result_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Manifest Yardımcısı
# ---------------------------------------------------------------------------

def write_golden_testset_manifest(
    parquet_path: str,
    n_queries: int,
    negatives_per_query: int,
    seed: int,
    source_hashes: dict,
) -> str:
    """
    Golden test set parquet dosyası için SHA-256 manifest yazar.

    Returns
    -------
    str
        Manifest dosyasının yolu.
    """
    base, _ = os.path.splitext(parquet_path)
    manifest_path = f"{base}_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "data_file": os.path.basename(parquet_path),
        "sha256": _sha256_file(parquet_path),
        "parameters": {
            "n_queries": n_queries,
            "negatives_per_query": negatives_per_query,
            "seed": seed,
        },
        "source_hashes": source_hashes,
        "warning": (
            "label=0 rows are BM25 hard negative CANDIDATES. "
            "Human validation required before use as ground truth."
        ),
        "ground_truth_status": "provisional: only known positives are verified; "
        "all generated negatives require human review before ranking metrics are reported",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path

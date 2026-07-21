"""
src/data_quality.py
===================
G.G.A Takımı — Veri Kalite Kontrol Modülü

Eğitim verisinin kalitesini ölçen fonksiyonlar:
  - Duplicate query tespiti (farklı term_id, aynı query)
  - Attribute coverage analizi (renk/beden/materyal parse oranı)
  - Label tutarlılık doğrulaması (aynı çift, çelişen etiket)
  - Metadata normalizasyon raporu (gender, age_group, brand)

Kullanım:
  >>> from src.data_quality import run_full_quality_report
  >>> report = run_full_quality_report(terms_df, items_df, training_pairs_df)
  >>> print(report["duplicate_queries"]["count"])
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Duplicate Query Tespiti
# ---------------------------------------------------------------------------

def detect_duplicate_queries(terms_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Farklı term_id'ye sahip ama aynı query metnine sahip terimleri tespit eder.

    Neden önemli?
    Training verisinde aynı sorgu farklı term_id ile görünüyorsa,
    StratifiedGroupKFold her birini farklı grup sayar → validation leakage.

    Parameters
    ----------
    terms_df : pd.DataFrame
        En az 'term_id' ve 'query' kolonlarını içermeli.

    Returns
    -------
    dict
        count       : Duplicate query metni sayısı
        total_affected_terms : Kaç term_id etkilendi
        examples    : İlk 10 duplicate örneği (query, term_ids)
        details_df  : Tüm duplicate'lerin DataFrame'i (yoksa boş)
    """
    required = {"term_id", "query"}
    missing = required - set(terms_df.columns)
    if missing:
        raise ValueError(f"detect_duplicate_queries: eksik sütunlar {sorted(missing)}")

    normalized = terms_df.copy()
    normalized["_query_norm"] = (
        terms_df["query"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    grouped = normalized.groupby("_query_norm")["term_id"].agg(list).reset_index()
    duplicates = grouped[grouped["term_id"].apply(len) > 1]

    affected_term_ids = sum(len(ids) for ids in duplicates["term_id"])
    examples = [
        {"query": row["_query_norm"], "term_ids": row["term_id"]}
        for _, row in duplicates.head(10).iterrows()
    ]

    # Detay DataFrame
    if len(duplicates) > 0:
        rows = []
        for _, row in duplicates.iterrows():
            for tid in row["term_id"]:
                rows.append({"query_normalized": row["_query_norm"], "term_id": tid})
        details_df = pd.DataFrame(rows)
    else:
        details_df = pd.DataFrame(columns=["query_normalized", "term_id"])

    return {
        "count": int(len(duplicates)),
        "total_affected_terms": int(affected_term_ids),
        "examples": examples,
        "details_df": details_df,
    }


# ---------------------------------------------------------------------------
# 2. Attribute Coverage Analizi
# ---------------------------------------------------------------------------

_ATTR_PATTERNS = {
    "renk": re.compile(r"renk\s*[:\|]?\s*\w+", re.IGNORECASE),
    "beden": re.compile(r"(?:beden|numara|size)\s*[:\|]?\s*[\w,]+", re.IGNORECASE),
    "materyal": re.compile(r"(?:materyal|kumaş|malzeme)\s*[:\|]?\s*\w+", re.IGNORECASE),
}


def detect_attribute_coverage(items_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Her kategori L1 için attribute sütununun parse edilebilir oranını hesaplar.

    Neden önemli?
    Attribute feature'ları (color, size, material) items.csv'deki attributes
    sütununun kalitesine bağlı. Hangi kategorilerde coverage düşük bilmek
    feature engineering önceliklendirmesi için kritik.

    Parameters
    ----------
    items_df : pd.DataFrame
        En az 'attributes' ve 'category' kolonlarını içermeli.

    Returns
    -------
    dict
        overall_non_null_rate : Tüm katalogda attributes != null oranı
        color_coverage        : Renk attribute parse edilebilirlik oranı
        size_coverage         : Beden attribute parse edilebilirlik oranı
        material_coverage     : Materyal attribute parse edilebilirlik oranı
        by_category_l1        : Her L1 kategorisi için oranlar
    """
    required = {"attributes", "category"}
    missing = required - set(items_df.columns)
    if missing:
        raise ValueError(f"detect_attribute_coverage: eksik sütunlar {sorted(missing)}")

    attrs = items_df["attributes"].astype(str).fillna("")
    non_null_mask = (attrs != "") & (attrs != "nan") & (attrs != "None")
    overall_non_null = float(non_null_mask.mean())

    # Global coverage
    coverage = {}
    for attr_name, pattern in _ATTR_PATTERNS.items():
        has_attr = attrs.str.contains(pattern, regex=True, na=False)
        coverage[f"{attr_name}_coverage"] = float(has_attr.mean())

    # L1 kategorisi bazında
    items_df = items_df.copy()
    items_df["_cat_l1"] = items_df["category"].astype(str).str.split("/").str[0].str.strip()

    by_category: Dict[str, dict] = {}
    for cat_l1, group in items_df.groupby("_cat_l1", sort=True):
        if len(group) < 10:  # Çok küçük kategori atla
            continue
        group_attrs = group["attributes"].astype(str).fillna("")
        cat_data: dict = {
            "n_items": int(len(group)),
            "non_null_rate": float(
                ((group_attrs != "") & (group_attrs != "nan") & (group_attrs != "None")).mean()
            ),
        }
        for attr_name, pattern in _ATTR_PATTERNS.items():
            cat_data[f"{attr_name}_rate"] = float(
                group_attrs.str.contains(pattern, regex=True, na=False).mean()
            )
        by_category[str(cat_l1)] = cat_data

    return {
        "overall_non_null_rate": overall_non_null,
        **coverage,
        "by_category_l1": by_category,
    }


# ---------------------------------------------------------------------------
# 3. Label Tutarlılık Doğrulaması
# ---------------------------------------------------------------------------

def validate_label_consistency(training_pairs: pd.DataFrame) -> Dict[str, Any]:
    """
    Aynı (term_id, item_id) çiftinin çelişen etiket alıp almadığını kontrol eder.

    Yarışma verisinde yalnızca pozitif (label=1) çiftler var.
    Bu fonksiyon sentetik negatifler eklendikten SONRA çalıştırılmalıdır.

    Parameters
    ----------
    training_pairs : pd.DataFrame
        En az 'term_id', 'item_id', 'label' kolonlarını içermeli.

    Returns
    -------
    dict
        violations    : Çelişen label sayısı (0 olmalı ideal)
        duplicate_pairs : Aynı çiftin birden fazla kez geçmesi
        examples      : İlk 5 ihlal örneği
    """
    required = {"term_id", "item_id", "label"}
    missing = required - set(training_pairs.columns)
    if missing:
        raise ValueError(f"validate_label_consistency: eksik sütunlar {sorted(missing)}")

    # Aynı çiftin farklı label aldığı satırlar
    pair_labels = (
        training_pairs.groupby(["term_id", "item_id"])["label"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    # Çelişki: min != max (aynı çift hem 0 hem 1 label'a sahip)
    conflicts = pair_labels[pair_labels["min"] != pair_labels["max"]]
    # Duplicate çiftler (birden fazla kez geçiyor ama aynı label)
    duplicates = pair_labels[pair_labels["count"] > 1]

    examples = conflicts.head(5).to_dict("records") if len(conflicts) > 0 else []

    return {
        "violations": int(len(conflicts)),
        "duplicate_pairs": int(len(duplicates)),
        "total_pairs": int(len(pair_labels)),
        "examples": examples,
        "is_consistent": len(conflicts) == 0,
    }


# ---------------------------------------------------------------------------
# 4. Metadata Normalizasyon Raporu
# ---------------------------------------------------------------------------

def metadata_normalization_report(items_df: pd.DataFrame) -> Dict[str, Any]:
    """
    items.csv'deki metadata sütunlarının dağılım ve normalizasyon istatistikleri.

    Hangi değerlerin unknown/null olduğunu ve temizleme önceliğini gösterir.

    Parameters
    ----------
    items_df : pd.DataFrame
        items.csv içeriği.

    Returns
    -------
    dict
        gender_dist      : gender değer dağılımı (normalized)
        age_group_dist   : age_group değer dağılımı
        brand_stats      : brand istatistikleri (unique count, null rate)
        category_depth   : Kategori derinlik dağılımı
    """
    report: Dict[str, Any] = {}

    # Gender dağılımı
    if "gender" in items_df.columns:
        gender_counts = (
            items_df["gender"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": "unknown", "none": "unknown", "": "unknown"})
            .value_counts(normalize=True)
            .round(4)
        )
        report["gender_dist"] = gender_counts.to_dict()
        report["gender_null_rate"] = float(
            items_df["gender"].isna().mean() +
            (items_df["gender"].astype(str).str.lower().isin(["nan", "none", ""])).mean()
        )

    # Age group dağılımı
    if "age_group" in items_df.columns:
        age_counts = (
            items_df["age_group"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": "unknown", "none": "unknown", "": "unknown"})
            .value_counts(normalize=True)
            .round(4)
        )
        report["age_group_dist"] = age_counts.to_dict()

    # Brand istatistikleri
    if "brand" in items_df.columns:
        brands = items_df["brand"].astype(str)
        report["brand_stats"] = {
            "unique_count": int(brands.nunique()),
            "null_rate": float(brands.isin(["nan", "None", ""]).mean()),
            "top_10_brands": brands.value_counts().head(10).to_dict(),
        }

    # Kategori derinliği
    if "category" in items_df.columns:
        depths = items_df["category"].astype(str).str.split("/").apply(len)
        report["category_depth"] = {
            "mean": float(depths.mean()),
            "min": int(depths.min()),
            "max": int(depths.max()),
            "distribution": depths.value_counts().sort_index().to_dict(),
        }

    return report


# ---------------------------------------------------------------------------
# 5. Tam Rapor — Tüm Kontrolleri Tek Seferde Çalıştırır
# ---------------------------------------------------------------------------

def run_full_quality_report(
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    training_pairs_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Tüm veri kalite kontrollerini çalıştırır ve özet rapor üretir.

    Parameters
    ----------
    terms_df : pd.DataFrame
    items_df : pd.DataFrame
    training_pairs_df : pd.DataFrame, optional
        None ise label consistency kontrolü atlanır.
    verbose : bool
        Sonuçları yazdır.

    Returns
    -------
    dict
        duplicate_queries, attribute_coverage, label_consistency,
        metadata_normalization bölümlerini içeren rapor.
    """
    report: Dict[str, Any] = {}

    if verbose:
        print("\n--------------------------------------------------------")
        print("  Veri Kalite Raporu")
        print("--------------------------------------------------------")

    # 1. Duplicate queries
    try:
        dup = detect_duplicate_queries(terms_df)
        report["duplicate_queries"] = {k: v for k, v in dup.items() if k != "details_df"}
        if verbose:
            print(f"  Duplicate query metni      : {dup['count']:,}")
            print(f"  Etkilenen term_id sayısı   : {dup['total_affected_terms']:,}")
    except Exception as exc:
        report["duplicate_queries"] = {"error": str(exc)}
        if verbose:
            print(f"  Duplicate query: HATA - {exc}")

    # 2. Attribute coverage
    try:
        cov = detect_attribute_coverage(items_df)
        report["attribute_coverage"] = {k: v for k, v in cov.items() if k != "by_category_l1"}
        report["attribute_coverage_by_category"] = cov.get("by_category_l1", {})
        if verbose:
            print(f"  Attribute non-null oranı   : {cov['overall_non_null_rate']:.1%}")
            print(f"  Renk coverage              : {cov.get('renk_coverage', 0):.1%}")
            print(f"  Beden coverage             : {cov.get('beden_coverage', 0):.1%}")
            print(f"  Materyal coverage          : {cov.get('materyal_coverage', 0):.1%}")
    except Exception as exc:
        report["attribute_coverage"] = {"error": str(exc)}
        if verbose:
            print(f"  Attribute coverage: HATA - {exc}")

    # 3. Label consistency (opsiyonel)
    if training_pairs_df is not None:
        try:
            lc = validate_label_consistency(training_pairs_df)
            report["label_consistency"] = lc
            status = "[OK] Tutarlı" if lc["is_consistent"] else f"[X] {lc['violations']} ihlal"
            if verbose:
                print(f"  Label tutarlılığı          : {status}")
        except Exception as exc:
            report["label_consistency"] = {"error": str(exc)}
            if verbose:
                print(f"  Label consistency: HATA - {exc}")

    # 4. Metadata normalization
    try:
        meta = metadata_normalization_report(items_df)
        report["metadata_normalization"] = meta
        if verbose and "gender_null_rate" in meta:
            print(f"  Gender unknown oranı       : {meta['gender_null_rate']:.1%}")
        if verbose and "brand_stats" in meta:
            print(f"  Unique brand sayısı        : {meta['brand_stats']['unique_count']:,}")
    except Exception as exc:
        report["metadata_normalization"] = {"error": str(exc)}
        if verbose:
            print(f"  Metadata normalization: HATA - {exc}")

    if verbose:
        print("--------------------------------------------------------\n")

    return report

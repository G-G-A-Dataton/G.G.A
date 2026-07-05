"""
src/data_quality.py
===================
G.G.A Takımı — Veri Kalite Kontrol Fonksiyonları

Mustafa Mert Çevik tarafından hazırlanmıştır.

Pozitif çiftlerde aşağıdaki kontrolleri yapar:
  1. Tekrar eden satırlar (duplicate) tespiti
  2. Bilinmeyen / eksik item_id veya term_id
  3. Merge sonrasında kaybolan satır kontrolü
"""

import pandas as pd
from src.data import load_terms, load_items, merge_pairs


def check_duplicates(df, id_col="id", pair_cols=("term_id", "item_id")):
    """
    Eğitim çiftlerindeki tekrar eden satırları tespit eder.

    Parametreler
    ----------
    df : pd.DataFrame
        Eğitim çiftleri (training_pairs veya merged).
    id_col : str
        Tekil kimlik kolonu.
    pair_cols : tuple of str
        Eşleşmeyi temsil eden kolon çifti.

    Döndürür
    -------
    dict
        {
          "duplicate_ids": int,         — Tekrar eden id sayısı
          "duplicate_pairs": int,        — Tekrar eden (term_id, item_id) çift sayısı
          "duplicate_pair_examples": df  — İlk 5 tekrar eden çift
        }
    """
    dup_ids = df.duplicated(subset=[id_col]).sum()
    dup_pairs = df.duplicated(subset=list(pair_cols)).sum()
    dup_examples = df[df.duplicated(subset=list(pair_cols), keep=False)].head(5)

    return {
        "duplicate_ids": int(dup_ids),
        "duplicate_pairs": int(dup_pairs),
        "duplicate_pair_examples": dup_examples,
    }


def check_id_coverage(pairs_df, terms_df, items_df):
    """
    Eğitim çiftlerindeki term_id ve item_id'lerin kataloglarda
    karşılığının olup olmadığını kontrol eder.

    Parametreler
    ----------
    pairs_df : pd.DataFrame
        training_pairs.csv içeriği.
    terms_df : pd.DataFrame
        terms.csv içeriği.
    items_df : pd.DataFrame
        items.csv içeriği.

    Döndürür
    -------
    dict
        {
          "unknown_term_ids": int,
          "unknown_item_ids": int,
          "unknown_term_examples": list,
          "unknown_item_examples": list,
        }
    """
    known_terms = set(terms_df["term_id"].tolist())
    known_items = set(items_df["item_id"].tolist())

    pairs_terms = set(pairs_df["term_id"].tolist())
    pairs_items = set(pairs_df["item_id"].tolist())

    unknown_terms = pairs_terms - known_terms
    unknown_items = pairs_items - known_items

    return {
        "unknown_term_ids": len(unknown_terms),
        "unknown_item_ids": len(unknown_items),
        "unknown_term_examples": list(unknown_terms)[:5],
        "unknown_item_examples": list(unknown_items)[:5],
    }


def check_merge_loss(pairs_df, merged_df):
    """
    Merge işleminden önce ve sonraki satır sayılarını karşılaştırır.
    Herhangi bir satır kaybı olup olmadığını ve nedenini raporlar.

    Parametreler
    ----------
    pairs_df : pd.DataFrame
        Merge öncesi ham eğitim çiftleri.
    merged_df : pd.DataFrame
        Merge sonrası birleştirilmiş DataFrame.

    Döndürür
    -------
    dict
        {
          "pairs_count": int,
          "merged_count": int,
          "lost_rows": int,
          "is_clean": bool
        }
    """
    pairs_count  = len(pairs_df)
    merged_count = len(merged_df)
    lost_rows    = pairs_count - merged_count

    return {
        "pairs_count": pairs_count,
        "merged_count": merged_count,
        "lost_rows": lost_rows,
        "is_clean": lost_rows == 0,
    }


def run_full_quality_check(pairs_path, terms_path, items_path, verbose=True):
    """
    Tüm kalite kontrollerini sırayla çalıştırır ve sonuçları özetler.

    Döndürür
    -------
    dict
        Tüm kontrollerin birleşik sonuçları.
    """
    terms_df  = load_terms(terms_path)
    items_df  = load_items(items_path)
    pairs_df  = pd.read_csv(pairs_path, dtype={"id": "string", "term_id": "string",
                                                "item_id": "string", "label": "int8"})
    merged_df = merge_pairs(pairs_path, terms_df, items_df, is_train=True)

    dup_result   = check_duplicates(pairs_df)
    cov_result   = check_id_coverage(pairs_df, terms_df, items_df)
    merge_result = check_merge_loss(pairs_df, merged_df)

    if verbose:
        print("=" * 50)
        print("  VERİ KALİTE KONTROL RAPORU")
        print("=" * 50)
        print(f"\n[1] TEKRAR EDEN SATIRLAR")
        print(f"  Tekrar eden ID         : {dup_result['duplicate_ids']}")
        print(f"  Tekrar eden çift       : {dup_result['duplicate_pairs']}")

        print(f"\n[2] ID KAPSAM KONTROLÜ")
        print(f"  Bilinmeyen term_id     : {cov_result['unknown_term_ids']}")
        print(f"  Bilinmeyen item_id     : {cov_result['unknown_item_ids']}")
        if cov_result["unknown_term_examples"]:
            print(f"  Örnek bilinmeyen term  : {cov_result['unknown_term_examples']}")
        if cov_result["unknown_item_examples"]:
            print(f"  Örnek bilinmeyen item  : {cov_result['unknown_item_examples']}")

        print(f"\n[3] MERGE KAYIP KONTROLÜ")
        print(f"  Pairs satır sayısı     : {merge_result['pairs_count']}")
        print(f"  Merged satır sayısı    : {merge_result['merged_count']}")
        print(f"  Kayıp satır            : {merge_result['lost_rows']}")
        print(f"  Temiz mi?              : {'[TEMIZ]' if merge_result['is_clean'] else '[HATA] Kayip var!'}")
        print("=" * 50)

    return {**dup_result, **cov_result, **merge_result}


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    run_full_quality_check(
        pairs_path=os.path.join(data_dir, "training_pairs.csv"),
        terms_path=os.path.join(data_dir, "terms.csv"),
        items_path=os.path.join(data_dir, "items.csv"),
    )

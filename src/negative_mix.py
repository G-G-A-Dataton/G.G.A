"""
src/negative_mix.py
====================
G.G.A Takımı — Random + Hard Negative Karışımı (Gün 9 görevi: train_mix_v2)

Mustafa Mert Çevik tarafından hazırlanmıştır. (9 Temmuz görevi)

Neden karışım gerekiyor?
  `src/bm25_hard_negative.py` her sorgu için BM25 top_n adayları arasından
  pozitif olmayanları hard negative olarak seçiyor. Ama bazı sorgularda
  (kelimeleri kataloğun çok küçük bir kısmında geçen, nadir sorgular gibi)
  top_n aday havuzunda `ratio` kadar pozitif-olmayan bulunamayabilir — o
  modülün docstring'inde de belirtildiği gibi bu durumda eksik tamamlanmaz.

  Bu modül tam olarak o boşluğu dolduruyor: önce her sorgu için BM25 hard
  negative dener, sonra eksik kalan kısmı random negative ile TAMAMLAR.
  Sonuç: her sorgu için garanti `ratio` adet negatif, mümkün olduğunca
  "zor" (hard) örneklerle, gerekirse "kolay" (random) örneklerle
  tamamlanmış. Çıktıdaki `source` kolonu hangi örneğin nereden geldiğini
  işaretler — böylece Ömer modele hard/random karışım oranının etkisini
  ayrıca inceleyebilir.
"""

import numpy as np
import pandas as pd

from src.bm25_hard_negative import BM25Index, standardize_item_text
from src.negative_sampling import verify_no_leakage


# ─────────────────────────────────────────────────────────────────────────────
# 1. Karışık (hard + random-fill) Negatif Üretici
# ─────────────────────────────────────────────────────────────────────────────

def build_mixed_negative_set(
    train_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    ratio: int = 3,
    top_n: int = 50,
    max_df_ratio: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Her sorgu için tam olarak `ratio` adet negatif üretir: önce BM25 hard
    negative dener, top_n adayları yetmezse random negative ile tamamlar.

    Parametreler
    ----------
    train_df, terms_df, items_df : bkz. bm25_hard_negative / negative_sampling
    ratio : int, default=3
        Sorgu başına hedeflenen (garanti edilen) negatif sayısı.
    top_n : int, default=50
        BM25 ile getirilecek aday sayısı.
    max_df_ratio : float, default=0.15
        BM25Index için yaygın kelime filtresi (bkz. BM25Index docstring'i).
    random_state : int, default=42
        Tekrar üretilebilirlik için seed (sadece random doldurma adımında
        kullanılır — BM25 kısmı deterministiktir).
    verbose : bool, default=True
        İlerleme bilgisi yazdır.

    Döndürür
    -------
    pd.DataFrame
        Kolonlar: term_id, item_id, label (0), source ("hard" / "random_fill")
    """
    if verbose:
        print("[negative_mix] Urun metni standardize ediliyor, BM25 indeks kuruluyor...")
    item_texts = standardize_item_text(items_df).tolist()
    index = BM25Index(items_df["item_id"].to_numpy(), item_texts, max_df_ratio=max_df_ratio)

    pos_by_term = train_df.groupby("term_id")["item_id"].apply(set)
    term_to_query = terms_df.set_index("term_id")["query"]
    unique_terms = train_df["term_id"].unique()

    if verbose:
        print(f"[negative_mix] {len(unique_terms):,} sorgu icin hard negative "
              f"deneniyor (top_n={top_n}, hedef ratio={ratio})...")

    hard_rows = []
    shortfall_terms = []  # eksik kalan her (term_id) icin bir kayit
    for i, term_id in enumerate(unique_terms):
        query = term_to_query.get(term_id, "")
        pos_items = pos_by_term.get(term_id, set())

        added = 0
        for item_id in index.top_n(query, n=top_n):
            if item_id in pos_items:
                continue
            hard_rows.append((term_id, item_id))
            added += 1
            if added >= ratio:
                break

        if added < ratio:
            shortfall_terms.extend([term_id] * (ratio - added))

        if verbose and (i + 1) % 10_000 == 0:
            print(f"  ... {i + 1:,}/{len(unique_terms):,} sorgu islendi")

    hard_df = pd.DataFrame(hard_rows, columns=["term_id", "item_id"])
    hard_df["source"] = "hard"

    if verbose:
        print(f"[negative_mix] {len(hard_df):,} hard negative uretildi, "
              f"{len(shortfall_terms):,} eksik random ile tamamlanacak "
              f"({len(shortfall_terms) / (len(unique_terms) * ratio):.1%} "
              f"toplam hedefin).")

    # Eksikleri random negatif ile tamamla. Yasak kume: hem gercek
    # pozitifler hem de o sorgu icin zaten secilmis hard negatifler
    # (ayni cifti iki kez uretmemek icin).
    pos_keys = set(train_df["term_id"] + "|" + train_df["item_id"])
    hard_keys = set(hard_df["term_id"] + "|" + hard_df["item_id"]) if len(hard_df) else set()
    forbidden_keys = pos_keys | hard_keys

    rng = np.random.default_rng(random_state)
    all_item_ids = items_df["item_id"].to_numpy()
    remaining_terms = np.array(shortfall_terms, dtype=object)

    filled = []
    filled_keys = set()
    tur = 0
    while len(remaining_terms) > 0:
        tur += 1
        adaylar = pd.DataFrame({
            "term_id": remaining_terms,
            "item_id": rng.choice(all_item_ids, size=len(remaining_terms)),
        })
        anahtar = adaylar["term_id"] + "|" + adaylar["item_id"]

        gecerli = (
            ~anahtar.isin(forbidden_keys)
            & ~anahtar.isin(filled_keys)
            & ~anahtar.duplicated()
        )
        filled.append(adaylar[gecerli])
        filled_keys.update(anahtar[gecerli])

        if verbose:
            print(f"  [negative_mix/fill] tur {tur}: {gecerli.sum():,} kabul, "
                  f"kalan {len(remaining_terms) - gecerli.sum():,}")

        remaining_terms = adaylar.loc[~gecerli, "term_id"].to_numpy()

    fill_df = (
        pd.concat(filled, ignore_index=True) if filled
        else pd.DataFrame(columns=["term_id", "item_id"])
    )
    fill_df["source"] = "random_fill"

    mixed = pd.concat([hard_df, fill_df], ignore_index=True)
    mixed["label"] = 0

    if verbose:
        hard_n = (mixed["source"] == "hard").sum()
        fill_n = (mixed["source"] == "random_fill").sum()
        toplam = len(mixed)
        oran = hard_n / toplam if toplam else 0.0
        print(f"[negative_mix] Toplam {toplam:,} negatif: {hard_n:,} hard "
              f"+ {fill_n:,} random_fill ({oran:.1%} hard oran).")

    return mixed[["term_id", "item_id", "label", "source"]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tam Eğitim Seti (Pozitif + Karışık Negatif) — train_mix_v2
# ─────────────────────────────────────────────────────────────────────────────

def build_mixed_training_set(
    train_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    ratio: int = 3,
    top_n: int = 50,
    max_df_ratio: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pozitif çiftlere karışık (hard + random-fill) negatifleri ekleyerek
    `train_mix_v2` eğitim setini oluşturur. `negative_sampling.build_training_set`
    ile aynı desen, negatif kaynağı BM25+random karışımı olması dışında.

    Döndürür
    -------
    pd.DataFrame
        Karıştırılmış pozitif + negatif çiftler.
        Kolonlar: term_id, item_id, label, source ("positive" pozitifler için)
    """
    negatives_df = build_mixed_negative_set(
        train_df, terms_df, items_df,
        ratio=ratio, top_n=top_n, max_df_ratio=max_df_ratio,
        random_state=random_state, verbose=verbose,
    )

    positives_df = train_df[["term_id", "item_id"]].copy()
    positives_df["label"] = 1
    positives_df["source"] = "positive"

    full_df = pd.concat([positives_df, negatives_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if verbose:
        pos_n = (full_df["label"] == 1).sum()
        neg_n = (full_df["label"] == 0).sum()
        print(f"[build_mixed_training_set] Sonuc: {pos_n:,} pozitif + "
              f"{neg_n:,} negatif = {len(full_df):,} toplam "
              f"(pozitif oran {pos_n / len(full_df):.1%})")

    return full_df


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data import load_terms, load_items

    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(data_dir, "terms.csv"))
    items_df = load_items(os.path.join(data_dir, "items.csv"))
    train_df = pd.read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    print(f"Pozitif cift sayisi: {len(train_df):,}")
    print(f"Katalog urun sayisi: {len(items_df):,}")
    print(f"Sorgu sayisi: {len(terms_df):,}")

    RATIO = 3
    SEED = 42
    mixed_train = build_mixed_training_set(
        train_df, terms_df, items_df,
        ratio=RATIO, top_n=50, random_state=SEED, verbose=True,
    )

    print("\nSizinti kontrolu yapiliyor...")
    neg_only = mixed_train[mixed_train["label"] == 0]
    verify_no_leakage(neg_only, train_df)

    out_path = os.path.join(output_dir, f"train_mix_v2_ratio{RATIO}_seed{SEED}.csv")
    mixed_train.to_csv(out_path, index=False)
    print(f"\nKaydedildi: {out_path} ({len(mixed_train):,} satir)")

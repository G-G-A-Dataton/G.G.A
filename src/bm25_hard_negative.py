"""
src/bm25_hard_negative.py
==========================
G.G.A Takımı — BM25 Hard Negative Üretimi (Gün 6-7 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (6-7 Temmuz görevi)

Neden random negative yetmiyor?
  Rastgele seçilen ürünler sorguyla neredeyse hiç kelime paylaşmıyor
  ("spor ayakkabı" sorgusuna karşı rastgele "bebek maması" gibi) — bunlar
  modelin öğrenmesi için çok kolay. BM25 hard negative, sorguyla KELİME
  BENZERLİĞİ yüksek ama pozitif OLMAYAN ürünleri seçer ("spor ayakkabı"
  sorgusuna karşı başka markadan bir spor ayakkabı gibi). Model bu şekilde
  gerçekten alakalı/alakasız ayrımını öğrenmek zorunda kalır.

Neden requirements.txt'deki rank_bm25'i doğrudan kullanmıyoruz?
  rank_bm25.BM25Okapi.get_scores() her çağrıda TÜM kataloğu tarar
  (~963K ürün). Bunu ~50K benzersiz sorgu için tekrarlamak, random
  negative'de düzelttiğimiz "term x katalog" performans tuzağının
  aynısı — pratikte bitmiyor. Bunun yerine ters indeks (inverted index)
  kuruyoruz: her kelimenin hangi ürünlerde geçtiğini önceden hesaplıyoruz.
  Bir sorgu geldiğinde sadece o sorgunun kelimelerini İÇEREN ürünleri
  puanlıyoruz (gerçek arama motorlarının yaptığı gibi) — kataloğun
  tamamını değil. Puanlama formülü (Okapi BM25, k1=1.5, b=0.75) rank_bm25
  ile birebir aynı; değişen sadece hangi belgelerin puanlandığı.

  Ek olarak: bir kelime kataloğun çok büyük bir kısmında geçiyorsa
  ("kadın", "erkek" gibi) IDF'i zaten sıfıra yakın/negatiftir, yani
  puana neredeyse katkısı yoktur ama posting listesi devasa olduğu için
  taraması pahalıdır. `max_df_ratio` ile bu tarz aşırı yaygın kelimeleri
  indeksten (dolayısıyla taramadan) baştan eliyoruz.

Bu modül şunları sağlar:
  1. Ürün metnini standardize etme (title + category + brand)
  2. Ters indeks + BM25 (Okapi) puanlama kurulumu
  3. Her sorgu için top-N BM25 adayından pozitif olmayanları hard
     negative olarak seçme
  4. Sızıntı kontrolü (src.negative_sampling.verify_no_leakage ile ortak)

Not (kapsam sınırı): Bir sorgunun BM25 adayları arasında yeterli sayıda
pozitif-olmayan bulunamazsa (nadir/boş sorgu metni gibi) o sorgu için
`ratio`'dan az hard negative üretilir, eksik tamamlanmaz. Random ve
hard negative'i karıştırıp eksikleri random ile tamamlamak Gün 9
görevi (`train_mix_v2`) kapsamına giriyor, burada değil.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ürün Metni Standardizasyonu
# ─────────────────────────────────────────────────────────────────────────────

def standardize_item_text(items_df: pd.DataFrame) -> pd.Series:
    """
    BM25 indeksi için ürün metnini tek bir alanda birleştirir: title + category + brand.

    tfidf_features.py'deki aynı birleşimi kullanır (kategori patikasındaki
    "/" boşlukla değiştirilir) — böylece BM25 ve TF-IDF aynı kelime
    dağarcığı mantığından besleniyor.
    """
    category_flat = items_df["category"].astype(str).str.replace("/", " ", regex=False)
    return (
        items_df["title"].fillna("") + " " +
        category_flat.fillna("") + " " +
        items_df["brand"].fillna("")
    )


def _tokenize(text) -> list:
    if not text or not isinstance(text, str):
        return []
    return text.lower().split()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ters İndeksli BM25 (Okapi)
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Ters indeksli BM25 (Okapi) puanlayıcı.

    rank_bm25.BM25Okapi ile aynı skor formülünü kullanır ama her sorguyu
    sadece ilgili adaylar üzerinde puanlar (bkz. modül docstring'i —
    "Neden requirements.txt'deki rank_bm25'i doğrudan kullanmıyoruz?").
    """

    def __init__(
        self,
        item_ids: np.ndarray,
        texts: list,
        k1: float = 1.5,
        b: float = 0.75,
        max_df_ratio: float = 0.15,
    ):
        self.item_ids = np.asarray(item_ids)
        self.k1 = k1
        self.b = b

        doc_tokens = [_tokenize(t) for t in texts]
        self.doc_len = np.array([len(toks) for toks in doc_tokens], dtype=np.float64)
        self.n_docs = len(doc_tokens)
        self.avgdl = self.doc_len.mean() if self.n_docs else 0.0

        # Ters indeks: token -> {doc_idx: term_frequency} (henüz filtresiz)
        raw_index = defaultdict(dict)
        for doc_idx, toks in enumerate(doc_tokens):
            tf = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                raw_index[tok][doc_idx] = freq

        # IDF: rank_bm25.BM25Okapi ile birebir aynı formül (ATIRE varyantı,
        # epsilon tabanlı taban değeriyle) — idf = log(N - n + 0.5) - log(n + 0.5).
        # Negatif idf'ler (kelime dokümanların yarısından fazlasında geçiyorsa
        # olur) epsilon * ortalama_idf ile taban değerine sabitlenir. Bu hesap,
        # sonraki adımda taramadan atacağımız yaygın token' lar dahil TÜM
        # token'lar üzerinden yapılır (rank_bm25 ile ayni average_idf için).
        idf, negative_idf_tokens, idf_sum = {}, [], 0.0
        for tok, postings in raw_index.items():
            n = len(postings)
            val = np.log(self.n_docs - n + 0.5) - np.log(n + 0.5)
            idf[tok] = val
            idf_sum += val
            if val < 0:
                negative_idf_tokens.append(tok)
        average_idf = idf_sum / len(idf) if idf else 0.0
        eps = 0.25 * average_idf
        for tok in negative_idf_tokens:
            idf[tok] = eps
        self.idf = idf

        # Taramayı hızlandırmak için indeksten SADECE çok yaygın token'ları
        # atıyoruz (bkz. modül docstring'i) — idf değerleri yukarıda tüm
        # token'lar üzerinden zaten hesaplandı, burada sadece posting
        # listeleri (asıl tarama maliyeti) filtreleniyor.
        max_df = max_df_ratio * self.n_docs if self.n_docs else 0
        self.inverted_index = {
            tok: postings for tok, postings in raw_index.items()
            if len(postings) <= max_df
        }

    def top_n(self, query: str, n: int = 50) -> np.ndarray:
        """Sorgu metnine göre en yüksek BM25 skorlu n ürünün item_id'sini döndürür."""
        candidate_scores = defaultdict(float)

        for tok in _tokenize(query):
            postings = self.inverted_index.get(tok)
            if not postings:
                continue
            idf = self.idf[tok]
            for doc_idx, tf in postings.items():
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                candidate_scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom

        if not candidate_scores:
            return np.array([], dtype=self.item_ids.dtype)

        ranked_doc_idx = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:n]
        return self.item_ids[ranked_doc_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hard Negative Üretimi
# ─────────────────────────────────────────────────────────────────────────────

def generate_bm25_hard_negatives(
    train_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    top_n: int = 50,
    ratio: int = 3,
    max_df_ratio: float = 0.15,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Her benzersiz sorgu (term_id) için BM25 ile en benzer top_n üründen,
    pozitif olmayanların arasından en fazla `ratio` adet hard negative seçer.

    Parametreler
    ----------
    train_df : pd.DataFrame
        Pozitif çiftler (term_id, item_id). training_pairs.csv'den gelir.
    terms_df : pd.DataFrame
        Sorgular (term_id, query). load_terms() ile yüklenir.
    items_df : pd.DataFrame
        Ürün kataloğu. load_items() ile yüklenir.
    top_n : int, default=50
        BM25 ile getirilecek aday sayısı (bu adaylardan pozitif olanlar elenir).
    ratio : int, default=3
        Sorgu başına üretilecek hedef hard negative sayısı (üst sınır; top_n
        adayları arasında yeterli pozitif-olmayan bulunamazsa daha az üretilir).
    max_df_ratio : float, default=0.15
        BM25Index için — kataloğun bu orandan fazlasında geçen kelimeler
        indekslenmez (bkz. BM25Index docstring'i).
    verbose : bool, default=True
        İlerleme bilgisi yazdır.

    Döndürür
    -------
    pd.DataFrame
        Kolonlar: term_id, item_id, label (hepsi 0)
    """
    if verbose:
        print("[bm25_hard_negative] Ürün metni standardize ediliyor...")
    item_texts = standardize_item_text(items_df).tolist()

    if verbose:
        print("[bm25_hard_negative] Ters indeks kuruluyor...")
    index = BM25Index(
        items_df["item_id"].to_numpy(), item_texts, max_df_ratio=max_df_ratio,
    )
    if verbose:
        print(f"[bm25_hard_negative] Indeks hazir: {index.n_docs:,} urun, "
              f"ort. belge uzunlugu {index.avgdl:.1f} kelime, "
              f"{len(index.inverted_index):,} benzersiz token indekslendi.")

    # Her term için pozitif item_id kümesi — hard negative bunlarla asla çakışmamalı.
    pos_by_term = train_df.groupby("term_id")["item_id"].apply(set)
    term_to_query = terms_df.set_index("term_id")["query"]

    unique_terms = train_df["term_id"].unique()
    if verbose:
        print(f"[bm25_hard_negative] {len(unique_terms):,} benzersiz sorgu icin "
              f"BM25 taramasi basliyor (top_n={top_n}, ratio={ratio})...")

    negatives = []
    for i, term_id in enumerate(unique_terms):
        query = term_to_query.get(term_id, "")
        pos_items = pos_by_term.get(term_id, set())

        added = 0
        for item_id in index.top_n(query, n=top_n):
            if item_id in pos_items:
                continue
            negatives.append((term_id, item_id))
            added += 1
            if added >= ratio:
                break

        if verbose and (i + 1) % 5_000 == 0:
            print(f"  ... {i + 1:,}/{len(unique_terms):,} sorgu islendi "
                  f"({len(negatives):,} hard negative uretildi)")

    negatives_df = pd.DataFrame(negatives, columns=["term_id", "item_id"])
    negatives_df["label"] = 0

    if verbose:
        avg_found = len(negatives_df) / len(unique_terms) if len(unique_terms) else 0.0
        # reindex: hic hard negative uretilemeyen (0 satirlik) terimler
        # groupby sonucunda hic gorunmez, reindex ile bunlari da 0 olarak
        # sayima katiyoruz (bkz. notebooks/03_negatif_kalite_mert.py'deki
        # ayni desen).
        term_counts = (
            negatives_df.groupby("term_id").size().reindex(unique_terms, fill_value=0)
            if len(negatives_df) else pd.Series(0, index=unique_terms)
        )
        eksik = (term_counts < ratio).sum()
        print(f"[bm25_hard_negative] Toplam {len(negatives_df):,} hard negative uretildi "
              f"(sorgu basina ort. {avg_found:.2f} / hedef {ratio}).")
        print(f"[bm25_hard_negative] Hedefin altinda kalan sorgu sayisi: {eksik:,} "
              f"(top_n adaylari arasinda yeterli pozitif-olmayan bulunamadi).")

    return negatives_df


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data import load_terms, load_items
    from src.negative_sampling import verify_no_leakage

    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    print("Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(data_dir, "terms.csv"))
    items_df = load_items(os.path.join(data_dir, "items.csv"))
    train_df = pd.read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    # Hızlı deneme: küçük bir sorgu örneğiyle çalış
    sample_terms = train_df["term_id"].drop_duplicates().sample(n=200, random_state=42)
    sample_train = train_df[train_df["term_id"].isin(sample_terms)]

    hard_negatives = generate_bm25_hard_negatives(
        sample_train, terms_df, items_df, top_n=50, ratio=3,
    )

    print("\nSizinti kontrolu yapiliyor...")
    verify_no_leakage(hard_negatives, train_df)

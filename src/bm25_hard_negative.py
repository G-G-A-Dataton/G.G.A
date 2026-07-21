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

from array import array
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.retrieval.bm25 import BM25Index
from src.text_utils import normalize_for_matching


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
    category_flat = items_df["category"].astype("string").str.replace("/", " ", regex=False)
    return (
        items_df["title"].fillna("") + " " +
        category_flat.fillna("") + " " +
        items_df["brand"].fillna("")
    )


def _tokenize(text) -> list:
    if not text or not isinstance(text, str):
        return []
    return normalize_for_matching(text).split()


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
        if not 0.0 < max_df_ratio <= 1.0:
            raise ValueError("max_df_ratio must be in (0, 1]")
        if k1 <= 0 or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 requires k1 > 0 and b in [0, 1]")
        self.item_ids = np.asarray(item_ids)
        if self.item_ids.ndim != 1 or len(self.item_ids) != len(texts):
            raise ValueError("item_ids and texts must be aligned one-dimensional data")
        self.k1 = k1
        self.b = b

        self.n_docs = len(texts)
        self.doc_len = np.empty(self.n_docs, dtype=np.float32)
        document_frequency = Counter()
        for doc_idx, text in enumerate(texts):
            tokens = _tokenize(text)
            self.doc_len[doc_idx] = len(tokens)
            document_frequency.update(set(tokens))
        self.avgdl = self.doc_len.mean() if self.n_docs else 0.0

        # IDF: rank_bm25.BM25Okapi ile birebir aynı formül (ATIRE varyantı,
        # epsilon tabanlı taban değeriyle) — idf = log(N - n + 0.5) - log(n + 0.5).
        # Negatif idf'ler (kelime dokümanların yarısından fazlasında geçiyorsa
        # olur) epsilon * ortalama_idf ile taban değerine sabitlenir. Bu hesap,
        # sonraki adımda taramadan atacağımız yaygın token' lar dahil TÜM
        # token'lar üzerinden yapılır (rank_bm25 ile ayni average_idf için).
        idf, negative_idf_tokens, idf_sum = {}, [], 0.0
        for tok, n in document_frequency.items():
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

        max_df = max_df_ratio * self.n_docs if self.n_docs else 0
        eligible_tokens = {
            token for token, frequency in document_frequency.items()
            if frequency <= max_df
        }
        postings = defaultdict(lambda: (array("I"), array("I")))
        for doc_idx, text in enumerate(texts):
            frequencies = Counter(_tokenize(text))
            for token, frequency in frequencies.items():
                if token in eligible_tokens:
                    postings[token][0].append(doc_idx)
                    postings[token][1].append(frequency)
        self.inverted_index = {
            token: (
                np.frombuffer(documents, dtype=np.uint32),
                np.frombuffer(frequencies, dtype=np.uint32),
            )
            for token, (documents, frequencies) in postings.items()
        }

    def top_n(self, query: str, n: int = 50) -> np.ndarray:
        """Sorgu metnine göre en yüksek BM25 skorlu n ürünün item_id'sini döndürür."""
        if n <= 0:
            raise ValueError("n must be positive")
        document_parts = []
        score_parts = []
        if self.avgdl <= 0:
            return np.array([], dtype=self.item_ids.dtype)
        for tok in _tokenize(query):
            postings = self.inverted_index.get(tok)
            if postings is None:
                continue
            doc_indices, term_frequencies = postings
            idf = self.idf[tok]
            tf = term_frequencies.astype(np.float32, copy=False)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * self.doc_len[doc_indices] / self.avgdl
            )
            document_parts.append(doc_indices)
            score_parts.append(idf * (tf * (self.k1 + 1)) / denominator)

        if not document_parts:
            return np.array([], dtype=self.item_ids.dtype)
        documents = np.concatenate(document_parts)
        contributions = np.concatenate(score_parts)
        unique_documents, inverse = np.unique(documents, return_inverse=True)
        scores = np.zeros(len(unique_documents), dtype=np.float64)
        np.add.at(scores, inverse, contributions)
        order = np.lexsort((unique_documents, -scores))[:n]
        return self.item_ids[unique_documents[order]]


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
    positive_reference_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Her sorgu (term_id) için BM25 ile en benzer ürünlerden, o sorgunun eğitim
    alt kümesindeki pozitif çift sayısı x `ratio` kadar hard negative seçer.

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
        Her pozitif çift başına üretilecek hedef hard negative sayısı. BM25
        adayları yetmezse daha az üretilebilir; karma üretici kalan kotayı
        random negatiflerle doldurur.
    max_df_ratio : float, default=0.15
        BM25Index için — kataloğun bu orandan fazlasında geçen kelimeler
        indekslenmez (bkz. BM25Index docstring'i).
    verbose : bool, default=True
        İlerleme bilgisi yazdır.
    positive_reference_df : pd.DataFrame, optional
        Negatiflerden dışlanacak tüm bilinen pozitif çiftler. Örneklemli
        çalışmada tam pozitif veri seti burada verilmelidir.

    Döndürür
    -------
    pd.DataFrame
        Kolonlar: term_id, item_id, label (hepsi 0)
    """
    if not isinstance(ratio, int) or ratio <= 0:
        raise ValueError(f"ratio must be a positive integer, got {ratio}")
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n}")
    required_pairs = {"term_id", "item_id"}
    if not required_pairs.issubset(train_df.columns):
        raise ValueError(f"train_df must contain {sorted(required_pairs)}")
    if train_df.empty:
        return pd.DataFrame(columns=["term_id", "item_id", "label"])
    if not {"term_id", "query"}.issubset(terms_df.columns):
        raise ValueError("terms_df must contain term_id and query")
    if terms_df["term_id"].duplicated().any():
        raise ValueError("terms_df contains duplicate term_id values")
    required_items = {"item_id", "title", "category", "brand"}
    if not required_items.issubset(items_df.columns):
        raise ValueError(f"items_df must contain {sorted(required_items)}")
    if items_df.empty or items_df["item_id"].isna().any():
        raise ValueError("items_df must contain non-null catalog items")
    if items_df["item_id"].duplicated().any():
        raise ValueError("items_df contains duplicate item_id values")
    positive_reference_df = (
        train_df if positive_reference_df is None else positive_reference_df
    )
    if not required_pairs.issubset(positive_reference_df.columns):
        raise ValueError(
            f"positive_reference_df must contain {sorted(required_pairs)}"
        )

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
    pos_by_term = positive_reference_df.groupby("term_id")["item_id"].apply(set)
    term_to_query = terms_df.set_index("term_id")["query"]
    target_by_term = train_df.groupby("term_id").size().mul(ratio)

    unique_terms = train_df["term_id"].unique()
    if verbose:
        print(f"[bm25_hard_negative] {len(unique_terms):,} benzersiz sorgu icin "
              f"BM25 taramasi basliyor (top_n={top_n}, ratio={ratio})...")

    negatives = []
    for i, term_id in enumerate(unique_terms):
        query = term_to_query.get(term_id, "")
        pos_items = pos_by_term.get(term_id, set())

        target = int(target_by_term.loc[term_id])
        candidate_limit = max(top_n, target + len(pos_items))
        added = 0
        for item_id in index.top_n(query, n=candidate_limit):
            if item_id in pos_items:
                continue
            negatives.append((term_id, item_id))
            added += 1
            if added >= target:
                break

        if verbose and (i + 1) % 5_000 == 0:
            print(f"  ... {i + 1:,}/{len(unique_terms):,} sorgu islendi "
                  f"({len(negatives):,} hard negative uretildi)")

    negatives_df = pd.DataFrame(negatives, columns=["term_id", "item_id"])
    negatives_df["label"] = 0

    if verbose:
        expected_total = int(target_by_term.sum())
        avg_found = len(negatives_df) / len(unique_terms) if len(unique_terms) else 0.0
        # reindex: hic hard negative uretilemeyen (0 satirlik) terimler
        # groupby sonucunda hic gorunmez, reindex ile bunlari da 0 olarak
        # sayima katiyoruz (bkz. notebooks/03_negatif_kalite_mert.py'deki
        # ayni desen).
        term_counts = (
            negatives_df.groupby("term_id").size().reindex(unique_terms, fill_value=0)
            if len(negatives_df) else pd.Series(0, index=unique_terms)
        )
        target_counts = target_by_term.reindex(unique_terms)
        eksik = (term_counts < target_counts).sum()
        print(f"[bm25_hard_negative] Toplam {len(negatives_df):,} hard negative uretildi "
              f"(sorgu basina ort. {avg_found:.2f}; toplam hedef {expected_total:,}).")
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
        positive_reference_df=train_df,
    )

    print("\nSizinti kontrolu yapiliyor...")
    verify_no_leakage(hard_negatives, train_df)

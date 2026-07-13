"""
src/tfidf_features.py
=====================
G.G.A Takımı — TF-IDF Cosine Similarity Feature Modülü

Muhammed Köseoğlu tarafından hazırlanmıştır.
3 Temmuz: Temel modül
6 Temmuz: Hiperparametre deneyleri sonucu varsayılan parametreler güncellendi
  Deney detayları: docs/tfidf_deney_tablosu.md

TF-IDF Nedir?
  TF-IDF (Term Frequency - Inverse Document Frequency):
  Her kelimenin bir belgede ne kadar önemli olduğunu gösteren istatistiksel bir ölçüm.
  "spor" kelimesi her yerde geçiyorsa önemsiz, "kerastase" sadece az yerde geçiyorsa çok önemlidir.

Cosine Similarity Nedir?
  İki metin vektörü arasındaki açının cosinüsüdür.
  1.0 = tamamen aynı yönde (çok benzer)
  0.0 = dik açıda (hiç benzer değil)
  -1.0 = tam tersi yönde (anlamsız negatif metinler için)

Bu modül şunları sağlar:
  - Sorgu ve ürün başlığı/kategorisi arasında TF-IDF cosine similarity hesaplama
  - Büyük veri setleri için batch (toplu) işleme ile bellek dostu yaklaşım
  - Vectorizer'ı (eğitilmiş modeli) kaydetme/yükleme (bir kez eğit, sürekli kullan)
"""

import os
import pickle
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.item_text import build_item_texts, clean_text, iter_item_texts


# ─────────────────────────────────────────────────────────────────────────────
# 1. TF-IDF Vectorizer Oluşturma ve Eğitme
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    max_features: int = 10_000,   # 6 Tem deney: 10K unigram > 30K/50K bigram (sep: 0.4464 vs 0.3307)
    ngram_range: tuple = (1, 1),  # 6 Tem deney: unigram kazandı (Türkçe çekimli dil — bigram sulanıyor)
    min_df: int = 2,              # 6 Tem deney: min_df etkisi ihmal edilebilir, 2 seçildi
) -> TfidfVectorizer:
    """
    Tüm sorgu ve ürün metinlerini kullanarak TF-IDF vectorizer'ı eğitir.

    Neden hem sorgu hem ürün metinleriyle eğitiyoruz?
      Vocabulary (kelime dağarcığı) her iki tarafın kelimelerini içermelidir.
      Yoksa bir tarafın kelimeleri bilinmez ve sıfır vektör üretilir.

    Parametreler
    ----------
    terms_df : pd.DataFrame
        Sorgular (query kolonu içerir).
    items_df : pd.DataFrame
        Ürün kataloğu (title, category, brand kolonları içerir).
    max_features : int, default=10_000
        En sık geçen kaç kelime tutulacak.
        6 Temmuz deneyi: 10K, 30K, 50K karşılaştırıldı — 10K en iyi separation verdi.
        Büyük vocab'da IDF dağılımı düzleşiyor, ayırt edicilik düşüyor.
    ngram_range : tuple, default=(1, 1)
        Unigram kullanılıyor.
        6 Temmuz deneyi: bigram ve trigram unigram'dan belirgin düşük çıktı.
        Türkçe'de kelimeler zaten çekimli — bigram oluşturmak kelime dağarcığını sulandırıyor.
        "spor ayakkabı" gibi çift kelimeli anlamlı kombinasyonları yakalar.
    min_df : int, default=2
        En az kaç belgede geçen kelimeler alınsın. Nadir yazım hatalarını eler.

    Döndürür
    -------
    TfidfVectorizer
        Eğitilmiş (fitted) sklearn TF-IDF vectorizer nesnesi.
    """
    if not isinstance(max_features, int) or max_features <= 0:
        raise ValueError("max_features must be a positive integer")
    if (
        not isinstance(ngram_range, tuple)
        or len(ngram_range) != 2
        or not all(isinstance(value, int) and value > 0 for value in ngram_range)
        or ngram_range[0] > ngram_range[1]
    ):
        raise ValueError("ngram_range must be a pair of positive increasing integers")
    if not isinstance(min_df, int) or min_df <= 0:
        raise ValueError("min_df must be a positive integer")
    if "query" not in terms_df.columns:
        raise ValueError("terms_df must contain query")

    print("[tfidf] Metin korpusu hazirlaniyor...")

    # Sorgu metinlerini topla
    query_texts = [clean_text(query) for query in terms_df["query"]]

    # Stream the catalog instead of materializing ~963K Python strings. Query
    # texts already expose searched attribute values to the vocabulary; item
    # attributes are still included later when pair cosine scores are computed.
    item_texts = iter_item_texts(items_df, include_attrs=False)

    print(
        f"[tfidf] Toplam {len(query_texts) + len(items_df):,} metin ile "
        "egitim basliyor..."
    )

    # TF-IDF vectorizer'ı eğit
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,   # TF'yi log(1+tf) ile yumuşat (çok tekrar eden kelimelerin etkisini azalt)
        analyzer="word",
        strip_accents="unicode",
        lowercase=True,
    )
    vectorizer.fit(chain(query_texts, item_texts))
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[tfidf] Egitim tamamlandi. Kelime dagarciği boyutu: {vocab_size:,}")

    return vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vectorizer Kaydetme ve Yükleme
# ─────────────────────────────────────────────────────────────────────────────

def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """
    Eğitilmiş TF-IDF vectorizer'ı diske kaydeder.

    Neden kaydetmek önemli?
      50K metni eğitmek birkaç dakika sürer. Her seferinde tekrar yapmak yerine
      bir kez kaydet, sürekli yükle.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary_path = os.path.abspath(path) + ".tmp"
    with open(temporary_path, "wb") as f:
        pickle.dump(vectorizer, f)
    os.replace(temporary_path, path)
    print(f"[tfidf] Vectorizer kaydedildi: {path}")


def load_vectorizer(path: str) -> TfidfVectorizer:
    """
    Daha önce kaydedilmiş TF-IDF vectorizer'ı diskten yükler.
    """
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[tfidf] Vectorizer yuklendi: {path}")
    return vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cosine Similarity Hesaplama
# ─────────────────────────────────────────────────────────────────────────────

def compute_tfidf_cosine_batch(
    query_texts: list,
    item_texts: list,
    vectorizer: TfidfVectorizer,
    batch_size: int = 5_000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Sorgu ve ürün metinleri arasında TF-IDF cosine similarity hesaplar.

    Neden batch (toplu) işleme?
      Milyonlarca satır için tüm matrisi bir anda belleğe almak imkansız.
      Batch yöntemi: 5000 satır al, hesapla, kaydet, bir sonraki 5000'e geç.

    Parametreler
    ----------
    query_texts : list of str
        Her satır için sorgu metni.
    item_texts : list of str
        Her satır için ürün metni (title + category + brand).
    vectorizer : TfidfVectorizer
        build_tfidf_vectorizer() ile eğitilmiş vectorizer.
    batch_size : int, default=5_000
        Her seferinde işlenecek satır sayısı.

    Döndürür
    -------
    np.ndarray, shape=(n_samples,)
        Her satır için [0.0, 1.0] arasında bir cosine similarity skoru.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    n = len(query_texts)
    if n != len(item_texts):
        raise ValueError("query_texts and item_texts must have equal length")
    similarities = np.zeros(n, dtype=np.float32)

    if verbose:
        print(
            f"[tfidf] {n:,} satir icin cosine similarity hesaplaniyor "
            f"(batch={batch_size:,})..."
        )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # Bu batch'teki sorgu ve ürün metinlerini TF-IDF vektörlerine çevir
        q_batch = vectorizer.transform(query_texts[start:end])  # Seyrek matris (sparse)
        i_batch = vectorizer.transform(item_texts[start:end])

        # Satır satır cosine similarity: her (sorgu, ürün) çifti için bir skor
        # Element-wise çarpım sonra satır toplamı alınır (sparse matris dostu yöntem)
        batch_sim = q_batch.multiply(i_batch).sum(axis=1)
        similarities[start:end] = np.asarray(batch_sim).flatten()

        if verbose and (start // batch_size + 1) % 10 == 0:
            print(f"  ... {end:,}/{n:,} satir islendi")

    if verbose:
        mean_similarity = float(similarities.mean()) if n else 0.0
        print(f"[tfidf] Tamamlandi. Ortalama cosine: {mean_similarity:.4f}")
    return similarities


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataFrame'e Feature Ekleme
# ─────────────────────────────────────────────────────────────────────────────

def add_tfidf_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    batch_size: int = 5_000,
    verbose: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Birleştirilmiş DataFrame'e TF-IDF cosine similarity feature'larını ekler.

    DataFrame'in şu kolonları içermesi gerekir:
      - query    : Arama sorgusu
      - title    : Ürün başlığı
      - category : Ürün kategorisi
      - brand    : Marka

    Parametreler
    ----------
    df : pd.DataFrame
        merge_pairs() ile üretilmiş birleştirilmiş veri seti.
    vectorizer : TfidfVectorizer
        Eğitilmiş TF-IDF vectorizer.
    batch_size : int, default=5_000
        Batch boyutu.

    Döndürür
    -------
    pd.DataFrame
        'tfidf_cosine' kolonu eklenmiş DataFrame.
    """
    required = {"query", "title", "category", "brand"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"TF-IDF input is missing required columns: {missing}")
    out = df.copy() if copy else df

    # Sorgu metinleri
    query_texts = [clean_text(query) for query in out["query"]]

    # Ürün metinleri: title + category (/ → boşluk) + brand
    item_texts = build_item_texts(out, include_attrs=True)

    # Cosine similarity hesapla
    out["tfidf_cosine"] = compute_tfidf_cosine_batch(
        query_texts,
        item_texts,
        vectorizer,
        batch_size=batch_size,
        verbose=verbose,
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hızlı PoC (Proof of Concept) — 3 Temmuz görevi
# ─────────────────────────────────────────────────────────────────────────────

def run_tfidf_poc(terms_df, items_df, pairs_df, sample_size=500):
    """
    Küçük bir örnek üzerinde TF-IDF cosine similarity'nin çalıştığını gösterir.

    3 Temmuz görevine göre: "küçük örnek üzerinde cosine similarity hesapla"

    Parametreler
    ----------
    terms_df, items_df : DataFrame
        Sorgu ve ürün verileri.
    pairs_df : pd.DataFrame
        Birleştirilmiş eğitim çiftleri (query, title, category, brand içermeli).
    sample_size : int
        PoC için kullanılacak satır sayısı.
    """
    print("=" * 55)
    print("  TF-IDF PoC (Proof of Concept)")
    print(f"  Ornek boyutu: {sample_size:,} satir")
    print("=" * 55)

    # Küçük örnek al
    sample = pairs_df.sample(n=min(sample_size, len(pairs_df)), random_state=42)

    # Vectorizer eğit
    vectorizer = build_tfidf_vectorizer(
        terms_df, items_df, max_features=10_000, ngram_range=(1, 1), min_df=2
    )

    # Feature ekle
    sample = add_tfidf_features(sample, vectorizer)

    # Sonuçları göster
    print("\nOrnek TF-IDF Cosine Similarity Degerleri:")
    print("-" * 55)
    cols_to_show = ["query", "title", "tfidf_cosine"]
    if "label" in sample.columns:
        cols_to_show.append("label")

    for _, row in sample[cols_to_show].head(10).iterrows():
        label_str = f"  label={int(row['label'])}" if "label" in row else ""
        print(f"  query  : {str(row['query'])[:50]}")
        print(f"  title  : {str(row['title'])[:50]}")
        print(f"  cosine : {row['tfidf_cosine']:.4f}{label_str}")
        print()

    if "label" in sample.columns:
        pos_avg = sample[sample["label"] == 1]["tfidf_cosine"].mean()
        neg_avg = sample[sample["label"] == 0]["tfidf_cosine"].mean()
        print(f"Pozitif (label=1) ort. cosine: {pos_avg:.4f}")
        print(f"Negatif (label=0) ort. cosine: {neg_avg:.4f}")
        sep = pos_avg - neg_avg
        print(f"Ayiricilik (fark)            : {sep:.4f}  {'[IYI]' if sep > 0 else '[ZAYIF]'}")

    return sample, vectorizer


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data import load_terms, load_items, merge_pairs
    from src.negative_sampling import build_training_set

    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    print("Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(data_dir, "terms.csv"))
    items_df  = load_items(os.path.join(data_dir, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    # Küçük örnek üzerinde eğitim seti oluştur (hız için)
    print("\nKucuk ornek egitim seti olusturuluyor (1000 pozitif, ratio=1)...")
    small_train = build_training_set(
        train_raw.sample(1000, random_state=42), items_df,
        ratio=1, random_state=42, verbose=False,
        positive_reference_df=train_raw,
    )

    # Merge et
    print("Merge yapiliyor...")
    merged = small_train.merge(
        terms_df, on="term_id", how="left", validate="many_to_one"
    ).merge(items_df, on="item_id", how="left", validate="many_to_one")

    # TF-IDF PoC çalıştır
    run_tfidf_poc(terms_df, items_df, merged, sample_size=500)

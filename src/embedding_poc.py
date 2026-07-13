"""
src/embedding_poc.py
====================
G.G.A Takımı — Sentence-Transformers Embedding PoC (8 Temmuz Görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Bu modül sentence-transformers kütüphanesini kullanarak
sorgu ve ürün metinleri için yoğun (dense) embedding vektörleri üretir.

TF-IDF vs Embedding Farkı:
  - TF-IDF: Kelime sayımına dayalı seyrek (sparse) vektör
    "adidas koşu ayakkabı" → [0, 0.3, 0, 0.7, 0, ...]
  - Embedding: Anlamsal (semantic) yoğun vektör
    "adidas koşu ayakkabı" → [-0.12, 0.45, 0.83, ...]
    "nike run shoe" de benzer vektör olur — semantik benzerlik!

Bu PoC (Proof of Concept) şunları kanıtlar:
  1. sentence-transformers kurulumu çalışıyor mu?
  2. Küçük bir batch için embedding hızı ne kadar?
  3. Embedding cosine similarity, TF-IDF'e göre ne kadar ayırt edici?

NOT: GPU varsa otomatik kullanılır. CPU'da yavaş olabilir.
Tam embedding üretimi için 10 Temmuz'da GPU batch işleme yapılacak.

Kurulum (yoksa):
  pip install sentence-transformers
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Kullanılacak model — Türkçe ve çok dilli metinler için iyi çalışan
# Hafif (117MB) ve hızlı bir model seçildi
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_model(model_name=EMBEDDING_MODEL):
    """
    Sentence-transformers modelini yükler.

    İlk çalıştırmada model Hugging Face'den indirilir (~117MB).
    Sonraki çalıştırmalarda önbellekten (cache) yüklenir.

    Parametreler
    ----------
    model_name : str
        Kullanılacak sentence-transformers model adı.

    Döndürür
    -------
    SentenceTransformer
        Yüklenmiş model nesnesi.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers kurulu degil.\n"
            "Kurmak icin: pip install sentence-transformers"
        )

    print(f"[embedding] Model yukleniyor: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[embedding] Model yuklendi. "
          f"Cihaz: {'GPU' if model.device.type != 'cpu' else 'CPU'}")
    return model


def encode_texts(model, texts, batch_size=64, show_progress=True):
    """
    Metin listesini embedding vektörlerine dönüştürür.

    Her metin için sabit boyutlu (384 boyut) bir vektör üretilir.
    Bu vektör metnin anlamsal içeriğini temsil eder.

    Parametreler
    ----------
    model : SentenceTransformer
        Yüklü model.
    texts : list of str
        Encode edilecek metin listesi.
    batch_size : int
        Kaçar kaçar işlenecek (GPU belleğine göre ayarla).
    show_progress : bool
        İlerleme çubuğu göster.

    Döndürür
    -------
    np.ndarray
        Shape: (len(texts), embedding_dim). Her satır bir metnin vektörü.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize → cosine = dot product
    )
    return embeddings


def compute_cosine_similarities(query_embs, item_embs):
    """
    Normalize edilmiş embedding'ler için cosine similarity hesaplar.

    L2 normalize edilmiş vektörler için:
      cosine_similarity(a, b) = dot_product(a, b)
    Bu çok daha hızlı hesaplanır.

    Parametreler
    ----------
    query_embs : np.ndarray  shape (N, D)
    item_embs  : np.ndarray  shape (N, D)

    Döndürür
    -------
    np.ndarray  shape (N,)
        Her (sorgu, ürün) çifti için cosine similarity skoru.
    """
    # Satır bazlı dot product — her çift için ayrı hesaplama
    return np.einsum("nd,nd->n", query_embs, item_embs)


def run_poc(n_samples=200, batch_size=32):
    """
    Küçük bir veri seti üzerinde embedding PoC çalıştırır.

    Ayırıcılık metrikleri (separation):
      separation = pos_cosine_mean - neg_cosine_mean
      Bu değerin yüksek olması embeding'in iyi çalıştığını gösterir.
      Karşılaştırma: TF-IDF unigram separation = 0.4464 (6 Temmuz ölçümü)

    Parametreler
    ----------
    n_samples : int
        Test için kullanılacak çift sayısı.
    batch_size : int
        Encoding batch boyutu.
    """
    from src.data              import load_terms, load_items
    from src.negative_sampling import build_training_set

    DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
    SEED     = 42

    print("=" * 60)
    print("  G.G.A - Embedding PoC (8 Temmuz)")
    print(f"  Model: {EMBEDDING_MODEL}")
    print("=" * 60)

    # 1. Model yükle
    model = load_model()

    # 2. Küçük veri seti hazırla
    print(f"\n[1/4] Veri hazirlaniyor ({n_samples} poz + {n_samples} neg)...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    pos_sample = train_raw.sample(n=n_samples, random_state=SEED)
    full_set   = build_training_set(
        pos_sample, items_df, ratio=1,
        random_state=SEED, verbose=False,
        positive_reference_df=train_raw,
    )
    merged = full_set.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    merged = merged.reset_index(drop=True)

    # 3. Metin hazırlama — ürün metni: title + category + brand birleşimi
    query_texts = merged["query"].fillna("").tolist()
    item_texts  = (
        merged["title"].fillna("") + " " +
        merged["category"].astype(str).str.replace("/", " ", regex=False).fillna("") + " " +
        merged["brand"].fillna("")
    ).tolist()

    # 4. Encode et ve süreyi ölç
    print(f"\n[2/4] Sorgu metinleri encode ediliyor ({len(query_texts)} adet)...")
    t0 = time.time()
    query_embs = encode_texts(model, query_texts, batch_size=batch_size)
    query_time = time.time() - t0

    print(f"\n[3/4] Urun metinleri encode ediliyor ({len(item_texts)} adet)...")
    t0 = time.time()
    item_embs  = encode_texts(model, item_texts,  batch_size=batch_size)
    item_time  = time.time() - t0

    total_time = query_time + item_time
    rate       = len(merged) / total_time

    print(f"\n  Encoding suresi: {total_time:.1f}s  ({rate:.0f} metin/s)")
    print(f"  Embedding boyutu: {query_embs.shape[1]} boyut")

    # 5. Ayırıcılık ölç
    print("\n[4/4] Cosine similarity ve ayiricilik olculuyor...")
    cosines = compute_cosine_similarities(query_embs, item_embs)
    labels  = merged["label"].values

    pos_cos = cosines[labels == 1].mean()
    neg_cos = cosines[labels == 0].mean()
    sep     = pos_cos - neg_cos

    print("\n" + "=" * 60)
    print("  POC SONUCU")
    print("=" * 60)
    print(f"  Pozitif ciftler cosine ort : {pos_cos:.4f}")
    print(f"  Negatif ciftler cosine ort : {neg_cos:.4f}")
    print(f"  Separation                 : {sep:.4f}")
    print(f"  (Karsilastirma: TF-IDF unigram separation = 0.4464)")
    if sep > 0.4464:
        print("  Embedding TF-IDF'den DAHA iyi ayirt ediyor!")
    else:
        print("  Embedding TF-IDF'den daha zayif — daha buyuk model gerekebilir.")
    print(f"\n  Hiz  : {rate:.0f} metin/s")
    print(f"  Boyut: {query_embs.shape[1]}")
    print("=" * 60)

    return {
        "model"      : EMBEDDING_MODEL,
        "n_samples"  : len(merged),
        "emb_dim"    : query_embs.shape[1],
        "pos_cosine" : round(float(pos_cos), 4),
        "neg_cosine" : round(float(neg_cos), 4),
        "separation" : round(float(sep), 4),
        "speed_per_s": round(rate, 1),
    }


if __name__ == "__main__":
    result = run_poc(n_samples=200, batch_size=32)
    print("\n[embedding_poc] Tamamlandi.")
    print(f"  Separation: {result['separation']}")
    print(f"  Hiz: {result['speed_per_s']} metin/s")

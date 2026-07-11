"""
src/embedding_features.py
=========================
G.G.A Takımı — Sentence Embedding ve Cosine Similarity Features (Gün 12 görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Amaç:
  Sentence Transformers (pre-trained BERT modeli) kullanarak query ve item
  başlıklarını dense vector'lere dönüştürüp cosine similarity hesapla.
  
  TF-IDF'ten farkı: Semantik anlam yakalaması (TF-IDF sadece kelime sıklığı).
  
  Örnek:
    - Sorgu: "spor ayakkabı" 
    - Item title: "koşu ayakkabısı"
    - TF-IDF cosine: 0.3 (sadece "ayakkabı" ortak)
    - Embedding cosine: 0.85 (semantic similarity — her ikisi de footwear)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Yükleme (Cached)
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> Optional:
    """
    Sentence Transformers modelini yükle (cache'den).
    
    Model: paraphrase-multilingual-MiniLM-L12-v2
      - Hızlı (~60ms per 100 texts GPU'da)
      - Compact (66M parameters, CPU'da çalışabiliyor)
      - Multilingual (Türkçe destekli)
      - Açık kaynak, offline çalışabiliyor
    
    İlk kez yükleme ~ 500 MB indir ve .cache/huggingface/ içinde sakla.
    Sonraki yüklemeler anında (disk'ten).
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        print("[embedding] SentenceTransformers yüklü değil. pip install sentence-transformers yapın.")
        return None
    
    try:
        print(f"[embedding] Model yükleniyor: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"[embedding] Model yükle: dimension={model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"[embedding] Hata: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding Batch Üretimi
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts_batch(texts: list, model, batch_size: int = 32) -> np.ndarray:
    """
    Metinleri batch'ler halinde embedding'e dönüştür.
    
    Parametreler
    ----------
    texts : list of str
        Embedding'e dönüştürülecek metinler.
    model : SentenceTransformer
        Eğitilmiş embedding modeli.
    batch_size : int
        Batch boyutu (bellek & hız tradeoff).
    
    Döndürür
    -------
    np.ndarray, shape=(n, embedding_dim)
        Embedding matris.
    """
    n = len(texts)
    print(f"[embedding] {n:,} metin embedding'e dönüştürülüyor (batch_size={batch_size})...")
    
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cosine Similarity Hesaplama (Batch)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_pair(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    İki embedding sütunu arasında cosine similarity hesapla.
    
    Parametreler
    ----------
    emb1 : np.ndarray, shape=(n, d)
        Query embedding'leri.
    emb2 : np.ndarray, shape=(n, d)
        Item embedding'leri.
    
    Döndürür
    -------
    np.ndarray, shape=(n,)
        Her (query, item) çifti için cosine similarity [0.0, 1.0].
    """
    # Embedding'leri normalize et (birim vektör)
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    
    # Satır satır dot product
    similarities = np.sum(emb1_norm * emb2_norm, axis=1)
    return similarities


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataFrame'e Feature Ekleme
# ─────────────────────────────────────────────────────────────────────────────

def add_embedding_cosine_feature(
    df: pd.DataFrame,
    model,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    DataFrame'e embedding-based cosine similarity feature'ı ekle.
    
    İnput DataFrame kolonları:
      - query   : Sorgu metni
      - title   : Ürün başlığı
    
    Çıktı Kolonu:
      - embedding_cosine : [0.0, 1.0] between query ve title
    """
    if model is None:
        print("[embedding] Model None — feature eklenemiyor.")
        return df
    
    out = df.copy()
    
    # Query ve title embedding'lerini üret
    queries = out["query"].fillna("").tolist()
    titles = out["title"].fillna("").tolist()
    
    print(f"[embedding] Query embedding'leri üretiliyor...")
    query_embs = embed_texts_batch(queries, model, batch_size=batch_size)
    
    print(f"[embedding] Title embedding'leri üretiliyor...")
    title_embs = embed_texts_batch(titles, model, batch_size=batch_size)
    
    # Cosine similarity hesapla
    print(f"[embedding] Cosine similarity hesaplaniyor...")
    out["embedding_cosine"] = cosine_similarity_pair(query_embs, title_embs)
    
    print(f"[embedding] Tamamlandi. Ortalama cosine: {out['embedding_cosine'].mean():.4f}")
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. PoC Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("ERROR: sentence-transformers yüklü değil.")
        print("Kurulum: pip install sentence-transformers")
        sys.exit(1)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    from src.data import load_terms, load_items
    
    # Test verisi
    print("[TEST] Embedding feature PoC başlıyor...")
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    terms_df = load_terms(os.path.join(data_dir, "terms.csv"))
    items_df = load_items(os.path.join(data_dir, "items.csv"))
    
    train_df = pd.read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )
    sample = train_df.head(500).copy()
    sample = sample.merge(terms_df, on="term_id")
    sample = sample.merge(items_df, on="item_id")
    
    # Model yükle
    model = load_embedding_model()
    if model is None:
        print("Model yükleme başarısız.")
        sys.exit(1)
    
    # Feature ekle
    sample_with_emb = add_embedding_cosine_feature(sample, model, batch_size=32)
    
    print("\nSample embedding_cosine values:")
    print(sample_with_emb[["query", "title", "embedding_cosine"]].head(10).to_string())
    
    print(f"\nEmbedding cosine istatistikleri:")
    print(sample_with_emb["embedding_cosine"].describe())

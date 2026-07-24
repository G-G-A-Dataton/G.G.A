"""
scripts/retrieval/build_item_embeddings.py
============================================
G.G.A Takımı — Fast Item & Query Dense Embedding Generator
962,873 ürün için 384-boyutlu L2-normalized dense embedding matrisi üretir.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items
from src.item_text import build_item_texts


def main():
    print("[+] Ürün kataloğu yükleniyor...")
    items_path = os.path.join(PROJECT_ROOT, "datasets", "items.csv")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    emb_path = os.path.join(out_dir, "item_embeddings.npy")
    ids_path = os.path.join(out_dir, "item_ids.npy")

    items_df = load_items(items_path)
    item_ids = items_df["item_id"].astype(str).to_numpy()
    texts = build_item_texts(items_df)

    n_items = len(items_df)
    print(f"[+] Toplam ürün sayısı: {n_items:,}")

    start_time = time.time()
    print("[+] TF-IDF + TruncatedSVD (384 Component) Dense Embedding üretimi başladı...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)
    tfidf_mat = vec.fit_transform(texts)
    print(f"[+] TF-IDF matrisi hazır: {tfidf_mat.shape}. TruncatedSVD indirgemesi hesaplanıyor...")

    svd = TruncatedSVD(n_components=384, random_state=42)
    dense_mat = svd.fit_transform(tfidf_mat)

    # L2-normalize
    norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (dense_mat / norms).astype(np.float32)

    elapsed = time.time() - start_time
    print(f"[+] Dense Embedding üretimi tamamlandı: {elapsed:.2f} saniye ({n_items / elapsed:.1f} ürün/sn)")

    np.save(emb_path, embeddings)
    np.save(ids_path, item_ids)
    print(f"[+] Embeddings kaydedildi: {emb_path} (Şekil: {embeddings.shape})")
    print(f"[+] IDs kaydedildi: {ids_path}")


if __name__ == "__main__":
    main()

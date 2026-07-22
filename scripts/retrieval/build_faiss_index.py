import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.faiss_index import FAISSIndex
from src.embedding_cosine import EmbeddingIndex

def main():
    emb_path = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "item_embeddings.npy")
    ids_path = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "item_ids.npy")
    output_path = os.path.join(PROJECT_ROOT, "outputs", "indexes", "faiss_items.index")
    
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        print(f"HATA: Embedding dosyaları bulunamadı! Lütfen önce embedding oluşturun.")
        sys.exit(1)
        
    print(f"Embedding dosyaları yükleniyor...")
    try:
        idx = EmbeddingIndex()
        idx.load(emb_path, ids_path)
        embeddings = idx.embeddings
        item_ids = idx.item_ids
    except Exception as e:
        print("EmbeddingIndex ile yükleme başarısız, np.load ile yükleniyor:", e)
        embeddings = np.load(emb_path)
        item_ids = np.load(ids_path, allow_pickle=True)
        
    print(f"Index inşa ediliyor (dimension=384, n_lists=256, n_probes=32)...")
    start_time = time.time()
    
    # 384 assumes a standard embedding size like all-MiniLM-L6-v2
    dimension = embeddings.shape[1] if len(embeddings.shape) == 2 else 384
    index = FAISSIndex(dimension=dimension, n_lists=256, n_probes=32)
    index.build(embeddings, item_ids)
    
    elapsed = time.time() - start_time
    n_items = index.n_items
    items_per_sec = n_items / elapsed if elapsed > 0 else 0
    
    print(f"FAISS Index kaydediliyor: {output_path}")
    index.save(output_path)
    
    print("\n--- İstatistikler ---")
    print(f"İndekslenen öğe: {n_items}")
    print(f"Geçen süre: {elapsed:.2f} saniye")
    print(f"Hız: {items_per_sec:.2f} öğe/saniye")

if __name__ == '__main__':
    main()

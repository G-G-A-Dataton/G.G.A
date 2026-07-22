import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items
from src.retrieval.bm25_retriever import BM25Retriever

def main():
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "indexes")
    output_path = os.path.join(output_dir, "bm25_index.pkl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Veri yükleniyor...")
    items_path = os.path.join(PROJECT_ROOT, "datasets", "items.csv")
    items_df = load_items(items_path)
    
    print("BM25 Index inşa ediliyor (max_df_ratio=0.3)...")
    start_time = time.time()
    
    retriever = BM25Retriever(items_df, max_df_ratio=0.3)
    
    elapsed = time.time() - start_time
    n_items = len(items_df)
    items_per_sec = n_items / elapsed if elapsed > 0 else 0
    
    print(f"BM25 Index kaydediliyor: {output_path}")
    retriever.save(output_path)
    
    print("\n--- İstatistikler ---")
    print(f"İndekslenen öğe: {n_items}")
    print(f"Geçen süre: {elapsed:.2f} saniye")
    print(f"Hız: {items_per_sec:.2f} öğe/saniye")

if __name__ == '__main__':
    main()

"""
run_term_embeddings.py
======================
G.G.A Takımı — Term (Sorgu) Embedding Uretimi (11 Temmuz Gorevi)

Muhammed Koseoğlu tarafından hazırlanmıştır.

Bu script tum sorgu metinleri (terms.csv) icin embedding uretir ve
outputs/embeddings/term_embeddings.npy dosyasina kaydeder.

Item embedding'lerden fark:
  - Sorgular kisa (1-5 kelime), urunler uzun (10-50 kelime)
  - Sorgu embedding boyutu = urun embedding boyutu (384)
  - Her sorgu bir query_id ile eslenir

Cikti dosyalari:
  outputs/embeddings/term_embeddings.npy   -> shape (N_terms, 384)
  outputs/embeddings/term_ids.npy          -> hangi indeks hangi term_id
  outputs/embeddings/production.log        -> uretim logu

Calistirmak icin:
  python run_term_embeddings.py

Not: Buyuk katalogu da uretemek icin:
  python src/embedding_batch.py --target items
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
EMB_DIR  = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Term Embedding Uretimi (11 Temmuz)")
    print("=" * 60)

    # src/embedding_batch.py'daki tam modulu cagir
    from src.embedding_batch import load_model, produce_term_embeddings

    t0 = time.time()
    model = load_model()
    embeddings = produce_term_embeddings(model)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  OZET")
    print("=" * 60)
    print(f"  Embedding shape  : {embeddings.shape}")
    print(f"  Toplam sure      : {elapsed:.1f}s")
    print(f"  Cikti dizini     : {EMB_DIR}")
    print("=" * 60)

    # Uretim loguna ozet ekle
    log_path = os.path.join(EMB_DIR, "production.log")
    with open(log_path, "a", encoding="utf-8") as f:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"\n[{ts}] Term embeddings tamamlandi: shape={embeddings.shape}, sure={elapsed:.1f}s\n")

    print(f"\n  Log  : {log_path}")

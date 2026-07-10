"""
src/embedding_batch.py
======================
G.G.A Takımı — Item/Term Embedding Toplu Uretimi (10 Temmuz Gorevi)

Muhammed Koseoğlu tarafından hazırlanmıştır.

Bu modul sentence-transformers kullanarak buyuk olcekli embedding uretir:
  - ~963K urun (items.csv) icin item embeddingleri
  - ~XXX sorgu (terms.csv) icin term embeddingleri

Cihaz onceligi:
  1. CUDA GPU (varsa, en hizli)
  2. CPU (yedek)

Bellek yonetimi:
  - Embeddingler chunk'lar halinde uretilir (varsayilan 10K)
  - Her chunk NumPy .npy dosyasina kaydedilir
  - Islem kesilirse kaldigi yerden devam eder (checkpoint)

Cikti dosyalari:
  outputs/embeddings/item_embeddings.npy    -> shape (N_items, 384)
  outputs/embeddings/term_embeddings.npy    -> shape (N_terms, 384)
  outputs/embeddings/item_ids.npy           -> hangi indeks hangi item_id
  outputs/embeddings/term_ids.npy           -> hangi indeks hangi term_id
  outputs/embeddings/production.log         -> uretim logu

Calistirmak icin:
  python src/embedding_batch.py --target items     # Sadece urunler
  python src/embedding_batch.py --target terms     # Sadece sorgular
  python src/embedding_batch.py --target both      # Ikisi birden
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.item_text import add_item_text_column, build_query_text

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
EMB_DIR    = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# Hangi model kullanilacak — Turkce destekli cok dilli model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Kac satirlik chunk'lar halinde calisacak
# GPU: 10_000-50_000, CPU: 5_000-10_000
CHUNK_SIZE  = 10_000
BATCH_SIZE  = 64   # Encoding batch boyutu (GPU bellegine gore)


def load_model():
    """
    Sentence-transformers modelini cihaza yukler.

    GPU varsa otomatik kullanir. Cihaz bilgisini loglar.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers kurulu degil.\n"
            "Kurmak icin: pip install sentence-transformers"
        )

    print(f"[embedding_batch] Model yukleniyor: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    device = model.device.type
    print(f"[embedding_batch] Cihaz: {'GPU (' + str(model.device) + ')' if device != 'cpu' else 'CPU'}")
    return model


def encode_in_chunks(model, texts, ids, out_prefix, chunk_size=CHUNK_SIZE):
    """
    Buyuk metin listesini chunk'lar halinde encode edip .npy dosyalarına kaydeder.

    Neden chunk? items.csv ~963K satir; hepsini belleğe almak crash'e neden olabilir.
    Chunk yaklaşımı ile bellek kullanimi sabit kalir.

    Checkpoint sistemi:
      {out_prefix}_chunk_0000.npy, {out_prefix}_chunk_0001.npy, ...
    Islem kesilirse hangi chunk'in eksik oldugu kontrol edilir ve kaldigi yerden baslar.

    Parametreler
    ----------
    model : SentenceTransformer
        Yuklenmis model.
    texts : list of str
        Encode edilecek metinler.
    ids : np.ndarray
        Her metine karsilik gelen ID (item_id veya term_id).
    out_prefix : str
        Cikti dosyalarinin on eki. Ornek: outputs/embeddings/item
    chunk_size : int
        Kac satirlik parcalar halinde isle.
    """
    n_total  = len(texts)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    log_path = os.path.join(EMB_DIR, "production.log")
    t_start  = time.time()

    print(f"\n[embedding_batch] {n_total:,} metin, {n_chunks} chunk, chunk_size={chunk_size:,}")

    for i in range(n_chunks):
        chunk_path = f"{out_prefix}_chunk_{i:04d}.npy"

        # Checkpoint: bu chunk zaten uretilmisse atla
        if os.path.exists(chunk_path):
            print(f"  Chunk {i+1}/{n_chunks} zaten var, atlanıyor...")
            continue

        s = i * chunk_size
        e = min((i + 1) * chunk_size, n_total)
        chunk_texts = texts[s:e]

        t0 = time.time()
        chunk_embs = model.encode(
            chunk_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize - cosine = dot product
        )
        elapsed = time.time() - t0
        rate    = len(chunk_texts) / elapsed

        np.save(chunk_path, chunk_embs)

        msg = (
            f"  Chunk {i+1}/{n_chunks} tamamlandi: "
            f"{s:,}-{e:,} ({elapsed:.1f}s, {rate:.0f} metin/s)"
        )
        print(msg)

        # Uretim loguna da yaz
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Tum chunk'lari birlestir
    print("\n[embedding_batch] Chunk'lar birlestiriliyor...")
    all_embs = []
    for i in range(n_chunks):
        chunk_path = f"{out_prefix}_chunk_{i:04d}.npy"
        all_embs.append(np.load(chunk_path))

    final_embs = np.concatenate(all_embs, axis=0)

    # Ana dosyaya kaydet
    final_path  = f"{out_prefix}_embeddings.npy"
    ids_path    = f"{out_prefix}_ids.npy"

    np.save(final_path, final_embs)
    np.save(ids_path,   ids)

    total_time = time.time() - t_start
    print(f"[embedding_batch] Tamamlandi!")
    print(f"  Dosya    : {final_path}")
    print(f"  Shape    : {final_embs.shape}")
    print(f"  Toplam   : {total_time:.1f}s  ({n_total/total_time:.0f} metin/s)")

    # Chunk dosyalarini temizle (opsiyonel — yer tasarrufu)
    for i in range(n_chunks):
        chunk_path = f"{out_prefix}_chunk_{i:04d}.npy"
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

    return final_embs


def produce_item_embeddings(model):
    """
    Tum urunler icin embedding uretir.
    items.csv -> item_text -> embedding -> outputs/embeddings/item_embeddings.npy
    """
    print("\n[embedding_batch] === ITEM EMBEDDINGS ===")
    items_df = pd.read_csv(
        os.path.join(DATA_DIR, "items.csv"),
        dtype={"item_id": "string"}
    )
    print(f"  {len(items_df):,} urun yuklendi")

    # Item metnini standartlastir (src/item_text.py)
    items_df = add_item_text_column(items_df, include_attrs=True)
    texts    = items_df["item_text"].tolist()
    ids      = items_df["item_id"].values

    out_prefix = os.path.join(EMB_DIR, "item")
    return encode_in_chunks(model, texts, ids, out_prefix)


def produce_term_embeddings(model):
    """
    Tum sorgular icin embedding uretir.
    terms.csv -> query -> embedding -> outputs/embeddings/term_embeddings.npy
    """
    print("\n[embedding_batch] === TERM EMBEDDINGS ===")
    terms_df = pd.read_csv(
        os.path.join(DATA_DIR, "terms.csv"),
        dtype={"term_id": "string"}
    )
    print(f"  {len(terms_df):,} sorgu yuklendi")

    texts = terms_df.apply(build_query_text, axis=1).tolist()
    ids   = terms_df["term_id"].values

    out_prefix = os.path.join(EMB_DIR, "term")
    return encode_in_chunks(model, texts, ids, out_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Item ve/veya term embeddinglerini toplu uret"
    )
    parser.add_argument(
        "--target",
        choices=["items", "terms", "both"],
        default="both",
        help="Hangi embeddingler uretilsin? (varsayilan: both)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Kac satirlik chunk'larla isle (varsayilan: {CHUNK_SIZE:,})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Encoding batch boyutu (varsayilan: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  G.G.A - Embedding Toplu Uretim (10 Temmuz)")
    print(f"  Hedef: {args.target}")
    print("=" * 60)

    model = load_model()

    if args.target in ("items", "both"):
        produce_item_embeddings(model)

    if args.target in ("terms", "both"):
        produce_term_embeddings(model)

    print("\n[embedding_batch] Tum islemler tamamlandi.")
    print(f"  Ciktilar: {EMB_DIR}")

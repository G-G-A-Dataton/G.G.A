"""
src/embedding_cosine.py
=======================
G.G.A Takimi -- Query-Item Cosine Similarity Feature (12 Temmuz Gorevi)

Muhammed Koseoğlu tarafından hazırlanmıştır.

Bu modul, onceden uretilmis embedding vektorlerini kullanarak
her (query, item) cifti icin cosine similarity skoru uretir.

Neden embedding cosine?
  - TF-IDF kelime eslemesi yapar: "adidas kosu" -> "adidas" ve "kosu" token eslemesi
  - Embedding anlamsal esleme yapar: "kosu ayakkabisi" ~ "spor ayakkabi" (farkli kelimeler, ayni anlam)
  - Bu iki sinyal birbirini tamamlar -> ikisini birden kullanmak daha iyi F1 verir

Cache stratejisi:
  Tam submission seti icin ~3.36M (query, item) cifti var.
  Her cifti aninda embedding hesaplamak yerine:
    1. Term embedding'leri bir kez uret ve kaydet (outputs/embeddings/term_embeddings.npy)
    2. Item embedding'leri bir kez uret ve kaydet (outputs/embeddings/item_embeddings.npy)
    3. Submission sirasinda: cosine = dot_product(query_emb[term_id], item_emb[item_id])
  Bu yaklasim ~3.36M hesaplama yerine sadece lookup islemi yapar -> ~100x hizlanma.

Ciktı feature'lari:
  - embedding_cosine : Query ve item embedding vektorleri arasindaki cosine similarity (0-1)
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedding Index (Cache)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingIndex:
    """
    Onceden uretilmis embedding dosyalarini belleğe yukler ve
    ID -> embedding vektoru seklinde hizli arama saglar.

    Kullanim:
        index = EmbeddingIndex(
            emb_path="outputs/embeddings/item_embeddings.npy",
            ids_path="outputs/embeddings/item_ids.npy"
        )
        vec = index.get("123456")  # -> shape (384,)

    Cache mantigi:
        numpy dosyalari disk uzerinde sifirdan okunur (mmap_mode='r').
        Bu buyuk dosyalari (>1GB) tum RAM'e yuklenmeden kullanir.
    """

    def __init__(self, emb_path, ids_path):
        """
        Parametreler
        ----------
        emb_path : str
            Embedding matrisinin yolu. Shape: (N, D).
        ids_path : str
            ID dizisinin yolu. Shape: (N,). emb_path ile ayni siralamayi paylasiyor.
        """
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Embedding dosyasi bulunamadi: {emb_path}\n"
                f"Once 'python run_term_embeddings.py' ya da "
                f"'python src/embedding_batch.py --target both' calistirin."
            )
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"ID dosyasi bulunamadi: {ids_path}")

        # Buyuk dosyalar icin memory-mapped okuma (RAM tasarrufu)
        self.embeddings = np.load(emb_path, mmap_mode='r')
        ids_raw = np.load(ids_path, allow_pickle=True)

        # ID -> indeks sozlugu (hizli arama: O(1))
        self.id_to_idx = {str(id_): i for i, id_ in enumerate(ids_raw)}
        print(f"[embedding_cosine] Index yuklendi: {len(self.id_to_idx):,} kayit, dim={self.embeddings.shape[1]}")

    def get(self, id_):
        """
        Bir ID icin embedding vektorunu dondurur.

        Bilinmeyen ID icin None yerine sifir vektor doner (model bozulmasin).

        Parametreler
        ----------
        id_ : str
            Aranacak ID.

        Dondurur
        -------
        np.ndarray  shape (D,)
            Normalize edilmis embedding vektoru. Bilinmeyen ID: sifir vektor.
        """
        idx = self.id_to_idx.get(str(id_))
        if idx is None:
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)
        return self.embeddings[idx]

    def get_batch(self, ids):
        """
        Birden fazla ID icin topluca embedding matrisi dondurur.

        Parametreler
        ----------
        ids : list or array-like
            ID listesi.

        Dondurur
        -------
        np.ndarray  shape (N, D)
        """
        return np.stack([self.get(id_) for id_ in ids])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cosine Similarity Hesaplama
# ─────────────────────────────────────────────────────────────────────────────

def compute_cosine_batch(query_embs, item_embs):
    """
    Normalize edilmis embedding ciftleri icin cosine similarity hesaplar.

    L2 normalize edilmis vektorlerde cosine = dot product.
    Bu islem matris carpimi ile vektorize sekilde yapilir -> cok hizli.

    Parametreler
    ----------
    query_embs : np.ndarray  shape (N, D)
    item_embs  : np.ndarray  shape (N, D)

    Dondurur
    -------
    np.ndarray  shape (N,)
        Her (query, item) cifti icin cosine similarity [-1, 1] araliginda.
        L2 normalize vektorler icin [0, 1] araliginda olmasi beklenir.
    """
    # Satir bazli dot product
    return np.einsum("nd,nd->n", query_embs, item_embs)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DataFrame'e Feature Ekleme
# ─────────────────────────────────────────────────────────────────────────────

def add_embedding_cosine_feature(
    df,
    term_index,
    item_index,
    term_id_col="term_id",
    item_id_col="item_id",
    out_col="embedding_cosine",
):
    """
    DataFrame'deki her (term_id, item_id) cifti icin cosine similarity ekler.

    Embedding dosyalari mevcutsa hizli cache lookup yapar.
    Embedding dosyalari yoksa feature'i 0.0 olarak doldurur (fallback).

    Parametreler
    ----------
    df : pd.DataFrame
        Isleme alinacak veri.
    term_index : EmbeddingIndex | None
        Sorgu embedding indeksi.
    item_index : EmbeddingIndex | None
        Urun embedding indeksi.
    term_id_col : str
        Sorgu ID kolonu adi.
    item_id_col : str
        Urun ID kolonu adi.
    out_col : str
        Olusturulacak feature kolonu adi.

    Dondurur
    -------
    pd.DataFrame
        out_col eklenenmis kopya.
    """
    out = df.copy()

    # Embedding yoksa 0.0 doldur (pipeline bozulmasin)
    if term_index is None or item_index is None:
        print(f"[embedding_cosine] Embedding bulunamadi, {out_col}=0.0 olarak dolduruldu.")
        out[out_col] = 0.0
        return out

    print(f"[embedding_cosine] {len(df):,} cift icin cosine hesaplaniyor...")
    term_ids = out[term_id_col].astype(str).tolist()
    item_ids = out[item_id_col].astype(str).tolist()

    query_embs = term_index.get_batch(term_ids)
    item_embs  = item_index.get_batch(item_ids)

    cosines         = compute_cosine_batch(query_embs, item_embs)
    out[out_col]    = cosines.astype(np.float32)

    pos_mean = out.loc[out["label"] == 1, out_col].mean() if "label" in out.columns else None
    neg_mean = out.loc[out["label"] == 0, out_col].mean() if "label" in out.columns else None

    if pos_mean is not None:
        print(f"[embedding_cosine] Pozitif cosine ort: {pos_mean:.4f}")
        print(f"[embedding_cosine] Negatif cosine ort: {neg_mean:.4f}")
        print(f"[embedding_cosine] Separation: {pos_mean - neg_mean:.4f}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Index Yukleyici (Yardimci)
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_indexes(project_root):
    """
    Embedding index'lerini yukler. Dosya yoksa None dondurur.

    Parametreler
    ----------
    project_root : str
        Proje kok dizini.

    Dondurur
    -------
    (EmbeddingIndex | None, EmbeddingIndex | None)
        (term_index, item_index)
    """
    emb_dir = os.path.join(project_root, "outputs", "embeddings")

    term_emb_path = os.path.join(emb_dir, "term_embeddings.npy")
    term_ids_path = os.path.join(emb_dir, "term_ids.npy")
    item_emb_path = os.path.join(emb_dir, "item_embeddings.npy")
    item_ids_path = os.path.join(emb_dir, "item_ids.npy")

    term_index = None
    item_index = None

    if os.path.exists(term_emb_path) and os.path.exists(term_ids_path):
        try:
            term_index = EmbeddingIndex(term_emb_path, term_ids_path)
        except Exception as e:
            print(f"[embedding_cosine] Term index yuklenemedi: {e}")
    else:
        print(f"[embedding_cosine] Term embedding dosyasi yok: {term_emb_path}")

    if os.path.exists(item_emb_path) and os.path.exists(item_ids_path):
        try:
            item_index = EmbeddingIndex(item_emb_path, item_ids_path)
        except Exception as e:
            print(f"[embedding_cosine] Item index yuklenemedi: {e}")
    else:
        print(f"[embedding_cosine] Item embedding dosyasi yok: {item_emb_path}")

    return term_index, item_index


# ─────────────────────────────────────────────────────────────────────────────
# 5. Birim Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== EMBEDDING_COSINE BIRIM TESTI ===")

    # Kucuk sentetik veri ile test
    N, D = 10, 384
    rng = np.random.default_rng(42)

    # Rastgele embedding'ler uret ve normalize et
    fake_embs = rng.standard_normal((N, D)).astype(np.float32)
    norms = np.linalg.norm(fake_embs, axis=1, keepdims=True)
    fake_embs = fake_embs / norms

    fake_ids = np.array([str(i) for i in range(N)])

    # Gecici dosyalara kaydet
    import tempfile
    # ignore_cleanup_errors=True: Windows mmap kilit sorununu yonetir (Python 3.10+)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        emb_path = os.path.join(tmpdir, "emb.npy")
        ids_path = os.path.join(tmpdir, "ids.npy")
        np.save(emb_path, fake_embs)
        np.save(ids_path, fake_ids)

        index = EmbeddingIndex(emb_path, ids_path)

        # Ayni vektor ile cosine = 1.0 olmali
        v = index.get("3")
        self_cos = float(np.dot(v, v))
        print(f"\n  Ayni vektor cosine (beklenen 1.0): {self_cos:.6f}")
        assert abs(self_cos - 1.0) < 1e-4, "Normalizasyon hatasi!"

        # Batch cosine
        q_embs = index.get_batch(["0", "1", "2"])
        i_embs = index.get_batch(["0", "1", "2"])
        cosines = compute_cosine_batch(q_embs, i_embs)
        print(f"  Batch cosine (ayni cifti, hepsi ~1.0): {cosines}")
        assert all(abs(c - 1.0) < 1e-4 for c in cosines), "Batch cosine hatasi!"

        # Bilinmeyen ID -> sifir vektor
        unknown = index.get("999")
        print(f"  Bilinmeyen ID vektoru (beklenen [0,0,...]): norm={np.linalg.norm(unknown):.4f}")
        assert np.linalg.norm(unknown) == 0.0, "Bilinmeyen ID hatasi!"

        # Windows mmap kilidi: context manager kapanmadan once serbest birak
        del index

    print("\n  Tum testler gecti!")

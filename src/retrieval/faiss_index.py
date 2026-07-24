import os
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors
from .index_versioning import IndexVersion


class FAISSIndex:
    """
    FAISS veya Scikit-Learn NearestNeighbors tabanlı dense retrieval.
    faiss-cpu kurulu olmadığında otomatik Scikit-Learn fallback sunar.
    """
    def __init__(self, dimension: int, n_lists: int = 256, n_probes: int = 32):
        self.dimension = dimension
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.index = None
        self.nn_fallback = None
        self.embeddings = None
        self._ids = None

    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """
        Dense index inşa et.
        embeddings: (N, D) float32, L2-normalized
        ids: (N,) — string veya int array of item IDs
        """
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings 2D olmalıdır. Verilen boyut: {embeddings.shape}")
            
        if embeddings.dtype != np.float32:
            try:
                embeddings = embeddings.astype(np.float32)
            except Exception as e:
                raise ValueError(f"Embeddings float32 tipine dönüştürülemedi: {e}")
                
        if len(embeddings) != len(ids):
            raise ValueError("Embeddings ve ids boyutları eşleşmiyor.")

        self._ids = np.array(ids, dtype=str)

        if HAS_FAISS:
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.n_lists)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.n_probes
        else:
            print("[!] faiss-cpu yüklü değil, Scikit-Learn NearestNeighbors fallback indeksi kullanılıyor...")
            self.embeddings = embeddings
            self.nn_fallback = NearestNeighbors(n_neighbors=50, algorithm="brute", metric="euclidean")
            self.nn_fallback.fit(embeddings)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        En yakın k komşuyu döndür.
        Returns: (scores: float32 array (Q,K)), (item_ids: str array (Q,K))
        """
        if self.index is None and self.nn_fallback is None:
            raise ValueError("Arama yapmadan önce index'i oluşturmak için build() çağrılmalıdır.")
            
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        if HAS_FAISS:
            scores, indices = self.index.search(query_vectors, k)
        else:
            scores, indices = self.nn_fallback.kneighbors(query_vectors, n_neighbors=k)

        item_ids = np.full(indices.shape, 'MISSING', dtype=object)
        valid_mask = indices != -1
        
        if valid_mask.any():
            item_ids[valid_mask] = self._ids[indices[valid_mask]]
            
        return scores.astype(np.float32), item_ids.astype(str)

    def save(self, index_path: str, manifest: bool = True) -> str:
        """Index'i binary dosyaya kaydet + manifest yaz."""
        if self.index is None and self.nn_fallback is None:
            raise ValueError("Kaydedilecek index bulunmuyor (build() çağrılmamış).")
            
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        ids_path = os.path.splitext(index_path)[0] + ".ids.npy"
        np.save(ids_path, self._ids)

        if HAS_FAISS:
            faiss.write_index(self.index, index_path)
        else:
            emb_path = os.path.splitext(index_path)[0] + ".embeddings.npy"
            np.save(emb_path, self.embeddings)

        manifest_path = ""
        if manifest:
            manifest_path = IndexVersion.write(
                index_path=index_path if HAS_FAISS else os.path.splitext(index_path)[0] + ".ids.npy",
                model_name="FAISS_IVFFlat" if HAS_FAISS else "Sklearn_NN",
                n_items=self.n_items,
                dimension=self.dimension,
                index_type="IVFFlat" if HAS_FAISS else "NearestNeighbors"
            )
            
        return manifest_path

    @classmethod
    def load(cls, index_path: str, verify: bool = True) -> 'FAISSIndex':
        """Kaydedilmiş index'i yükle."""
        ids_path = os.path.splitext(index_path)[0] + ".ids.npy"
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"ID dosyası bulunamadı: {ids_path}")

        item_ids = np.load(ids_path)
        
        if HAS_FAISS and os.path.exists(index_path):
            if verify:
                IndexVersion.verify(index_path)
            loaded_index = faiss.read_index(index_path)
            instance = cls(dimension=loaded_index.d)
            instance.index = loaded_index
            instance._ids = item_ids
            return instance
        else:
            emb_path = os.path.splitext(index_path)[0] + ".embeddings.npy"
            if os.path.exists(emb_path):
                embeddings = np.load(emb_path)
                instance = cls(dimension=embeddings.shape[1])
                instance.build(embeddings, item_ids)
                return instance
            else:
                raise FileNotFoundError("Index yüklenemedi: FAISS index veya sklearn embeddings dosyası eksik.")

    @property
    def n_items(self) -> int:
        """Index'teki öğe sayısı."""
        if self.index is not None:
            return self.index.ntotal
        elif self._ids is not None:
            return len(self._ids)
        return 0

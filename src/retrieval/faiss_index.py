import os
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

from .index_versioning import IndexVersion

class FAISSIndex:
    """
    IVF (Inverted File Index) tabanlı FAISS dense retrieval.
    ~1M item embedding için optimize edilmiş.
    """
    def __init__(self, dimension: int, n_lists: int = 256, n_probes: int = 32):
        if not HAS_FAISS:
            raise ImportError("faiss-cpu veya faiss-gpu yüklü değil: pip install faiss-cpu")
        self.dimension = dimension
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.index = None
        self._ids = None

    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """
        FAISS IVF index inşa et.
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

        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.n_lists)
        
        # Eğit ve ekle
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = self.n_probes
        
        self._ids = np.array(ids, dtype=str)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        En yakın k komşuyu döndür.
        Returns: (scores: float32 array (Q,K)), (item_ids: str array (Q,K))
        -1 dönen index slotları (yetersiz aday) için id 'MISSING' set et.
        """
        if self.index is None:
            raise ValueError("Arama yapmadan önce index'i oluşturmak için build() çağrılmalıdır.")
            
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
            
        scores, indices = self.index.search(query_vectors, k)
        
        item_ids = np.full(indices.shape, 'MISSING', dtype=object)
        valid_mask = indices != -1
        
        if valid_mask.any():
            item_ids[valid_mask] = self._ids[indices[valid_mask]]
            
        return scores, item_ids.astype(str)

    def save(self, index_path: str, manifest: bool = True) -> str:
        """Index'i binary dosyaya kaydet + manifest yaz. Manifest yolunu döndür."""
        if self.index is None:
            raise ValueError("Kaydedilecek index bulunmuyor (build() çağrılmamış).")
            
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        
        ids_path = os.path.splitext(index_path)[0] + ".ids.npy"
        np.save(ids_path, self._ids)
        
        manifest_path = ""
        if manifest:
            manifest_path = IndexVersion.write(
                index_path=index_path,
                model_name="FAISS_IVFFlat",
                n_items=self.n_items,
                dimension=self.dimension,
                index_type="IVFFlat"
            )
            
        return manifest_path

    @classmethod
    def load(cls, index_path: str, verify: bool = True) -> 'FAISSIndex':
        """Kaydedilmiş index'i yükle. verify=True ise SHA-256 doğrula."""
        if verify:
            IndexVersion.verify(index_path)
            
        loaded_index = faiss.read_index(index_path)
        dimension = loaded_index.d
        
        instance = cls(dimension=dimension)
        instance.index = loaded_index
        
        ids_path = os.path.splitext(index_path)[0] + ".ids.npy"
        if os.path.exists(ids_path):
            instance._ids = np.load(ids_path)
        else:
            raise FileNotFoundError(f"ID dosyası bulunamadı: {ids_path}")
            
        return instance

    @property
    def n_items(self) -> int:
        """Index'teki öğe sayısı."""
        if self.index is None:
            return 0
        return self.index.ntotal

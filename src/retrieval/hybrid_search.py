import pandas as pd
import numpy as np

from typing import Any, Optional

from .bm25_retriever import BM25Retriever

try:
    from .faiss_index import FAISSIndex
except ImportError:
    FAISSIndex = Any  # type: ignore

def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = 60,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    RRF score for each item = sum(weight_i / (k + rank_i))
    where rank_i is 1-based rank in each list.
    Returns sorted list of (item_id, rrf_score) descending.
    """
    rrf_scores = {}
    
    for rank, (item_id, _) in enumerate(bm25_results, 1):
        if item_id not in rrf_scores:
            rrf_scores[item_id] = 0.0
        rrf_scores[item_id] += bm25_weight / (k + rank)
        
    for rank, (item_id, _) in enumerate(dense_results, 1):
        if item_id not in rrf_scores:
            rrf_scores[item_id] = 0.0
        rrf_scores[item_id] += dense_weight / (k + rank)
        
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def hybrid_retrieve(
    query_text: str,
    query_embedding: np.ndarray,
    bm25_retriever: BM25Retriever,
    faiss_index: FAISSIndex,
    top_k: int = 100,
    rrf_k: int = 60,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    bm25_k: int = 200,
    dense_k: int = 200,
) -> pd.DataFrame:
    """
    Hibrit retrieval: BM25 + FAISS, RRF ile birleştirilmiş.
    Returns DataFrame with columns:
      item_id, bm25_score, dense_score, rrf_score, bm25_rank, dense_rank
    Eksik sütunlar (sadece birinde bulunan item) NaN.
    """
    bm25_results = bm25_retriever.retrieve(query_text, k=bm25_k)
    
    query_embedding_2d = np.expand_dims(query_embedding, axis=0) if len(query_embedding.shape) == 1 else query_embedding
    distances, item_ids = faiss_index.search(query_embedding_2d, k=dense_k)
    
    dense_results = []
    for i in range(len(item_ids[0])):
        item_id = item_ids[0][i]
        score = float(distances[0][i])
        if item_id != 'MISSING':
            dense_results.append((item_id, score))
            
    rrf_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight
    )
    
    data = []
    bm25_dict = {item_id: (score, rank) for rank, (item_id, score) in enumerate(bm25_results, 1)}
    dense_dict = {item_id: (score, rank) for rank, (item_id, score) in enumerate(dense_results, 1)}
    
    for item_id, rrf_score in rrf_results[:top_k]:
        b_score, b_rank = bm25_dict.get(item_id, (np.nan, np.nan))
        d_score, d_rank = dense_dict.get(item_id, (np.nan, np.nan))
        
        data.append({
            "item_id": item_id,
            "bm25_score": b_score,
            "dense_score": d_score,
            "rrf_score": rrf_score,
            "bm25_rank": b_rank,
            "dense_rank": d_rank
        })
        
    df = pd.DataFrame(data, columns=["item_id", "bm25_score", "dense_score", "rrf_score", "bm25_rank", "dense_rank"])
    return df

def linear_fusion(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Score normalization + ağırlıklı toplam.
    Her listedeki skorlar min-max ile [0,1]'e normalize edilir.
    """
    def normalize(results):
        if not results:
            return {}
        scores = [s for _, s in results]
        min_s = min(scores)
        max_s = max(scores)
        norm_dict = {}
        for item_id, score in results:
            if max_s > min_s:
                norm_dict[item_id] = (score - min_s) / (max_s - min_s)
            else:
                norm_dict[item_id] = 1.0 if score > 0 else 0.0
        return norm_dict
        
    bm25_norm = normalize(bm25_results)
    
    dense_norm = {}
    if dense_results:
        scores = [s for _, s in dense_results]
        min_s = min(scores)
        max_s = max(scores)
        for item_id, score in dense_results:
            if max_s > min_s:
                dense_norm[item_id] = (score - min_s) / (max_s - min_s)
            else:
                dense_norm[item_id] = 1.0
                
    all_items = set(bm25_norm.keys()) | set(dense_norm.keys())
    combined = []
    
    for item_id in all_items:
        b_score = bm25_norm.get(item_id, 0.0)
        d_score = dense_norm.get(item_id, 0.0)
        
        final_score = (b_score * bm25_weight) + (d_score * dense_weight)
        combined.append((item_id, final_score))
        
    return sorted(combined, key=lambda x: x[1], reverse=True)

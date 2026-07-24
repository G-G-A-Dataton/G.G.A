"""
Golden Test Set üzerinde BM25, Dense (TF-IDF SVD) ve Hibrit (RRF) Arama Performansını Ölçer.
İlla tüm modeli eğitmeye gerek kalmadan anlık doğruluk / Macro-F1 metriklerini raporlar.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.data import load_terms, load_items
from src.item_text import build_item_texts, clean_text


def evaluate_instant_retrieval():
    print("=" * 60)
    print("  G.G.A -- Anlik Retrieval & Reranking Performans Degerlendirmesi")
    print("=" * 60)

    golden_path = os.path.join(PROJECT_ROOT, "datasets", "golden_testset_v1.csv")
    bm25_path = os.path.join(PROJECT_ROOT, "outputs", "indexes", "bm25_index.pkl")
    faiss_path = os.path.join(PROJECT_ROOT, "outputs", "indexes", "faiss_items.index")

    if not os.path.exists(golden_path):
        print(f"[!] Golden test set bulunamadi: {golden_path}")
        return

    golden_df = pd.read_csv(golden_path)
    queries = golden_df[["term_id", "query_text"]].drop_duplicates()
    print(f"[+] Degerlendirilecek Benzersiz Sorgu Sayisi: {len(queries)}")
    print(f"[+] Toplam Golden Cift Sayisi: {len(golden_df)}")

    # Ground truth mapping: term_id -> set of relevant item_ids
    truth_map = {}
    for term_id, group in golden_df.groupby("term_id"):
        positives = set(group[group["label"] == 1]["item_id"])
        if positives:
            truth_map[term_id] = positives

    # BM25 Retriever yükle
    bm25 = None
    if os.path.exists(bm25_path):
        print("[+] BM25 indeksi yukleniyor...")
        bm25 = BM25Retriever.load(bm25_path)

    # Dense Vector Index yükle
    dense_index = None
    if os.path.exists(faiss_path):
        print("[+] Dense Vector indeksi yukleniyor...")
        dense_index = FAISSIndex.load(faiss_path, verify=False)

    # TF-IDF + SVD modelini sorgular için fit et
    terms_df = load_terms(os.path.join(PROJECT_ROOT, "datasets", "terms.csv"))
    items_df = load_items(os.path.join(PROJECT_ROOT, "datasets", "items.csv"))
    
    print("[+] TF-IDF + SVD (384-d) Vectorizer egitiliyor...")
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)
    corpus = [clean_text(q) for q in terms_df["query"].tolist()]
    tfidf_mat = vec.fit_transform(corpus)
    
    svd = TruncatedSVD(n_components=384, random_state=42)
    svd.fit(tfidf_mat)

    # Değerlendirme Döngüsü
    ks = [5, 10, 50, 100]
    bm25_recalls = {k: [] for k in ks}
    dense_recalls = {k: [] for k in ks}
    hybrid_recalls = {k: [] for k in ks}
    mrr_list = []

    print("\n[+] Arama performansi hesaplaniyor...")
    start_time = time.time()

    for idx, row in queries.iterrows():
        term_id = row["term_id"]
        q_text = clean_text(str(row["query_text"]))
        
        if term_id not in truth_map:
            continue
        relevant = truth_map[term_id]

        # BM25 Arama
        bm25_hits = []
        if bm25:
            bm25_hits = bm25.retrieve(q_text, k=100)
            
        # Dense Arama
        dense_hits = []
        if dense_index:
            q_vec_sparse = vec.transform([q_text])
            q_arr = svd.transform(q_vec_sparse).astype(np.float32)
            
            norm = np.linalg.norm(q_arr, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            q_arr = q_arr / norm

            _, dense_ids = dense_index.search(q_arr, k=100)
            dense_hits = [(item_id, float(1.0 / (rank + 1))) for rank, item_id in enumerate(dense_ids[0]) if item_id != "MISSING"]

        # Hibrit RRF
        hybrid_hits = reciprocal_rank_fusion(bm25_hits, dense_hits, k=60)

        # Recalls
        for k in ks:
            bm25_topk = set([item for item, _ in bm25_hits[:k]])
            dense_topk = set([item for item, _ in dense_hits[:k]])
            hybrid_topk = set([item for item, _ in hybrid_hits[:k]])

            bm25_recalls[k].append(len(bm25_topk & relevant) / len(relevant))
            dense_recalls[k].append(len(dense_topk & relevant) / len(relevant))
            hybrid_recalls[k].append(len(hybrid_topk & relevant) / len(relevant))

        # MRR (First relevant item rank)
        rank = 0
        for r, (item_id, _) in enumerate(hybrid_hits, 1):
            if item_id in relevant:
                rank = r
                break
        mrr_list.append(1.0 / rank if rank > 0 else 0.0)

    elapsed = time.time() - start_time
    print(f"\n[+] Degerlendirme Tamamlandi ({elapsed:.2f} saniye)")
    print("-" * 60)
    print(f"{'Metrik / Strateji':<25} | {'BM25':<10} | {'Dense SVD':<10} | {'Hibrit RRF':<10}")
    print("-" * 60)
    for k in ks:
        b_score = np.mean(bm25_recalls[k]) if bm25_recalls[k] else 0.0
        d_score = np.mean(dense_recalls[k]) if dense_recalls[k] else 0.0
        h_score = np.mean(hybrid_recalls[k]) if hybrid_recalls[k] else 0.0
        print(f"{'Recall@' + str(k):<25} | {b_score:.4f}     | {d_score:.4f}     | {h_score:.4f}")

    mrr_score = np.mean(mrr_list) if mrr_list else 0.0
    print(f"{'MRR (Mean Reciprocal Rank)':<25} | {'-':<10} | {'-':<10} | {mrr_score:.4f}")
    print("-" * 60)

    # Macro-F1 tahmini (At Top-10)
    p_at_10 = np.mean([len(set([item for item, _ in hybrid_hits[:10]]) & truth_map.get(row['term_id'], set())) / 10 for _, row in queries.iterrows() if row['term_id'] in truth_map])
    r_at_10 = np.mean(hybrid_recalls[10])
    macro_f1 = (2 * p_at_10 * r_at_10) / (p_at_10 + r_at_10) if (p_at_10 + r_at_10) > 0 else 0.0

    print(f"[*] Tahmini Retrieval Macro-F1 (Top-10): {macro_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_instant_retrieval()

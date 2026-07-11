"""
run_tfidf_experiments.py
========================
G.G.A Takımı — TF-IDF Hiperparametre Deney Tablosu (6 Temmuz Görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Bu script, TF-IDF vectorizer'ın temel hiperparametrelerini sistematik
olarak dener ve hangi kombinasyonun en iyi cosine ayırıcılığını
sağladığını raporlar.

Denenen parametreler:
  - ngram_range : (1,1), (1,2), (1,3)
  - max_features: 10_000, 30_000, 50_000
  - min_df      : 1, 2, 5

Çalıştırmak için:
  python run_tfidf_experiments.py
"""

import os
import sys
import itertools
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.negative_sampling import build_training_set
from src.tfidf_features    import build_tfidf_vectorizer, compute_tfidf_cosine_batch

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Deney parametreleri ──────────────────────────────────────────────────────
# Her parametre kombinasyonu denenecek (toplam 3 × 3 × 3 = 27 deney)
NGRAM_RANGES  = [(1, 1), (1, 2), (1, 3)]  # Unigram, bigram, trigram
MAX_FEATURES  = [10_000, 30_000, 50_000]  # Kelime dağarcığı boyutu
MIN_DFS       = [1, 2, 5]                 # En az kaç belgede geçmeli

# Deney boyutu (hız için küçük tutuyoruz)
SAMPLE_POS  = 1_000   # Kaç pozitif çift kullanılacak
NEG_RATIO   = 1       # Hız için 1:1
RANDOM_SEED = 42


def eval_tfidf_params(
    query_texts, item_texts, labels,
    terms_df, items_df,
    ngram_range, max_features, min_df,
):
    """
    Bir parametre kombinasyonu için TF-IDF cosine ayırıcılığını ölçer.

    Ayırıcılık = pozitif çiftlerin ortalama cosinesi - negatif çiftlerin ortalama cosinesi
    Bu fark ne kadar büyükse, TF-IDF bu iki grubu o kadar iyi ayırt edebiliyor.

    Parametreler
    ----------
    query_texts, item_texts : list of str
        Her satır için sorgu ve ürün metinleri.
    labels : array-like
        Her satır için label (1=pozitif, 0=negatif).
    terms_df, items_df : pd.DataFrame
        Tüm veri — vectorizer eğitimi için kullanılır.
    ngram_range : tuple
        TF-IDF n-gram aralığı.
    max_features : int
        Kelime dağarcığı boyutu.
    min_df : int
        Minimum belge frekansı.

    Döndürür
    -------
    dict
        Deney sonuçları sözlüğü.
    """
    t0 = time.time()

    # Vectorizer eğit
    vectorizer = build_tfidf_vectorizer(
        terms_df, items_df,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    train_sec = time.time() - t0

    # Cosine similarity hesapla
    t1 = time.time()
    cosines = compute_tfidf_cosine_batch(
        query_texts, item_texts, vectorizer, batch_size=5_000
    )
    infer_sec = time.time() - t1

    labels = np.array(labels)
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Ayırıcılık metrikleri
    pos_mean = cosines[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
    neg_mean = cosines[neg_mask].mean() if neg_mask.sum() > 0 else 0.0
    separation = pos_mean - neg_mean  # Ana metrik: ne kadar iyi ayırt ediyor?

    # Vocabulary boyutu (gerçekte kaç kelime öğrenildi)
    vocab_size = len(vectorizer.vocabulary_)

    return {
        "ngram_range"  : str(ngram_range),
        "max_features" : max_features,
        "min_df"       : min_df,
        "vocab_size"   : vocab_size,
        "pos_cosine"   : round(pos_mean, 4),
        "neg_cosine"   : round(neg_mean, 4),
        "separation"   : round(separation, 4),   # Yüksek = iyi
        "train_sec"    : round(train_sec, 1),
        "infer_sec"    : round(infer_sec, 1),
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  G.G.A — TF-IDF Hiperparametre Deneyleri (6 Temmuz)")
    print(f"  {len(NGRAM_RANGES)} ngram x {len(MAX_FEATURES)} max_feat x {len(MIN_DFS)} min_df"
          f" = {len(NGRAM_RANGES)*len(MAX_FEATURES)*len(MIN_DFS)} deney")
    print("=" * 65)

    # ─── 1. Veri hazırlama ────────────────────────────────────────────────────
    print("\n[1/3] Veri yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    # Küçük örnek eğitim seti oluştur
    print(f"[2/3] {SAMPLE_POS} pozitif + {SAMPLE_POS*NEG_RATIO} negatif ornek hazirlaniyor...")
    sample = build_training_set(
        train_raw.sample(SAMPLE_POS, random_state=RANDOM_SEED),
        items_df, ratio=NEG_RATIO, random_state=RANDOM_SEED, verbose=False
    )
    merged = sample.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")

    # Metin sütunlarını hazırla
    query_texts = merged["query"].fillna("").tolist()
    item_texts  = (
        merged["title"].fillna("") + " " +
        merged["category"].astype(str).str.replace("/", " ", regex=False).fillna("") + " " +
        merged["brand"].fillna("")
    ).tolist()
    labels = merged["label"].values

    # ─── 2. Tüm kombinasyonları dene ─────────────────────────────────────────
    print("\n[3/3] Deneyler basliyor...\n")
    results = []
    total = len(NGRAM_RANGES) * len(MAX_FEATURES) * len(MIN_DFS)

    for i, (ngram, maxf, mindf) in enumerate(
        itertools.product(NGRAM_RANGES, MAX_FEATURES, MIN_DFS), start=1
    ):
        print(f"  [{i:02d}/{total}] ngram={ngram}, max_feat={maxf:,}, min_df={mindf} ...", end=" ", flush=True)
        try:
            result = eval_tfidf_params(
                query_texts, item_texts, labels,
                terms_df, items_df,
                ngram_range=ngram, max_features=maxf, min_df=mindf
            )
            results.append(result)
            print(f"separation={result['separation']:.4f}  [{result['train_sec']}s eğit]")
        except Exception as e:
            print(f"HATA: {e}")

    # ─── 3. Sonuçları sırala ve göster ───────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("separation", ascending=False)

    print("\n" + "=" * 65)
    print("  SONUCLAR (Ayiriciliga gore sirali — yuksek = iyi)")
    print("=" * 65)
    print(results_df[[
        "ngram_range", "max_features", "min_df",
        "vocab_size", "pos_cosine", "neg_cosine", "separation", "train_sec"
    ]].to_string(index=False))

    # En iyi 3'ü vurgula
    print("\n  En iyi 3 kombinasyon:")
    for rank, (_, row) in enumerate(results_df.head(3).iterrows(), start=1):
        print(f"  {rank}. ngram={row['ngram_range']}, max_feat={row['max_features']:,}, "
              f"min_df={row['min_df']}  -> separation={row['separation']:.4f}")

    # Önerilen kombinasyon
    best = results_df.iloc[0]
    print(f"\n  ONERI: ngram={best['ngram_range']}, max_features={best['max_features']:,}, "
          f"min_df={best['min_df']}")
    print(f"  Bu kombinasyon src/tfidf_features.py icin kullanilmali.")

    # CSV olarak kaydet
    out_path = os.path.join(OUTPUT_DIR, "tfidf_deney_sonuclari.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Sonuclar kaydedildi: {out_path}")
    print("=" * 65)

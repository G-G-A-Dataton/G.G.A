"""
notebooks/05_bm25_hard_negative_mert.py
========================================
G.G.A Takımı — BM25 Hard Negative Üretimi (Gün 6-7 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (6-7 Temmuz görevi)

Üretim mantığı src/bm25_hard_negative.py'de yaşıyor, burada tekrar
yazılmıyor. Bu script sadece:
  1. Veriyi yükler
  2. Hard negative'leri üretir (varsayılan: top_n=50, ratio=3)
  3. Sızıntı kontrolü yapar (src.negative_sampling.verify_no_leakage)
  4. outputs/negative_bm25_v1.csv olarak kaydeder — Ömer'in
     run_hard_neg_comparison.py scripti bu dosyayı varsayılan olarak arıyor:
       python run_hard_neg_comparison.py --bm25 outputs/negative_bm25_v1.csv

Kullanım:
  python notebooks/05_bm25_hard_negative_mert.py            # tam veri
  python notebooks/05_bm25_hard_negative_mert.py --sample 2000  # hızlı deneme
    (2000 benzersiz sorguyla, tüm 50K yerine — indeks kurulumu ve tarama
    süresini görmek için önce bunu çalıştırmanı öneririm)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # src.* import edebilmek için

from src.data import load_terms, load_items  # noqa: E402
from src.bm25_hard_negative import generate_bm25_hard_negatives  # noqa: E402
from src.negative_sampling import verify_no_leakage  # noqa: E402

DATA = ROOT / "datasets"
OUT = ROOT / "outputs"

SEED = 42
TOP_N = 50
RATIO = 3


def main(sample: int | None) -> None:
    print("Veriler yukleniyor...")
    terms_df = load_terms(DATA / "terms.csv")
    items_df = load_items(DATA / "items.csv")
    train_df = pd.read_csv(
        DATA / "training_pairs.csv",
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )
    print(f"pozitif cift: {len(train_df):,} | urun katalogu: {len(items_df):,} | "
          f"benzersiz sorgu: {train_df['term_id'].nunique():,}")

    if sample:
        sample_terms = train_df["term_id"].drop_duplicates().sample(n=sample, random_state=SEED)
        train_df = train_df[train_df["term_id"].isin(sample_terms)]
        print(f"[--sample {sample}] {len(train_df):,} pozitif cift ile hizli deneme modunda calisiyor.")

    hard_negatives = generate_bm25_hard_negatives(
        train_df, terms_df, items_df, top_n=TOP_N, ratio=RATIO,
    )

    print("\nSizinti kontrolu yapiliyor...")
    ok = verify_no_leakage(hard_negatives, train_df)
    if not ok:
        raise SystemExit("Sizinti tespit edildi, dosya kaydedilmedi.")

    hard_negatives.insert(0, "id", [f"NEG_BM25_{i:07d}" for i in range(len(hard_negatives))])

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / ("negative_bm25_v1_sample.csv" if sample else "negative_bm25_v1.csv")
    hard_negatives.to_csv(out_path, index=False)
    print(f"\n-> {out_path}: {len(hard_negatives):,} hard negative kaydedildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 hard negative uretimi")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sadece bu kadar benzersiz sorguyla hizli deneme yap (tam kosu icin verme)",
    )
    args = parser.parse_args()
    main(sample=args.sample)

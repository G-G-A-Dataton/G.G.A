"""
scripts/data/run_golden_testset_build.py
=========================================
G.G.A Takımı — Golden Test Set Builder CLI

Eğitim verisinden stratified sorgu örneklemesi ve BM25 hard negativeler
kullanarak golden test set parquet dosyası üretir.

Kullanım:
  python scripts/data/run_golden_testset_build.py
  python scripts/data/run_golden_testset_build.py --n-queries 500 --negatives-per-query 20
"""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.training.run_train_full_v2 import sha256_file
from src.data import load_items, load_terms
from src.golden_testset import build_golden_testset, write_golden_testset_manifest

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Golden test set parquet üretici")
    parser.add_argument("--n-queries", type=int, default=500, help="Benzersiz sorgu sayısı")
    parser.add_argument("--negatives-per-query", type=int, default=20, help="Sorgu başı BM25 hard negatif")
    parser.add_argument("--seed", type=int, default=42, help="Rastgele seed")
    parser.add_argument(
        "--output-parquet",
        default=os.path.join(DATA_DIR, "golden_testset_v1.parquet"),
        help="Çıktı parquet dosyası yolu",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 60)
    print("  G.G.A — Golden Test Set Builder")
    print("=" * 60)

    print("\n[1/3] Kaynak veriler yükleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))
    pairs_path = os.path.join(DATA_DIR, "training_pairs.csv")
    positives_df = pd.read_csv(
        pairs_path,
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )
    print(f"  terms.csv         : {len(terms_df):,} satır")
    print(f"  items.csv         : {len(items_df):,} satır")
    print(f"  training_pairs.csv: {len(positives_df):,} satır")

    print("\n[2/3] Golden test set oluşturuluyor...")
    golden_df = build_golden_testset(
        training_pairs=positives_df,
        items_df=items_df,
        terms_df=terms_df,
        n_queries=args.n_queries,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
        verbose=True,
    )

    print("\n[3/3] Parquet/CSV ve SHA-256 manifest kaydediliyor...")
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    out_file = args.output_parquet
    try:
        golden_df.to_parquet(out_file, index=False)
    except ImportError:
        out_file = args.output_parquet.replace(".parquet", ".csv")
        golden_df.to_csv(out_file, index=False)
        print(f"[!] pyarrow/fastparquet bulunamadı. CSV olarak kaydediliyor: {out_file}")

    source_hashes = {
        "terms.csv": sha256_file(os.path.join(DATA_DIR, "terms.csv")),
        "items.csv": sha256_file(os.path.join(DATA_DIR, "items.csv")),
        "training_pairs.csv": sha256_file(pairs_path),
    }

    manifest_path = write_golden_testset_manifest(
        parquet_path=out_file,
        n_queries=args.n_queries,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
        source_hashes=source_hashes,
    )

    print(f"[+] Golden test set kaydedildi : {out_file}")
    print(f"[+] Manifest kaydedildi        : {manifest_path}")


if __name__ == "__main__":
    main()

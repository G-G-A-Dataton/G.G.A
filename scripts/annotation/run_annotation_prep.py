"""
scripts/annotation/run_annotation_prep.py
==========================================
G.G.A Takımı — Annotation Export Script

Model tahminlerinden ve aday havuzundan belirsizlik örneklemesi (Active Learning)
yaparak insan doğrulaması (Labeling / Annotation) için CSV/JSON çıktı üretir.

Kullanım:
  python scripts/annotation/run_annotation_prep.py --n-samples 500 --strategy entropy
"""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.active_learning import compute_uncertainty_scores, sample_uncertain_pairs
from src.data import load_items, load_terms

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Active learning annotation export")
    parser.add_argument("--n-samples", type=int, default=500, help="Örnekleme sayısı")
    parser.add_argument("--strategy", default="entropy", choices=["entropy", "margin", "least_conf", "hybrid"])
    parser.add_argument(
        "--output-csv",
        default=os.path.join(OUTPUT_DIR, "annotation_batch_v1.csv"),
        help="Çıktı CSV dosya yolu",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 60)
    print("  G.G.A — Active Learning Annotation Export")
    print("=" * 60)
    print(f"  Strateji : {args.strategy}")
    print(f"  Örnekleme: {args.n_samples:,}")

    # OOF veya aday olasılıklarını yükle
    oof_preds_path = os.path.join(OUTPUT_DIR, "ensemble_artifacts", "oof_lgbm.npy")
    oof_meta_path = os.path.join(OUTPUT_DIR, "ensemble_artifacts", "oof_metadata.csv")

    if not os.path.exists(oof_preds_path) or not os.path.exists(oof_meta_path):
        print("⚠️  OOF tahminleri veya metadata bulunamadı.")
        print("   Önce 'python scripts/run_production.py --stage train' çalıştırın.")
        sys.exit(1)

    print("\n[1/3] OOF tahminleri ve metadata yükleniyor...")
    import numpy as np
    scores = np.load(oof_preds_path)
    meta = pd.read_csv(oof_meta_path)
    meta["proba"] = scores

    print("\n[2/3] Belirsizlik skorları hesaplanıyor ve örnekleniyor...")
    meta_with_unc = compute_uncertainty_scores(meta, proba_col="proba")
    sampled = sample_uncertain_pairs(
        meta_with_unc,
        n_samples=args.n_samples,
        strategy=args.strategy,
        max_per_query=5,
    )

    # Detayları zenginleştir (query_text, item_title)
    print("\n[3/3] Ürün ve sorgu detayları ekleniyor...")
    terms = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items = load_items(os.path.join(DATA_DIR, "items.csv"))

    terms_map = terms.set_index("term_id")["query"].to_dict()
    items_map = items.set_index("item_id")["title"].to_dict()

    sampled["query_text"] = sampled["term_id"].astype(str).map(terms_map)
    sampled["item_title"] = sampled["item_id"].astype(str).map(items_map)

    # Annotator için temiz kolon seçimi
    cols = ["term_id", "item_id", "query_text", "item_title", "proba", "uncertainty_entropy", "uncertainty_margin"]
    out_cols = [c for c in cols if c in sampled.columns]
    final_df = sampled[out_cols].copy()

    # İnsan etiketleme için boş label kolonu ekle
    final_df["human_verified_label"] = ""  # Annotator 1 veya 0 yazacak
    final_df["annotator_notes"] = ""

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    final_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ Annotation batch kaydedildi: {args.output_csv}")
    print(f"   Satır sayısı: {len(final_df):,}")


if __name__ == "__main__":
    main()

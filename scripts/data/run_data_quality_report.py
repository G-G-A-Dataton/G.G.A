"""
scripts/data/run_data_quality_report.py
========================================
G.G.A Takımı — Veri Kalite Raporu

Tüm data quality kontrollerini çalıştırır ve JSON rapor üretir.

Kullanım:
  python scripts/data/run_data_quality_report.py
  python scripts/data/run_data_quality_report.py --output outputs/data_quality_report.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items, load_terms
from src.data_quality import run_full_quality_report


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Veri kalite raporu üret")
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "data_quality_report.json"),
        help="Rapor çıktı yolu (.json)",
    )
    parser.add_argument(
        "--include-training-pairs", action="store_true",
        help="training_pairs.csv label consistency kontrolü yap",
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Çıktıyı bastır",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.no_verbose

    print("=" * 60)
    print("  G.G.A — Veri Kalite Raporu")
    print("=" * 60)

    # Veri yükleme
    print("\n[1/2] Veriler yükleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))
    print(f"  terms.csv  : {len(terms_df):,} satır")
    print(f"  items.csv  : {len(items_df):,} satır")

    training_pairs_df = None
    if args.include_training_pairs:
        pairs_path = os.path.join(DATA_DIR, "training_pairs.csv")
        if os.path.exists(pairs_path):
            training_pairs_df = pd.read_csv(
                pairs_path,
                dtype={"term_id": "string", "item_id": "string", "label": "int8"},
            )
            print(f"  training_pairs.csv: {len(training_pairs_df):,} satır")
        else:
            print(f"  ⚠️  training_pairs.csv bulunamadı: {pairs_path}")

    # Kalite kontrolü
    print("\n[2/2] Kalite kontrolleri çalıştırılıyor...")
    report = run_full_quality_report(
        terms_df=terms_df,
        items_df=items_df,
        training_pairs_df=training_pairs_df,
        verbose=verbose,
    )

    # Metadata ekle
    report["_meta"] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "terms_count": len(terms_df),
        "items_count": len(items_df),
        "training_pairs_count": len(training_pairs_df) if training_pairs_df is not None else None,
    }

    # JSON kaydet
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def _json_serializable(obj):
        """JSON serialize edilemeyen tipleri dönüştür."""
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        return str(obj)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_serializable)

    print(f"\n[+] Rapor kaydedildi: {args.output}")

    # Özet istatistikler
    dup = report.get("duplicate_queries", {})
    if "count" in dup:
        status = "[OK] Temiz" if dup["count"] == 0 else f"[!] {dup['count']} duplicate"
        print(f"  Duplicate queries  : {status}")

    lc = report.get("label_consistency", {})
    if "violations" in lc:
        status = "[OK] Tutarlı" if lc["violations"] == 0 else f"[X] {lc['violations']} ihlal"
        print(f"  Label consistency  : {status}")

    cov = report.get("attribute_coverage", {})
    if "overall_non_null_rate" in cov:
        print(f"  Attribute coverage : {cov['overall_non_null_rate']:.1%}")

    return report


if __name__ == "__main__":
    main()

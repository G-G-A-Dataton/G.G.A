"""
scripts/analysis/run_segment_report.py
=======================================
G.G.A Takımı — Segment Analysis Report CLI

Kategori, sorgu uzunluğu ve öznitelik segmentlerinde metrik raporu üretir.

Kullanım:
  python scripts/analysis/run_segment_report.py
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.segment_analysis import compute_segment_breakdown

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Segment analysis report runner")
    parser.add_argument(
        "--eval-dataset",
        default=os.path.join(DATA_DIR, "golden_testset_v1.parquet"),
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(OUTPUT_DIR, "segment_report.json"),
    )
    parser.add_argument(
        "--output-md",
        default=os.path.join(DOCS_DIR, "segment_report.md"),
    )
    return parser.parse_args(argv)


def _load_data(path: str) -> pd.DataFrame:
    full_path = os.path.abspath(path)
    if os.path.exists(full_path):
        try:
            return pd.read_parquet(full_path)
        except Exception:
            csv_path = full_path.replace(".parquet", ".csv")
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
    # Dummy fallback
    print(f"[!] {path} bulunamadı. Sentetik eval verisi kullanılıyor.")
    return pd.DataFrame({
        "term_id": ["q1"] * 5 + ["q2"] * 5,
        "item_id": [f"i{i}" for i in range(1, 11)],
        "score": [0.9, 0.8, 0.7, 0.6, 0.1, 0.85, 0.75, 0.65, 0.2, 0.1],
        "label": [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "query_text": ["siyah bot 42 beden"] * 5 + ["kırmızı elbise"] * 5,
        "item_category": ["Ayakkabı/Erkek"] * 5 + ["Giyim/Kadın"] * 5,
    })


def main(argv=None):
    args = parse_args(argv)
    df = _load_data(args.eval_dataset)

    print("=" * 65)
    print("  G.G.A — Segment Analysis Report")
    print("=" * 65)

    breakdowns = compute_segment_breakdown(df)

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)

    json_dict = {name: df_b.to_dict("records") for name, df_b in breakdowns.items()}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)

    # Save Markdown
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Segment Analysis Raporu\n",
        f"**Tarih:** {now}  \n\n---\n",
    ]

    for title, df_b in breakdowns.items():
        lines.append(f"## {title.replace('_', ' ').title()}\n\n")
        lines.append("| Segment | Sorgu Sayısı | Çift Sayısı | Recall@10 | NDCG@10 | MRR | ECE |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for _, row in df_b.iterrows():
            lines.append(
                f"| **{row['segment_name']}** | {row['n_queries']:,} | {row['n_pairs']:,} | "
                f"{row['recall@10']:.4f} | {row['ndcg@10']:.4f} | {row['mrr']:.4f} | {row['ece']:.4f} |\n"
            )
        lines.append("\n")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[+] Segment JSON kaydedildi: {args.output_json}")
    print(f"[+] Segment Rapor kaydedildi: {args.output_md}")


if __name__ == "__main__":
    main()

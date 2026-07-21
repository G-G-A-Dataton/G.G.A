"""Run ablations over recorded outputs from real retrieval and ranking stages.

No variant score is synthesized from labels. A missing stage output is a
configuration error, not a reason to invent a score and continue.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration import evaluate_calibration
from src.experiment_tracker import ExperimentTracker
from src.retrieval_metrics import evaluation_report

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

VARIANTS = (
    ("A", "BM25", "bm25_score"),
    ("B", "Dense Retrieval", "dense_score"),
    ("C", "Hybrid RRF", "rrf_score"),
    ("D", "Hybrid Linear", "linear_score"),
    ("E", "Hybrid + Cross-Encoder", "rerank_score"),
    ("F", "Hybrid + Reranker + Calibration", "calibrated_score"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate materialized ablation outputs")
    parser.add_argument("--eval-dataset", default=os.path.join(DATA_DIR, "golden_testset_verified_v1.parquet"))
    parser.add_argument("--output-csv", default=os.path.join(OUTPUT_DIR, "ablation_matrix.csv"))
    parser.add_argument("--output-md", default=os.path.join(DOCS_DIR, "ablation_matrix_report.md"))
    return parser.parse_args(argv)


def _load_and_validate(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    frame = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    required = {"term_id", "item_id", "label", "label_status", *(score for _, _, score in VARIANTS)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Ablation dataset lacks materialized outputs: " + ", ".join(missing))
    if frame.empty or not frame["label"].isin([0, 1]).all():
        raise ValueError("Evaluation dataset must contain non-empty binary labels")
    if not set(frame["label_status"].dropna()).issubset({"verified_positive", "verified_negative"}):
        raise ValueError("Ablation metrics require fully verified labels")
    return frame


def _evaluate_variant(code: str, name: str, frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    report = evaluation_report(frame, ks=[10, 50, 100], score_col=score_column, verbose=False)
    scores = frame[score_column].to_numpy(dtype=float)
    ece = float("nan")
    if np.isfinite(scores).all() and ((scores >= 0.0) & (scores <= 1.0)).all():
        ece = evaluate_calibration(frame["label"], scores)["ece"]
    elapsed = time.perf_counter() - started_at
    queries = frame["term_id"].nunique()
    return {
        "code": code,
        "variant": name,
        "score_column": score_column,
        "recall@50": report.get("recall@50", 0.0),
        "recall@100": report.get("recall@100", 0.0),
        "precision@10": report.get("precision@10", 0.0),
        "ndcg@10": report.get("ndcg@10", 0.0),
        "mrr": report.get("mrr", 0.0),
        "ece": ece,
        "latency_ms": elapsed / queries * 1000.0 if queries else 0.0,
        "throughput_qps": queries / elapsed if elapsed else 0.0,
    }


def run_ablation_suite(dataset_path: str) -> pd.DataFrame:
    frame = _load_and_validate(dataset_path)
    return pd.DataFrame([_evaluate_variant(code, name, frame, score) for code, name, score in VARIANTS])


def write_ablation_reports(results: pd.DataFrame, output_csv: str, output_md: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    results.to_csv(output_csv, index=False)
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = "| Code | Variant | Recall@50 | Recall@100 | Precision@10 | NDCG@10 | MRR | ECE | Latency ms | QPS |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = "".join(
        f"| {row.code} | {row.variant} | {row['recall@50']:.4f} | {row['recall@100']:.4f} | {row['precision@10']:.4f} | {row['ndcg@10']:.4f} | {row.mrr:.4f} | {row.ece:.4f} | {row.latency_ms:.2f} | {row.throughput_qps:.1f} |\n"
        for _, row in results.iterrows()
    )
    with open(output_md, "w", encoding="utf-8") as report_file:
        report_file.write(f"# Real-output Ablation Matrix\n\nGenerated: {generated_at}\n\n{header}{rows}\n")


def main(argv=None):
    args = parse_args(argv)
    results = run_ablation_suite(args.eval_dataset)
    write_ablation_reports(results, args.output_csv, args.output_md)
    tracker = ExperimentTracker.from_config(os.path.join(PROJECT_ROOT, "configs", "mlflow.yaml"))
    with tracker.start_run(run_name="ablation_matrix_eval"):
        for _, row in results.iterrows():
            tracker.log_metrics({f"ablation_{row.code}_{key}": float(row[key]) for key in ("recall@50", "recall@100", "ndcg@10", "mrr", "ece") if pd.notna(row[key])})
        tracker.log_artifact(args.output_md)


if __name__ == "__main__":
    main()

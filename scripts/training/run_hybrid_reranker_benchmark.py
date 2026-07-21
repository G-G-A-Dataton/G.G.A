"""Evaluate materialized hybrid-retrieval and reranking stage outputs.

This command is deliberately an evaluator, not a model simulator.  It accepts
only a dataset whose candidates were scored by the real BM25, dense, hybrid,
reranker, and calibration stages.  It also rejects unverified hard-negative
candidates, because they are not evaluation ground truth.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration import evaluate_calibration
from src.experiment_tracker import ExperimentTracker
from src.reranker.listwise import rerank_by_combination
from src.retrieval_metrics import evaluation_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate real hybrid/reranker outputs")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "benchmark_e2e.yaml"))
    parser.add_argument("--eval-dataset", default=None)
    parser.add_argument("--output-md", default=os.path.join(PROJECT_ROOT, "docs", "latest_benchmark_report.md"))
    parser.add_argument("--output-csv", default=os.path.join(PROJECT_ROOT, "outputs", "latest_benchmark_metrics.csv"))
    return parser.parse_args(argv)


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    import yaml

    with open(path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _load_eval_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _validate_evaluation_contract(frame: pd.DataFrame, config: dict) -> dict:
    """Ensure every reported number comes from real outputs and verified labels."""
    contract = config.get("evaluation_contract", {})
    columns = {
        "bm25": contract.get("bm25_score_column", "bm25_score"),
        "dense": contract.get("dense_score_column", "dense_score"),
        "hybrid": contract.get("hybrid_score_column", "hybrid_score"),
        "rerank": contract.get("rerank_score_column", "rerank_score"),
        "calibrated": contract.get("calibrated_score_column", "calibrated_score"),
    }
    required = {"term_id", "item_id", "label", *columns.values()}
    if contract.get("require_verified_labels", True):
        status_column = contract.get("label_status_column", "label_status")
        required.add(status_column)
        valid_statuses = set(contract.get("accepted_label_statuses", []))
        observed_statuses = set(frame.get(status_column, pd.Series(dtype="string")).dropna())
        if not valid_statuses or not observed_statuses.issubset(valid_statuses):
            raise ValueError(
                "Evaluation labels are not fully verified. BM25 hard-negative "
                "candidates must be human-annotated before metrics are reported."
            )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Evaluation dataset lacks real stage outputs: " + ", ".join(missing))
    if frame.empty or not frame["label"].isin([0, 1]).all():
        raise ValueError("Evaluation dataset must contain non-empty binary labels")
    for name in ("rerank", "calibrated"):
        values = frame[columns[name]]
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise ValueError(f"{columns[name]} must contain finite probabilities in [0, 1]")
    return columns


def run_benchmark(config_path: str, dataset_override: str | None = None) -> dict[str, Any]:
    started_at = time.time()
    config = _load_config(config_path)
    dataset_path = dataset_override or config.get("eval_dataset")
    if not dataset_path:
        raise ValueError("eval_dataset must be set in the benchmark config")
    frame = _load_eval_data(dataset_path)
    columns = _validate_evaluation_contract(frame, config)
    n_queries = int(frame["term_id"].nunique())

    retrieval = evaluation_report(frame, ks=[10, 50, 100], score_col=columns["hybrid"], verbose=False)
    reranker_config = config.get("reranker", {})
    reranked = rerank_by_combination(
        frame,
        first_stage_score_col=columns["hybrid"],
        rerank_score_col=columns["rerank"],
        weight_first=reranker_config.get("blend_weight_first", 0.3),
        weight_rerank=reranker_config.get("blend_weight_reranker", 0.7),
    )
    rerank = evaluation_report(reranked, ks=[10, 50], score_col="combined_rank_score", verbose=False)
    n_bins = config.get("calibration", {}).get("n_bins", 10)
    raw_calibration = evaluate_calibration(frame["label"], frame[columns["rerank"]], n_bins=n_bins)
    calibrated = evaluate_calibration(frame["label"], frame[columns["calibrated"]], n_bins=n_bins)
    elapsed = time.time() - started_at
    metrics = {
        **{f"retrieval_{key}": float(value) for key, value in retrieval.items()},
        **{f"rerank_{key}": float(value) for key, value in rerank.items()},
        "ece_raw": raw_calibration["ece"],
        "mce_raw": raw_calibration["mce"],
        "ece_calibrated": calibrated["ece"],
        "mce_calibrated": calibrated["mce"],
        "latency_total_sec": elapsed,
        "throughput_qps": n_queries / elapsed if elapsed else 0.0,
    }
    return {"metrics": metrics, "n_queries": n_queries, "n_rows": len(frame), "dataset_path": dataset_path, "config": config}


def write_benchmark_report(results: dict[str, Any], output_md: str, output_csv: str) -> None:
    metrics = results["metrics"]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    pd.DataFrame([{"metric": name, "value": value} for name, value in metrics.items()]).to_csv(output_csv, index=False)
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Hybrid Retrieval and Reranker Benchmark

**Generated:** {generated_at}  
**Dataset:** `{results['dataset_path']}` ({results['n_queries']:,} queries, {results['n_rows']:,} pairs)  
**Validation rule:** all labels were verified and all scores came from materialized model stages.

| Metric | Value |
|---|---:|
| Recall@10 | {metrics.get('retrieval_recall@10', 0):.4f} |
| Recall@50 | {metrics.get('retrieval_recall@50', 0):.4f} |
| Recall@100 | {metrics.get('retrieval_recall@100', 0):.4f} |
| Rerank NDCG@10 | {metrics.get('rerank_ndcg@10', 0):.4f} |
| Rerank Precision@10 | {metrics.get('rerank_precision@10', 0):.4f} |
| Rerank MRR | {metrics.get('rerank_mrr', 0):.4f} |
| Raw ECE | {metrics.get('ece_raw', 0):.4f} |
| Calibrated ECE | {metrics.get('ece_calibrated', 0):.4f} |
| Throughput (QPS) | {metrics.get('throughput_qps', 0):.1f} |
"""
    with open(output_md, "w", encoding="utf-8") as report_file:
        report_file.write(report)


def main(argv=None):
    args = parse_args(argv)
    results = run_benchmark(args.config, args.eval_dataset)
    write_benchmark_report(results, args.output_md, args.output_csv)
    tracker = ExperimentTracker.from_config(os.path.join(PROJECT_ROOT, "configs", "mlflow.yaml"))
    with tracker.start_run(run_name="e2e_hybrid_reranker_benchmark"):
        tracker.log_params({"eval_dataset": results["dataset_path"], "n_queries": results["n_queries"]})
        tracker.log_metrics(results["metrics"])
        tracker.log_artifact(args.output_md)


if __name__ == "__main__":
    main()

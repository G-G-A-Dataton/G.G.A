"""Calibrate hard-negative mixtures against unlabeled submission candidates."""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.bm25_hard_negative import BM25Index, standardize_item_text
from src.candidate_sampling import build_test_shaped_training_set, sample_complete_terms
from src.data import load_items, load_terms
from src.features import build_features
from src.tfidf_features import add_tfidf_features, build_tfidf_vectorizer


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "candidate_shift_analysis.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "candidate_shift_analysis.md")
RANDOM_SEED = 42
SHIFT_FEATURES = [
    "query_title_overlap",
    "query_title_coverage",
    "query_category_overlap",
    "tfidf_cosine",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Match training candidate hardness to unlabeled test candidates"
    )
    parser.add_argument("--sample-train-terms", type=int, default=1_000)
    parser.add_argument("--sample-test-terms", type=int, default=1_000)
    parser.add_argument(
        "--bm25-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument("--category-hard-fraction", type=float, default=0.50)
    parser.add_argument("--bm25-top-n", type=int, default=200)
    parser.add_argument("--bm25-max-df-ratio", type=float, default=0.15)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def normalized_quantile_distance(reference, candidate, quantiles=None):
    """Return a scale-normalized marginal distribution distance."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.ndim != 1
        or candidate.ndim != 1
        or len(reference) == 0
        or len(candidate) == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("distribution inputs must be non-empty finite vectors")
    quantiles = (
        np.linspace(0.0, 1.0, 101)
        if quantiles is None
        else np.asarray(quantiles, dtype=np.float64)
    )
    if (
        quantiles.ndim != 1
        or len(quantiles) == 0
        or not np.isfinite(quantiles).all()
        or ((quantiles < 0.0) | (quantiles > 1.0)).any()
    ):
        raise ValueError("quantiles must be finite values in [0, 1]")
    raw_distance = float(
        np.abs(
            np.quantile(reference, quantiles)
            - np.quantile(candidate, quantiles)
        ).mean()
    )
    scale = max(float(np.std(reference)), 1e-12)
    return raw_distance, raw_distance / scale


def _load_test_sample(terms, training_term_ids, sample_size):
    eligible = terms.loc[
        ~terms["term_id"].isin(training_term_ids), "term_id"
    ].to_numpy()
    if sample_size > len(eligible):
        raise ValueError("--sample-test-terms exceeds available submission terms")
    rng = np.random.default_rng(RANDOM_SEED)
    selected = set(rng.choice(eligible, size=sample_size, replace=False).tolist())
    chunks = []
    reader = pd.read_csv(
        os.path.join(DATA_DIR, "submission_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
        chunksize=250_000,
    )
    try:
        for chunk in reader:
            selected_rows = chunk[chunk["term_id"].isin(selected)]
            if len(selected_rows):
                chunks.append(selected_rows)
    finally:
        reader.close()
    result = pd.concat(chunks, ignore_index=True)
    if result["term_id"].nunique() != sample_size:
        raise RuntimeError("test sampling did not recover every selected term")
    return result


def _feature_frame(pairs, terms, items, vectorizer):
    frame = pairs.merge(
        terms, on="term_id", how="left", validate="many_to_one"
    ).merge(items, on="item_id", how="left", validate="many_to_one")
    if frame[["query", "title"]].isna().any().any():
        raise ValueError("candidate sample contains unresolved term or item IDs")
    frame = build_features(frame, verbose=False, copy=False)
    return add_tfidf_features(frame, vectorizer, verbose=False, copy=False)


def _distribution_rows(fraction, train_frame, test_frame, source_rows):
    rows = []
    for feature in SHIFT_FEATURES:
        train_values = train_frame[feature].to_numpy(dtype=np.float64)
        test_values = test_frame[feature].to_numpy(dtype=np.float64)
        raw_distance, normalized_distance = normalized_quantile_distance(
            test_values, train_values
        )
        rows.append(
            {
                "bm25_hard_fraction": float(fraction),
                "feature": feature,
                "train_rows": len(train_frame),
                "test_rows": len(test_frame),
                "train_mean": float(train_values.mean()),
                "test_mean": float(test_values.mean()),
                "train_zero_rate": float((train_values == 0).mean()),
                "test_zero_rate": float((test_values == 0).mean()),
                "quantile_mae": raw_distance,
                "normalized_quantile_mae": normalized_distance,
                "bm25_rows": int(source_rows.get("bm25", 0)),
                "category_rows": int(source_rows.get("category", 0)),
                "random_rows": int(source_rows.get("random", 0)),
                "positive_rows": int(source_rows.get("positive", 0)),
            }
        )
    return rows


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_report(frame, args):
    summary = (
        frame.groupby("bm25_hard_fraction", as_index=False)
        .agg(
            mean_normalized_distance=("normalized_quantile_mae", "mean"),
            max_normalized_distance=("normalized_quantile_mae", "max"),
            train_rows=("train_rows", "first"),
            test_rows=("test_rows", "first"),
            bm25_rows=("bm25_rows", "first"),
            category_rows=("category_rows", "first"),
            random_rows=("random_rows", "first"),
        )
        .sort_values(
            ["mean_normalized_distance", "bm25_hard_fraction"]
        )
    )
    best = summary.iloc[0]
    lines = [
        "# Candidate Distribution Shift",
        "",
        "This analysis uses no submission labels. It compares retrieval-sensitive "
        "feature marginals from deterministic training candidates with unseen "
        "submission-term candidates.",
        "",
        f"- Sampled training terms: `{args.sample_train_terms:,}`",
        f"- Sampled submission terms: `{args.sample_test_terms:,}`",
        f"- Fixed category-hard fraction: `{args.category_hard_fraction:.2f}`",
        "- Selected BM25 fraction by mean normalized quantile distance: "
        f"`{best['bm25_hard_fraction']:.2f}`",
        "",
        "| BM25 fraction | Mean distance | Max distance | BM25 rows | "
        "Category rows | Random rows |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.bm25_hard_fraction:.2f} | "
            f"{row.mean_normalized_distance:.6f} | "
            f"{row.max_normalized_distance:.6f} | "
            f"{row.bm25_rows:,} | {row.category_rows:,} | {row.random_rows:,} |"
        )
    lines.extend(
        [
            "",
            "The result is a covariate-shift diagnostic, not a label-based model "
            "score. Final promotion also requires grouped OOF behavior and "
            "submission sanity checks.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    temporary_path = args.report + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    os.replace(temporary_path, args.report)
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.sample_train_terms < 5 or args.sample_test_terms <= 0:
        raise ValueError("sample term counts must be positive and train must be at least 5")
    fractions = np.unique(np.asarray(args.bm25_fractions, dtype=np.float64))
    if (
        len(fractions) == 0
        or not np.isfinite(fractions).all()
        or (fractions < 0.0).any()
        or (fractions + args.category_hard_fraction > 1.0).any()
    ):
        raise ValueError("BM25 fractions must be finite and fit the negative quota")

    terms = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items = load_items(os.path.join(DATA_DIR, "items.csv"))
    positives = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={
            "id": "string",
            "term_id": "string",
            "item_id": "string",
            "label": "int8",
        },
    )
    selected_positives = sample_complete_terms(
        positives, args.sample_train_terms, RANDOM_SEED
    )
    test_pairs = _load_test_sample(
        terms, set(positives["term_id"].unique()), args.sample_test_terms
    )
    vectorizer = build_tfidf_vectorizer(
        terms, items, max_features=10_000, ngram_range=(1, 1), min_df=2
    )
    test_frame = _feature_frame(test_pairs, terms, items, vectorizer)

    print("[candidate_shift] Building the shared full-catalog BM25 index")
    index = BM25Index(
        items["item_id"].to_numpy(),
        standardize_item_text(items).tolist(),
        max_df_ratio=args.bm25_max_df_ratio,
    )
    rows = []
    for fraction in fractions:
        candidates = build_test_shaped_training_set(
            selected_positives,
            items,
            terms_df=terms,
            positive_reference_df=positives,
            bm25_hard_fraction=float(fraction),
            category_hard_fraction=args.category_hard_fraction,
            bm25_top_n=args.bm25_top_n,
            bm25_max_df_ratio=args.bm25_max_df_ratio,
            bm25_index=index,
            random_state=RANDOM_SEED,
            verbose=False,
        )
        source_rows = candidates["neg_source"].value_counts().to_dict()
        train_frame = _feature_frame(candidates, terms, items, vectorizer)
        rows.extend(
            _distribution_rows(fraction, train_frame, test_frame, source_rows)
        )
        print(
            f"[candidate_shift] fraction={fraction:.2f} "
            f"rows={len(candidates):,} tfidf_mean={train_frame['tfidf_cosine'].mean():.6f}"
        )
        del candidates, train_frame
        gc.collect()

    result = pd.DataFrame(rows)
    _atomic_write_frame(result, args.output)
    summary = _write_report(result, args)
    print(summary.to_string(index=False))
    print(f"analysis={args.output}\nreport={args.report}")
    return args.output


if __name__ == "__main__":
    main()

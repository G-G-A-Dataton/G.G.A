"""Build an evidence-based error taxonomy from verified grouped OOF artifacts."""

import argparse
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.candidate_sampling import (
    build_test_shaped_training_set,
    candidate_distribution,
    sample_complete_terms,
)
from src.data import load_items, load_terms
from src.features import build_features
from src.metrics import macro_f1
from src.modeling import (
    predictions_from_cross_fitted_selection,
    probabilities_from_cross_fitted_selection,
    select_cross_fitted_candidate,
)
from src.oof_artifacts import load_oof_artifacts


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ensemble_artifacts")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "error_taxonomy.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "error_taxonomy.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify held-out errors using observable feature conflicts"
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-sample", action="store_true")
    return parser.parse_args(argv)


def classify_error_signals(frame):
    """Assign one primary observable signal without claiming error causality."""
    required = {
        "label",
        "query_model_token_conflict",
        "demographic_conflict",
        "query_color_match",
        "query_size_match",
        "query_material_match",
        "query_title_overlap",
        "query_category_overlap",
        "query_title_coverage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"error taxonomy input is missing columns: {missing}")
    conditions = [
        frame["query_model_token_conflict"] == 1,
        frame["demographic_conflict"] == 1,
        frame["query_color_match"] == -1,
        frame["query_size_match"] == -1,
        frame["query_material_match"] == -1,
        (frame["query_title_overlap"] == 0)
        & (frame["query_category_overlap"] == 0),
        (frame["label"] == 0) & (frame["query_title_coverage"] >= 0.5),
    ]
    labels = [
        "MODEL_CODE_CONFLICT",
        "DEMOGRAPHIC_CONFLICT",
        "COLOR_CONFLICT",
        "SIZE_CONFLICT",
        "MATERIAL_CONFLICT",
        "NO_LEXICAL_EVIDENCE",
        "LEXICAL_DECOY",
    ]
    return pd.Series(
        np.select(conditions, labels, default="OTHER"),
        index=frame.index,
        dtype="string",
    )


def candidate_sampling_kwargs(manifest):
    """Extract every deterministic reconstruction parameter from a manifest."""
    config = manifest.get("candidate_sampling", {})
    required = {
        "strategy",
        "min_candidates",
        "dense_multiplier",
        "bm25_hard_fraction",
        "category_hard_fraction",
        "bm25_top_n",
        "bm25_max_df_ratio",
        "random_state",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"candidate sampling manifest is missing: {missing}")
    if config["strategy"] != "test_shaped_bm25_category_random":
        raise ValueError(f"unsupported candidate strategy: {config['strategy']}")
    return {key: config[key] for key in required if key != "strategy"}


def _rebuild_candidates(manifest, terms, items):
    positives = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={
            "id": "string",
            "term_id": "string",
            "item_id": "string",
            "label": "int8",
        },
    )
    if manifest["training_mode"] == "sample":
        selected = sample_complete_terms(
            positives,
            manifest["training"]["terms"],
            random_state=42,
        )
    elif manifest["training_mode"] == "full":
        selected = positives
    else:
        raise ValueError(f"unsupported training mode: {manifest['training_mode']}")
    candidates = build_test_shaped_training_set(
        selected,
        items,
        terms_df=terms,
        positive_reference_df=positives,
        **candidate_sampling_kwargs(manifest),
    )
    actual = candidate_distribution(candidates)
    for field in (
        "terms",
        "rows",
        "positive_rows",
        "negative_rows",
        "source_rows",
    ):
        if actual[field] != manifest["training"][field]:
            raise ValueError(
                f"reconstructed candidate {field} mismatch: "
                f"{actual[field]} != {manifest['training'][field]}"
            )
    return candidates


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _markdown_cell(value, limit=80):
    return str(value).replace("|", "\\|").replace("\n", " ")[:limit]


def _write_report(path, errors, selection, score, manifest):
    counts = (
        errors.groupby(["error_type", "observed_signal"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    lines = [
        "# Error Taxonomy",
        "",
        "Errors are generated with fold-specific model weights and thresholds selected without the evaluated fold. Taxonomy labels describe observable feature evidence; they do not assert root cause.",
        "",
        f"- Artifact mode: `{manifest['training_mode']}`",
        f"- Selected candidate: `{selection['deploy']['selected_model']}`",
        f"- Cross-fitted Macro-F1: `{score:.6f}`",
        f"- False positives: `{int((errors['error_type'] == 'FP').sum()):,}`",
        f"- False negatives: `{int((errors['error_type'] == 'FN').sum()):,}`",
        "",
        "## Distribution",
        "",
        "| Error type | Observed signal | Rows |",
        "|---|---|---:|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.error_type} | {row.observed_signal} | {row.rows:,} |")

    lines.extend(
        [
            "",
            "## Highest-confidence False Positives",
            "",
            "| Query | Product title | Probability | Signal |",
            "|---|---|---:|---|",
        ]
    )
    false_positives = errors[errors["error_type"] == "FP"].nlargest(
        10, "oof_probability"
    )
    for row in false_positives.itertuples(index=False):
        lines.append(
            f"| {_markdown_cell(row.query)} | {_markdown_cell(row.title)} | "
            f"{row.oof_probability:.6f} | {row.observed_signal} |"
        )
    lines.extend(
        [
            "",
            "## Lowest-confidence False Negatives",
            "",
            "| Query | Product title | Probability | Signal |",
            "|---|---|---:|---|",
        ]
    )
    false_negatives = errors[errors["error_type"] == "FN"].nsmallest(
        10, "oof_probability"
    )
    for row in false_negatives.itertuples(index=False):
        lines.append(
            f"| {_markdown_cell(row.query)} | {_markdown_cell(row.title)} | "
            f"{row.oof_probability:.6f} | {row.observed_signal} |"
        )
    lines.extend(
        [
            "",
            "The row-level evidence is stored in `outputs/error_taxonomy.csv`.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or (
        os.path.join(PROJECT_ROOT, "outputs", "error_taxonomy_sample.csv")
        if args.allow_sample
        else DEFAULT_OUTPUT
    )
    report = args.report or (
        os.path.join(PROJECT_ROOT, "docs", "error_taxonomy_sample.md")
        if args.allow_sample
        else DEFAULT_REPORT
    )
    manifest, arrays = load_oof_artifacts(
        args.artifact_dir,
        require_full=not args.allow_sample,
        source_data_dir=DATA_DIR,
    )
    selection = select_cross_fitted_candidate(
        arrays["y_true.npy"],
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        arrays["fold_ids.npy"],
    )
    probabilities = probabilities_from_cross_fitted_selection(
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        arrays["fold_ids.npy"],
        selection,
    )
    predictions = predictions_from_cross_fitted_selection(
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        arrays["fold_ids.npy"],
        selection,
    )
    y_true = np.asarray(arrays["y_true.npy"], dtype=np.int8)
    score = macro_f1(y_true, predictions)
    error_mask = predictions != y_true

    terms = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items = load_items(os.path.join(DATA_DIR, "items.csv"))
    candidates = _rebuild_candidates(manifest, terms, items)
    if not np.array_equal(candidates["label"].to_numpy(dtype=np.int8), y_true):
        raise ValueError("reconstructed candidate labels do not align with OOF rows")
    errors = candidates.loc[
        error_mask, ["term_id", "item_id", "label"]
    ].copy()
    errors["prediction"] = predictions[error_mask]
    errors["oof_probability"] = probabilities[error_mask]
    errors["error_type"] = np.where(errors["label"] == 0, "FP", "FN")

    errors = errors.merge(
        terms, on="term_id", how="left", validate="many_to_one"
    ).merge(items, on="item_id", how="left", validate="many_to_one")
    errors = build_features(errors, copy=False)
    errors["observed_signal"] = classify_error_signals(errors)
    export_columns = [
        "term_id",
        "item_id",
        "label",
        "prediction",
        "oof_probability",
        "error_type",
        "observed_signal",
        "query",
        "title",
        "category",
        "brand",
        "query_title_overlap",
        "query_title_coverage",
        "query_category_overlap",
        "query_model_token_conflict",
        "demographic_conflict",
        "query_color_match",
        "query_size_match",
        "query_material_match",
    ]
    errors = errors[export_columns].sort_values(
        ["error_type", "oof_probability"], ascending=[True, False]
    )
    _atomic_write_frame(errors, output)
    _write_report(report, errors, selection, score, manifest)
    print(
        f"selected={selection['deploy']['selected_model']} "
        f"cross_fitted_macro_f1={score:.6f} errors={len(errors):,}"
    )
    print(f"taxonomy={output}\nreport={report}")
    return output


if __name__ == "__main__":
    main()

"""Load the frozen ensemble and produce a submission without network access."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items, load_terms
from src.oof_artifacts import EXPECTED_TEST_ROWS, sha256_file
from src.out_of_core_features import (
    build_base_feature_store,
    build_context_feature_store,
    load_feature_batch,
    remove_feature_stores,
)
from src.solution_contract import load_inference_bundle
from src.validate_submission import validate_submission


def _xgb_predict(model, matrix):
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (
        (0, best_iteration + 1) if best_iteration is not None else (0, 0)
    )
    return model.predict(matrix, iteration_range=iteration_range)


def _predict_batch(features, lightgbm_models, xgboost_models, deploy):
    lightgbm_values = np.mean(
        [model.predict(features) for model in lightgbm_models],
        axis=0,
    )
    xgboost_matrix = xgb.DMatrix(features)
    xgboost_values = np.mean(
        [_xgb_predict(model, xgboost_matrix) for model in xgboost_models],
        axis=0,
    )
    probabilities = (
        float(deploy["lightgbm_weight"]) * lightgbm_values
        + float(deploy["xgboost_weight"]) * xgboost_values
    )
    return (probabilities >= float(deploy["threshold"])).astype(np.int8)


def run_inference(
    model_dump_path,
    competition_data_path,
    out_path,
    *,
    batch_size=100_000,
    require_full=True,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_dump_path = os.path.abspath(model_dump_path)
    competition_data_path = os.path.abspath(competition_data_path)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    manifest, decision, lightgbm_models, xgboost_models, vectorizer = (
        load_inference_bundle(
            model_dump_path,
            competition_data_path,
            require_full=require_full,
        )
    )
    test_rows = int(manifest["test_rows"])
    if require_full and test_rows != EXPECTED_TEST_ROWS:
        raise ValueError("Production inference requires the complete test set")
    terms = load_terms(os.path.join(competition_data_path, "terms.csv"))
    items = load_items(os.path.join(competition_data_path, "items.csv"))
    pairs_path = os.path.join(competition_data_path, "submission_pairs.csv")
    sample_path = os.path.join(competition_data_path, "sample_submission.csv")
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Missing sample submission: {sample_path}")
    with open(
        os.path.join(PROJECT_ROOT, "configs", "final_v1.json"),
        encoding="utf-8",
    ) as config_file:
        expected_sample_hash = json.load(config_file)["data_freeze"]["files"][
            "sample_submission.csv"
        ]["sha256"]
    if sha256_file(sample_path) != expected_sample_hash:
        raise ValueError("sample_submission.csv SHA-256 does not match data freeze")

    store_prefix = os.path.join(
        os.path.dirname(out_path),
        f".offline_features_{os.getpid()}",
    )
    store_paths = []
    temporary_output = out_path + ".tmp"
    offset = 0
    positive_rows = 0
    try:
        base_path, codes_path = build_base_feature_store(
            pairs_path,
            terms,
            items,
            vectorizer,
            row_count=test_rows,
            batch_size=batch_size,
            output_prefix=store_prefix,
        )
        store_paths.extend([base_path, codes_path])
        context_path = build_context_feature_store(
            base_path,
            codes_path,
            store_prefix,
        )
        store_paths.append(context_path)
        base_store = np.load(base_path, mmap_mode="r")
        context_store = np.load(context_path, mmap_mode="r")
        pair_reader = pd.read_csv(
            pairs_path,
            usecols=["id"],
            dtype={"id": "string"},
            nrows=test_rows,
            chunksize=batch_size,
        )
        try:
            for id_chunk in pair_reader:
                end = offset + len(id_chunk)
                features = load_feature_batch(
                    base_store,
                    context_store,
                    offset,
                    end,
                )
                predictions = _predict_batch(
                    features,
                    lightgbm_models,
                    xgboost_models,
                    decision["deploy"],
                )
                positive_rows += int(predictions.sum())
                pd.DataFrame(
                    {"id": id_chunk["id"].to_numpy(), "prediction": predictions}
                ).to_csv(
                    temporary_output,
                    mode="w" if offset == 0 else "a",
                    header=offset == 0,
                    index=False,
                )
                offset = end
                print(f"offline inference: {offset:,}/{test_rows:,}")
        finally:
            pair_reader.close()
        if offset != test_rows:
            raise RuntimeError(
                f"Offline inference row mismatch: {offset:,} != {test_rows:,}"
            )
        if not validate_submission(
            temporary_output,
            sample_submission_path=sample_path,
            expected_rows=test_rows,
            chunk_size=batch_size,
            verbose=True,
        ):
            raise RuntimeError("Offline inference output failed submission QA")
        os.replace(temporary_output, out_path)
    except Exception:
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
        raise
    finally:
        remove_feature_stores(*store_paths)
    print(
        f"output={out_path} rows={test_rows} positives={positive_rows} "
        f"positive_rate={positive_rows / test_rows:.6%}"
    )
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run verified offline inference")
    parser.add_argument("--model-dump-path", required=True)
    parser.add_argument("--competition-data-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--allow-sample", action="store_true")
    args = parser.parse_args(argv)
    return run_inference(
        args.model_dump_path,
        args.competition_data_path,
        args.out_path,
        batch_size=args.batch_size,
        require_full=not args.allow_sample,
    )


if __name__ == "__main__":
    main()

"""Validation and model loading for the finalist delivery workflow."""

import json
import os

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from src.modeling import MODEL_FEATURE_COLS, select_cross_fitted_candidate
from src.oof_artifacts import load_oof_artifacts
from src.tfidf_features import load_vectorizer


DECISION_FILENAME = "ensemble_decision.json"


def _finite_probability(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return value


def build_deploy_decision(artifact_dir, source_data_dir, require_full=True):
    """Recompute the deploy decision from hash-verified OOF artifacts."""
    manifest, arrays = load_oof_artifacts(
        artifact_dir,
        require_full=require_full,
        source_data_dir=source_data_dir,
    )
    selection = select_cross_fitted_candidate(
        arrays["y_true.npy"],
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        arrays["fold_ids.npy"],
    )
    deploy = selection["deploy"]
    lightgbm_weight = float(deploy["lightgbm_weight"])
    threshold = float(deploy["threshold"])
    probabilities = (
        lightgbm_weight * arrays["test_lgbm.npy"]
        + (1.0 - lightgbm_weight) * arrays["test_xgb.npy"]
    )
    positive_rows = int(np.count_nonzero(probabilities >= threshold))
    return {
        "validation": {
            "lightgbm": selection["validation"]["lightgbm"],
            "xgboost": selection["validation"]["xgboost"],
            "ensemble": selection["validation"]["weighted_blend"],
        },
        "deploy": {
            "selected_model": deploy["selected_model"],
            "lightgbm_weight": lightgbm_weight,
            "xgboost_weight": 1.0 - lightgbm_weight,
            "threshold": threshold,
            "positive_rate": positive_rows / manifest["test_rows"],
            "rows": manifest["test_rows"],
        },
    }


def write_deploy_decision(artifact_dir, source_data_dir, require_full=True):
    decision = build_deploy_decision(
        artifact_dir,
        source_data_dir,
        require_full=require_full,
    )
    output_path = os.path.join(artifact_dir, DECISION_FILENAME)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(decision, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, output_path)
    return output_path, decision


def validate_deploy_decision(artifact_dir, source_data_dir, require_full=True):
    """Bind the checked decision file to the current OOF/model bundle."""
    decision_path = os.path.join(artifact_dir, DECISION_FILENAME)
    if not os.path.isfile(decision_path):
        raise FileNotFoundError(f"Missing deploy decision: {decision_path}")
    with open(decision_path, encoding="utf-8") as decision_file:
        decision = json.load(decision_file)
    expected = build_deploy_decision(
        artifact_dir,
        source_data_dir,
        require_full=require_full,
    )
    deploy = decision.get("deploy")
    if not isinstance(deploy, dict):
        raise ValueError("Deploy decision is missing deploy parameters")
    if deploy.get("selected_model") not in {
        "lightgbm",
        "xgboost",
        "weighted_blend",
    }:
        raise ValueError("Deploy decision has an invalid selected_model")
    for field in (
        "lightgbm_weight",
        "xgboost_weight",
        "threshold",
        "positive_rate",
    ):
        _finite_probability(deploy.get(field), f"deploy.{field}")
    if not np.isclose(
        float(deploy["lightgbm_weight"]) + float(deploy["xgboost_weight"]),
        1.0,
        atol=1e-12,
    ):
        raise ValueError("Deploy model weights must sum to one")
    expected_deploy = expected["deploy"]
    if deploy.get("selected_model") != expected_deploy["selected_model"]:
        raise ValueError("Deploy decision model does not match OOF selection")
    if deploy.get("rows") != expected_deploy["rows"]:
        raise ValueError("Deploy decision row count does not match artifacts")
    for field in (
        "lightgbm_weight",
        "xgboost_weight",
        "threshold",
        "positive_rate",
    ):
        if not np.isclose(
            float(deploy[field]),
            float(expected_deploy[field]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Deploy decision {field} does not match OOF selection")
    return decision


def load_inference_bundle(artifact_dir, source_data_dir, require_full=True):
    """Validate and load all ensemble models and the TF-IDF vectorizer."""
    decision = validate_deploy_decision(
        artifact_dir,
        source_data_dir,
        require_full=require_full,
    )
    manifest_path = os.path.join(artifact_dir, "oof_manifest.json")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    expected_lgbm = [f"lgbm_fold_{fold}.txt" for fold in range(1, 6)]
    expected_xgb = [f"xgb_fold_{fold}.json" for fold in range(1, 6)]
    if manifest.get("model_files") != [
        name
        for pair in zip(expected_lgbm, expected_xgb)
        for name in pair
    ]:
        raise ValueError("Model bundle does not contain the canonical fold ordering")

    lightgbm_models = [
        lgb.Booster(model_file=os.path.join(artifact_dir, filename))
        for filename in expected_lgbm
    ]
    xgboost_models = []
    for filename in expected_xgb:
        model = xgb.Booster()
        model.load_model(os.path.join(artifact_dir, filename))
        xgboost_models.append(model)
    for model in lightgbm_models:
        if model.feature_name() != MODEL_FEATURE_COLS:
            raise ValueError("LightGBM feature columns do not match the solution")
    for model in xgboost_models:
        if model.feature_names != MODEL_FEATURE_COLS:
            raise ValueError("XGBoost feature columns do not match the solution")
    vectorizer = load_vectorizer(
        os.path.join(artifact_dir, "tfidf_vectorizer.pkl")
    )
    return manifest, decision, lightgbm_models, xgboost_models, vectorizer

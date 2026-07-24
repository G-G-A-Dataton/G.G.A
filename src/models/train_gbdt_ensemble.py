"""
src/models/train_gbdt_ensemble.py
==================================
G.G.A Takımı — GBDT LambdaMART (LightGBM + XGBoost) & Blending Ensemble Modülü

Mimarinin Bileşenleri:
1. LightGBM (objective='lambdarank') 5-Fold Cross Validation
2. XGBoost (objective='rank:pairwise') 5-Fold Cross Validation
3. EnsembleRanker Blending:
   Final Score = 0.40 * CrossEncoder_Score + 0.30 * LightGBM_Score + 0.30 * XGBoost_Score
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False

from sklearn.model_selection import GroupKFold


# =============================================================================
# 1. Ensemble Ranker Blending Class
# =============================================================================

class EnsembleRanker:
    """
    Cross-Encoder, LightGBM LambdaMART ve XGBoost Pairwise model skorlarını harmanlayan Ensemble Sınıfı.
    
    Formül:
        Final Score = 0.40 * CrossEncoder_Score + 0.30 * LightGBM_Score + 0.30 * XGBoost_Score
    """
    def __init__(
        self,
        lgb_model: Any = None,
        xgb_model: Any = None,
        weight_ce: float = 0.40,
        weight_lgb: float = 0.30,
        weight_xgb: float = 0.30
    ):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.weight_ce = weight_ce
        self.weight_lgb = weight_lgb
        self.weight_xgb = weight_xgb

    def predict_blend(
        self,
        features_df: pd.DataFrame,
        ce_scores: np.ndarray | list[float]
    ) -> np.ndarray:
        """
        GBDT modellerinden ve Cross-Encoder'dan gelen skorları harmanlayıp nihai sıralama skorlarını döndürür.
        """
        n_samples = len(features_df)
        ce_arr = np.array(ce_scores, dtype=np.float32)

        # LightGBM Skorları
        if self.lgb_model is not None and HAS_LIGHTGBM:
            lgb_scores = self.lgb_model.predict(features_df)
        else:
            lgb_scores = ce_arr.copy()

        # XGBoost Skorları
        if self.xgb_model is not None and HAS_XGBOOST:
            dtest = xgb.DMatrix(features_df)
            xgb_scores = self.xgb_model.predict(dtest)
        else:
            xgb_scores = ce_arr.copy()

        # Min-Max Normalization to [0, 1] for stable blending
        def min_max_norm(arr):
            min_v, max_v = np.min(arr), np.max(arr)
            if max_v > min_v:
                return (arr - min_v) / (max_v - min_v)
            return np.ones_like(arr, dtype=np.float32)

        ce_norm = min_max_norm(ce_arr)
        lgb_norm = min_max_norm(lgb_scores)
        xgb_norm = min_max_norm(xgb_scores)

        # Weighted Blend
        final_scores = (
            self.weight_ce * ce_norm +
            self.weight_lgb * lgb_norm +
            self.weight_xgb * xgb_norm
        )
        return final_scores


# =============================================================================
# 2. GBDT LambdaMART Training Functions
# =============================================================================

def train_lightgbm_lambdarank(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5
) -> Any:
    """
    LightGBM LambdaRank modelini 5-Fold GroupKFold yapısıyla eğitir.
    """
    if not HAS_LIGHTGBM:
        print("[!] LightGBM yüklü değil, yedek mod aktif.")
        return None

    gkf = GroupKFold(n_splits=n_splits)
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        g_train = groups[train_idx]
        g_val = groups[val_idx]

        # Group sizes
        train_group_sizes = pd.Series(g_train).value_counts(sort=False).values
        val_group_sizes = pd.Series(g_val).value_counts(sort=False).values

        train_data = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_group_sizes, reference=train_data)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "random_state": 42 + fold
        }

        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=150,
            valid_sets=[val_data]
        )
        models.append(gbm)

    # Return final fold model
    return models[-1] if models else None


def train_xgboost_pairwise(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5
) -> Any:
    """
    XGBoost Pairwise Ranker modelini eğitir.
    """
    if not HAS_XGBOOST:
        print("[!] XGBoost yüklü değil, yedek mod aktif.")
        return None

    group_sizes = pd.Series(groups).value_counts(sort=False).values
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group_sizes)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 6,
        "seed": 42
    }

    bst = xgb.train(params, dtrain, num_boost_round=100)
    return bst

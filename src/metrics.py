"""
src/metrics.py
==============
G.G.A Takımı — Yarışma Metrik ve Validasyon Şeması

Ömer Faruk Kara tarafından hazırlanmıştır.

Yarışmanın resmi metriği: Macro-F1
Doğrulama şeması: 5-Fold Stratified Group K-Fold (group=term_id)
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


# ─────────────────────────────────────────────
# 1. Macro-F1 Hesaplama
# ─────────────────────────────────────────────

def _validate_binary_inputs(y_true, values, value_name):
    """Return validated one-dimensional arrays for binary metrics."""
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    if y_true.ndim != 1 or values.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional")
    if len(y_true) == 0 or len(y_true) != len(values):
        raise ValueError("Metric inputs must be non-empty and have equal length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(values).all():
        raise ValueError(f"{value_name} must contain only finite values")
    return y_true.astype(np.int8, copy=False), values


def macro_f1(y_true, y_pred):
    """
    Yarışmanın resmi değerlendirme metriği olan Macro-F1'i hesaplar.
    Her iki sınıfa eşit ağırlık verir (0: alakasız, 1: alakalı).

    Parametreler
    ----------
    y_true : array-like of int
        Gerçek etiketler (0 veya 1).
    y_pred : array-like of int
        Model tahminleri (0 veya 1). Olasılık değil, binary tahmin.

    Döndürür
    -------
    float
        Macro-F1 skoru (0.0 ile 1.0 arasında).

    Örnek
    -----
    >>> y_true = [1, 1, 0, 0, 1]
    >>> y_pred = [1, 0, 0, 1, 1]
    >>> macro_f1(y_true, y_pred)
    0.5833333333333333
    """
    y_true, y_pred = _validate_binary_inputs(y_true, y_pred, "y_pred")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only binary labels 0 and 1")
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )


def macro_f1_from_proba(y_true, y_proba, threshold=0.5):
    """
    Olasılık tahminlerini eşik değeri kullanarak binary'e çevirip Macro-F1 hesaplar.

    Parametreler
    ----------
    y_true : array-like of int
        Gerçek etiketler (0 veya 1).
    y_proba : array-like of float
        Pozitif sınıf (label=1) olasılıkları. 0.0 ile 1.0 arasında.
    threshold : float, default=0.5
        Olasılığı binary tahmine dönüştürmek için eşik değeri.

    Döndürür
    -------
    float
        Macro-F1 skoru.

    Örnek
    -----
    >>> y_true   = [1, 1, 0, 0, 1]
    >>> y_proba  = [0.8, 0.3, 0.2, 0.7, 0.9]
    >>> macro_f1_from_proba(y_true, y_proba, threshold=0.5)
    0.5833333333333333
    """
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise TypeError("threshold must be a number")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    y_true, y_proba = _validate_binary_inputs(y_true, y_proba, "y_proba")
    if ((y_proba < 0.0) | (y_proba > 1.0)).any():
        raise ValueError("y_proba must be between 0 and 1")
    y_pred = (y_proba >= float(threshold)).astype(np.int8)
    return macro_f1(y_true, y_pred)


def find_best_threshold(y_true, y_proba, thresholds=None):
    """
    Validation seti üzerinde Macro-F1'i maksimize eden eşik değerini arar.

    Parametreler
    ----------
    y_true : array-like of int
        Gerçek etiketler.
    y_proba : array-like of float
        Pozitif sınıf olasılıkları.
    thresholds : list of float, optional
        Denenecek eşik değerleri. None ise tahminlerin değiştiği tüm benzersiz
        olasılık sınırları kayıpsız ve O(n log n) zamanda değerlendirilir.

    Döndürür
    -------
    best_threshold : float
    best_score : float
    results : list of (threshold, score) tuples
    """
    y_true, y_proba = _validate_binary_inputs(y_true, y_proba, "y_proba")
    y_proba = y_proba.astype(np.float64, copy=False)
    if ((y_proba < 0.0) | (y_proba > 1.0)).any():
        raise ValueError("y_proba must be between 0 and 1")

    if thresholds is None:
        # Predictions only change at observed probability values. Compute every
        # boundary in O(n log n), without rescoring the full vector per boundary.
        order = np.argsort(-y_proba, kind="stable")
        sorted_proba = y_proba[order]
        sorted_labels = y_true[order]
        boundary_indices = np.flatnonzero(
            np.r_[sorted_proba[:-1] != sorted_proba[1:], True]
        )
        true_positives = np.cumsum(sorted_labels, dtype=np.int64)[boundary_indices]
        predicted_positives = boundary_indices + 1
        false_positives = predicted_positives - true_positives
        total_positives = int(sorted_labels.sum())
        total_negatives = len(sorted_labels) - total_positives
        false_negatives = total_positives - true_positives
        true_negatives = total_negatives - false_positives

        positive_denominator = 2 * true_positives + false_positives + false_negatives
        negative_denominator = 2 * true_negatives + false_positives + false_negatives
        positive_f1 = np.divide(
            2 * true_positives,
            positive_denominator,
            out=np.zeros_like(true_positives, dtype=np.float64),
            where=positive_denominator != 0,
        )
        negative_f1 = np.divide(
            2 * true_negatives,
            negative_denominator,
            out=np.zeros_like(true_negatives, dtype=np.float64),
            where=negative_denominator != 0,
        )
        thresholds_array = sorted_proba[boundary_indices]
        scores_array = (positive_f1 + negative_f1) / 2.0
        results = list(zip(thresholds_array.tolist(), scores_array.tolist()))
    else:
        thresholds_array = np.asarray(list(thresholds), dtype=np.float64)
        if thresholds_array.ndim != 1 or len(thresholds_array) == 0:
            raise ValueError("thresholds must be a non-empty one-dimensional iterable")
        if not np.isfinite(thresholds_array).all() or (
            (thresholds_array < 0.0) | (thresholds_array > 1.0)
        ).any():
            raise ValueError("thresholds must contain finite values in [0, 1]")
        thresholds_array = np.unique(thresholds_array)
        results = [
            (float(threshold), macro_f1_from_proba(y_true, y_proba, threshold))
            for threshold in thresholds_array
        ]
    best_score = max(score for _, score in results)
    tied = [
        threshold
        for threshold, score in results
        if np.isclose(score, best_score, rtol=0.0, atol=1e-12)
    ]
    # A deterministic central tie-break is less brittle than selecting the
    # first edge of a flat optimum.
    best_threshold = min(tied, key=lambda value: (abs(value - 0.5), value))
    return float(best_threshold), float(best_score), results


# ─────────────────────────────────────────────
# 2. Stratified Group K-Fold Validasyon Şeması
# ─────────────────────────────────────────────

def get_stratified_group_kfold(n_splits=5, shuffle=True, random_state=42):
    """
    Aynı term_id'nin train ve validation'a bölünmesini engelleyen
    StratifiedGroupKFold nesnesi döndürür.

    Parametreler
    ----------
    n_splits : int, default=5
        Fold sayısı.
    shuffle : bool, default=True
        Karıştırma. Tekrarlanabilirlik için random_state ile birlikte kullanılır.
    random_state : int, default=42
        Rastgelelik tohumu (proje standardı: 42).

    Döndürür
    -------
    StratifiedGroupKFold
        Scikit-learn StratifiedGroupKFold nesnesi.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer greater than one")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean")
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )


def cross_validate_macro_f1(
    model,
    X,
    y,
    groups,
    n_splits=5,
    random_state=42,
    verbose=True,
):
    """
    5-Fold Stratified Group CV ile Macro-F1 çapraz doğrulaması yapar.

    Parametreler
    ----------
    model : sklearn-uyumlu model
        fit() ve predict_proba() metodları olan herhangi bir model.
    X : array-like veya pd.DataFrame
        Özellik matrisi.
    y : array-like of int
        Hedef etiketler (0 veya 1).
    groups : array-like
        Aynı sorguya ait satırları tek fold'da tutan grup kimlikleri.
    n_splits : int, default=5
        Fold sayısı.
    random_state : int, default=42
        Rastgelelik tohumu.
    verbose : bool, default=True
        Her fold için skoru yazdır.

    Döndürür
    -------
    fold_scores : list of float
        Her fold'un Macro-F1 skoru.
    mean_score : float
        Ortalama Macro-F1 skoru.
    std_score : float
        Standart sapma.
    """
    skf = get_stratified_group_kfold(
        n_splits=n_splits, random_state=random_state
    )
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X, y, groups=groups), start=1
    ):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold   = X.iloc[val_idx]   if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold   = y.iloc[val_idx]   if hasattr(y, 'iloc') else y[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
        score = macro_f1_from_proba(y_val_fold, y_proba)
        fold_scores.append(score)

        if verbose:
            print(f"  Fold {fold_idx}/{n_splits}  →  Macro-F1: {score:.4f}")

    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))

    if verbose:
        print(f"  ─────────────────────────────────")
        print(f"  Ortalama Macro-F1 : {mean_score:.4f}")
        print(f"  Std Sapma         : {std_score:.4f}")

    return fold_scores, mean_score, std_score

"""
src/metrics.py
==============
G.G.A Takımı — Yarışma Metrik ve Validasyon Şeması

Ömer Faruk Kara tarafından hazırlanmıştır.

Yarışmanın resmi metriği: Macro-F1
Doğrulama şeması: 5-Fold Stratified K-Fold
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────
# 1. Macro-F1 Hesaplama
# ─────────────────────────────────────────────

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
    0.75
    """
    return f1_score(y_true, y_pred, average='macro')


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
    0.75
    """
    y_pred = (np.array(y_proba) >= threshold).astype(int)
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
        Denenecek eşik değerleri. None ise [0.05, 0.10, ..., 0.95] kullanılır.

    Döndürür
    -------
    best_threshold : float
    best_score : float
    results : list of (threshold, score) tuples
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05).tolist()

    results = []
    for t in thresholds:
        score = macro_f1_from_proba(y_true, y_proba, threshold=t)
        results.append((round(t, 2), round(score, 4)))

    best_threshold, best_score = max(results, key=lambda x: x[1])
    return best_threshold, best_score, results


# ─────────────────────────────────────────────
# 2. Stratified K-Fold Validasyon Şeması
# ─────────────────────────────────────────────

def get_stratified_kfold(n_splits=5, shuffle=True, random_state=42):
    """
    Takım standartlarına uygun 5-Fold Stratified K-Fold nesnesi döndürür.
    Seed olarak proje geneli 42 kullanılır.

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
    StratifiedKFold
        Scikit-learn StratifiedKFold nesnesi.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def cross_validate_macro_f1(model, X, y, n_splits=5, random_state=42, verbose=True):
    """
    5-Fold Stratified CV ile Macro-F1 çapraz doğrulaması yapar.

    Parametreler
    ----------
    model : sklearn-uyumlu model
        fit() ve predict_proba() metodları olan herhangi bir model.
    X : array-like veya pd.DataFrame
        Özellik matrisi.
    y : array-like of int
        Hedef etiketler (0 veya 1).
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
    skf = get_stratified_kfold(n_splits=n_splits, random_state=random_state)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold   = X.iloc[val_idx]   if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold   = y.iloc[val_idx]   if hasattr(y, 'iloc') else y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_proba = model.predict_proba(X_val_fold)[:, 1]
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

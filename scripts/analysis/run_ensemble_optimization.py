"""
scripts/analysis/run_ensemble_optimization.py
=============================================
G.G.A Takımı — Ensemble Ağırlıklandırma & Ortak Optimizasyon (15 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script:
  1. Dün dışa aktarılan Out-of-Fold (OOF) ve test tahminlerini (LGBM ve XGB) yükler.
  2. SciPy minimize (Powell metodu ile, non-differentiable / türevsiz optimizasyon)
     kullanarak model ağırlıklarını ve karar eşiğini (threshold) AYNI ANDA optimize eder.
  3. Lokal Macro-F1 skorunu maksimize eden en iyi parametreleri belirler.
  4. Bu en iyi parametrelerle "Final Aday Model v1" tahminlerini ve Kaggle
     formatına uygun submission_v2.csv dosyasını üretir.

Çalıştırmak için:
  python scripts/analysis/run_ensemble_optimization.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.validate_submission import validate_submission

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
SUB_OUTPUT = os.path.join(OUTPUT_DIR, "submission_v2.csv")


def f1_objective(params, oof_lgbm, oof_xgb, y_true):
    """
    SciPy optimizasyonu için minimize edilecek negatif Macro-F1 fonksiyonu.
    
    params[0] = w_lgbm (LightGBM ağırlığı)
    params[1] = threshold (karar eşiği)
    """
    w_lgbm = params[0]
    threshold = params[1]
    
    # Ağırlık sınırları kontrolü (ceza fonksiyonu olarak)
    if w_lgbm < 0.0 or w_lgbm > 1.0 or threshold < 0.05 or threshold > 0.95:
        return 1.0  # F1'in olabilecek en kötü değerinin negatifi (aslında minimize için pozitif)
        
    w_xgb = 1.0 - w_lgbm
    
    # Ağırlıklı kombinasyon
    prob_blend = w_lgbm * oof_lgbm + w_xgb * oof_xgb
    pred_blend = (prob_blend >= threshold).astype(int)
    
    # Macro-F1 hesapla
    score = f1_score(y_true, pred_blend, average="macro", zero_division=0)
    return -score  # Minimize etmek istediğimiz için negatifi


def main():
    print("=" * 65)
    print("  G.G.A — Ensemble Agirliklandirma & Ortak Optimizasyon")
    print("  15 Temmuz 2026 — Omer Faruk Kara")
    print("=" * 65)

    # 1. Tahminleri Yükle
    print("\n[1/4] Tahmin dosyaları yukleniyor...")
    paths = {
        "oof_lgbm"  : os.path.join(OUTPUT_DIR, "oof_lgbm.npy"),
        "test_lgbm" : os.path.join(OUTPUT_DIR, "test_lgbm.npy"),
        "oof_xgb"   : os.path.join(OUTPUT_DIR, "oof_xgb.npy"),
        "test_xgb"  : os.path.join(OUTPUT_DIR, "test_xgb.npy"),
        "y_true"    : os.path.join(OUTPUT_DIR, "y_true.npy"),
        "metadata"  : os.path.join(OUTPUT_DIR, "test_metadata.csv")
    }

    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print("  [HATA] Aşağıdaki dosyalar bulunamadı:")
        for m in missing:
            print(f"    - {paths[m]}")
        print("\n  Lütfen önce shortlist scriptini çalıştırın:")
        print("  python scripts/training/run_model_shortlist.py")
        sys.exit(1)

    oof_lgbm = np.load(paths["oof_lgbm"])
    test_lgbm = np.load(paths["test_lgbm"])
    oof_xgb = np.load(paths["oof_xgb"])
    test_xgb = np.load(paths["test_xgb"])
    y_true = np.load(paths["y_true"])
    sub_df = pd.read_csv(paths["metadata"], dtype={"id": "string", "term_id": "string", "item_id": "string"})

    print(f"  LightGBM OOF Boyutu: {oof_lgbm.shape}")
    print(f"  XGBoost OOF Boyutu  : {oof_xgb.shape}")
    print(f"  True Label Boyutu   : {y_true.shape}")

    # Tekil model başarılarını yazdır
    f1_lgb_init = f1_score(y_true, (oof_lgbm >= 0.5).astype(int), average="macro")
    f1_xgb_init = f1_score(y_true, (oof_xgb >= 0.5).astype(int), average="macro")
    print(f"\n  Tekil Model Başarıları (Threshold = 0.50):")
    print(f"    - LightGBM F1: {f1_lgb_init:.4f}")
    print(f"    - XGBoost  F1: {f1_xgb_init:.4f}")

    # 2. Ortak Optimizasyon (SciPy)
    print("\n[2/4] SciPy ile ortak optimizasyon baslatiliyor...")
    
    # Başlangıç tahmin değerleri: w_lgbm=0.5, threshold=0.5
    initial_guess = [0.5, 0.5]
    
    # Sınırlar (Bounds): w_lgbm in [0, 1], threshold in [0.05, 0.95]
    bounds = [(0.0, 1.0), (0.05, 0.95)]

    # F1 gibi türevsiz ve süreksiz fonksiyonlar için Powell veya Nelder-Mead en iyisidir
    res = minimize(
        f1_objective,
        initial_guess,
        args=(oof_lgbm, oof_xgb, y_true),
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-4, "ftol": 1e-4, "disp": False}
    )

    best_w_lgbm = float(res.x[0])
    best_w_xgb = 1.0 - best_w_lgbm
    best_threshold = float(res.x[1])
    best_f1 = -float(res.fun)

    print("\n  Optimizasyon Tamamlandı:")
    print(f"    - En İyi LightGBM Ağırlığı: {best_w_lgbm:.4f}")
    print(f"    - En İyi XGBoost Ağırlığı : {best_w_xgb:.4f}")
    print(f"    - En İyi Karar Eşiği (TH) : {best_threshold:.4f}")
    print(f"    - En İyi Lokal Macro-F1   : **{best_f1:.4f}**")

    # 3. Final Aday Tahminleri Üret
    print("\n[3/4] En iyi parametrelerle test tahminleri birlestiriliyor...")
    test_prob_blend = best_w_lgbm * test_lgbm + best_w_xgb * test_xgb
    sub_df["prediction"] = (test_prob_blend >= best_threshold).astype(int)

    # 4. Submission Kaydet & QA Kontrol
    print("\n[4/4] Submission CSV dosyası olusturuluyor...")
    sub_df[["id", "prediction"]].to_csv(SUB_OUTPUT, index=False)
    print(f"  Kaydedildi: {SUB_OUTPUT}")

    # Basit QA kontrolü yap
    validate_submission(SUB_OUTPUT, os.path.join(DATA_DIR, "submission_pairs.csv"))

    # Rapor yazdır
    report_path = os.path.join(PROJECT_ROOT, "docs", "ensemble_karsilastirma.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Ensemble Ağırlıklandırma & Optimizasyon Raporu (15 Temmuz)\n\n")
        f.write("**Hazırlayan:** Ömer Faruk Kara  \n")
        f.write("**Tarih:** 15 Temmuz 2026  \n\n")
        f.write("## 1. Optimizasyon Sonuçları\n\n")
        f.write(f"| Metrik / Parametre | Değer |\n")
        f.write(f"|---|---|\n")
        f.write(f"| LightGBM Ağırlığı | {best_w_lgbm:.4f} |\n")
        f.write(f"| XGBoost Ağırlığı | {best_w_xgb:.4f} |\n")
        f.write(f"| Karar Eşiği (Threshold) | {best_threshold:.4f} |\n")
        f.write(f"| **En İyi Macro-F1** | **{best_f1:.4f}** |\n\n")
        f.write("## 2. Karşılaştırma\n\n")
        f.write(f"- LightGBM Tekil (0.50): `{f1_lgb_init:.4f}`\n")
        f.write(f"- XGBoost Tekil (0.50): `{f1_xgb_init:.4f}`\n")
        f.write(f"- **Weighted Ensemble (Optimized): `{best_f1:.4f}`**\n\n")
        f.write(f"Ensemble ve threshold ortak optimizasyonu sayesinde **+{best_f1 - max(f1_lgb_init, f1_xgb_init):.4f}** F1 artışı sağlanmıştır.\n")

    print(f"  Rapor kaydedildi: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

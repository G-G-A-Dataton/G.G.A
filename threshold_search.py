"""
scripts/analysis/run_threshold_search.py
========================================
G.G.A Takımı — Karar Eşiği (Threshold) Arama Scripti (14 Temmuz Görevi)

Ahmet Emin Işın tarafından hazırlanmıştır.

Bu script, kaydedilen OOF olasılık tahminleri (predict_proba) ve gerçek etiketler
üzerinde 0.00 ile 1.00 arasında 0.01 adımlarla tarama yaparak Macro-F1 skorunu
en tepeye çıkaran karar eşiğini (threshold) belirler.

Kullanım:
  python scripts/analysis/run_threshold_search.py
"""

import os
import sys
import warnings
import numpy as np
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)


def scan_thresholds(y_true, y_prob):
    """
    0.01 adımlarla threshold araması yapar.
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    scores = []
    
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        scores.append((th, score))
        
    return scores


def main():
    print("=" * 65)
    print("  G.G.A — Eşik Değer (Threshold) Tarama Modülü")
    print("  14 Temmuz 2026 — Ahmet Emin Isin")
    print("=" * 65)

    # 1. Tahminleri Yükle
    oof_path = os.path.join(OUTPUT_DIR, "oof_lgbm.npy")
    y_true_path = os.path.join(OUTPUT_DIR, "y_true.npy")

    if not os.path.exists(oof_path) or not os.path.exists(y_true_path):
        print("[HATA] OOF tahminleri bulunamadı!")
        print("  Lütfen önce modelleri eğitin: python scripts/training/run_model_shortlist.py")
        sys.exit(1)

    y_prob = np.load(oof_path)
    y_true = np.load(y_true_path)

    print(f"  Yüklenen Tahmin Sayısı: {len(y_prob):,}")

    # 2. Tarama Yap
    print("  0.00 ile 1.00 arasında 0.01 adımlarla taranıyor...")
    scores = scan_thresholds(y_true, y_prob)

    # En iyi threshold
    best_th, best_f1 = max(scores, key=lambda x: x[1])
    
    # En iyi aralığı belirle (peak F1'in %0.5 tolerans aralığı)
    tolerance = 0.005
    ideal_candidates = [th for th, score in scores if score >= best_f1 - tolerance]
    min_ideal_th = min(ideal_candidates)
    max_ideal_th = max(ideal_candidates)

    print("\n  Tarama Sonuçları:")
    print(f"    - En İyi Karar Eşiği (Threshold): **{best_th:.2f}**")
    print(f"    - En Yüksek Macro-F1 Skoru      : **{best_f1:.4f}**")
    print(f"    - Toleranslı Optimal Eşik Aralığı : **[{min_ideal_th:.2f} - {max_ideal_th:.2f}]**")

    # Kısa bir tablo yazdır
    print("\n  Eşik Değerlerden Örnekler:")
    print("    Threshold | Macro-F1")
    print("    --------------------")
    for th in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
        # En yakın skoru bul
        score_val = next(s for t, s in scores if abs(t - th) < 1e-5)
        marker = " <<" if abs(th - best_th) < 0.02 else ""
        print(f"      {th:<7.2f} | {score_val:.4f}{marker}")

    # Rapor olarak kaydet
    out_md = os.path.join(DOCS_DIR, "threshold_tarama_raporu.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Karar Eşiği Tarama Raporu (14 Temmuz)\n\n")
        f.write("**Hazırlayan:** Ahmet Emin Işın  \n")
        f.write("**Tarih:** 14 Temmuz 2026  \n\n")
        f.write("## 1. Özet Bulgular\n\n")
        f.write(f"- **En İyi Karar Eşiği:** `{best_th:.2f}`\n")
        f.write(f"- **En Yüksek Macro-F1:** `{best_f1:.4f}`\n")
        f.write(f"- **Optimal Eşik Aralığı:** `[{min_ideal_th:.2f} - {max_ideal_th:.2f}]`\n\n")
        f.write("## 2. Detaylı Skor Tablosu\n\n")
        f.write("| Eşik Değer (Threshold) | Macro-F1 |\n")
        f.write("|---|---|\n")
        for t, s in scores:
            bold = "**" if abs(t - best_th) < 1e-5 else ""
            # Sadece 0.05 adımları ve en iyi threshold değerini rapora yaz
            if abs(t % 0.05) < 1e-5 or abs(t - best_th) < 1e-5:
                f.write(f"| {bold}{t:.2f}{bold} | {bold}{s:.4f}{bold} |\n")
                
    print(f"\n  Rapor kaydedildi: {out_md}")
    print("=" * 65)


if __name__ == "__main__":
    main()

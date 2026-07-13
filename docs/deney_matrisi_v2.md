# Deney Matrisi v2 (11 Temmuz)

> [!CAUTION]
> **Historical, invalidated matrix.** These scores used row-level CV and the old
> sampling/attribute contracts. Rerun the matrix with grouped validation before
> selecting a ratio or feature set.

**Hazırlayan:** Ömer Faruk Kara  
**Tarih:** 11 Temmuz 2026  
**Yöntem:** Legacy 5-Fold row-level Stratified CV, seed=42 (invalidated)
**Pozitif örnek:** 2,000

## Eksenler

- **Negatif Oran:** 1:1, 2:1, 3:1, 5:1
- **Feature Seti:**
  - A: Temel 7 feature (3 Temmuz)
  - B: Temel + TF-IDF
  - C: Temel + Kategori L2/L3/depth
  - D: Tam 15 feature (mevcut)

---

## Sonuç Matrisi (Best F1)

| Neg. Oran | A_temel7 | B_tfidf | C_kategori | D_tam15 |
|---|---|---|---|---|
| 1:1 | 0.9610 | 0.9610 | 0.9610 | 0.9610 |
| 2:1 | 0.9611 | 0.9611 | **0.9632** | **0.9632** |
| 3:1 | 0.9591 | 0.9591 | 0.9631 | 0.9631 |
| 5:1 | 0.9548 | 0.9548 | 0.9555 | 0.9555 |

---

## En İyi Kombinasyon

| Parametre | Değer |
|---|---|
| Feature seti | C_kategori |
| Negatif oran | 2:1 |
| Best F1 | **0.9632** |
| Best Threshold | 0.45 |
| Feature sayısı | 12 |

---

## Ham Sonuclar

| neg_ratio | feature_set | mean_f1 | std_f1 | best_threshold | best_f1 | n_features | train_sec |
|---|---|---|---|---|---|---|---|
| 1 | A_temel7 | 0.961 | 0.0041 | 0.45 | 0.961 | 7 | 0.8 |
| 1 | B_tfidf | 0.961 | 0.0041 | 0.45 | 0.961 | 7 | 0.8 |
| 1 | C_kategori | 0.9607 | 0.0038 | 0.55 | 0.961 | 12 | 0.9 |
| 1 | D_tam15 | 0.9607 | 0.0038 | 0.55 | 0.961 | 15 | 0.9 |
| 2 | A_temel7 | 0.9607 | 0.0023 | 0.45 | 0.9611 | 7 | 0.9 |
| 2 | B_tfidf | 0.9607 | 0.0023 | 0.45 | 0.9611 | 7 | 0.9 |
| 2 | C_kategori | 0.9628 | 0.0021 | 0.45 | 0.9632 | 12 | 1.0 |
| 2 | D_tam15 | 0.9628 | 0.0021 | 0.45 | 0.9632 | 15 | 1.0 |
| 3 | A_temel7 | 0.9586 | 0.0063 | 0.4 | 0.9591 | 7 | 0.9 |
| 3 | B_tfidf | 0.9586 | 0.0063 | 0.4 | 0.9591 | 7 | 0.9 |
| 3 | C_kategori | 0.9615 | 0.0039 | 0.45 | 0.9631 | 12 | 1.0 |
| 3 | D_tam15 | 0.9615 | 0.0039 | 0.45 | 0.9631 | 15 | 1.0 |
| 5 | A_temel7 | 0.9508 | 0.0034 | 0.35 | 0.9548 | 7 | 1.1 |
| 5 | B_tfidf | 0.9508 | 0.0034 | 0.35 | 0.9548 | 7 | 1.0 |
| 5 | C_kategori | 0.9546 | 0.002 | 0.3 | 0.9555 | 12 | 1.0 |
| 5 | D_tam15 | 0.9546 | 0.002 | 0.3 | 0.9555 | 15 | 1.1 |

*CSV: `outputs/deney_matrisi_v2.csv`*

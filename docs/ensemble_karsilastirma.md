# Ensemble Ağırlıklandırma & Optimizasyon Raporu (15 Temmuz)

> [!CAUTION]
> **Historical, invalidated result.** The OOF inputs were generated under the
> former validation and sampling contracts. The `0.7198` score and weights are
> not approved for production.
> Current runners write `docs/ensemble_comparison.md` and
> `docs/ensemble_selection.md` from the accepted hash-verified full run.

**Hazırlayan:** Ömer Faruk Kara  
**Tarih:** 15 Temmuz 2026  

## 1. Optimizasyon Sonuçları

| Metrik / Parametre | Değer |
|---|---|
| LightGBM Ağırlığı | 0.3875 |
| XGBoost Ağırlığı | 0.6125 |
| Karar Eşiği (Threshold) | 0.7142 |
| **En İyi Macro-F1** | **0.7198** |

## 2. Karşılaştırma

- LightGBM Tekil (0.50): `0.6456`
- XGBoost Tekil (0.50): `0.6521`
- **Weighted Ensemble (Optimized): `0.7198`**

Ensemble ve threshold ortak optimizasyonu sayesinde **+0.0678** F1 artışı sağlanmıştır.

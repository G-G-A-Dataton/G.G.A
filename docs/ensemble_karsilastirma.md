# Ensemble Aday Karsilastirma Tablosu (13 Temmuz)

**Hazırlayan:** Ömer Faruk Kara  
**Tarih:** 13 Temmuz 2026

---

| Model | F1 (0.5) | Best F1 | Threshold | Sure |
|---|---|---|---|---|
| LGBM_BASE | 0.9628 | 0.9632 | 0.45 | 6.4s |
| **LGBM_TUNED** | 0.9628 | **0.9637** | 0.45 | 11.3s |
| XGB_BASE | 0.9624 | 0.9624 | 0.4 | 4.8s |
| ENS_LGBM_XGB | 0.9625 | 0.963 | 0.55 | 4.8s |

**En iyi aday:** `LGBM_TUNED`  
*CSV: `outputs/ensemble_karsilastirma.csv`*

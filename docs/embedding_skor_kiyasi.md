# Embedding Cosine Feature Skor Kiyasi (12 Temmuz)

> [!CAUTION]
> **Historical PoC only.** Synthetic embedding results and row-level CV scores
> in this document are not production validation evidence. The accepted 15 July
> model does not use embeddings.

**Hazırlayan:** Ömer Faruk Kara  
**Tarih:** 12 Temmuz 2026  
**Cosine kaynagi:** sentetik  

---

## Sonuc

| Model | N Feature | mean_F1 | best_F1 | Threshold |
|---|---|---|---|---|
| LGBM_BASE | 15 | 0.9628 | **0.9637** | 0.45 |
| LGBM_EMB | 16 | 0.9622 | **0.9622** | 0.5 |

**Embedding cosine etkisi:** `-0.0015`  
**Cosine separation:** `0.1346`  

> [!WARNING]
> Sonuclar sentetik cosine ile uretildi. A future promotion requires a real local
> checkpoint, full hash-manifested matrices, and a positive grouped ablation.

*CSV: `outputs/embedding_skor_kiyasi.csv`*

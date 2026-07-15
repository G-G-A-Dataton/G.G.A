# G.G.A — Deney Kayıt Tablosu (experiment_log.md)

> [!CAUTION]
> **EXP-001 through EXP-009 are legacy, non-comparable runs.** They used the
> former row-level CV and/or sampled-positive negative exclusion, and attribute
> features were inactive against the real flat catalog format. Retain the rows
> only as history; do not use their scores or thresholds for decisions. The
> accepted full run is EXP-010 and follows [`model_status.md`](model_status.md).

Tüm model deneyleri bu dosyada kayıt altına alınır.  
Her submission veya önemli local validation sonucu buraya eklenir.

**Format:** Her deney aşağıdaki bilgileri içermelidir:
- Deney ID (EXP-xxx)
- Tarih ve sahibi
- Kullanılan veri seti ve negatif strateji
- Feature listesi
- CV skoru (Macro-F1 ± std)
- Kaggle Public LB skoru (varsa)
- Notlar / sonraki adım

---

## Deney Tablosu

| Deney | Tarih | Sahibi | Model | Neg. Strateji | Feature | CV F1 | LB F1 | Notlar |
|---|---|---|---|---|---|---|---|---|
| **EXP-010** | **15 Tem** | Tüm ekip | **LGBM/XGB blend** | **Test-shaped BM25/category/random** | **33** | **0.837508 grouped cross-fitted** | — | **Kabul edilen full run; LGB 0.65, XGB 0.35, threshold 0.37181571** |
| EXP-001 | 3 Tem | Ömer Faruk | LightGBM | Random 3:1 / 5K poz | 7 temel | 0.9613 ± 0.0013 | — | Baseline v0, threshold opt. → 0.9621 |
| EXP-002 | 4 Tem | Ömer Faruk | LightGBM | Random 3:1 / 5K poz | 9 (7 + demografik) | TBD | — | age_group_match + demographic_conflict |
| EXP-003 | 4 Tem | Muhammed | LightGBM | Random 3:1 / 5K poz | 10 (9 + TF-IDF) | **0.9699 ± 0.0028** | — | TF-IDF #1 importance, +0.0086 kazanım |
| EXP-004 | 6 Tem | Ahmet Emin | LightGBM | Random 3:1 / 3K poz | 12 (10 + L2/L3/depth) | TBD | — | Kategori seviye feature'ları eklendi |
| EXP-005 | 7 Tem | Ömer Faruk | LightGBM | Random 3:1 / 3K poz | 12 temel | **0.9622 ± 0.0022** | — | Hard neg. baseline (BM25 bekleniyor), thresh=0.4 → 0.9625 |
| EXP-006 | 8 Tem | Ömer Faruk | LightGBM | BM25 Hard 3:1 / 5K sorgu | 15 (12+TF-IDF+attrübüt) | ⏳ Çalıştırılacak | — | `notebooks/06_bm25_karsilastirma_tam_omerfaruk.py` ile üret |
| EXP-007 | 9 Tem | Ömer Faruk | LightGBM | Mix(BM25+Rand) 3:1 / 250K poz | 16 (15+tfidf) | ⏳ Çalıştırılacak | — | `run_train_full_v2.py` → tam model, submission hazır |
| EXP-008 | 8 Tem | Ömer Faruk | LightGBM | Random 3:1 / 2K poz | 15 (12+attributes) | **0.9615 ± 0.0039** | — | Grid search: num_leaves=31, lr=0.05, min_child=20 → F1: 0.9631 |
| EXP-009 | 9 Tem | Ömer Faruk | LGBM vs XGB | Random 3:1 / 3K poz | 15 (12+attributes) | **LGBM: 0.9605 / XGB: 0.9597** | — | LGBM is faster and slightly higher F1 (0.9613 vs 0.9597) |


---

## Deney Detayları

### EXP-010 — Accepted Full Production Run (15 Temmuz)

| Parametre | Değer |
|---|---|
| Git revision | `f22e1e66a1e06879d29905637ecbfe6c0cfc6604` |
| Training candidates | 1,877,700 across 17,968 complete terms |
| Candidate sources | 250,000 positive; 316,893 BM25; 814,401 category; 496,406 random |
| Validation | 5-fold `StratifiedGroupKFold`, group=`term_id` |
| Models | 5 LightGBM + 5 XGBoost |
| Feature count | 33 |
| Cross-fitted scores | LGBM 0.837304; XGB 0.836820; blend **0.837508** |
| Deploy parameters | LGBM 0.65; XGB 0.35; threshold 0.3718157097697258 |
| Output | 3,359,679 rows; 645,783 positive; exact QA passed |
| Runtime | training 48:28.11; selection/inference 1:11.70 |
| Leaderboard | Not observed; team-account upload required |

The OOF, decision, and submission are hash-bound. See
[`july_15_delivery.md`](july_15_delivery.md) for the accepted integrity chain.

### EXP-001 — Baseline v0 (3 Temmuz)

**Amaç:** Sistemin uçtan uca çalıştığını doğrulamak, ilk kaba F1 skorunu almak.

| Parametre | Değer |
|---|---|
| Model | LightGBM |
| `num_leaves` | 31 |
| `learning_rate` | 0.05 |
| `min_child_samples` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `num_boost_round` | 500 (early stopping 30) |
| Negatif strateji | Random, ratio=3:1 |
| Eğitim seti | 5.000 pozitif + 15.000 negatif = 20.000 |
| Validation | Legacy 5-Fold row-level Stratified CV, seed=42 (invalidated) |

**Fold Sonuçları:**

| Fold | Macro-F1 | Best Iter |
|---|---|---|
| 1 | 0.9636 | 118 |
| 2 | 0.9615 | 124 |
| 3 | 0.9600 | 126 |
| 4 | 0.9603 | 118 |
| 5 | 0.9612 | 121 |
| **Ort.** | **0.9613 ± 0.0013** | — |

**Threshold Optimizasyonu:**
- Varsayılan (t=0.50): 0.9613
- En iyi (t=0.45): **0.9621**

**Feature Importance (gain):**
```
query_title_overlap       ██████████████████████████████ (89,620)
query_category_overlap    ██████████ (31,299)
query_brand_match         ████████ (24,145)
query_len                 █ (5,911)
title_len                  (2,582)
gender_match               (1,961)
query_cat_l1_overlap       (1,608)
```

**Çıkarımlar:**
- `query_title_overlap` dominant (toplam gain'in %58'i)
- Demografik feature'lar (gender_match) çok az katkı → %60 unknown nedeniyle beklenen
- Threshold 0.5 yerine 0.45 daha iyi → model biraz konservatif tahmin ediyor

**Sonraki adım:** TF-IDF ekleyip EXP-002/003 çalıştır.

---

### EXP-002 — Demografik Feature Seti (4 Temmuz)

> ⚠️ Code commit’de var (`src/features.py` satır 388-397), manuel CV sonucu
> kaydedilmemis. EXP-003 ile birlikte çalıştırılmıştı; TF-IDF (EXP-003) bastığı icin
> ayrı skor algılanmadı. Sonraki tam modelde feature importance'a bakılacak.

**Değişiklik:** `src/features.py`'ye `age_group_match` ve `demographic_conflict` eklendi.

| Parametre | EXP-001 | EXP-002 |
|---|---|---|
| Feature sayısı | 7 | 9 |
| Yeni feature'lar | — | `age_group_match`, `demographic_conflict` |

**Beklenti:** Küçük iyileşme (< +0.005). age_group %59 unknown olduğu için sınırlı etki.

---

### EXP-003 — TF-IDF Cosine Feature (4 Temmuz)

> ⚠️ Code commit'de sonucu kayitli (experiment_log satiri: 0.9699 ± 0.0028)
> EXP-002 ile birlikte ayni run'da mi yoksa ayri mi calistirildi netlestirilmeli.

**Değişiklik:** `src/tfidf_features.py` baseline pipeline'a bağlandi.

| Parametre | EXP-001 | EXP-003 |
|---|---|---|
| Feature sayısı | 7 | 10 |
| Yeni feature'lar | — | `tfidf_cosine` |
| TF-IDF vocab | — | 30K, ngram=(1,2) |

**Beklenti:** Orta düzey iyileşme (+0.005 — +0.02). TF-IDF PoC'ta pozitif/negatif cosine farkı 0.37 idi — güçlü sinyal.

---

## Notlar

- Tüm deneyler `seed=42` ile tekrar üretilebilir
- Local CV skoru Kaggle LB ile uyuşmayabilir — LB skoru her zaman not alınmalı
- **EXP-001 threshold 0.45 is historical and must not be used for submissions.**
- **EXP-003 best TF-IDF konfig: `ngram=(1,1), max_features=10_000`** — `run_baseline_tfidf.py` 8 Temmuz'da güncellendi
- **EXP-006** is retained only as an invalidated historical plan; EXP-010 is the
  accepted BM25/category/random result.

---

## EXP-006 Detayı (8 Temmuz)

**Amaç:** BM25 hard negative'in random negative'den daha iyi mi olduğunu ölçmek.

| Parametre | Değer |
|---|---|
| Script | `notebooks/06_bm25_karsilastirma_tam_omerfaruk.py` |
| BM25 top_n | 50 |
| Negatif oran | 3:1 |
| Örnek sorgu sayısı | 5.000 benzersiz sorgu |
| Feature seti | 15 (12 temel + TF-IDF + attributes) |

> **➡️ Sonucu** `python notebooks/06_bm25_karsilastirma_tam_omerfaruk.py` çalıştırınca `outputs/hard_neg_comparison.csv`'ye yazılır.

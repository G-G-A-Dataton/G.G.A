# G.G.A — Deney Kayıt Tablosu (experiment_log.md)

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
| EXP-001 | 3 Tem | Ömer Faruk | LightGBM | Random 3:1 / 5K poz | 7 temel | 0.9613 ± 0.0013 | — | Baseline v0, threshold opt. → 0.9621 |
| EXP-002 | 4 Tem | Ömer Faruk | LightGBM | Random 3:1 / 5K poz | 9 (7 + demografik) | TBD | — | age_group_match + demographic_conflict |
| EXP-003 | 4 Tem | Muhammed | LightGBM | Random 3:1 / 5K poz | 10 (9 + TF-IDF) | **0.9699 ± 0.0028** | — | TF-IDF #1 importance, +0.0086 kazanım |
| EXP-004 | 6 Tem | Ahmet Emin | LightGBM | Random 3:1 / 3K poz | 12 (10 + L2/L3/depth) | TBD | — | Kategori seviye feature'ları eklendi |
| EXP-005 | 7 Tem | Ömer Faruk | LightGBM | Random 3:1 / 3K poz | 12 temel | **0.9622 ± 0.0022** | — | Hard neg. baseline (BM25 bekleniyor), thresh=0.4 → 0.9625 |

---

## Deney Detayları

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
| Validation | 5-Fold Stratified CV, seed=42 |

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

> ⏳ Çalıştırıldıktan sonra doldurulacak

**Değişiklik:** `src/features.py`'ye `age_group_match` ve `demographic_conflict` eklendi.

| Parametre | EXP-001 | EXP-002 |
|---|---|---|
| Feature sayısı | 7 | 9 |
| Yeni feature'lar | — | `age_group_match`, `demographic_conflict` |

**Beklenti:** Küçük iyileşme (< +0.005). age_group %59 unknown olduğu için sınırlı etki.

---

### EXP-003 — TF-IDF Cosine Feature (4 Temmuz)

> ⏳ Çalıştırıldıktan sonra doldurulacak

**Değişiklik:** `src/tfidf_features.py` baseline pipeline'a bağlandı.

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
- **EXP-001 best threshold: 0.45** — submission'larda bu kullanılacak

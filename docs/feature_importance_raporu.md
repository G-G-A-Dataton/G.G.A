# Feature Importance Raporu (10 Temmuz)

> [!CAUTION]
> **Historical, invalidated result.** This run used the old attribute parser,
> which could not read the catalog's `key: value` format, and row-level CV.
> The zero attribute gains describe that broken run, not the current features.

**Hazırlayan:** Ömer Faruk Kara  
**Tarih:** 10 Temmuz 2026  
**Feature sayısı:** 15  
**Yöntem:** 5-Fold CV ortalama Gain Importance

---

## 1. Feature Önem Sıralaması (Gain'e Göre)

| Sıra | Feature | Gain Ort. | Gain % | Split Ort. | Split % |
|---|---|---|---|---|---|
| 1 | `query_title_overlap` | 46669 | 48.8% ██████████████ | 470 | 12.6% |
| 2 | `query_category_overlap` | 23056 | 24.1% ███████ | 471 | 12.7% |
| 3 | `query_brand_match` | 14660 | 15.3% ████ | 194 | 5.2% |
| 4 | `query_len` | 3507 | 3.7% █ | 752 | 20.2% |
| 5 | `title_len` | 2063 | 2.2%  | 829 | 22.3% |
| 6 | `query_cat_l3_overlap` | 2013 | 2.1%  | 180 | 4.8% |
| 7 | `query_cat_l1_overlap` | 786 | 0.8%  | 128 | 3.4% |
| 8 | `query_cat_l2_overlap` | 767 | 0.8%  | 122 | 3.3% |
| 9 | `gender_match` | 715 | 0.7%  | 167 | 4.5% |
| 10 | `cat_depth` | 689 | 0.7%  | 265 | 7.1% |
| 11 | `age_group_match` | 362 | 0.4%  | 82 | 2.2% |
| 12 | `demographic_conflict` | 289 | 0.3%  | 65 | 1.7% |
| 13 | `query_color_match` | 0 | 0.0%  | 0 | 0.0% |
| 14 | `query_size_match` | 0 | 0.0%  | 0 | 0.0% |
| 15 | `query_material_match` | 0 | 0.0%  | 0 | 0.0% |

---

## 2. Bulgular ve Uyarılar

> [!WARNING]
> **Kritik Bulgu:** `query_color_match`, `query_size_match`, `query_material_match` — **3 attribute feature'ının gain importance'ı tam sıfır!**

**Sıfır importance'ın olası nedenleri:**

1. **Attributes doluluk sorunu:** ~%32 ürünün hiç attributes bilgisi yok → feature çoğunlukla 0 kalıyor, model öğrenemiyor
2. **Random negative yetersizliği:** Random örnekleme attribute çelişkisi yaratan zor negatifler üretmiyor — model attribute olmadan da ayırt edebiliyor
3. **Sığ model:** 3K pozitif + 9K negatif küçük eğitim seti; attributes sinyali ancak büyük veri + hard negative ile ortaya çıkabilir

> [!NOTE]
> `age_group_match` (0.4%) ve `demographic_conflict` (0.3%) de çok düşük. `gender=unknown` %61 oranında olduğu için bu feature'lar sınırlı kalıyor.

**Temel dağılım: İlk 3 feature gain'in %88.2'sini taşıyor:**
- `query_title_overlap`: **%48.8**
- `query_category_overlap`: **%24.1**  
- `query_brand_match`: **%15.3**


---

## 3. Öneri

### Tutulması Gereken (Top feature'lar)
- `query_title_overlap`
- `query_category_overlap`
- `query_brand_match`
- `query_len`
- `title_len`

### Çıkarılabilir veya İzlenecek Düşük Önemli Feature'lar
- `query_cat_l1_overlap` (gain < %1)
- `query_cat_l2_overlap` (gain < %1)
- `gender_match` (gain < %1)
- `cat_depth` (gain < %1)
- `age_group_match` (gain < %1)
- `demographic_conflict` (gain < %1)
- `query_color_match` (gain < %1)
- `query_size_match` (gain < %1)
- `query_material_match` (gain < %1)

---

## 4. Sonraki Adımlar

- Düşük önemli feature'lar Sprint 2'de çıkarılarak model yeniden eğitilebilir
- `tfidf_cosine` — TF-IDF'in 10K unigram konfigürasyonuyla yeniden ölçülmeli
- Embedding cosine feature eklendikten sonra bu analiz tekrarlanmalı (12 Temmuz)

*Ham CSV: `outputs/feature_importance.csv`*

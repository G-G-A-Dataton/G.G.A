# Submission Feature Dry-Run Raporu (13 Temmuz)

**Hazırlayan:** Mustafa Mert Çevik  
**Tarih:** 13 Temmuz 2026  
**Mod:** Dry-Run (10,000 satir)  

---

## 1. Join Sonucu

| Metrik | Deger |
|---|---|
| Submission satir | 10,000 |
| Katalogsuz term_id | 0 |
| Katalogsuz item_id | 0 |

---

## 2. Feature Dagilimi

| Feature | Min | Max | Mean | Null |
|---|---|---|---|---|
| query_title_overlap | 0.0 | 1.0 | 0.0572 | 0 |
| query_category_overlap | 0.0 | 1.0 | 0.0704 | 0 |
| query_brand_match | 0.0 | 1.0 | 0.0754 | 0 |
| query_cat_l1_overlap | 0.0 | 0.5 | 0.0096 | 0 |
| title_len | 4.0 | 100.0 | 55.0297 | 0 |
| query_len | 2.0 | 50.0 | 16.0622 | 0 |
| gender_match | -1.0 | 1.0 | 0.0332 | 0 |
| age_group_match | -1.0 | 1.0 | 0.0079 | 0 |
| demographic_conflict | 0.0 | 1.0 | 0.0248 | 0 |
| query_cat_l2_overlap | 0.0 | 1.0 | 0.0268 | 0 |
| query_cat_l3_overlap | 0.0 | 1.0 | 0.0736 | 0 |
| cat_depth | 2.0 | 6.0 | 3.858 | 0 |
| query_color_match | 0.0 | 0.0 | 0.0 | 0 |
| query_size_match | 0.0 | 0.0 | 0.0 | 0 |
| query_material_match | 0.0 | 0.0 | 0.0 | 0 |

---

## 3. Performans

| Metrik | Deger |
|---|---|
| Toplam sure | 39.7s |
| Feature uretim suresi | 9.0s |
| Isleme hizi | 1,107 satir/s |
| RAM kullanimi | 1254 MB |

## 4. Sorunlar

- [SIFIR VARYANS] query_color_match: Tum degerler 0
- [SIFIR VARYANS] query_size_match: Tum degerler 0
- [SIFIR VARYANS] query_material_match: Tum degerler 0
> [!WARNING]
> Yukaridaki sorunlar cozulmeden submission uretilmemeli!

> [!NOTE]
> Dry-run hizi: **1,107 satir/s**. Tam submission icin tahmini sure: **50.6 dakika**.
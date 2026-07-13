# Teknik Rapor Bölümü v1 — EDA ve Feature Engineering Bulguları (10 Temmuz)

> [!NOTE]
> Historical draft. Model scores and feature conclusions that depend on legacy
> CV or the former attribute parser are not current validation evidence.

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 10 Temmuz 2026  
**Kapsam:** Sprint 1-2 boyunca elde edilen tüm EDA ve feature bulguları

---

## 1. Veri Seti Genel Bakış

| Dosya | Boyut | Kritik Not |
|---|---|---|
| `items.csv` | ~963K ürün | 79K+ benzersiz marka, yüksek kardinalite |
| `terms.csv` | ~50K sorgu | Kısa, net sorgular ağırlıklı |
| `training_pairs.csv` | ~250K çift | Tamamı pozitif (label=1) — negatif örnekleme zorunlu |
| `submission_pairs.csv` | ~3.36M çift | Tahmin edilecek asıl test seti |

**Önemli:** Negatif örneklerin tamamı yapay üretilmektedir. Bu, negatif örnekleme stratejisinin model kalitesini doğrudan belirlediği anlamına gelir.

---

## 2. Keşifsel Veri Analizi (EDA) Bulguları

### 2.1 Kategori Dağılımı

Katalogda en kalabalık 5 L1 kategorisi:

| Kategori (L1) | Ürün Sayısı | Oran |
|---|---|---|
| Ev & Mobilya | ~250K | %26 |
| Giyim | ~164K | %17 |
| Elektronik | ~130K | %13 |
| Spor & Outdoor | ~105K | %11 |
| Kozmetik | ~87K | %9 |

> [!NOTE]
> Ev & Mobilya + Giyim beraber kataloğun ~%43'ünü oluşturuyor. Bu iki kategorideki 
> model performansı genel skoru belirleyici şekilde etkiler.

### 2.2 Cinsiyet ve Yaş Grubu Dağılımı

| Alan | Değer | Oran |
|---|---|---|
| `gender` | `unknown` | **%61** |
| `gender` | `erkek` | %22 |
| `gender` | `kadın` | %17 |
| `age_group` | `unknown` | **%59** |
| `age_group` | `yetişkin` | %26 |
| `age_group` | `çocuk` | %15 |

**Sonuç:** Cinsiyet ve yaş grubu bilgisi eksik verinin çoğunlukta olması nedeniyle hard filter olarak kullanılamaz. Soft feature (sürekli değer, -1/0/1) olarak modele verildi.

### 2.3 Marka Sinyali

Top 10 marka sorgu metinlerinde sık geçiyor:
- adidas, nike, koton, defacto, puma, columbia, tommy hilfiger, mavi, lc waikiki, zara

`query_brand_match` feature'ı bu sinyali başarıyla yakalıyor (feature importance'ta sürekli top 5'te).

### 2.4 Attributes (Nitelik) Doluluğu

| Nitelik | Dolu Ürün | Oran |
|---|---|---|
| `attributes` (herhangi biri) | ~654K | **%68** |
| Renk bilgisi | ~520K | ~%54 |
| Materyal bilgisi | ~299K | ~%31 |
| Beden/Numara | ~280K | ~%29 |

> [!NOTE]
> ~%32 ürünün hiç attributes bilgisi yok → Bu ürünlerde `query_color_match`, 
> `query_size_match`, `query_material_match` feature'ları 0 değer alır.

### 2.5 Title Uzunluk Analizi

- Çok kısa title'lar (< 10 karakter): ~%8 — çoğunlukla model kodları (örn. "530", "990v6")
- Bu ürünlerde TF-IDF yetersiz → Attributes + category feature'ları kritik

---

## 3. Feature Engineering Özeti (Sprint 1-2)

### 3.1 Geliştirilen Feature'lar

| Feature | Açıklama | Eklenme | Önem (tahmini) |
|---|---|---|---|
| `query_title_overlap` | Sorgu-başlık Jaccard benzerliği | 3 Tem | ⭐⭐⭐⭐⭐ |
| `query_category_overlap` | Sorgu-kategori Jaccard benzerliği | 3 Tem | ⭐⭐⭐ |
| `query_brand_match` | Sorguda marka adı geçiyor mu? | 3 Tem | ⭐⭐⭐⭐ |
| `query_cat_l1_overlap` | L1 kategori örtüşmesi | 3 Tem | ⭐⭐ |
| `title_len` | Başlık uzunluğu | 3 Tem | ⭐ |
| `query_len` | Sorgu uzunluğu | 3 Tem | ⭐ |
| `gender_match` | Cinsiyet uyum skoru (-1/0/1) | 4 Tem | ⭐⭐ |
| `age_group_match` | Yaş grubu uyum skoru | 4 Tem | ⭐ |
| `demographic_conflict` | Cinsiyet/yaş çelişkisi var mı? | 4 Tem | ⭐⭐ |
| `tfidf_cosine` | TF-IDF cosine similarity | 4 Tem | ⭐⭐⭐⭐⭐ |
| `query_cat_l2_overlap` | L2 (orta) kategori örtüşmesi | 6 Tem | ⭐⭐⭐ |
| `query_cat_l3_overlap` | L3 (en spesifik) kategori örtüşmesi | 6 Tem | ⭐⭐ |
| `cat_depth` | Kategori derinliği (1/2/3+) | 6 Tem | ⭐ |
| `query_color_match` | Sorguda renk bilgisi ürünle uyuşuyor mu? | 8 Tem | ⭐⭐⭐ |
| `query_size_match` | Sorguda beden bilgisi ürünle uyuşuyor mu? | 8 Tem | ⭐⭐ |
| `query_material_match` | Sorguda materyal bilgisi ürünle uyuşuyor mu? | 8 Tem | ⭐⭐ |

**Toplam:** 16 feature (tfidf_cosine dahil)

### 3.2 Feature Tasarım Prensipleri

1. **Asimetrik eşleşme kontrolü:** Sorgunun tüm kelimelerinin başlıkta geçip geçmediği ölçülür (Jaccard overlap). Sıra önemsiz, sadece içerik önemli.

2. **Soft sinyal:** Cinsiyet, renk, materyal gibi alanlar sert kural değil, yumuşak sinyal olarak verildi. Eksik bilgi durumunda ceza yok (0 değer).

3. **Hiyerarşik kategori:** L1 (genel) → L2 (orta) → L3 (spesifik) her biri ayrı feature. Model hangi seviyenin ne kadar önemli olduğunu öğreniyor.

4. **Anlamsal genişleme (TF-IDF):** Kelime düzeyinde benzerlik ötesinde, IDF ağırlıklı cosine similarity. 6 Temmuz deneyi: unigram (1,1) + 10K vocab en iyi ayırıcılığı verdi.

---

## 4. Negatif Örnekleme Stratejileri

### 4.1 Random Negative (Sprint 1)

- Katalogdan tamamen rastgele ürün seçimi
- Avantaj: Basit, hızlı, tekrar üretilebilir
- Dezavantaj: "Kolay negatifler" — model çok hızlı öğrenir ama gerçek dünyada yanılır

### 4.2 BM25 Hard Negative (Sprint 2, 6-7 Temmuz)

- BM25 index ile her sorgu için en benzer ama pozitif olmayan ürünler seçilir
- Avantaj: Model daha ayırt edici öğrenir
- Dezavantaj: Yavaş üretim, çakışma kontrolü gerekir
- `src/bm25_hard_negative.py` → `src/negative_sampling.py` → `verify_no_leakage()`

### 4.3 Karışık Dataset v2 (Sprint 2, 9 Temmuz)

- Random + BM25 karışımı → `src/train_mix_v2.py`
- Hem kolay hem zor negatifler → daha dengeli öğrenme

---

## 5. Model Sonuçları Özeti

| Deney | Tarih | Yenilik | CV F1 |
|---|---|---|---|
| EXP-001 | 3 Tem | Baseline (7 feature, random neg) | 0.9613 |
| EXP-003 | 4 Tem | +TF-IDF cosine | **0.9699** |
| EXP-005 | 7 Tem | +12 feature, hard neg baseline | 0.9625 |
| EXP-008 | 8 Tem | Tuned: num_leaves=31, lr=0.05 | **0.9631** |
| EXP-009 | 9 Tem | XGBoost vs LightGBM | LGBM: **0.9613** > XGB: 0.9597 |

**Temel bulgular:**
- TF-IDF cosine en yüksek katkıyı sağladı (+0.0086)
- LightGBM, XGBoost'u hem hız hem F1 açısından geçti
- Optimal threshold ~0.40-0.45 aralığında (varsayılan 0.5'ten düşük)

---

## 6. Açık Sorular ve Sprint 3 Yönlendirmesi

| Soru | Durum | Sprint |
|---|---|---|
| Embedding cosine feature ne kadar katkı sağlar? | Planlandı | Sprint 3 (12 Tem) |
| BM25 hard negative random'dan ne kadar iyi? | Mert çıktısı bekleniyor | Sprint 2 devam |
| Optimal negatif oran (1:1 vs 3:1 vs 5:1)? | Planlandı | Sprint 3 (11 Tem) |
| Renk feature'ının %32 boş veri sorunu nasıl çözülür? | Araştırılacak | Sprint 3 |
| Ensemble (LGBM + XGB) avantaj sağlar mı? | Planlandı | Sprint 3 (13 Tem) |

---

*Bu rapor Sprint 1-2 bulgularını kapsamaktadır. Sprint 3 tamamlandığında güncellenecektir.*

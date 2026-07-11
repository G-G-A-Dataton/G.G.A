# Rapor Yöntem Bölümü v1 (13 Temmuz)

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 13 Temmuz 2026  
**Kapsam:** Sprint 1-2 Yöntem Özeti (Negatif Örnekleme, Feature Engineering, Validation)

---

## 1. Problem Tanımı

Bu çalışmada, bir e-ticaret platformunda kullanıcı arama sorguları (queries) ile ürün kataloğu arasındaki uyumu tahmin eden ikili sınıflandırma modeli geliştirilmektedir.

- **Girdi:** (query, item) çifti
- **Çıktı:** 1 (uyumlu) / 0 (uyumsuz)
- **Değerlendirme:** Macro-F1 skoru

Veri setinde yalnızca pozitif çiftler sağlanmıştır (`training_pairs.csv`). Negatif örneklerin tamamı yapay olarak üretilmektedir.

---

## 2. Negatif Örnekleme Stratejileri

### 2.1 Random Negative Örnekleme

**Yöntem:** Her pozitif sorgu için, eğitim setinde o sorguya ait olmayan rastgele ürünler seçilir.

**Uygulama:**
```python
build_training_set(pos_pairs, items_df, ratio=2, random_state=42)
```

**Avantaj:** Hızlı ve tekrar üretilebilir. Baseline olarak kullanılabilir.

**Dezavantaj:** Üretilen negatifler genellikle çok kolaydır (farklı kategori, farklı marka). Model gerçek dünya zor negatiflerinde yanılabilir.

### 2.2 BM25 Hard Negative Örnekleme

**Yöntem:** BM25 metin benzerlik skoruyla her sorgu için en benzer ama pozitif olmayan ürünler seçilir. Bu ürünler modeli daha fazla zorlar.

**Uygulama:**
```python
BM25HardNegativeSampler(items_df).sample(pos_pairs, ratio=1)
```

**Avantaj:** Model daha ayırt edici öğrenir, gerçek dünya dağılımına daha yakın.

**Dezavantaj:** BM25 index kurulumu ve örnekleme daha yavaştır (~5-10x).

### 2.3 Karışık Dataset v2 (Random + BM25)

**Yöntem:** Eğitim setinin bir kısmı BM25 hard negative, kalanı random negative olarak oluşturulur.

**Uygulama:** `src/train_mix_v2.py`

**11 Temmuz Sonucu:** 2:1 negatif oran en yüksek F1 (0.9632) verdi. 5:1 oranında F1 düşüşü gözlemlendi.

---

## 3. Feature Engineering

### 3.1 Metin Benzerliği Feature'ları (3 Temmuz)

| Feature | Formül / Mantık |
|---|---|
| `query_title_overlap` | Jaccard(query_words ∩ title_words) |
| `query_category_overlap` | Jaccard(query_words ∩ category_words) |
| `query_brand_match` | brand ∈ query → 1, değil → 0 |
| `query_cat_l1_overlap` | L1 kategori kelimelerinin sorguda geçme oranı |
| `title_len` | Başlık kelime sayısı |
| `query_len` | Sorgu kelime sayısı |

**Önem:** `query_title_overlap` ve `query_category_overlap` gain importance'ın **%73**'ünü taşıyor (10 Temmuz analizi).

### 3.2 TF-IDF Cosine Similarity (4 Temmuz)

BM25 ile indekslenen ürün metinleri üzerinde her sorgu için cosine similarity hesaplanır.

```
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=10_000)
tfidf_cosine = cosine_similarity(query_vec, item_vec)
```

**6 Temmuz deneyi:** Unigram + 10K vocab en iyi konfigürasyon seçildi (EXP-006).

### 3.3 Demografik Feature'lar (4 Temmuz)

| Feature | Değerler |
|---|---|
| `gender_match` | -1 (çelişki) / 0 (unknown) / 1 (uyum) |
| `age_group_match` | -1 / 0 / 1 |
| `demographic_conflict` | 1 (herhangi bir çelişki) / 0 |

> **Not:** `gender=unknown` %61 oranında olduğu için bu feature'lar sınırlı sinyal taşıyor.

### 3.4 Kategori Hiyerarşisi (6 Temmuz)

| Feature | Mantık |
|---|---|
| `query_cat_l2_overlap` | Orta kategori seviyesi örtüşmesi |
| `query_cat_l3_overlap` | En spesifik kategori örtüşmesi |
| `cat_depth` | Kategori derinliği (1/2/3+) |

### 3.5 Attributes Feature'ları (8 Temmuz)

| Feature | Kaynak |
|---|---|
| `query_color_match` | items.csv → attributes.renk |
| `query_size_match` | items.csv → attributes.beden |
| `query_material_match` | items.csv → attributes.materyal |

> **10 Temmuz Bulgusu:** Bu 3 feature'ın gain importance'ı sıfır çıktı. BM25 hard negative ile tekrar test edilecek.

---

## 4. Model ve Validation Yöntemi

### 4.1 Model Seçimi

**9 Temmuz EXP-009 sonucuna göre LightGBM seçildi:**

| Model | CV F1 | Eğitim Süresi |
|---|---|---|
| LightGBM (tuned) | **0.9613** | ~12s |
| XGBoost | 0.9597 | ~85s |

### 4.2 Hiperparametre Tuning (8 Temmuz, EXP-008)

```python
LGBM_PARAMS = {
    "num_leaves"       : 31,
    "learning_rate"    : 0.05,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
}
```

### 4.3 Validation Stratejisi

**5-Fold Stratified Cross-Validation:**
- Her fold pozitif/negatif oranını korur
- OOF (Out-of-Fold) tahminleri ile threshold optimizasyonu yapılır
- Seed sabit tutularak tekrarlanabilirlik sağlanır

**11 Temmuz Threshold Analizi:** Optimal threshold = **0.35** (varsayılan 0.5 değil).

---

## 5. Deney Geçmişi Özeti

| Deney | Tarih | F1 | Yenilik |
|---|---|---|---|
| EXP-001 | 3 Tem | 0.9613 | Baseline (7 feature) |
| EXP-003 | 4 Tem | **0.9699** | +TF-IDF cosine |
| EXP-005 | 7 Tem | 0.9625 | Hard negative baseline |
| EXP-008 | 8 Tem | 0.9631 | LGBM tuning |
| EXP-009 | 9 Tem | 0.9613 | LGBM vs XGBoost |

---

## 6. Açık Sorular (Sprint 3 İçin)

1. Gerçek embedding cosine feature ne kadar katkı sağlar? (12 Temmuz sonucu sentetikti)
2. BM25 hard negative vs random negatif tam veri setinde ne fark yaratır?
3. Ensemble (LGBM + XGB) gerçek anlamda F1 artırır mı? (13 Temmuz)
4. Optimal threshold tam eğitim setinde de 0.35 mi kalır?

---

*Bu belge Sprint 1-2 yöntemlerini özetlemektedir. Sprint 3 tamamlandığında güncellenecektir.*

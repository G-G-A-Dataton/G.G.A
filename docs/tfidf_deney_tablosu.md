# TF-IDF Hiperparametre Deney Tablosu (6 Temmuz)

**Hazırlayan:** Muhammed Köseoğlu  
**Tarih:** 6-7 Temmuz 2026  
**Script:** `run_tfidf_experiments.py`

---

## Deney Tasarımı

**Ölçülen metrik:** Cosine Similarity Separation = `pozitif_ortalama - negatif_ortalama`
- Yüksek separation → TF-IDF bu parametreyle pozitif/negatif çiftleri daha iyi ayırt ediyor
- Küçük bir eğitim seti üzerinde test edildi (1.000 poz + 1.000 neg, seed=42)

**Denenen parametreler:**

| Parametre | Değerler |
|---|---|
| `ngram_range` | (1,1) · (1,2) · (1,3) |
| `max_features` | 10.000 · 30.000 · 50.000 |
| `min_df` | 1 · 2 · 5 |
| **Toplam deney** | **27 kombinasyon** |

---

## Tam Sonuç Tablosu (Separation'a Göre Sıralı)

| ngram | max_feat | min_df | vocab | pos_cos | neg_cos | **separation** | süre(s) |
|---|---|---|---|---|---|---|---|
| (1,1) | 10.000 | 5 | 10.000 | 0.4524 | 0.0060 | **0.4464** 🥇 | 27.7 |
| (1,1) | 10.000 | 1 | 10.000 | 0.4522 | 0.0060 | **0.4462** 🥈 | 27.5 |
| (1,1) | 10.000 | 2 | 10.000 | 0.4522 | 0.0060 | **0.4462** 🥉 | 27.6 |
| (1,1) | 30.000 | 5 | 30.000 | 0.4421 | 0.0056 | 0.4365 | 40.9 |
| (1,1) | 30.000 | 2 | 30.000 | 0.4418 | 0.0056 | 0.4362 | 27.7 |
| (1,1) | 30.000 | 1 | 30.000 | 0.4416 | 0.0056 | 0.4360 | 31.1 |
| (1,1) | 50.000 | 5 | 50.000 | 0.4329 | 0.0055 | 0.4275 | 50.0 |
| (1,1) | 50.000 | 2 | 50.000 | 0.4328 | 0.0055 | 0.4273 | 39.6 |
| (1,1) | 50.000 | 1 | 50.000 | 0.4327 | 0.0055 | 0.4272 | 27.5 |
| (1,2) | 10.000 | 1 | 10.000 | 0.3687 | 0.0041 | 0.3646 | 96.9 |
| (1,2) | 10.000 | 2 | 10.000 | 0.3687 | 0.0041 | 0.3646 | 78.2 |
| (1,2) | 10.000 | 5 | 10.000 | 0.3686 | 0.0041 | 0.3645 | 100.8 |
| (1,2) | 30.000 | 5 | 30.000 | 0.3454 | 0.0034 | 0.3421 | 61.5 |
| (1,2) | 30.000 | 1 | 30.000 | 0.3454 | 0.0034 | 0.3420 | 50.1 |
| (1,2) | 30.000 | 2 | 30.000 | 0.3454 | 0.0034 | 0.3420 | 47.5 |
| (1,2) | 50.000 | 1 | 50.000 | 0.3337 | 0.0031 | 0.3307 | 49.1 |
| (1,2) | 50.000 | 5 | 50.000 | 0.3336 | 0.0031 | 0.3305 | 46.5 |
| (1,2) | 50.000 | 2 | 50.000 | 0.3332 | 0.0031 | 0.3301 | 47.6 |
| (1,3) | 10.000 | 1 | 10.000 | 0.3323 | 0.0038 | 0.3284 | 73.5 |
| (1,3) | 10.000 | 2 | 10.000 | 0.3323 | 0.0038 | 0.3284 | 75.7 |
| (1,3) | 10.000 | 5 | 10.000 | 0.3323 | 0.0038 | 0.3284 | 125.6 |
| (1,3) | 30.000 | 1 | 30.000 | 0.3160 | 0.0029 | 0.3131 | 90.5 |
| (1,3) | 30.000 | 2 | 30.000 | 0.3160 | 0.0029 | 0.3131 | 75.4 |
| (1,3) | 30.000 | 5 | 30.000 | 0.3160 | 0.0029 | 0.3131 | 78.6 |
| (1,3) | 50.000 | 2 | 50.000 | 0.2990 | 0.0027 | 0.2964 | 74.8 |
| (1,3) | 50.000 | 1 | 50.000 | 0.2990 | 0.0027 | 0.2963 | 75.3 |
| (1,3) | 50.000 | 5 | 50.000 | 0.2990 | 0.0027 | 0.2963 | 75.0 |

---

## Bulgular ve Yorumlar

### 1. Unigram (1,1) açık ara kazandı

- Unigram ortalama separation: **~0.43**
- Bigram  ortalama separation: **~0.34** (-%21 düşüş)
- Trigram ortalama separation: **~0.31** (-%28 düşüş)

**Neden?** Türkçe çekimli bir dil; "koşu ayakkabısı" ifadesinde "koşu" ve "ayakkabısı" zaten birer token. Bigram oluşturmak kelime dağarcığını gererek sulandırıyor ve IDF ağırlıkları zayıflıyor.

### 2. Küçük kelime dağarcığı (10K) daha iyi

- 10K vocab: separation = **0.4464**
- 30K vocab: separation = **0.4365** (-%2.2)
- 50K vocab: separation = **0.4272** (-%4.3)

**Neden?** Daha küçük kelime dağarcığında IDF değerleri daha anlamlı: ürün başlıklarında geçen nadir kelimeler daha yüksek ağırlık alıyor. Büyük vocab'da IDF dağılımı düzleşiyor.

### 3. min_df'nin etkisi ihmal edilebilir

Aynı ngram/max_feat kombinasyonu içinde `min_df=1`, `2`, `5` arasındaki separation farkı < 0.0002. Uygulama açısından önemsiz.

---

## Öneri

> **Kullanılacak konfigürasyon:** `ngram_range=(1,1)`, `max_features=10_000`, `min_df=2`

`min_df=2` seçildi çünkü:
- `min_df=5` ile neredeyse aynı separation (**0.4462 vs 0.4464**)
- Çok nadir terimleri (örn. yazım hataları, benzersiz model kodları) filtreler
- `min_df=1` kadar gürültülü değil

**Bu konfigürasyon `src/tfidf_features.py`'daki varsayılan değerlere yansıtılmalıdır.**

> [!NOTE]
> Mevcut `run_baseline_tfidf.py`'da `max_features=30_000, ngram_range=(1,2)` kullanılıyordu.
> 10K unigram konfigürasyonu ile separation **0.34 → 0.45** seviyesine çıkıyor.
> Bir sonraki baseline çalıştırmasında bu parametreler güncellenmelidir.

---

## Ham Sonuçlar

Ham CSV çıktısı: `outputs/tfidf_deney_sonuclari.csv`

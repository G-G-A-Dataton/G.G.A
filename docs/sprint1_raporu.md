# Sprint 1 Raporu — İlk 5 Gün Özeti (1-5 Temmuz 2026)

**Hazırlayan:** Ahmet Emin Işın (İletişim ve Rapor Sorumlusu)  
**Tarih:** 5 Temmuz 2026  
**Kapsam:** 1 Temmuz – 5 Temmuz 2026 (Sprint 1)

---

## 1. Sprint Hedefleri ve Tamamlanma Durumu

| Tarih | Görev | Sahibi | Durum |
|---|---|---|---|
| 1 Temmuz | Repo standardı ve EDA notları | Tüm ekip | ✅ Tamamlandı |
| 2 Temmuz | Macro-F1, EDA ön raporu, veri kalite, submission QA | Tüm ekip | ✅ Tamamlandı |
| 3 Temmuz | Baseline notebook, feature'lar, negatif örnekleme, TF-IDF | Tüm ekip | ✅ Tamamlandı |
| 4 Temmuz | Demografik feature'lar, TF-IDF pipeline bağlantısı | Ahmet Emin, Muhammed | ✅ Tamamlandı |
| 5 Temmuz | Submission pipeline, QA checklist, bu rapor | Ömer Faruk, Muhammed, Ahmet Emin | ✅ Tamamlandı |

**Sprint 1 Sonucu:** Tüm hedefler zamanında tamamlandı. Teknik altyapı eksiksiz kuruldu.

---

## 2. Veri Analizi (EDA) Bulguları

### 2.1 Veri Seti Özeti

| Dosya | Boyut | Önemli Not |
|---|---|---|
| `items.csv` | ~963K ürün | 79K+ benzersiz marka — yüksek kardinalite |
| `terms.csv` | ~50K sorgu | Kısa ve net sorgular ağırlıklı |
| `training_pairs.csv` | 250K çift | **Tamamı pozitif (label=1)** — negatif üretmek gerekiyor |
| `submission_pairs.csv` | 3.36M çift | Tahmin edilecek çiftler |

### 2.2 Kritik Veri Gözlemleri

**Kategori Dağılımı:**
- Ev & Mobilya (%26) ve Giyim (%17) kataloğun yaklaşık yarısını oluşturuyor
- Kategori hiyerarşisi 3+ seviyeli (`l1/l2/l3`) — her seviye ayrı feature olabilir

**Cinsiyet ve Yaş Grubu Skewness:**
- `gender = unknown` → **%61** oranında — çok yüksek
- `age_group = unknown` → **%59** oranında — çok yüksek
- Bu yüzden cinsiyet/yaş kuralı hard filter olarak kullanılamaz, **soft feature** olarak kullanıldı

**Marka Sinyali:**
- Top 15 marka (adidas, koton, defacto, nike...) sorgu metinlerinde sık geçiyor
- `query_brand_match` feature'ı bu sinyali yakalıyor

### 2.3 Pozitif Çift Manuel İnceleme (50 örnek)

**Güçlü sinyal durumları:**
- Marka uyumu: `"kerastase şampuan"` → `brand: kerastase` — çok güvenilir
- Model kodu uyumu: `"new balance 408"` → `title: ml408wn-r 408 spor ayakkabı` — yüksek özgüllük
- Kategori uyumu: `"erkek spor ayakkabı"` → `category: ayakkabı/spor ayakkabı/...`

**Zayıf/dikkat gerektiren durumlar:**
- Genel sorgular: `"yüz bakım"` → çok farklı ürünler eşleşebilir
- Kısa başlıklar: `title: '530'` (sadece model kodu) — TF-IDF için anlamsız
- Cinsiyet çakışması: `"çocuk abiye elbise"` → `gender: kadın` — gender etiketine güvenmek riskli

---

## 3. Feature Analizi

### 3.1 Sprint 1'de Üretilen Feature'lar

| Feature | Tür | Gün | Feature Importance |
|---|---|---|---|
| `query_title_overlap` | Jaccard benzerliği | 3 Tem | 🥇 En yüksek (89,620) |
| `query_category_overlap` | Jaccard benzerliği | 3 Tem | 🥈 İkinci (31,299) |
| `query_brand_match` | Binary eşleşme | 3 Tem | 🥉 Üçüncü (24,145) |
| `query_len` | Sayısal | 3 Tem | 4. (5,911) |
| `title_len` | Sayısal | 3 Tem | 5. (2,582) |
| `gender_match` | Ternary (-1/0/1) | 3 Tem | 6. (1,961) |
| `query_cat_l1_overlap` | Jaccard benzerliği | 3 Tem | 7. (1,608) |
| `age_group_match` | Ternary (-1/0/1) | 4 Tem | Test edilecek |
| `demographic_conflict` | Binary (0/1) | 4 Tem | Test edilecek |
| `tfidf_cosine` | Cosine similarity | 3-4 Tem | Test edilecek |

### 3.2 Feature Kalite Değerlendirmesi

**Güçlü feature'lar (kesinlikle tutulacak):**
- `query_title_overlap` — tartışmasız en güçlü sinyal
- `query_brand_match` — yüksek precision, özellikle brand aramaları için kritik

**İyileştirilmesi gereken feature'lar:**
- `gender_match` ve `age_group_match` — %60 `unknown` nedeniyle çok sayıda `0` üretiyor, etki sınırlı
- `query_cat_l1_overlap` — baseline'da en düşük importance, kategorinin sadece L1'ini kullanmak zayıf

**Sonraki sprint'te eklenecek feature'lar:**
- TF-IDF cosine similarity (4-5 Temmuz'da bağlandı)
- Embedding cosine similarity (12 Temmuz)
- Attributes parse (renk, materyal) (8 Temmuz)
- BM25 benzerlik skoru (7 Temmuz'dan itibaren)

---

## 4. Model Sonuçları

### 4.1 Baseline LightGBM v0 (3-4 Temmuz)

| Parametre | Değer |
|---|---|
| Model | LightGBM |
| Negatif strateji | Random (ratio=3:1) |
| Eğitim seti boyutu | 20.000 satır (5K poz + 15K neg) |
| Feature sayısı | 7 (temel) |
| Validation şeması | 5-Fold Stratified CV |
| **Ort. Macro-F1** | **0.9613 ± 0.0013** |
| En iyi threshold | 0.45 |
| Optimized F1 | 0.9621 |

> **Not:** Bu sonuç SADECE 5.000 pozitif örnek ile elde edilmiştir. Tüm 250K pozitif ile tam eğitim (4-5 Temmuz) çok daha güçlü bir model üretecektir.

> **Önemli:** Yüksek local validation skoru (0.96) Kaggle public LB ile uyuşmayabilir. Asıl değerlendirme Kaggle submission'ından sonra yapılacak.

---

## 5. Teknik Altyapı Durumu

### Oluşturulan Modüller

```
src/
├── data.py              ✅ Bellek optimizasyonlu veri yükleme
├── features.py          ✅ 9 feature (7 temel + 2 demografik)
├── metrics.py           ✅ Macro-F1 + Stratified KFold + threshold opt.
├── negative_sampling.py ✅ Sızıntısız random negative (1:1/3:1/5:1)
├── tfidf_features.py    ✅ TF-IDF cosine similarity (ayırıcılık: +0.37)
├── submission.py        ✅ Batch submission üretimi + ensemble
└── validate_submission.py ✅ 5+2 aşamalı format kontrolü

notebooks/
└── 01_veri_kalite_mert.py ✅ Veri kalite kontrol (0 sızıntı, 0 merge kaybı)
```

### Çalıştırma Scriptleri

| Script | Ne Yapar |
|---|---|
| `verify_pipeline.py` | Veri yükleme ve merge testini doğrular |
| `run_baseline.py` | Temel 7 feature ile LightGBM eğitir |
| `run_baseline_tfidf.py` | TF-IDF dahil 8 feature ile eğitir |
| `run_submission_qa.py` | Submission formatını doğrular |

---

## 6. Sprint 2 Planı (6-12 Temmuz)

| Tarih | Öncelik | Görev |
|---|---|---|
| 6 Temmuz | Yüksek | Hata analizi: false positive/negative örüntüleri |
| 7 Temmuz | Yüksek | BM25 hard negative mining |
| 8 Temmuz | Orta | Attributes parse (renk, materyal, beden) |
| 10 Temmuz | Orta | Feature importance analizi (tam modelde) |
| 12 Temmuz | Düşük | Embedding cosine PoC |

---

## 7. Açık Sorular ve Riskler

| Risk | Açıklama | Önlem |
|---|---|---|
| Local/LB uçurumu | 0.96 local skor Kaggle'da çok düşebilir | BM25 hard negative ile daha gerçekçi negatifler üret |
| Unknown skewness | %60 `unknown` gender/age feature etkisini sınırlıyor | Soft feature olarak bırak, hard filter yapma |
| Kısa title | Bazı ürünlerin başlığı sadece model kodu (örn. "530") | Attributes ve category bilgisini title'a ekle |
| Eğitim süresi | 250K pozitif + 3:1 negatif = 1M satır → apply() yavaş | Feature'ları vektörize hesaplamaya geç |

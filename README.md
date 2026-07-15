# 🛒 G.G.A — E-Ticaret Ürün-Terim İlişkilendirme

**TEKNOFEST 2026 E-Ticaret Hackathon Yarışması** (Kaggle Datathon Aşaması) için geliştirilen; arama terimleri ile ürün özellikleri arasındaki anlamsal bağı çözmeyi amaçlayan **metin tabanlı ikili sınıflandırma (Binary Classification)** projesi.

> T3 Vakfı, Sanayi ve Teknoloji Bakanlığı ve **Trendyol** iş birliğiyle düzenlenmektedir.

## 15 Temmuz Üretim Durumu

Tam üretim akışı 17.968 sorgu ve 1.877.700 eğitim adayı üzerinde çalıştırıldı.
Beş LightGBM ve beş XGBoost modelinden seçilen `%65/%35` blend, sorgu bazlı
cross-fitted Macro-F1 `0.837508` elde etti. `outputs/submission_v2.csv` tam
3.359.679 satır için format, binary değer, benzersiz/tam ID sırası ve hash
zinciri kontrollerini geçti. Ayrıntılı kanıt:
[`docs/july_15_delivery.md`](docs/july_15_delivery.md).

16 Temmuz temiz ortam dry-run'ı, 158 hash-kilitli paketle `102/102` testi geçti
ve tam submission dosyasını kabul edilen SHA-256 ile byte düzeyinde aynı üretti:
[`docs/reproducibility_dry_run.md`](docs/reproducibility_dry_run.md).

Bu değer yerel validasyon sonucudur, Kaggle leaderboard skoru değildir.

---

## 📌 Problem Tanımı

Bir kullanıcının arama terimi (query) ile bir ürün (item) arasında **alaka var mı yok mu?** sorusuna cevap veren bir model geliştirmek.

| Girdi | Çıktı |
|---|---|
| `("siyah erkek spor ayakkabı", "Nike erkek koşu ayakkabısı siyah")` | `1` (alakalı) |
| `("siyah erkek spor ayakkabı", "Kadın pembe topuklu sandalet")` | `0` (alakasız) |

**Değerlendirme Metriği:** Macro-F1

**Kritik Not:** Eğitim verisinde **yalnızca pozitif (label=1) örnekler** bulunmaktadır. Negatif örneklerin üretilmesi yarışmacılara bırakılmıştır.

---

## 📊 Veri Seti

| Dosya | Satır Sayısı | Açıklama |
|---|---|---|
| `items.csv` | 962,873 | Ürün kataloğu (title, category, brand, gender, age_group, attributes) |
| `terms.csv` | 50,153 | Arama terimleri (term_id, query) |
| `training_pairs.csv` | 250,000 | Eğitim çiftleri — sadece label=1 |
| `submission_pairs.csv` | 3,359,679 | Tahmin edilecek çiftler |
| `sample_submission.csv` | 3,359,679 | Gönderim formatı örneği |

---

## 📁 Proje Yapısı

```
G.G.A/
├── datasets/                    # Yarışma veri setleri (Git LFS)
│   ├── items.csv
│   ├── terms.csv
│   ├── training_pairs.csv
│   └── submission_pairs.csv
├── src/                         # Üretim (Production) modülleri
│   ├── data.py                  # Bellek dostu veri yükleme
│   ├── features.py              # 23 lexical/demographic/category/attribute features
│   ├── context_features.py      # 9 candidate-relative rank/gap features
│   ├── candidate_sampling.py    # BM25/category/random test-shaped candidates
│   ├── modeling.py              # Shared grouped OOF/threshold/ensemble contracts
│   ├── oof_artifacts.py         # Hash-verified shortlist artifact contract
│   ├── out_of_core_features.py  # Global context features with bounded RAM
│   ├── data_freeze.py           # Frozen source-data verifier
│   ├── metrics.py               # Macro-F1, threshold, term_id gruplu CV
│   ├── negative_sampling.py     # Random & BM25 hard negative üretimi
│   ├── bm25_hard_negative.py    # BM25 hard negative örnekleyici
│   ├── attributes.py            # Renk/beden/materyal attribute parse
│   ├── tfidf_features.py        # TF-IDF cosine similarity feature
│   ├── item_text.py             # Embedding için item metin standardizasyonu
│   ├── embedding_batch.py       # Toplu embedding üretimi (chunk + checkpoint)
│   ├── embedding_cosine.py      # Query-item cosine similarity feature + cache
│   ├── embedding_poc.py         # Embedding PoC (küçük batch test)
│   ├── train_mix_v2.py          # Random+BM25 karışık eğitim seti (v2)
│   ├── submission.py            # Submission üretim araçları
│   ├── validate_submission.py   # Submission format doğrulayıcı
│   └── error_analysis.py        # FP/FN hata analizi
├── scripts/                     # Çalıştırılabilir scriptler
│   ├── training/                # Model eğitimi
│   │   ├── run_model_shortlist.py   # Final LGBM/XGB grouped shortlist
│   │   └── run_train_full_v2.py     # Single-LGBM fallback
│   ├── analysis/                # Analiz & deney
│   │   ├── run_feature_importance.py    # Feature importance (5-Fold gain)
│   │   ├── run_threshold_analysis.py    # Verified OOF threshold diagnostics
│   │   ├── run_deney_matrisi_v2.py      # 4x4 oran × feature seti deneyi
│   │   ├── run_ensemble_comparison.py   # Cross-fitted shortlist comparison
│   │   ├── run_ensemble_optimization.py # Final model/weight/threshold selection
│   │   └── run_hata_taksonomisi.py      # Fold-external error taxonomy
│   ├── embedding/               # Embedding üretimi
│   │   ├── run_term_embeddings.py           # Term embedding üretim runner
│   │   └── run_embedding_score_comparison.py # Embedding cosine feature etkisi
│   ├── submission/              # Submission
│   │   ├── run_full_submission_v2.py    # Tam submission dosyası üretimi
│   │   └── run_submission_qa.py         # Submission QA kontrolü
│   └── data/                    # Veri hazırlama & kalite
│       ├── run_hard_neg_comparison.py   # Hard vs random negative karşılaştırma
│       ├── run_tfidf_experiments.py     # TF-IDF parametre deneyleri
│       └── verify_pipeline.py           # Pipeline doğrulama
├── docs/                        # Dokümantasyon & raporlar
│   ├── experiment_log.md            # Tarihsel deneyler + kabul edilen full run
│   ├── feature_importance_raporu.md # Feature önem analizi (10 Temmuz)
│   ├── threshold_analysis.md        # Current threshold diagnostics (generated)
│   ├── candidate_shift_analysis.md  # BM25 oranı dağılım analizi
│   ├── ensemble_selection.md        # Final shortlist decision (generated)
│   ├── feature_importance.md        # Current full-run importance (generated)
│   ├── error_taxonomy.md            # Current full-run taxonomy (generated)
│   ├── embedding_skor_kiyasi.md     # Embedding cosine etkisi (12 Temmuz)
│   ├── teknik_rapor_v1.md           # EDA & feature bulguları (10 Temmuz)
│   ├── rapor_yontem_v1.md           # Kabul edilen yöntem (15 Temmuz)
│   ├── offline_dependency.md        # Offline hazırlık rehberi (13 Temmuz)
│   ├── sprint1_raporu.md            # Sprint 1 özeti
│   ├── sprint2_raporu.md            # Sprint 2 özeti
│   └── yarışma/                     # Resmi yarışma belgeleri
├── outputs/                     # Model çıktıları & embedding dosyaları
│   ├── embeddings/              # term_embeddings.npy, item_embeddings.npy
│   └── *.csv                    # Deney sonuç tabloları
├── notebooks/                   # Jupyter Notebooklar
├── requirements.txt              # Doğrudan bağımlılık pinleri
├── requirements.lock             # Hash'li tam transitif ortam kilidi
└── README.md
```

---

## 🛠️ Nasıl Çalıştırılır?

> **Not:** Tüm komutları proje kök dizininden çalıştırın:
> ```bash
> cd G.G.A
> ```

### Temel Akış (Hızlı Başlangıç)

> Güncel model durumu ve eski deneylerin geçerlilik sınırı için
> [`docs/model_status.md`](docs/model_status.md) belgesini okuyun.

```bash
# 1. Test, environment, frozen-data and relationship verification
python scripts/run_production.py --stage verify

# 2. LGBM/XGB shortlist artefaktlarını tam veriyle eğit
python scripts/run_production.py --stage train

# 3. Cross-fitted model seçimiyle manifest doğrulamalı submission üret
python scripts/run_production.py --stage predict
```

Hızlı smoke eğitimleri doğrudan `run_model_shortlist.py --sample-terms ...`
ile çalıştırılır, ayrı sample artefakt dizinine yazılır ve üretim seçimi
tarafından kabul edilmez. Tam operasyon adımları için [`RUNBOOK.md`](RUNBOOK.md)
kanonik kaynaktır.

Varsayılan akış LightGBM, XGBoost ve weighted blend adaylarını karşılaştırır.
Tekil LightGBM fallback'i yalnızca `--pipeline lightgbm` ile açıkça seçilir.

### Analiz Scriptleri

```bash
# Hash doğrulamalı OOF threshold tanılaması
python scripts/analysis/run_threshold_analysis.py

# Negatif oran × feature kombinasyon deneyi (4×4 grid)
python scripts/analysis/run_deney_matrisi_v2.py

# LGBM vs XGBoost vs weighted blend
python scripts/analysis/run_ensemble_comparison.py

# Hatalı tahminlerin sınıflandırılması
python scripts/analysis/run_hata_taksonomisi.py
```

### Embedding Scriptleri

```bash
# Term (sorgu) embeddinglerini üret
python scripts/embedding/run_term_embeddings.py

# Item embeddinglerini üret (~963K ürün)
python src/embedding_batch.py --target items

# Embedding cosine feature etkisini ölç
python scripts/embedding/run_embedding_score_comparison.py
```

### Veri & Kalite

```bash
python scripts/data/verify_data_freeze.py
python scripts/data/verify_pipeline.py
```

---

## 🚀 Kurulum

### Gereksinimler

- **Python** 3.13.5 (see `.python-version`)
- **Git LFS** (büyük veri dosyaları için)
- GPU (opsiyonel, embedding hesaplama için önerilir)

### Adım 1: Repoyu Klonla

```bash
git lfs install
git clone https://github.com/G-G-A-Dataton/G.G.A.git
cd G.G.A
```

### Adım 2: Sanal Ortam Oluştur

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Adım 3: Bağımlılıkları Yükle

```bash
pip install --upgrade pip
pip install --require-hashes -r requirements.lock
```

### Adım 4: Kurulumu Doğrula

```bash
PYTHONNOUSERSITE=1 python scripts/verify_environment.py \
  --lock requirements.lock
```

### 📦 Temel Bağımlılıklar

| Paket | Versiyon | Açıklama |
|---|---|---|
| `pandas` | 3.0.3 | Veri işleme |
| `numpy` | 2.5.0 | Sayısal hesaplama |
| `scikit-learn` | 1.9.0 | ML araçları & metrikler |
| `lightgbm` | 4.6.0 | Gradient boosting modeli |
| `xgboost` | 3.3.0 | Gradient boosting modeli |
| `sentence-transformers` | 5.6.0 | Metin embedding'leri |
| `transformers` | 5.12.1 | NLP modelleri |
| `torch` | 2.12.1 | Deep learning backend |
| `rank_bm25` | 0.2.2 | BM25 metin araması |
| `matplotlib` | 3.11.0 | Görselleştirme |
| `seaborn` | 0.13.2 | İstatistiksel görselleştirme |

> Doğrudan bağımlılıklar [requirements.txt](requirements.txt), kurulabilir tam
> ortam ise [requirements.lock](requirements.lock) dosyasında tutulur. Offline
> wheelhouse ve temiz ortam doğrulaması için
> [offline dependency rehberini](docs/offline_dependency.md) kullanın.

---

## 📅 Yarışma Takvimi

| Tarih | Aşama |
|---|---|
| 26 Haziran – 17 Temmuz 2026 | 🟢 **Online Kaggle Yarışması** |
| 18 Temmuz – 1 Ağustos 2026 | Finalist adayı çözümlerin incelenmesi |
| 2 Ağustos 2026 | Finalist takımların açıklanması (ilk 10) |
| 14 Ağustos 2026 | Finalistlerle çevrimiçi buluşma |
| 5 – 6 Eylül 2026 | 🏁 Fiziksel Hackathon (İstanbul, Trendyol Maslak) |
| 30 Eylül – 4 Ekim 2026 | 🏆 Ödül Töreni (TEKNOFEST Şanlıurfa) |

---

## 🏆 Ödüller

| Derece | Ödül |
|---|---|
| 🥇 Birinci | ₺150.000 |
| 🥈 İkinci | ₺120.000 |
| 🥉 Üçüncü | ₺100.000 |

---

## 👥 Takım — G.G.A Dataton

| # | İsim | Rol |
|---|---|---|
| 1 | Ömer Faruk Kara | Kaptan |
| 2 | Ahmet Emin Işın | İletişim Sorumlusu |
| 3 | Mustafa Mert Çevik | Üye |
| 4 | Muhammed Köseoğlu | Üye |

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

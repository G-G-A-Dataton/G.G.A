# 🛒 G.G.A — E-Ticaret Ürün-Terim İlişkilendirme

**TEKNOFEST 2026 E-Ticaret Hackathon Yarışması** (Kaggle Datathon Aşaması) için geliştirilen; arama terimleri ile ürün özellikleri arasındaki anlamsal bağı çözmeyi amaçlayan **metin tabanlı ikili sınıflandırma (Binary Classification)** projesi.

> T3 Vakfı, Sanayi ve Teknoloji Bakanlığı ve **Trendyol** iş birliğiyle düzenlenmektedir.

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
| `items.csv` | ~966K | Ürün kataloğu (title, category, brand, gender, age_group, attributes) |
| `terms.csv` | ~50K | Arama terimleri (term_id, query) |
| `training_pairs.csv` | ~250K | Eğitim çiftleri — sadece label=1 |
| `submission_pairs.csv` | ~3.36M | Tahmin edilecek çiftler |
| `sample_submission.csv` | ~3.36M | Gönderim formatı örneği |

---

## 📁 Proje Yapısı

```
G.G.A/
├── datasets/                # Yarışma veri setleri (Git LFS)
│   ├── items.csv
│   ├── terms.csv
│   ├── training_pairs.csv
│   ├── submission_pairs.csv
│   └── sample_submission.csv
├── docs/                    # Dokümantasyon
│   ├── Rehber.md            # Sıfırdan başlayanlar için tam rehber
│   ├── EDA_notlari_v0.md    # İlk veri analizi ve veri sözlüğü
│   ├── EDA_on_raporu.md     # Kategori, marka dağılımları ve 50 pozitif çift analizi
│   ├── REPO_CALISMA_STANDARDI.md # Ekip çalışma ve Git yönergeleri
│   └── yarışma/             # Resmi yarışma belgeleri
├── notebooks/               # Jupyter Notebooklar (Deneyler)
│   ├── 01_veri_kalite_mert.py        # Veri kalite kontrolü (K1-K6)
│   ├── 02_negative_sampling_mert.py  # Random negative üretimi (1:1/3:1/5:1)
│   ├── 03_negatif_kalite_mert.py     # Negatif örneklerin bağımsız doğrulaması
│   └── 04_baseline_lgbm_omerfaruk.ipynb # Baseline LightGBM modeli eğitim süreci
├── src/                     # Üretim (Production) Kodları
│   ├── __init__.py
│   ├── data.py              # Bellek dostu veri yükleme ve merge pipeline'ı
│   ├── features.py          # Kelime örtüşmesi ve cinsiyet uyumu feature'ları
│   ├── metrics.py           # Macro-F1 ve 5-Fold Stratified K-Fold validation şeması
│   ├── negative_sampling.py # Sızıntı korumalı rastgele negatif örnek üretici
│   ├── tfidf_features.py    # TF-IDF cosine similarity feature üretimi
│   └── validate_submission.py # Kaggle submission format doğrulaması
├── verify_pipeline.py       # Veri yükleme doğrulama scripti
├── run_baseline.py          # LightGBM baseline model çalıştırma scripti
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Nasıl Çalıştırılır?

Projenin veri yükleme hattını veya baseline model eğitimini yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### 1. Veri Yükleme Hattını Test Etme
Bellek optimizasyonlu veri yükleme ve birleştirme (merge) aşamasını test etmek için:
```bash
python verify_pipeline.py
```

### 2. LightGBM Baseline Modelini Çalıştırma
5-Fold Çapraz Doğrulama (Cross-Validation) ile 20.000 satırlık örnek veri üzerinde LightGBM baseline eğitimini başlatmak, threshold optimizasyonu yapmak ve feature importance değerlerini görmek için:
```bash
python run_baseline.py
```

### 3. Notebook Üzerinden Adım Adım İnceleme
Görsel sonuçları ve eğitim detaylarını Jupyter Notebook üzerinden izlemek için:
```bash
jupyter notebook
# Tarayıcıda notebooks/04_baseline_lgbm_omerfaruk.ipynb dosyasını açıp çalıştırın.
```


---

## 🚀 Kurulum

### Gereksinimler

- **Python** 3.10+
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
pip install -r requirements.txt
```

### Adım 4: Kurulumu Doğrula

```bash
python -c "import pandas, numpy, torch, sentence_transformers, lightgbm; print('✅ Tüm paketler başarıyla yüklendi!')"
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

> Tüm bağımlılıkların tam listesi için [requirements.txt](requirements.txt) dosyasına bakınız.

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

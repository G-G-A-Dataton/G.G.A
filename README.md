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
│   └── yarışma/             # Resmi yarışma belgeleri
│       ├── şartname.md
│       ├── genel-kurallar.md
│       ├── etik-kuralar.md
│       ├── Yarışma Hakkında.md
│       ├── Yarışma Takvimi.md
│       └── takım.md
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Kurulum

```bash
# Repoyu klonla (Git LFS gerekli)
git lfs install
git clone https://github.com/G-G-A-Dataton/G.G.A.git
cd G.G.A

# Python bağımlılıkları
pip install pandas numpy scikit-learn lightgbm xgboost
pip install sentence-transformers rank_bm25
pip install matplotlib seaborn
```

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

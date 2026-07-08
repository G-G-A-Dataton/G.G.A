# G.G.A Repo Çalışma Standartları ve Kuralları

Bu doküman, G.G.A takımının yarışma boyunca kod kalitesini, tekrarlanabilirliğini ve veri bütünlüğünü koruması için uyması gereken temel kuralları ve repo standartlarını belirler.

---

## 1. Klasör ve Dosya Yapısı

Repo içerisinde çalışmaların karışmaması için aşağıdaki klasör şeması takip edilecektir:

```text
G.G.A/
├── datasets/                # Veri setleri (items.csv, terms.csv, training_pairs.csv vb.)
├── docs/                    # Raporlar, kılavuzlar ve resmi belgeler
│   ├── yarışma/             # Yarışma kuralları, takvim ve planlar
│   ├── REPO_CALISMA_STANDARDI.md
│   └── EDA_notlari_v0.md
├── notebooks/               # Geliştirme, veri analizi ve deneme notebook'ları
│   └── ...                  # [Sıra No]_[Açıklama]_[Yazar].ipynb formatında
├── src/                     # Tekrar kullanılabilir Python scriptleri
│   ├── __init__.py
│   ├── data.py              # Veri okuma, merge ve veri tipi optimizasyonu
│   ├── features.py          # Özellik çıkarma (TF-IDF, overlap, vb.)
│   ├── train.py             # Model eğitimi
│   └── predict.py           # Tahmin ve submission üretimi
├── requirements.txt         # Kilitlenmiş bağımlılık paketi listesi
└── README.md                # Kurulum ve genel proje tanıtımı
```

---

## 2. Notebook Adlandırma Standardı

Tüm Jupyter Notebook dosyaları aşağıdaki formata uygun olarak adlandırılacaktır:
Format: `[SIRA_NO]_[GÖREV_ADI]_[ADINIZ].ipynb`

Örnekler:
*   `01_eda_ahmet.ipynb`
*   `02_baseline_lgbm_omer.ipynb`
*   `03_tfidf_features_muhammed.ipynb`

*Not: Notebook'lar paylaşıma sunulmadan önce tüm hücreleri temizlenmiş (kernel restarted and cleared) veya yukarıdan aşağıya sırayla hatasız çalıştırılmış şekilde commit edilecektir.*

---

## 3. Deney Kayıt Standardı

Yapılan her model veya özellik (feature) deneyi, takım içi karşılaştırma yapılabilmesi adına ortak bir formata kaydedilecektir. Deneyler, `docs/deneyler.md` veya ana deney tablosuna aşağıdaki kolonlar ile işlenmelidir:

| Kolon | Açıklama |
|---|---|
| **Deney ID** | `EXP-YYYY-MM-DD-SIRA` (Örn: `EXP-2026-07-03-01`) |
| **Geliştirici** | Deneyi koşan kişi (Ömer, Ahmet, Mustafa, Muhammed) |
| **Veri Versiyonu** | Kullanılan negatif örnekleme oranı ve yöntemi |
| **Model Tipi** | LightGBM, XGBoost, CatBoost vb. |
| **Kullanılan Özellikler** | Eklenen veya çıkarılan özellik gruplarının özeti |
| **Validation Şeması** | Örn: `5-Fold Stratified K-Fold` (Holdout, CV vb.) |
| **Local Macro-F1** | Modelin yerel doğrulama skoru |
| **Kaggle Public F1** | Kaggle Leaderboard üzerindeki skor (eğer submit edildiyse) |
| **Karar & Notlar** | Sonraki adımlarda bu modelin/özelliklerin kullanılıp kullanılmayacağı |

---

## 4. Submission QA (Kalite Kontrol) Kuralları

Herhangi bir tahmin dosyası Kaggle'a yüklenmeden önce aşağıdaki kontrollere tabi tutulacaktır. Bu kontrolleri otomatik koşan `src/validate_submission.py` scripti kullanılabilir:

- [ ] **Kolon Uyumu:** Sadece `id` ve `prediction` kolonları bulunmalıdır.
- [ ] **Satır Sayısı:** Satır sayısı tam olarak `3.359.681` (header dahil `3.359.682`) olmalıdır.
- [ ] **ID Sırası:** `sample_submission.csv` ile tahmin dosyasındaki `id` değerlerinin sırası birebir eşleşmelidir.
- [ ] **Tahmin Değerleri:** `prediction` kolonu sadece `0` veya `1` (binary) değerlerini içermelidir. `NaN` veya float değer olmamalıdır.
- [ ] **Index Kontrolü:** CSV dosyası kaydedilirken Pandas'ın `index=False` parametresi kesinlikle kullanılmalıdır.

---

## 5. Tekrarlanabilirlik (Reproducibility) Kuralları

1.  **Sabit Seed:** Rastgelelik içeren tüm süreçlerde (veri bölme, negatif örnekleme, model eğitimi vb.) rastgelelik tohumu (seed) olarak `42` kullanılacaktır.
2.  **Modüler Yapı:** Notebook'larda doğrulanan kodlar hızlıca `src/` altına taşınacak ve parametrik fonksiyonlar haline getirilecektir.
3.  **Çevre Tutarlılığı:** Her yeni kütüphane ekleme ihtiyacında `requirements.txt` güncellenecektir.

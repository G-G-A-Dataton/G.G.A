# Trendyol E-Ticaret Datathon 2026 - G.G.A Takım Görev Planı

Bu doküman G.G.A takımının Kaggle datathon aşamasını, finalist adaylığı durumunda teslim edilecek çözüm paketini ve hackathon hazırlık sürecini profesyonel bir çalışma planına bağlar.

> [!NOTE]
> **15 Temmuz gerçekleşme durumu:** Gün 1-15 arasındaki 60 üye-görev hücresinin
> 58'i repository kanıtı ile tamamlanmış veya teknik kararla kapatılmıştır
> (`%96,7`). Kalan iki hücre Kaggle takım erişimi ve submission/public score
> kaydıdır; yetkili takım hesabı gerektirir. Günlük kanıt ve kabul edilen full-run
> sonucu için [`july_15_task_audit.md`](july_15_task_audit.md) kullanılır.
> Bu planın ileriki günleri ve tarihsel görev metinleri değiştirilmeden korunur.
>
> **16 Temmuz gerçekleşme durumu:** Dört ekip üyesinin görevi `4/4` tamamlandı.
> Gün 1-16 ekip toplamı `62/64` (`%96,9`); kalan iki hücre önceki günlerden
> taşınan yetkili Kaggle hesap işlemleridir. Kanıt:
> [`july_16_task_audit.md`](july_16_task_audit.md).
>
> **17 Temmuz gerçekleşme durumu:** Dört üyenin repository çıktıları `4/4`
> tamamlandı. Yerelde yapılabilen hücreler Gün 1-17 için `65/65` (`%100`),
> tüm plan ise `65/68` (`%95,6`). Kalan üç hücre tek yetkili Kaggle akışında
> kapanır: takım erişimini doğrulama, iki CSV'yi yükleyip skorlarını kaydetme ve
> ikisini manuel final submission olarak seçme. Kanıt ve kesin dosya listesi:
> [`july_17_task_audit.md`](july_17_task_audit.md) ve
> [`final_kaggle_selection.md`](final_kaggle_selection.md).

Plan başlangıç tarihi **1 Temmuz 2026** olarak alınmıştır. Resmi Kaggle bitiş tarihi **17 Temmuz 2026** olduğu için ilk bölüm yoğun bir 17 günlük sprinttir. Devamındaki bölüm ise **18 Temmuz - 6 Eylül 2026** arası çözüm doğrulama, rapor ve hackathon hazırlığını kapsar.

---

## 1. Yarışma Hedefi

Amaç, verilen `term_id + item_id` çifti için ürünün arama terimiyle alakalı olup olmadığını tahmin eden, tekrarlanabilir ve açıklanabilir bir çözüm üretmektir.

| Başlık | Hedef |
|---|---|
| Problem | İkili sınıflandırma: `1 = alakalı`, `0 = alakasız` |
| Resmi metrik | Macro-F1 |
| Eğitim verisi | Sadece pozitif çiftler (`label=1`) |
| Kritik zorluk | Tutarlı negatif örnek üretimi |
| Teslim formatı | `id,prediction` kolonlu CSV |
| Günlük Kaggle limiti | Takım başına 5 submission |
| Finalist aday kontrolü | Kod, veri işleme, eğitim ve tahmin sürecinin tekrarlanabilir olması |
| Offline şartı | Tahmin aşaması internetsiz makinede çalışabilmeli |

> Önemli: Ücretli/online LLM veya dış servisler tahmin aşamasında kullanılmayacak. Geliştirme, fikir üretme, rapor ve sentetik veri üretiminde yardımcı araç kullanımı serbesttir; ancak finalist aday tesliminde süreç açıklanabilir ve yeniden üretilebilir olmalıdır.

---

## 2. Takım Rolleri ve Sorumluluklar

| Takım Üyesi | Ana Rol | Birincil Sorumluluk | Teslim Edeceği Çıktılar |
|---|---|---|---|
| **Ömer Faruk Kara** | Kaptan, Modelleme ve Submission Lideri | Kaggle takım yönetimi, model stratejisi, CV düzeni, final submission kararı | Baseline model, deney skor tablosu, ensemble kararı, submission onayı |
| **Ahmet Emin Işın** | İletişim Sorumlusu, EDA ve Rapor Lideri | Kaggle/TEKNOFEST iletişimi, veri keşfi, raporlaştırma, feature analizleri | EDA raporu, günlük durum notları, final çözüm raporu taslağı |
| **Mustafa Mert Çevik** | Veri Pipeline ve Negatif Örnekleme Lideri | Merge pipeline, negatif örnekleme, attributes parsing, veri kalite kontrolleri | `train_v*.csv/parquet`, negatif sampling kodları, veri doğrulama raporu |
| **Muhammed Köseoğlu** | Semantik Özellikler ve Reproducibility Lideri | TF-IDF, embedding, repo düzeni, offline çalışma paketi, otomasyon | TF-IDF/embedding feature'ları, requirements, runbook, tekrar üretim paketi |

### Yedek Sorumluluklar

| Kritik Alan | Ana Sorumlu | Yedek |
|---|---|---|
| Kaggle submission | Ömer Faruk Kara | Muhammed Köseoğlu |
| Kaggle/forum ve organizasyon iletişimi | Ahmet Emin Işın | Ömer Faruk Kara |
| Negatif örnek üretimi | Mustafa Mert Çevik | Ahmet Emin Işın |
| Embedding ve büyük feature üretimi | Muhammed Köseoğlu | Mustafa Mert Çevik |
| Final rapor ve çözüm anlatımı | Ahmet Emin Işın | Muhammed Köseoğlu |
| Offline tekrar üretim kontrolü | Muhammed Köseoğlu | Ömer Faruk Kara |

---

## 3. Çalışma İlkeleri

1. Her gün en az bir somut çıktı üretilecek: notebook, feature dosyası, skor kaydı, rapor notu veya doğrulama çıktısı.
2. Kaggle public leaderboard tek karar kaynağı olmayacak; takım içi validasyon skoru ana karar sinyali kabul edilecek.
3. Tüm deneyler `experiment_log.md` veya eşdeğer bir tabloya yazılacak.
4. Submission yapılmadan önce format, satır sayısı, ID sırası ve `0/1` değer kontrolü zorunlu olacak.
5. Veri seti takım dışına dağıtılmayacak. Sorular ve teknik tartışmalar Kaggle discussion gibi herkese açık kanallardan yürütülecek.
6. Finalist adaylığı hedeflendiği için kod ve çıktıların en baştan tekrarlanabilir olması sağlanacak.

---

## 4. Hedef Çıktılar

| Tarih | Çıktı | Sahibi | Kabul Kriteri |
|---|---|---|---|
| 2 Temmuz | EDA ön raporu | Ahmet Emin Işın | Kolonlar, eksikler, kategori dağılımları, örnek pozitif çiftler incelenmiş |
| 3 Temmuz | Random negative v1 | Mustafa Mert Çevik | En az 1:1 ve 3:1 negatif oranı üretilebiliyor |
| 4 Temmuz | Baseline LightGBM | Ömer Faruk Kara | Local validation Macro-F1 hesaplanıyor |
| 5 Temmuz | İlk Kaggle submission | Ömer Faruk Kara | Format doğrulandı, skor kaydedildi |
| 7 Temmuz | BM25 hard negative v1 | Mustafa Mert Çevik | Query'ye benzer ama pozitif olmayan ürünler üretildi |
| 8 Temmuz | TF-IDF feature seti | Muhammed Köseoğlu | `query-title`, `query-category` benzerliği modele eklenebilir |
| 10 Temmuz | Feature importance analizi | Ahmet Emin Işın | Zayıf/güçlü feature listesi çıkarıldı |
| 12 Temmuz | Embedding cosine v1 | Muhammed Köseoğlu | Kaydedilmiş embedding ve cosine feature üretildi |
| 14 Temmuz | Model karşılaştırma tablosu | Ömer Faruk Kara | LGBM/XGBoost/ensemble adayları karşılaştırıldı |
| 15 Temmuz | Threshold optimizasyonu | Ahmet Emin Işın + Ömer Faruk Kara | Validation üzerinde en iyi threshold belirlendi |
| 16 Temmuz | Final aday submission seti | Tüm ekip | En iyi 2 submission adayı seçildi |
| 17 Temmuz | Kaggle final günü kontrolü | Ömer Faruk Kara | Seçili submission'lar ve deney notları donduruldu |
| 18 Temmuz | Reproducibility paketi başlangıcı | Muhammed Köseoğlu | Baştan sona çalıştırma komutları yazıldı |
| 1 Ağustos | Finalist aday teslim paketi | Tüm ekip | Kod, model, requirements, rapor ve üretim adımları hazır |

---

## 5. Detaylı 17 Günlük Kaggle Sprint Planı

### Gün 1 - 1 Temmuz 2026, Çarşamba

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Kaggle takımının başvuru takım adıyla birebir uyumlu olduğunu kontrol eder. Submission limitlerini ve liderlik paneli kurallarını ekipte netleştirir. | Takım erişim kontrol listesi |
| Ahmet Emin Işın | `items.csv`, `terms.csv`, `training_pairs.csv`, `submission_pairs.csv` yapısını inceler. Veri sözlüğü taslağı çıkarır. | EDA notları v0 |
| Mustafa Mert Çevik | Merge akışını tasarlar: `training_pairs -> terms -> items`. Büyük dosyalar için bellek dostu okuma stratejisini belirler. | Merge planı ve ilk test notebook'u |
| Muhammed Köseoğlu | Repo çalışma düzenini oluşturur: notebook adlandırma, çıktı klasörü, deney log formatı, requirements kontrolü. | Repo çalışma standardı taslağı |

### Gün 2 - 2 Temmuz 2026, Perşembe

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Macro-F1 hesaplama fonksiyonunu ve stratified validation şemasını hazırlar. | `metrics` yardımcı fonksiyonu |
| Ahmet Emin Işın | Kategori, marka, cinsiyet, yaş grubu, attribute alanları için dağılım analizi yapar. 30-50 pozitif çifti manuel inceler. | EDA ön raporu |
| Mustafa Mert Çevik | Pozitif çiftlerde tekrar, eksik ID, merge kaybı kontrolü yapar. | Veri kalite kontrol çıktısı |
| Muhammed Köseoğlu | Submission format assert'lerini yazar: kolon, satır sayısı, ID sırası, binary prediction. | Submission doğrulama scripti |

### Gün 3 - 3 Temmuz 2026, Cuma

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | İlk LightGBM baseline iskeletini kurar. Modelin az feature ile çalıştığını doğrular. | Baseline notebook v0 |
| Ahmet Emin Işın | `query-title`, `query-category`, `query-brand` token overlap feature'larını tanımlar. | Feature fonksiyonları v1 |
| Mustafa Mert Çevik | Random negative mining üretir. 1:1, 3:1, 5:1 oranları için küçük örnek dataset çıkarır. | `negative_random_v1` |
| Muhammed Köseoğlu | TF-IDF yaklaşımını hazırlar, küçük örnek üzerinde cosine similarity hesaplar. | TF-IDF PoC |

### Gün 4 - 4 Temmuz 2026, Cumartesi

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Random negative v1 ile LightGBM eğitir, 5-fold veya holdout validation skorlarını kaydeder. | İlk validasyon skoru |
| Ahmet Emin Işın | Cinsiyet ve yaş grubu uyumu/çelişkisi feature'larını geliştirir. | Demografik feature seti |
| Mustafa Mert Çevik | Negatif örneklerin pozitif çiftlerle çakışmadığını garanti eden kontrol ekler. | Negatif kalite kontrolü |
| Muhammed Köseoğlu | TF-IDF feature'ını baseline veri setine bağlar. | Modele eklenebilir TF-IDF kolonu |

### Gün 5 - 5 Temmuz 2026, Pazar

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | İlk güvenli submission'ı üretir ve yükler. Skoru deney tablosuna işler. | Kaggle submission v1 |
| Ahmet Emin Işın | İlk 5 günün bulgularını kısa rapora dönüştürür. Hangi feature'ların mantıklı olduğunu listeler. | Sprint 1 raporu |
| Mustafa Mert Çevik | Random negative veri setini tekrar üretilebilir hale getirir, seed ve parametreleri yazar. | Tekrar üretilebilir random negative |
| Muhammed Köseoğlu | Submission doğrulama scriptini final akışına bağlar. | Submission QA checklist |

### Gün 6 - 6 Temmuz 2026, Pazartesi

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Baseline hata analizi yapar: false positive/false negative örüntüleri incelenir. | Hata analizi notları |
| Ahmet Emin Işın | Kategori seviyelerini ayrıştırır: `cat_l1`, `cat_l2`, `cat_l3`. | Kategori feature'ları |
| Mustafa Mert Çevik | BM25 index kurulumu yapar, item text alanını standardize eder. | BM25 index v1 |
| Muhammed Köseoğlu | TF-IDF için n-gram, max_features ve min_df denemelerine başlar. | TF-IDF deney tablosu |

### Gün 7 - 7 Temmuz 2026, Salı

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | BM25 hard negative ile küçük model denemesi yapar. Random negative'e göre etkisini ölçer. | Hard negative skor kıyası |
| Ahmet Emin Işın | Marka, renk, materyal gibi attribute sinyallerinin EDA'sını yapar. | Attribute sinyal notları |
| Mustafa Mert Çevik | Her query için top-N BM25 adayından pozitif olmayan hard negative üretir. | `negative_bm25_v1` |
| Muhammed Köseoğlu | TF-IDF feature'larını büyük submission akışında verimli üretmek için batch stratejisi tasarlar. | Batch TF-IDF planı |

### Gün 8 - 8 Temmuz 2026, Çarşamba

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | LGBM parametrelerini temel düzeyde ayarlar: `num_leaves`, `learning_rate`, `min_child_samples`. | Tuning v1 |
| Ahmet Emin Işın | Attribute parser çıktılarının anlamlılığını manuel örneklerle doğrular. | Attribute doğrulama notları |
| Mustafa Mert Çevik | `attributes` alanını parse eder; renk, materyal, beden, desen gibi kolonları çıkarır. | Parsed item attributes |
| Muhammed Köseoğlu | Sentence-transformers kurulumunu test eder, küçük batch embedding üretir. | Embedding PoC |

### Gün 9 - 9 Temmuz 2026, Perşembe

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | XGBoost veya alternatif gradient boosting modelini baseline ile karşılaştırır. | Model kıyas tablosu v1 |
| Ahmet Emin Işın | Renk ve materyal eşleşme/çelişki feature'larını tanımlar. | Renk/materyal feature'ları |
| Mustafa Mert Çevik | Random + hard negative karışımı için dataset v2 üretir. | `train_mix_v2` |
| Muhammed Köseoğlu | Item text alanını standartlaştırır: `title + category + brand + selected_attributes`. | Embedding item text |

### Gün 10 - 10 Temmuz 2026, Cuma

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Feature importance analizi yapar, gereksiz veya sızıntı riski taşıyan feature'ları işaretler. | Feature importance raporu |
| Ahmet Emin Işın | EDA ve feature bulgularını rapor formatına taşır. | Teknik rapor bölümü v1 |
| Mustafa Mert Çevik | Negatif oranı deneyleri için 1:1, 2:1, 3:1, 5:1 eğitim setlerini üretir. | Oran deney veri setleri |
| Muhammed Köseoğlu | Item embedding üretimini GPU üzerinde batch'ler halinde başlatır. | `item_embeddings` üretim logu |

### Gün 11 - 11 Temmuz 2026, Cumartesi

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Negatif oranı ve feature kombinasyonlarını kontrollü deneylerle karşılaştırır. | Deney matrisi v2 |
| Ahmet Emin Işın | Threshold adayları için validation tahminlerini analiz eder. | Threshold analiz hazırlığı |
| Mustafa Mert Çevik | Veri pipeline'ını parametreli hale getirir: seed, oran, hard-negative sayısı, çıktı yolu. | Parametreli data pipeline |
| Muhammed Köseoğlu | Term embedding üretimini tamamlar ve saklar. | `term_embeddings` |

### Gün 12 - 12 Temmuz 2026, Pazar

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Embedding cosine feature'ını modele ekler, skora etkisini ölçer. | Embedding skor kıyası |
| Ahmet Emin Işın | Hatalı tahmin örneklerini sınıflandırır: marka hatası, kategori hatası, renk hatası, semantik yakınlık hatası. | Hata taksonomisi |
| Mustafa Mert Çevik | Final aday eğitim seti için veri kalite raporu çıkarır. | Data QA raporu |
| Muhammed Köseoğlu | Query-item cosine similarity feature'ını üretir ve cache stratejisini yazar. | `embedding_cosine` feature |

### Gün 13 - 13 Temmuz 2026, Pazartesi

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | LGBM, XGBoost ve olası küçük ensemble adaylarını karşılaştırır. | Ensemble aday tablosu |
| Ahmet Emin Işın | Raporun yöntem bölümünü yazar: negatif sampling, feature engineering, validation. | Rapor yöntem v1 |
| Mustafa Mert Çevik | Submission seti için tüm feature üretim akışını uçtan uca dener. | Submission feature dry-run |
| Muhammed Köseoğlu | Offline çalışabilirlik için dosya bağımlılıklarını listeler. | Offline dependency listesi |

### Gün 14 - 14 Temmuz 2026, Salı

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | En iyi 2-3 model adayını belirler, validation tahminlerini dışa aktarır. | Model shortlist |
| Ahmet Emin Işın | Threshold taraması yapar; Macro-F1'i maksimize eden aralığı belirler. | Threshold v1 |
| Mustafa Mert Çevik | Büyük dosyalarda memory/time risklerini ölçer, gerekirse chunk yaklaşımı ekler. | Performans notları |
| Muhammed Köseoğlu | Tek komutla feature üretim ve prediction akışının taslağını yazar. | Runbook v1 |

### Gün 15 - 15 Temmuz 2026, Çarşamba

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Ensemble ağırlıklarını ve threshold'u birlikte optimize eder. | Final aday model v1 |
| Ahmet Emin Işın | Public leaderboard ile local validation farkını yorumlar. Overfit risklerini yazar. | Skor yorum raporu |
| Mustafa Mert Çevik | Final eğitim setini dondurur, hash/versiyon bilgisini kaydeder. | `train_final` versiyon notu |
| Muhammed Köseoğlu | Reproducibility için requirements ve çalışma ortamı notlarını düzenler. | Environment dokümanı |

### Gün 16 - 16 Temmuz 2026, Perşembe

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Final aday submission'ları üretir. En iyi iki adayın stratejisini belirler. | Submission adayları |
| Ahmet Emin Işın | Final rapor taslağını günceller; deney tablosunu temizler. | Rapor taslak v2 |
| Mustafa Mert Çevik | Submission dosyalarında kolon, satır, ID ve binary değer kontrollerini çalıştırır. | Submission QA onayı |
| Muhammed Köseoğlu | Tüm kodların yeniden çalıştırılabilir olduğunu temiz ortamda kontrol eder. | Dry-run sonucu |

### Gün 17 - 17 Temmuz 2026, Cuma

| Üye | İşler | Gün Sonu Çıktısı |
|---|---|---|
| Ömer Faruk Kara | Kaggle'da seçili final submission'ları kontrol eder ve son kararı verir. | Final Kaggle seçim notu |
| Ahmet Emin Işın | Güncel skor, deney ve karar özetini takım arşivine ekler. | Final sprint raporu |
| Mustafa Mert Çevik | Kullanılan veri seti ve negatif sampling parametrelerini kilitler. | Data freeze notu |
| Muhammed Köseoğlu | Kod, model dosyası, feature cache ve runbook bütünlüğünü doğrular. | Reproducibility freeze |

---

## 6. Kaggle Sonrası Genişletilmiş Yol Haritası

Bu bölüm, ilk 20 finalist adayı içine girilmesi veya çözüm talep edilmesi ihtimaline göre hazırlanmıştır.

### Faz 4 - Çözüm Teslimi ve Tekrar Üretilebilirlik, 18 Temmuz - 1 Ağustos 2026

| Tarih Aralığı | Odak | Sorumlular | Detay |
|---|---|---|---|
| 18-19 Temmuz | Kod temizliği | Muhammed Köseoğlu + Ömer Faruk Kara | Notebook'lar sıralanır, gereksiz hücreler temizlenir, tek çalıştırma sırası netleşir. |
| 20-21 Temmuz | Veri üretim dokümantasyonu | Mustafa Mert Çevik | Negatif sampling, sentetik veri, feature cache ve seed bilgileri açıklanır. |
| 22-23 Temmuz | Model tekrar eğitimi testi | Ömer Faruk Kara | Eğitim ve tahmin süreci temiz ortamda yeniden çalıştırılır. |
| 24-25 Temmuz | Offline tahmin denemesi | Muhammed Köseoğlu | İnternet kapalı varsayımıyla model dosyaları, tokenizer/model cache ve requirements kontrol edilir. |
| 26-27 Temmuz | Rapor taslağı | Ahmet Emin Işın | Problem, veri, yöntem, validasyon, sonuç, risk ve gelecek iş bölümleri yazılır. |
| 28-29 Temmuz | Skor tekrarı kontrolü | Ömer Faruk Kara + Mustafa Mert Çevik | Üretilen submission'ın Kaggle'daki final aday submission ile aynı formatta olduğu doğrulanır. |
| 30 Temmuz - 1 Ağustos | Teslim paketi | Tüm ekip | Kod, rapor, requirements, model ağırlıkları, çalıştırma talimatı ve karar logu paketlenir. |

### Faz 5 - Finalist Açıklaması ve Rapor Taslağı, 2 - 9 Ağustos 2026

| Tarih | İş | Sorumlu |
|---|---|---|
| 2 Ağustos | Finalist açıklamasını takip et, takım durumunu netleştir. | Ahmet Emin Işın + Ömer Faruk Kara |
| 3 Ağustos | Final rapor şablonunu resmi isterlere göre güncelle. | Ahmet Emin Işın |
| 4 Ağustos | Teknik yöntem bölümünü son modelle uyumlu hale getir. | Ömer Faruk Kara |
| 5 Ağustos | Veri ve negatif sampling bölümünü doğrula. | Mustafa Mert Çevik |
| 6 Ağustos | Offline çalışma ve kurulum bölümünü tamamla. | Muhammed Köseoğlu |
| 7 Ağustos | Grafikler, tablolar, feature importance ve hata analizi ekle. | Ahmet Emin Işın + Ömer Faruk Kara |
| 8 Ağustos | Takım içi rapor okuması, eksik/abartılı iddia kontrolü yap. | Tüm ekip |
| 9 Ağustos | Rapor taslağını teslim edilebilir hale getir. | Ahmet Emin Işın |

### Faz 6 - Hackathon Hazırlığı, 14 Ağustos - 4 Eylül 2026

| Tarih Aralığı | Hedef | Sorumlular | Detay |
|---|---|---|---|
| 14 Ağustos | Finalist toplantısı | Tüm ekip | Yeni problem kapsamı, çok sınıflı yapı, hız ve açıklanabilirlik isterleri not edilir. |
| 15-18 Ağustos | Yeni veri ve etiket yapısı analizi | Ahmet Emin Işın + Mustafa Mert Çevik | İlişkili, zayıf ilişkili, ilişkili değil sınıfları için veri okuması yapılır. |
| 19-22 Ağustos | Çok sınıflı model adaptasyonu | Ömer Faruk Kara | Binary model çok sınıflı yapıya taşınır, validation kurgusu güncellenir. |
| 23-25 Ağustos | Açıklanabilirlik arayüzü tasarımı | Muhammed Köseoğlu + Ahmet Emin Işın | Metin/görsel açıklama formatı, örnek karar gerekçeleri ve demo akışı hazırlanır. |
| 26-28 Ağustos | Servisleştirme hazırlığı | Muhammed Köseoğlu + Ömer Faruk Kara | API, inference süresi, batch/single request davranışı ve model yükleme süresi ölçülür. |
| 29 Ağustos - 4 Eylül | Ön inceleme ve son prova | Tüm ekip | Servis, rapor, demo, sunum ve hız testleri tekrar edilir. |

### Faz 7 - Fiziksel Hackathon ve Sunum, 5 - 6 Eylül 2026

| Tarih | İş | Sorumlu |
|---|---|---|
| 5 Eylül sabah | Ortam kurulumu, veri ve model kontrolü | Muhammed Köseoğlu |
| 5 Eylül gün içi | Final model iyileştirmeleri ve servis stabilizasyonu | Ömer Faruk Kara + Mustafa Mert Çevik |
| 5 Eylül akşam | Demo akışı ve açıklanabilirlik örnekleri | Ahmet Emin Işın + Muhammed Köseoğlu |
| 6 Eylül sabah | Sunum provası, süre kontrolü, yedek plan | Tüm ekip |
| 6 Eylül 14:00 sonrası | 10 dk sunum, 5 dk demo, 5 dk soru-cevap | Ömer Faruk Kara liderliğinde tüm ekip |

---

## 7. Deney Yönetimi

Her deney aşağıdaki formatta kaydedilecek.

| Alan | Örnek |
|---|---|
| Deney ID | `EXP-2026-07-08-01` |
| Veri versiyonu | `random3_bm25top50_v2` |
| Feature set | `overlap + tfidf + category + gender` |
| Model | `LightGBM` |
| Validation şeması | `5-fold StratifiedGroupKFold`, group=`term_id` |
| Threshold | `0.37` |
| Local Macro-F1 | `0.7421` |
| Public Kaggle skoru | `0.xx` |
| Karar | Devam / iptal / final adayı |
| Not | Hangi hata tiplerini iyileştirdi? |

---

## 8. Submission Kontrol Listesi

Submission yüklenmeden önce aşağıdaki kontroller yapılacak.

- [ ] Kolonlar tam olarak `id,prediction`.
- [ ] Satır sayısı `sample_submission.csv` ile aynı.
- [ ] ID sırası `sample_submission.csv` ile aynı.
- [ ] `prediction` değerleri sadece `0` veya `1`.
- [ ] Dosya UTF-8 CSV formatında ve index kolonu yok.
- [ ] Hangi model/threshold ile üretildiği deney tablosuna yazıldı.
- [ ] Takımın günlük 5 submission limitinde kaç hakkı kaldığı kontrol edildi.
- [ ] En iyi iki submission adayının stratejisi not edildi.

---

## 9. Kalite Kapıları

| Kapı | Tarih | Geçme Kriteri |
|---|---|---|
| QG-1 Baseline | 5 Temmuz | İlk submission yapılmış, local skor hesaplanmış |
| QG-2 Veri Kalitesi | 8 Temmuz | Random + hard negative üretimi tekrar edilebilir |
| QG-3 Feature Set | 12 Temmuz | TF-IDF veya embedding feature'ı modele eklenmiş |
| QG-4 Model Seçimi | 14 Temmuz | En iyi model ailesi ve negatif oranı belirlenmiş |
| QG-5 Final Adayı | 16 Temmuz | En iyi 2 submission adayı seçilmiş |
| QG-6 Reproducibility | 1 Ağustos | Kod, veri üretimi ve tahmin akışı yeniden çalıştırılabilir |

---

## 10. Riskler ve Önlemler

| Risk | Etki | Önlem | Sorumlu |
|---|---|---|---|
| Negatif örnekler çok kolay kalır | Model gerçek testte zayıf kalır | BM25 hard negative ve kategori içi negatif üret | Mustafa Mert Çevik |
| Public leaderboard'a aşırı uyum | Private skor düşer | Local validation ana karar sinyali olsun | Ömer Faruk Kara |
| Embedding üretimi çok uzun sürer | Son günlere feature yetişmez | Küçük modelle başla, cache kullan, batch üret | Muhammed Köseoğlu |
| Submission format hatası | Geçersiz submission | Otomatik assert ve manuel kontrol | Mustafa Mert Çevik + Muhammed Köseoğlu |
| Rapor yapılan işle uyuşmaz | Finalist adaylığında güven kaybı | Deney logunu günlük tut, raporu son güne bırakma | Ahmet Emin Işın |
| Offline çalışmama | Diskalifiye riski | Model/tokenizer/cache ve requirements paketini erken hazırla | Muhammed Köseoğlu |
| Takım-Kaggle üye uyumsuzluğu | Diskalifiye riski | Başvuru takımıyla Kaggle takımını ilk gün kontrol et | Ömer Faruk Kara + Ahmet Emin Işın |

---

## 11. Günlük Toplantı Formatı

Her gün 15 dakika kısa toplantı yapılacak.

1. Dün ne tamamlandı?
2. Bugün hangi çıktı teslim edilecek?
3. Engel veya karar ihtiyacı var mı?
4. Submission yapılacak mı, yapılacaksa hangi deneyle?
5. Deney logu ve rapor notları güncel mi?

Haftada iki kez 45 dakikalık teknik derinleşme yapılacak:

- Pazartesi: Model, veri ve feature kararları
- Perşembe: Skor, risk ve final aday submission stratejisi

---

## 12. Dosya ve Çalışma Standardı

Önerilen çalışma düzeni:

```text
notebooks/
  01_eda_ahmet.ipynb
  02_negative_sampling_mustafa.ipynb
  03_tfidf_embedding_muhammed.ipynb
  04_modeling_omer.ipynb
  05_submission_omer_muhammed.ipynb

src/
  data.py
  features.py
  negative_sampling.py
  train.py
  predict.py
  validate_submission.py

artifacts/
  features/
  models/
  submissions/
  reports/
```

Dosya adlarında tarih veya deney ID kullanılacak:

```text
submission_EXP-2026-07-16-02_threshold037.csv
model_lgbm_EXP-2026-07-15-01.txt
train_random3_bm25top50_seed42.parquet
```

---

## 13. İç Hedefler

Bu skorlar resmi taahhüt değildir; takım içi ilerleme göstergesi olarak kullanılacaktır.

| Tarih | İç Hedef | Beklenen İçerik |
|---|---|---|
| 5 Temmuz | İlk ölçülebilir skor | Random negative + temel overlap feature |
| 8 Temmuz | Belirgin iyileşme | BM25 hard negative + TF-IDF |
| 12 Temmuz | Güçlü aday model | Attribute feature + embedding cosine |
| 15 Temmuz | Final aday seviyesi | Ensemble + threshold optimizasyonu |
| 17 Temmuz | Kontrollü final | En iyi 2 submission seçili, deneyler belgeli |

---

## 14. Final Teslim Paketi İçeriği

Finalist adaylığı durumunda aşağıdaki paket hazırlanacak.

- Çalıştırma talimatı: ortam kurulumu, veri konumu, komut sırası
- `requirements.txt` veya environment dosyası
- Veri üretim kodu: negatif sampling, feature engineering, cache üretimi
- Eğitim kodu veya notebook'u
- Tahmin kodu veya notebook'u
- Kullanılan model ağırlıkları ve tokenizer/model cache dosyaları
- Final submission üretim adımları
- Deney logu ve skor tablosu
- Teknik çözüm raporu
- Offline çalışma notları

---

## 15. Kapanış Prensibi

Bu planın ana hedefi yalnızca yüksek skor almak değil; skoru **savunulabilir, tekrar üretilebilir ve raporlanabilir** bir çözümle almak. Yarışma sonunda takımın elinde çalışan bir model, düzgün bir veri pipeline'ı, temiz bir rapor ve finalist adaylığı durumunda teslim edilebilir bir paket olmalıdır.

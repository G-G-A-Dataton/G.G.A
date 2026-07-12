# Skor Yorum Raporu: Lokal CV vs. Public Leaderboard (15 Temmuz)

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 15 Temmuz 2026  
**Sürüm:** v1.0.0  
**Kapsam:** Güvenilirlik Analizi ve Aşırı Öğrenme (Overfitting) Raporu

---

## 1. Lokal CV ve Public Leaderboard Farkı (Gap) Analizi

Makine öğrenmesi projelerinde karşılaşılan en yaygın risklerden biri, lokal çapraz doğrulama (CV) skorunun çok yüksek çıkmasına rağmen, Kaggle Public Leaderboard (LB) skorunun beklenenden düşük olmasıdır (skor açılması / gap).

Lokal model geliştirme sürecinde gözlemlediğimiz F1 değerleri ve olası Kaggle yansıması aşağıda analiz edilmiştir:

| Model | Lokal CV (Macro-F1) | Tahmini Public LB | Durum |
|---|---|---|---|
| LightGBM Tuned | ~0.9637 | ~0.9580 - 0.9610 | Beklenen Küçük Sapma |
| XGBoost Tuned | ~0.9624 | ~0.9560 - 0.9590 | Beklenen Küçük Sapma |
| Optimized Weighted Ensemble | **~0.9650** | **~0.9610 - 0.9630** | En Güvenilir Aday |

---

## 2. Skor Açılmasının (Gap) Olası Sebepleri

Lokal ve genel skorlar arasında açılma olmasının temel nedenleri şunlardır:

### 2.1 Veri Sızıntısı (Data Leakage)
- **Problem:** Feature üretimi veya preprocessing adımlarının CV split'i yapılmadan önce tüm veri üzerinde yapılması (örn. TF-IDF'in tüm eğitim setinde fit edilmesi).
- **Mevcut Durumumuz:** `build_mixed_training_set` ve TF-IDF pipeline'ımız sızıntıyı önleyecek şekilde tasarlanmıştır.

### 2.2 Yanlış Negatif Örnekleme Dengesi
- **Problem:** Lokal eğitim setinde kullandığımız negatif/pozitif oranı (örn. 2:1 veya 3:1) ile gerçek test setindeki (`submission_pairs.csv`) negatif/pozitif oranının uyuşmaması. Test setinde çok daha fazla alakasız çift (yüksek negatif oranı) varsa, modelimiz çok fazla False Positive (FP) üreterek LB skorunu düşürecektir.
- **Çözüm:** Eşik değeri (threshold) optimizasyonunu test setinin potansiyel oranına göre kalibre etmek.

### 2.3 Hedef Dağılım Kayması (Target Distribution Shift)
- **Problem:** Eğitim setinde yer alan arama sorgularının sıklığı ve ürün kategorilerinin dağılımı ile test setindeki (`submission_pairs.csv`) dağılımın farklı olması.
- **Mevcut Durumumuz:** Kategori bazlı tabakalandırma (Stratification) kullanarak bu riski minimumda tutuyoruz.

---

## 3. Aşırı Öğrenme (Overfitting) Riskleri ve Önlemler

Modelin eğitim verisini ezberlemesini önlemek ve genelleyebilirliğini (generalization) artırmak için aşağıdaki stratejiler uygulanmaktadır:

### 3.1 Regularization (Düzenlileştirme) Parametreleri
LightGBM ve XGBoost modellerinde ezberlemeyi zorlaştırmak için şu parametreler aktifleştirilmiştir:
- `reg_alpha` (L1 Düzenlileştirmesi): Gereksiz katsayıları sıfırlayarak seyreklik sağlar.
- `reg_lambda` (L2 Düzenlileştirmesi): Katsayıların aşırı büyümesini engeller.
- `min_child_samples` (LightGBM) / `min_child_weight` (XGBoost): Yaprak başına minimum örnek sayısını artırarak küçük gruplara özel kural yazılmasını önler.

### 3.2 Öznitelik Azaltma (Feature Reduction)
- Gain önem değeri sıfır veya sıfıra çok yakın çıkan öznitelikler (`query_color_match`, `query_size_match`, `query_material_match` gibi) modelden çıkartılarak gürültü azaltılmalıdır.
- Sadece `query_title_overlap`, `query_category_overlap` ve `tfidf_cosine` gibi yüksek ayırt ediciliğe sahip temel 10-12 feature seti kullanılmalıdır.

### 3.3 Erken Durdurma (Early Stopping)
- Modellerin gereksiz yere binlerce tur eğitilmesi engellenmiş, validation kaybı 50 tur boyunca iyileşmediğinde eğitim durdurulacak şekilde ayarlanmıştır (`early_stopping_rounds=50`).

---

## 4. Aksiyon Planı

1. **Öznitelik Seçimi:** Gain importance analizi sonucunda model kararlarına katkısı olmayan feature'ları pipeline'dan çıkaracağız.
2. **Threshold Kalibrasyonu:** Lokal F1 optimizasyonu ile bulduğumuz `0.35` eşiğini test setindeki oran tahminlerine göre ince ayara tabi tutacağız.
3. **Güvenli Ensemble:** Tekil modellerin tahmin hatalarını birbirini dengeleyecek şekilde ağırlıklandırarak en kararlı skoru garanti altına alacağız.

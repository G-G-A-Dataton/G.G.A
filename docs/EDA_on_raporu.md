# EDA Ön Raporu — v1

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 2 Temmuz 2026  
**Veri:** Trendyol E-Ticaret Datathon 2026 — Kaggle Aşaması

---

## 1. Kategori (L1) Dağılımı — Top 15

Ürün kataloğunun en üst seviye kategorileri (cat_l1) incelendiğinde **Ev & Mobilya** ve **Giyim** birlikte katalogdaki ürünlerin yaklaşık %44'ünü oluşturmaktadır.

| Sıra | Kategori | Ürün Sayısı |
|---|---|---|
| 1 | ev & mobilya | 251.121 |
| 2 | giyim | 169.064 |
| 3 | aksesuar | 98.686 |
| 4 | ayakkabı | 85.418 |
| 5 | elektronik | 80.181 |
| 6 | kozmetik & kişisel bakım | 51.633 |
| 7 | süpermarket | 47.551 |
| 8 | otomobil & motosiklet | 35.915 |
| 9 | anne & bebek & çocuk | 23.745 |
| 10 | banyo yapı & hırdavat | 23.595 |
| 11 | hobi & eğlence | 23.440 |
| 12 | kırtasiye & ofis malzemeleri | 22.296 |
| 13 | kitap | 18.503 |
| 14 | spor & outdoor | 16.196 |
| 15 | bahçe & elektrikli el aletleri | 15.529 |

> **Not:** Hiyerarşik kategori patikası `/` ile ayrılmıştır (Örn: `giyim/üst giyim/gömlek`). Modelde `cat_l1`, `cat_l2`, `cat_l3` olarak ayrıştırılmalıdır.

---

## 2. Marka (Brand) Dağılımı — Top 15

Katalogda **79.791** benzersiz marka bulunmaktadır. Türk ve uluslararası fast-fashion/spor markaları baskın konumdadır.

| Sıra | Marka | Ürün Sayısı |
|---|---|---|
| 1 | adidas | 7.982 |
| 2 | koton | 7.807 |
| 3 | defacto | 7.756 |
| 4 | nike | 6.762 |
| 5 | u.s. polo assn. | 5.443 |
| 6 | puma | 5.417 |
| 7 | pierre cardin | 4.557 |
| 8 | karaca | 4.050 |
| 9 | teona ahşap | 3.762 |
| 10 | lc waikiki | 3.597 |
| 11 | guess | 2.967 |
| 12 | lumberjack | 2.895 |
| 13 | paşabahçe | 2.681 |
| 14 | özdilek | 2.446 |
| 15 | skechers | 2.415 |

> **Modelleme Notu:** Sorgu metni çoğunlukla doğrudan marka adını içeriyor (Örn: `"adidas erkek spor ayakkabı"`). `query` ve `brand` arasındaki birebir eşleşme güçlü bir özellik olacaktır.

---

## 3. Cinsiyet ve Yaş Grubu Dağılımı

| Cinsiyet | Sayı | Oran |
|---|---|---|
| **unknown** | 590.714 | ~%61.3 |
| kadın | 192.045 | ~%19.9 |
| erkek | 99.433 | ~%10.3 |
| unisex | 80.681 | ~%8.4 |

| Yaş Grubu | Sayı | Oran |
|---|---|---|
| **unknown** | 572.028 | ~%59.4 |
| yetişkin | 280.876 | ~%29.2 |
| çocuk | 52.876 | ~%5.5 |
| genç | 31.246 | ~%3.2 |
| bebek | 18.426 | ~%1.9 |
| bebek & çocuk | 7.421 | ~%0.8 |

> **Kritik Bulgu:** Her iki alanda da `unknown` oranı ~%60. Bu sebeple cinsiyet/yaş eşleşmesi için **kural tabanlı (rule-based) hard filter uygulamak risklidir**. Bunun yerine sorgu metnindeki demografik sinyaller (Örn: `"kadın", "erkek", "bebek"`) ile ürün başlığındaki sinyallerin **soft overlap feature olarak** kullanılması önerilir.

---

## 4. Manuel İnceleme: 50 Pozitif Çift Gözlem Notları

50 rastgele pozitif (alakalı) çift incelenmiştir. Öne çıkan gözlemler:

### Güçlü Alaka Sinyalleri
- **Marka uyumu:** Sorguda marka adı geçiyor ve üründe birebir eşleşiyor.  
  Örn: `"kerastase şampuan"` → `brand: kerastase`, `"pasabahçe iconic bardak"` → `brand: paşabahçe`
- **Kategori uyumu:** Sorgu ve ürünün L1/L2 kategorisi örtüşüyor.  
  Örn: `"erkek spor ayakkabı"` → `category: ayakkabı/spor ayakkabı/...`
- **Ürün modeli uyumu:** Sorgu model kodu/numarası içeriyor.  
  Örn: `"new balance 408"` → `title: ml408wn-r 408 spor ayakkabı`

### Zayıf/Dolaylı Alaka (Dikkat Edilmesi Gereken Durumlar)
- **Geniş sorgular:** Sorgu çok genel, ürün spesifik.  
  Örn: `"yüz bakım"` → `title: leke ve kırık karşıtı bitkisel kompleks bakım...`
- **Kategori yanlı eşleşme:** Sorgu ile başlık kelime örtüşmüyor, yalnızca kategori yoluyla alaka var.  
  Örn: `"dresuar üstü aksesuar"` → `title: dekoratif çanta zincir detaylı... vazo`
- **Ürün başlığı çok kısa:** Bazı ürün başlıkları anlamlı metin barındırmıyor.  
  Örn: `"new balance erkek spor ayakkabı"` → `title: '530'` (sadece model kodu!)

### Cinsiyet Çakışması Gözlemi
- `"çocuk abiye elbise"` sorgusu için `gender: kadın` etiketli bir ürün döndü — bu etiket hatalı veya üst kategori tanımı. Cinsiyet alanı kural tabanlı filtre için güvenilir değil.

---

## 5. Modelleme İçin Önerilen Feature'lar (Ön Değerlendirme)

| Feature | Açıklama | Beklenen Etki |
|---|---|---|
| `query_title_overlap` | Sorgu ile ürün başlığı arasındaki kelime örtüşmesi | ⭐⭐⭐⭐⭐ |
| `query_brand_match` | Sorgu metninde marka adı geçiyor mu? | ⭐⭐⭐⭐⭐ |
| `query_cat_l1_match` | Sorgunun olası L1 kategorisi ile ürün L1 kategorisi uyuşuyor mu? | ⭐⭐⭐⭐ |
| `query_gender_signal` | Sorguda cinsiyet kelimesi var mı? (kadın/erkek/bebek) | ⭐⭐⭐ |
| `tfidf_cosine` | TF-IDF vektörleri arasında cosine benzerliği | ⭐⭐⭐⭐ |
| `embedding_cosine` | Sentence-transformer embedding cosine benzerliği | ⭐⭐⭐⭐⭐ |

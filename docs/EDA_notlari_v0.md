# G.G.A EDA Notları v0 — Veri Sözlüğü ve İnceleme Raporu

Bu doküman, Trendyol E-Ticaret Datathon 2026 veri setinin (Kaggle Aşaması) temel boyutları, kolon yapıları, eksik değer analizleri ve modelleme öncesi elde edilen kritik semantik bulguları içerir.

---

## 1. Genel Veri Seti İstatistikleri

| Dosya Adı | Satır Sayısı | Sütun Sayısı | Temel Amacı | Eksik Değerler / Nulls |
|---|---|---|---|---|
| `terms.csv` | 50.153 | 2 | Arama terimleri kataloğu | Yok |
| `items.csv` | 962.873 | 7 | Ürün kataloğu ve özellikleri | `brand` kolonunda 2 satır null |
| `training_pairs.csv` | 250.000 | 4 | Model eğitimi için eşleşmeler | Yok |
| `submission_pairs.csv` | 3.359.679 | 3 | Test eşleşmeleri (Tahmin edilecek) | Yok |
| `sample_submission.csv` | 3.359.679 | 2 | Gönderim şablonu formatı | Yok |

---

## 2. Veri Sözlüğü (Data Dictionary)

### 2.1. `terms.csv`
Arama motoruna yazılan kullanıcı sorgularını içerir.
*   `term_id` (String): Sorgunun tekil kimliği (Örn: `TERM_f2b61db2`).
*   `query` (String): Kullanıcı sorgu metni (Örn: `defacto kız bebek elbise`).

### 2.2. `training_pairs.csv`
Eğitim seti çiftleridir. Yalnızca **pozitif** (alakalı) sorgu-ürün ilişkilerini barındırır.
*   `id` (String): Eşleşme tekil kimliği (Örn: `TRN_c639ed31a5`).
*   `term_id` (String): Sorgu kimliği.
*   `item_id` (String): Ürün kimliği.
*   `label` (Integer): Alaka derecesi. Eğitim setinde tamamı `1`'dir.

### 2.3. `items.csv`
Ürün kataloğu. Ürünün başlığı, kategorisi ve diğer ayırt edici özniteliklerini içerir.
*   `item_id` (String): Ürünün tekil kimliği (Örn: `ITEM_3a515fb7125d`).
*   `title` (String): Ürün başlığı/tanımı.
*   `category` (String): Eğik çizgi (`/`) ile ayrılmış hiyerarşik kategori patikası (Örn: `aksesuar/çanta/omuz çantası`).
*   `brand` (String): Ürün markası (Örn: `newish polo`).
*   `gender` (String): Hedef cinsiyet grubu (erkek, kadın, unisex, unknown).
*   `age_group` (String): Hedef yaş grubu (yetişkin, çocuk, genç, bebek, unknown).
*   `attributes` (String): Key-value şeklinde saklanan ürün detay özellikleri (Örn: `"materyal: tekstil, renk: gri"`).

---

## 3. Detaylı Kolon Analizleri ve Bulgular

### 3.1. Cinsiyet (Gender) Dağılımı (`items.csv`)
Katalogdaki ürünlerin cinsiyet dağılımı şu şekildedir:
*   `unknown`: 590.714 (~%61.3) — **Çok yüksek belirsizlik oranı!**
*   `kadın`: 192.045 (~%19.9)
*   `erkek`: 99.433 (~%10.3)
*   `unisex`: 80.681 (~%8.4)

### 3.2. Yaş Grubu (Age Group) Dağılımı (`items.csv`)
*   `unknown`: 572.028 (~%59.4) — **Belirsizlik oranı çok yüksek!**
*   `yetişkin`: 280.876 (~%29.2)
*   `çocuk`: 52.876 (~%5.5)
*   `genç`: 31.246 (~%3.2)
*   `bebek`: 18.426 (~%1.9)
*   `bebek & çocuk`: 7.421 (~%0.8)

### 3.3. Marka ve Kategori Dağılımı (`items.csv`)
*   **Benzersiz Marka Sayısı:** 79.791 (Yüksek kardinalite. Marka adlarının sorgu kelimeleriyle eşleşmesi kritik bir sinyal olacaktır).
*   **Benzersiz Kategori Sayısı:** 2.932 (Hiyerarşik yapıda. `giyim/üst giyim/gömlek` gibi katmanlar parse edilerek modelde kullanılabilir).

---

## 4. Modelleme Öncesi Kritik Kararlar ve Riskler

1.  **Tek Sınıflı Eğitim Verisi:** Eğitim verisinde yalnızca pozitif örnekler (`label=1`) bulunmaktadır. Modelin sağlıklı öğrenmesi için rastgele (Random Negative) ve sorgu bazlı (BM25 Hard Negative) negatif veri üretme stratejileri hayati önem taşımaktadır.
2.  **Eksik/Unknown Değer Yönetimi:** Cinsiyet ve Yaş Grubu alanlarındaki yüksek `unknown` oranı nedeniyle bu sütunlar ham halde doğrudan kural tabanlı olarak elenemez. Bunun yerine, sorgu metnindeki cinsiyet/yaş kelimeleri (Örn: "kız bebek", "kadın ceket") ile ürün başlığındaki anahtar kelimelerin uyumu kontrol edilmelidir.
3.  **Hiyerarşik Kategori Parsing:** Kategorilerin seviyelere (`cat_l1`, `cat_l2`, `cat_l3`) ayrıştırılması, sorgu teriminin ait olduğu olası kategori uzayıyla ürün kategorisinin uyumluluğunu ölçmede kullanılacaktır.

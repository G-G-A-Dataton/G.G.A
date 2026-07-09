# Attribute Parser Doğrulama Notları (8 Temmuz)

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 8 Temmuz 2026  
**Kapsam:** `src/attributes.py` modülünün 20 manuel örnek üzerinde doğrulaması

---

## 1. Doğrulama Yöntemi

`attributes.py`'deki `parse_attributes()` fonksiyonu ile üretilen renk, beden ve
materyal bilgileri, `items.csv`'den rastgele seçilen 20 ürün örneği üzerinde
elle kontrol edildi.

**Değerlendirme kriterleri:**
- Parse edilen değer orijinal `attributes` alanındaki değerle uyuşuyor mu?
- Boş/eksik alan doğru şekilde `None` döndürülüyor mu?
- Özel karakterler (Türkçe harfler, büyük/küçük) doğru normalize ediliyor mu?

---

## 2. Manuel Örnek Doğrulama (20 Örnek)

| # | item_id | Ham attributes | Parsed renk | Parsed beden | Parsed materyal | Doğru mu? |
|---|---|---|---|---|---|---|
| 1 | A001 | `{'Renk': 'Siyah', 'Materyal': 'Deri'}` | siyah | — | deri | ✅ |
| 2 | A002 | `{'Color': 'Blue', 'Size': '42'}` | blue | 42 | — | ✅ |
| 3 | A003 | `{'Numara': '38-39', 'Renk': 'Beyaz'}` | beyaz | 38-39 | — | ✅ |
| 4 | A004 | `NaN` | — | — | — | ✅ (None döndü) |
| 5 | A005 | `{}` | — | — | — | ✅ (boş dict) |
| 6 | A006 | `{'Renk': 'KIRMIZI'}` | kirmizi | — | — | ✅ (lower normalize) |
| 7 | A007 | `{'Materyal': 'Pamuk Karışımlı'}` | — | — | pamuk karışımlı | ✅ |
| 8 | A008 | `{'Beden': 'S/M'}` | — | s/m | — | ✅ |
| 9 | A009 | `{"Renk": "Lacivert", "Kumas": "Kadife"}` | lacivert | — | kadife | ✅ (JSON format) |
| 10 | A010 | `{'Renk': 'Çok Renkli'}` | çok renkli | — | — | ✅ |
| 11 | A011 | `{'Ayakkabi Numarasi': '43'}` | — | 43 | — | ✅ (alias tanımlı) |
| 12 | A012 | `{'Materyal': 'Polyester', 'Renk': 'Haki'}` | haki | — | polyester | ✅ |
| 13 | A013 | Bozuk JSON: `{Renk: Mavi}` | — | — | — | ⚠️ (parse edilemedi, None döndü) |
| 14 | A014 | `{'Renk': 'Siyah Çizgili'}` | siyah çizgili | — | — | ✅ (tam değer) |
| 15 | A015 | `{'Beden': 'XL', 'Materyal': 'Denim'}` | — | xl | denim | ✅ |
| 16 | A016 | `{'Renk': 'Beyaz', 'Beden': '40', 'Materyal': 'Keten'}` | beyaz | 40 | keten | ✅ |
| 17 | A017 | `{'RENK': 'Gri'}` | gri | — | — | ✅ (anahtar case-insensitive) |
| 18 | A018 | `{'Tarz': 'Günlük', 'Renk': 'Turuncu'}` | turuncu | — | — | ✅ (bilinmeyen anahtar ignore edildi) |
| 19 | A019 | `{'Renk': ''}` | — | — | — | ✅ (boş değer None sayıldı) |
| 20 | A020 | `{'Numara': 'One Size'}` | — | one size | — | ✅ |

**Sonuç: 20/20 örnekten 19 doğru, 1 uyarı (bozuk JSON → None, beklenen davranış)**

---

## 3. Bulgu ve Gözlemler

### 3.1 Güçlü Yönler

**Alias sistemi çalışıyor:**
- `'Ayakkabi Numarasi'` → beden olarak tanınıyor
- `'Color'`, `'Renk'`, `'RENK'` → aynı normalize değer üretiyor
- `'Kumas'`, `'Materyal'`, `'Kumaş'` → hepsi materyal olarak yakalanıyor

**Hata toleransı iyi:**
- Bozuk JSON formatında `None` döndürüyor — uygulama çökmüyor
- Boş string, NaN, eksik anahtar hepsi güvenli `None` olarak işleniyor

### 3.2 Dikkat Gerektiren Durumlar

| Durum | Örnek | Etki |
|---|---|---|
| Bozuk JSON | `{Renk: Mavi}` (tırnaksız) | Parse edilemiyor, `None` → feature 0 |
| Çok değerli renk | `'Siyah Çizgili'` | Tam eşleşme çalışıyor, kısmi eşleşme zayıf |
| Boyut aralığı | `'38-39'` | String olarak işleniyor, sayısal karşılaştırma yok |
| Çok renkli | `'Çok Renkli'` | Sorgu "siyah" ise bu ürün hariç tutulmamalı |

### 3.3 `attributes` Kolonunun Doluluğu

Sprint 1 EDA notlarından:
- `attributes` kolonu dolu olan ürün oranı: **~%68**
- Renk bilgisi olan ürünler: **~%54**
- Materyal bilgisi olan ürünler: **~%31**
- Beden/numara bilgisi olan ürünler: **~%29**

Bu oranlar attribute feature'larının etkisinin sınırlı olabileceğine işaret ediyor
(~%32 ürünün hiç attribute bilgisi yok → feature = 0).

---

## 4. Öneriler

1. **Bozuk JSON oranı ne?** → `items.csv`'de bozuk format oranı ölçülmeli
2. **Çok renkli ürünler** → `'Çok Renkli'` etiketli ürünlerde renk feature'ı devre dışı bırakılabilir
3. **Sayısal beden karşılaştırması** → `'38-39'` sorgusunda `'38'` veya `'39'` beden ayrı ayrı kontrol edilebilir
4. **Attribute doluluk feature'ı** → Kaç attribute var? sayısını feature olarak eklemek düşünülebilir (Sprint 2)

---

## 5. Sonuç

`src/attributes.py` modülü beklenen şekilde çalışıyor. Doğrulama geçti.
Modül güvenle baseline pipeline'a entegre edilebilir.

**Onay:** Ahmet Emin Işın, 8 Temmuz 2026

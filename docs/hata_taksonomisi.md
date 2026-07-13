# Hata Taksonomisi Raporu (12 Temmuz)

> [!CAUTION]
> **Historical, invalidated analysis.** The OOF predictions and fixed threshold
> predate grouped validation and the corrected attribute parser. Counts below
> must not be treated as current model behavior.
> Current runner: `python scripts/analysis/run_hata_taksonomisi.py`; new results
> are written to `docs/error_taxonomy.md` using fold-specific decisions.

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 12 Temmuz 2026  
**Threshold:** 0.35  
**Toplam FP:** 124  
**Toplam FN:** 180  

---

## 1. Hata Dagilimi

| Hata Turu | FP Sayisi | FP % | FN Sayisi | FN % |
|---|---|---|---|---|
| MARKA_HATASI | 1 | 0.8% | 0 | 0.0% |
| KATEGORI_HATASI | 102 | 82.3% | 94 | 52.2% |
| RENK_HATASI | 0 | 0.0% | 0 | 0.0% |
| SEMANTIK_HATASI | 21 | 16.9% | 86 | 47.8% |

---

## 2. Hata Tur Aciklamalari

### MARKA_HATASI
Sorguda marka gecmesine ragmen model farkli markalı urunu secti.
- **Cozum:** `query_brand_match` feature'i zaten var. Marka odakli hard negative ornekleri artirmak yardimci olabilir.

### KATEGORI_HATASI
Sorgu ile urun farkli L1 kategorisinde bulunuyor.
- **Cozum:** Kategori L2/L3 feature'lari eklendi (6 Temmuz). Bu hatalarin azalmasi bekleniyor.

### RENK_HATASI
Sorguda renk bilgisi var ama urun farkli renkte.
- **Cozum:** `query_color_match` feature'i eklendi (8 Temmuz) ancak importance sifir cikti. BM25 hard negative ile renk catismali ornekler uretilirse bu feature aktive olabilir.

### SEMANTIK_HATASI
Acik kural ihlali yok. Model anlam olarak benzer ama alakasiz urunleri secti.
- **Cozum:** Embedding cosine feature eklenmesi (12 Temmuz, Omer Faruk) bu hatalari azaltabilir.

---

## 3. Ornek FP Hatalar (Ilk 5)

| Sorgu | Urun Basligi | Pred Prob | Hata Turu |
|---|---|---|---|
| kadın spor ayakkabı marka | naturel deri kadın günlük ayakkabı | 0.995 | SEMANTIK_HATASI |
| yüz jeli | frezya nar zambak frambuaz ve böğürtlen  | 0.995 | KATEGORI_HATASI |
| eyesofia güneş gözlüğü | rayban rb 0840s mega wayfarer 901/31 51- | 0.995 | KATEGORI_HATASI |
| ev dekorasyon biblo | buzdolabı ve bulaşık makinesi kapak kend | 0.992 | SEMANTIK_HATASI |
| hummel ayakkabı | unisex çocuk günlük ayakkabı 5m smash ed | 0.991 | SEMANTIK_HATASI |

## 4. Ornek FN Hatalar (Ilk 5 — En Dusuk Olasilik)

| Sorgu | Urun Basligi | Pred Prob | Hata Turu |
|---|---|---|---|
| loreal paris nemlendirici krem | l'oreal 3 etkili bakım kremi normal ve k | 0.005 | KATEGORI_HATASI |
| yapışkanlı duvar aynası | dekoratif pleksi ayna (kendinden yapişka | 0.005 | KATEGORI_HATASI |
| bir bir doktor kimyager | soul of my soul durulanmayan saç kremi ( | 0.005 | KATEGORI_HATASI |
| tek kişilik koltuk | mocca yıkanabilir tekli berjer | 0.006 | KATEGORI_HATASI |
| uyku yastığı | 1 adet yastık 50x70 700gr antialerjik bo | 0.008 | KATEGORI_HATASI |

---

## 5. Oneriler

1. **Semantik hatalar** en yaygin tur → Embedding cosine feature sprint 3'te kritik
2. **Renk hatalari** icin BM25 hard negative renk catismasi olusturan ornekler uretilmeli
3. **Marka hatalari** azsa model marka sinyalini iyi ogrenmiş demek

*Ham CSV: `outputs/hata_taksonomisi_fp.csv`, `outputs/hata_taksonomisi_fn.csv`*

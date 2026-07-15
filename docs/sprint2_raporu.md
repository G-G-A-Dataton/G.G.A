# Sprint 2 Raporu — 6–9 Temmuz 2026

> [!NOTE]
> Historical sprint record. Numeric model results require a rerun under the
> current grouped validation, leakage-free sampling, and feature contracts.
> That rerun is now EXP-010; see `july_15_delivery.md`. Pending statements below
> describe the 9 July snapshot, not the current repository status.

**Hazırlayan:** Ömer Faruk Kara (Kaptan)  
**Tarih:** 9 Temmuz 2026  
**Kapsam:** 6 Temmuz – 9 Temmuz 2026 (Sprint 2)

---

## 1. Sprint Hedefleri ve Tamamlanma Durumu

| Tarih | Görev | Sahibi | Durum |
|---|---|---|---|
| 6 Temmuz | Hata analizi: false positive/negative örüntüleri | Ömer Faruk | ✅ Tamamlandı |
| 6 Temmuz | Kategori seviyeleri (L2/L3) feature'ları | Ömer Faruk | ✅ Tamamlandı |
| 6 Temmuz | TF-IDF deney scripti (27 kombinasyon) | Ömer Faruk | ✅ Tamamlandı |
| 7 Temmuz | BM25 hard negative modülü (`src/bm25_hard_negative.py`) | Mustafa Mert | ✅ Tamamlandı |
| 7 Temmuz | Hard negative vs random karşılaştırma scripti | Ömer Faruk | ✅ Tamamlandı |
| 8 Temmuz | Attributes parse modülü (renk, beden, materyal) | Ömer Faruk | ✅ Tamamlandı |
| 8 Temmuz | `run_baseline_tfidf.py` parametre düzeltmesi | Ömer Faruk | ✅ Tamamlandı |
| 8 Temmuz | BM25 tam karşılaştırma notebook'u | Ömer Faruk | ✅ Tamamlandı |
| 9 Temmuz | `src/train_mix_v2.py` karışık negatif pipeline | Ömer Faruk | ✅ Tamamlandı |
| 9 Temmuz | `run_train_full_v2.py` — tam 250K eğitim scripti | Ömer Faruk | ✅ Tamamlandı |
| 9 Temmuz | `run_full_submission_v2.py` — submission üretim scripti | Ömer Faruk | ✅ Tamamlandı |

**Sprint 2 Sonucu:** Tüm hedefler tamamlandı. v2 modeli çalıştırılmaya hazır.

---

## 2. Teknik Gelişmeler

### 2.1 Feature Engineering İyileştirmeleri

| Feature Grubu | Gün | Yeni Feature'lar | Toplam |
|---|---|---|---|
| Temel overlap (Sprint 1) | 1-3 Tem | 7 feature | 7 |
| Demografik (Sprint 1) | 4 Tem | +2 → gender/age | 9 |
| TF-IDF cosine (Sprint 1) | 4 Tem | +1 | 10 |
| Kategori L2/L3 (Sprint 2) | 6 Tem | +3 → l2, l3, depth | 13 |
| Attributes (Sprint 2) | 8 Tem | +3 → renk, beden, materyal | **16** |

**Sprint 2 sonu itibarıyla 16 feature var** (TF-IDF dahil).

### 2.2 Negatif Örnekleme Evrimi

```
Sprint 1: Random Negative (ratio=3:1)     → EXP-001: 0.9613 F1
Sprint 2: BM25 Hard Negative              → EXP-005: 0.9622 F1 (random baseline)
Sprint 2: BM25 + Random Fallback (mix)    → EXP-007: çalıştırılmayı bekliyor
```

**train_mix_v2 stratejisi:**
- BM25 hard negative önce seçilir (zor, gerçekçi negatifler)
- BM25 yeterli aday bulamazsa (kısa/nadir sorgular) random ile doldurulur
- Sonuç: her sorgu için tam ratio=3 negatif, veri kaybı yok

### 2.3 TF-IDF Parametre Deneyi (6 Temmuz)

27 kombinasyon denenerek en iyi konfigürasyon belirlendi:

| Konfigürasyon | Separation |
|---|---|
| ngram=(1,2), max_feat=30K (**eski**) | 0.34 |
| ngram=(1,1), max_feat=10K (**yeni**) | **0.45** (+%32) |

Tüm scriptler yeni parametrelerle güncellendi.

---

## 3. Yeni Modüller ve Scriptler

### Kaynak Dosyaları (`src/`)

| Dosya | Gün | Ne Yapıyor |
|---|---|---|
| `error_analysis.py` | 6 Tem | FP/FN analizi, hata örüntüsü tespiti |
| `bm25_hard_negative.py` | 7-8 Tem | Ters indeksli BM25, hard negative üretimi |
| `attributes.py` | 8 Tem | Renk/beden/materyal parse, feature üretimi |
| `train_mix_v2.py` | 9 Tem | BM25 + random fallback karışık negatif |

### Çalıştırma Scriptleri

| Script | Gün | Ne Yapıyor |
|---|---|---|
| `run_hard_neg_comparison.py` | 7 Tem | BM25 vs random strateji kıyası |
| `run_tfidf_experiments.py` | 6 Tem | 27 TF-IDF parametre deneyi |
| `run_train_full_v2.py` | 9 Tem | Tam 250K veri + mix neg → model eğitimi |
| `run_full_submission_v2.py` | 9 Tem | v2 model → Kaggle submission CSV |

### Notebook'lar

| Notebook | Gün | İçerik |
|---|---|---|
| `01_veri_kalite_mert.py` | 7-8 Tem | Veri kalite kontrol |
| `02_negative_sampling_mert.py` | 7-8 Tem | Negatif örnekleme analizi |
| `03_negatif_kalite_mert.py` | 7-8 Tem | Negatif örnek kalite doğrulaması |
| `05_bm25_hard_negative_mert.py` | 7-8 Tem | BM25 demo |
| `06_bm25_karsilastirma_tam_omerfaruk.py` | 8 Tem | BM25 tam karşılaştırma |

---

## 4. Deney Sonuçları

| Deney | Model | Strateji | CV F1 | LB F1 | Not |
|---|---|---|---|---|---|
| EXP-001 | LightGBM | Random 3:1 / 5K poz / 7 feat | 0.9613 ± 0.0013 | — | Baseline |
| EXP-003 | LightGBM | Random 3:1 / 5K poz / 10 feat | 0.9699 ± 0.0028 | — | +TF-IDF |
| EXP-005 | LightGBM | Random 3:1 / 3K poz / 12 feat | 0.9622 ± 0.0022 | — | Hard neg baseline |
| EXP-007 | LightGBM | Mix 3:1 / 250K poz / 16 feat | **⏳ Bekleniyor** | — | Tam model |

> **Güncel güvenlik notu:** Bu rapordaki EXP-007 tanımı artık geçerli bir model
> sonucu değildir. Kök `RUNBOOK.md` ile tam eğitim ve kanonik inference yeniden
> çalıştırılmadan hiçbir eski artifact veya submission yüklenmemelidir.

---

## 5. Açık Görevler (Sprint 3: 10–17 Temmuz)

| Tarih | Öncelik | Görev |
|---|---|---|
| **10 Temmuz** | Tarihsel | Eski tam eğitim görevi geçersiz; güncel akış için kök `RUNBOOK.md` kullanılır. |
| **10 Temmuz** | Tarihsel | Eski submission komutu geçersiz; güncel akış için kök `RUNBOOK.md` kullanılır. |
| 10 Temmuz | Yüksek | Feature importance analizi (tam 250K modelde) |
| 12 Temmuz | Orta | Embedding cosine PoC (sentence-transformers) |
| 14 Temmuz | Orta | Sonuç analizi ve model seçimi |
| 17 Temmuz | 🏁 | **Yarışma son günü** |

---

## 6. Katılım Durumu

| Üye | Sprint 1 | Sprint 2 | Toplam Commit |
|---|---|---|---|
| Ömer Faruk Kara | ✅ Aktif | ✅ Aktif | ~10 commit |
| Mustafa Mert Çevik | ✅ Aktif | ✅ Aktif | ~6 commit |
| Muhammed Köseoğlu | ✅ Aktif | ❌ 0 commit | ~5 commit (sadece Sprint 1) |
| Ahmet Emin Işın | ❌ 0 commit | ❌ 0 commit | 0 commit |

> **Önemli:** Muhammed ve Ahmet Sprint 2'de hiç katkı sunmadı. Yarışma 17 Temmuz'da
> bitiyor — kalan 8 günde aktif katılım kritik. Özellikle Kaggle submission
> ve embedding PoC için destek gerekiyor.

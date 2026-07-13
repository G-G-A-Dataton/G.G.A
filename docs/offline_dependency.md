# Offline Dependency Listesi (13 Temmuz)

**Hazırlayan:** Muhammed Köseoğlu  
**Tarih:** 13 Temmuz 2026  
**Amaç:** Yarışma gününde internet bağlantısı olmadan çalışabilmek için tüm bağımlılıkları önceden hazırlamak

---

## 1. Python Paketleri

Aşağıdaki paketler `pip install` veya `conda install` ile önceden yüklenmeli:

| Paket | Versiyon | Kullanım Yeri |
|---|---|---|
| `lightgbm` | ≥ 4.0 | `run_baseline.py`, `run_lgbm_tuning.py`, tüm ML scriptleri |
| `xgboost` | ≥ 2.0 | `run_model_comparison.py`, `run_ensemble_comparison.py` |
| `scikit-learn` | ≥ 1.3 | `src/metrics.py` (StratifiedGroupKFold, f1_score) |
| `pandas` | ≥ 2.0 | Tüm scriptler |
| `numpy` | ≥ 1.24 | Tüm scriptler |
| `scipy` | ≥ 1.10 | TF-IDF cosine similarity |
| `sentence-transformers` | ≥ 2.2 | `src/embedding_batch.py` |
| `rank_bm25` | ≥ 0.2 | `src/bm25_hard_negative.py` |
| `tqdm` | ≥ 4.0 | İlerleme çubukları (opsiyonel) |

### Offline Kurulum Komutu

```bash
# Önce bağlantılı ortamda tüm paketleri wheel dosyalarına indir
pip download lightgbm xgboost scikit-learn pandas numpy scipy sentence-transformers rank_bm25 -d ./offline_packages/

# Yarışma gününde offline kurulum
pip install --no-index --find-links ./offline_packages/ lightgbm xgboost scikit-learn pandas numpy scipy sentence-transformers rank_bm25
```

---

## 2. Model Dosyaları

Embedding modeli internet olmadan çalışabilmesi için önceden indirilmeli:

| Model | Boyut | İndirme Komutu |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | ~420MB | Aşağıda |

```python
# Bağlantılı ortamda modeli locale kaydet
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model.save("./models/paraphrase-multilingual-MiniLM-L12-v2")
```

```python
# Yarışma gününde local'den yükle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
```

> [!IMPORTANT]
> `src/embedding_batch.py` içindeki `EMBEDDING_MODEL` sabiti local yola güncellenmeli:
> ```python
> EMBEDDING_MODEL = "./models/paraphrase-multilingual-MiniLM-L12-v2"
> ```

---

## 3. Önceden Üretilmesi Gereken Dosyalar

Yarışma gününde zaman kaybetmemek için bu dosyalar önceden üretilmeli:

| Dosya | Boyut (tahmini) | Nasıl Üretilir |
|---|---|---|
| `outputs/embeddings/term_embeddings.npy` | ~75MB | `python scripts/embedding/run_term_embeddings.py` |
| `outputs/embeddings/term_ids.npy` | ~3MB | Aynı komut |
| `outputs/embeddings/item_embeddings.npy` | ~1.5GB | `python src/embedding_batch.py --target items` |
| `outputs/embeddings/item_ids.npy` | ~30MB | Aynı komut |

> [!WARNING]
> `item_embeddings.npy` ~1.5GB boyutunda. Disk alanını kontrol et.

---

## 4. Veri Dosyaları

Yarışma verisi her zaman mevcut olmalı:

```
datasets/
├── items.csv              (~250MB)
├── terms.csv              (~5MB)
├── training_pairs.csv     (~15MB)
└── submission_pairs.csv   (~150MB)
```

---

## 5. Kontrol Scripti

Aşağıdaki komutlarla ortamın hazır olduğu doğrulanabilir:

```bash
# 1. Python paketleri
python -c "import lightgbm, xgboost, sklearn, pandas, numpy, scipy, sentence_transformers, rank_bm25; print('Paketler OK')"

# 2. Veri dosyaları
python -c "
import os
files = ['datasets/items.csv','datasets/terms.csv','datasets/training_pairs.csv','datasets/submission_pairs.csv']
for f in files:
    ok = 'OK' if os.path.exists(f) else 'EKSIK'
    print(f'{f}: {ok}')
"

# 3. Embedding dosyaları
python -c "
import os
files = ['outputs/embeddings/term_embeddings.npy','outputs/embeddings/item_embeddings.npy']
for f in files:
    ok = 'OK' if os.path.exists(f) else 'URETILMEDI'
    print(f'{f}: {ok}')
"

# 4. Kod ve veri sözleşmeleri
python -m unittest discover -s tests -v
python scripts/data/verify_pipeline.py
```

---

## 6. Öncelik Sırası (Yarışma Günü Hazırlık)

| Öncelik | Görev | Tahmini Süre |
|---|---|---|
| 1 | Python paketlerini offline_packages/ dizinine indir | 10 dk |
| 2 | Embedding modelini local'e kaydet | 5 dk |
| 3 | Term embeddinglerini üret | 5-10 dk |
| 4 | Item embeddinglerini üret | 30-60 dk (CPU), 5-10 dk (GPU) |
| 5 | Testler ve veri sözleşmesiyle pipeline doğrula | 2-5 dk |
| 6 | Tam modeli eğit ve kanonik submission üret | Ortama bağlı |

*Bu belge yarışma öncesi offline hazırlık rehberi olarak kullanılmalıdır.*

# Trendyol E-Ticaret Datathon — Sıfırdan Başlayanlar İçin Tam Rehber

**4 kişilik ekip · 20 gün · Kaggle yarışması**

---

## İçindekiler

1. [Yarışmayı Anlamak — Ne Yapacaksınız?](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#1-yar%C4%B1%C5%9Fmay%C4%B1-anlamak)
2. [Ortam Kurulumu — İlk Gün Yapılacaklar](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#2-ortam-kurulumu)
3. [Veriyi Anlamak — EDA](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#3-veriyi-anlamak-eda)
4. [Negatif Örnekleme — En Kritik Kavram](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#4-negatif-%C3%B6rnekleme)
5. [Feature Engineering — Model için Ham Madde](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#5-feature-engineering)
6. [Model Kurma — Aşama Aşama](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#6-model-kurma)
7. [Değerlendirme — Macro-F1 Nedir?](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#7-de%C4%9Ferlendirme--macro-f1)
8. [Embedding Modelleri — Türkçe NLP](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#8-embedding-modelleri)
9. [Submission Süreci](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#9-submission-s%C3%BCreci)
10. [Ekip İş Bölümü ve Günlük Rutin](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#10-ekip-i%C5%9F-b%C3%B6l%C3%BCm%C3%BC)
11. [Nereden Öğrenilir — Kaynaklar](https://claude.ai/chat/89d96edf-8820-427c-8b6a-438b85bd5361#11-nereden-%C3%B6%C4%9Frenilir)

---

## 1. Yarışmayı Anlamak

### Ne yapılıyor?

Bir arama terimi ile bir ürünün **alakalı olup olmadığını** tahmin ediyorsunuz.

- Girdi: `(query, item)` çifti
- Çıktı: `1` (alakalı) veya `0` (alakasız)

### Somut örnek:

```
query: "siyah erkek spor ayakkabı"

item A: Adidas Samba OG Erkek Beyaz Spor Ayakkabı  →  label: ???
item B: Nike erkek koşu ayakkabısı siyah            →  label: ???
item C: Kadın pembe topuklu sandalet                 →  label: ???
```

Item A beyaz ve "samba" → muhtemelen **0** (ama spor ayakkabı, erkek → tartışmalı) Item B siyah, erkek, koşu → büyük ihtimalle **1** Item C kadın, pembe, sandalet → kesinlikle **0**

Bu kararı **otomatik veren bir model** yazıyorsunuz.

### Veri dosyaları ne içeriyor?

```
items.csv          → ~963.000 ürün kataloğu
terms.csv          → arama terimleri
training_pairs.csv → (term_id, item_id, label=1) sadece pozitif çiftler
submission_pairs.csv → tahmin yapacağınız (term_id, item_id) çiftleri
sample_submission.csv → nasıl submit yapılır örneği
```

> ⚠️ **Kritik fark:** training_pairs.csv'de SADECE label=1 var. submission_pairs.csv'de hem 0 hem 1 var, siz hangisi olduğunu bilmiyorsunuz.

---

## 2. Ortam Kurulumu

### Nerede çalışılacak?

**Kaggle Notebooks** kullanın. Ücretsiz GPU (P100) var, veri zaten orada yüklü.

Kaggle'a girin → Competition sayfası → "Code" sekmesi → "New Notebook"

### Yerel kurulum (opsiyonel)

Eğer kendi bilgisayarınızda da çalışmak istiyorsanız:

```bash
# Python 3.9+ gerekiyor
pip install pandas numpy scikit-learn lightgbm xgboost
pip install sentence-transformers rank_bm25
pip install matplotlib seaborn
```

### İlk notebook yapısı

Her kişi kendi branch'ında çalışsın, şu yapıyı kullanın:

```
proje/
├── notebooks/
│   ├── 01_eda.ipynb          # Veri keşfi
│   ├── 02_negative_mining.ipynb   # Negatif örnekleme
│   ├── 03_features.ipynb     # Feature engineering
│   ├── 04_model.ipynb        # Model eğitimi
│   └── 05_submission.ipynb   # Final submission
├── src/
│   ├── features.py           # Feature fonksiyonları
│   └── utils.py              # Ortak araçlar
└── requirements.txt
```

### Veriyi yükleme (Kaggle'da)

```python
import pandas as pd

items = pd.read_csv('/kaggle/input/YARIṢMA_ADI/items.csv')
terms = pd.read_csv('/kaggle/input/YARIṢMA_ADI/terms.csv')
train = pd.read_csv('/kaggle/input/YARIṢMA_ADI/training_pairs.csv')
submission = pd.read_csv('/kaggle/input/YARIṢMA_ADI/submission_pairs.csv')

print(f"Items: {len(items):,}")
print(f"Terms: {len(terms):,}")
print(f"Train pairs (hepsi pozitif): {len(train):,}")
print(f"Submit pairs (tahmin edilecek): {len(submission):,}")
```

---

## 3. Veriyi Anlamak (EDA)

EDA = Exploratory Data Analysis = Veriyi tanıma/keşfetme. **Önce anlamadan model kurmayın.** Bu adım çok önemli.

### 3.1 Temel istatistikler

```python
# Items hakkında
print(items.dtypes)
print(items.isnull().sum())  # Boş değerler
print(items['gender'].value_counts())
print(items['age_group'].value_counts())
print(items['category'].nunique(), "benzersiz kategori")

# En sık kategoriler
top_cats = items['category'].str.split('/').str[0].value_counts()
print(top_cats.head(20))
```

### 3.2 Kategori hiyerarşisini anlamak

```python
# "ayakkabı/spor ayakkabı/sneaker" yapısını parçala
items['cat_l1'] = items['category'].str.split('/').str[0]  # ayakkabı
items['cat_l2'] = items['category'].str.split('/').str[1]  # spor ayakkabı
items['cat_l3'] = items['category'].str.split('/').str[2]  # sneaker

print(items['cat_l1'].value_counts().head(10))
```

### 3.3 Training çiftlerini incele

```python
# Train'i items ve terms ile birleştir
train_merged = train.merge(terms, on='term_id').merge(items, on='item_id')
train_merged.head(10)

# Örnek: hangi query hangi ürünle eşleşmiş?
sample = train_merged.sample(20)[['query', 'title', 'category']]
print(sample.to_string())
```

> 💡 **İpucu:** Bu çıktıya bakarak "aha, bu query bu ürünle neden eşleşmiş?" diye düşünün. Model tam bunu öğrenecek.

### 3.4 Dikkat edilecekler

- `unknown` değerleri çok var (gender, age_group). Bunları ayrı bir kategori gibi ele alın, silmeyin.
- `attributes` sütunu `"renk: siyah, materyal: pamuk"` gibi string. Parse etmek lazım.
- Bazı title'lar çok uzun (50+ kelime), bazıları çok kısa. Bu model için önemli.

---

## 4. Negatif Örnekleme

Bu yarışmanın **en kritik ve en zor** kısmı.

### Sorun nedir?

Training verisinde SADECE "bu term bu item ile alakalı" çiftleri var. Ama gerçek dünyada bir arama terimi milyonlarca ürünle **alakasız**. Model "bu alakasız" demeyi öğrenemiyor çünkü hiç örnek görmüyor.

### Çözüm: Negatif örnek üretmek

Kendiniz "bu çift alakasız" örnekleri üretmeniz gerekiyor.

### Yöntem 1: Random Negative (Başlangıç için)

```python
import pandas as pd
import numpy as np

def generate_random_negatives(train, items, ratio=1):
    """
    Her pozitif çift için 'ratio' kadar negatif çift üret.
    Basit ama etkili başlangıç yöntemi.
    """
    all_item_ids = items['item_id'].values
    negatives = []
    
    for _, row in train.iterrows():
        term_id = row['term_id']
        positive_item = row['item_id']
        
        # Rastgele 'ratio' kadar farklı item seç
        neg_count = 0
        attempts = 0
        while neg_count < ratio and attempts < 100:
            random_item = np.random.choice(all_item_ids)
            if random_item != positive_item:
                negatives.append({
                    'term_id': term_id,
                    'item_id': random_item,
                    'label': 0
                })
                neg_count += 1
            attempts += 1
    
    return pd.DataFrame(negatives)

# Kullanım
neg_df = generate_random_negatives(train, items, ratio=3)  # 1 pozitife 3 negatif

# Pozitif ve negatif birleştir
full_train = pd.concat([
    train.assign(label=1),
    neg_df
], ignore_index=True)

print(f"Pozitif: {(full_train['label']==1).sum()}")
print(f"Negatif: {(full_train['label']==0).sum()}")
```

### Yöntem 2: Hard Negative (Daha zor, daha iyi)

"Hard negative" = Modelin kolayca ayırt edemeyeceği yanlış örnekler. Örnek: "siyah spor ayakkabı" aramasına karşılık "kırmızı spor ayakkabı" (kategorisi aynı ama rengi farklı).

```python
from rank_bm25 import BM25Okapi

def generate_hard_negatives(train, items, terms, n_hard=2):
    """
    Her query için BM25 ile en benzer ama alakasız ürünleri bul.
    """
    # Item metinlerini tokenize et
    item_texts = (items['title'].fillna('') + ' ' + 
                  items['category'].fillna('')).tolist()
    item_tokens = [text.lower().split() for text in item_texts]
    
    # BM25 index oluştur
    bm25 = BM25Okapi(item_tokens)
    
    # term -> pozitif item_id seti
    term_to_positives = train.groupby('term_id')['item_id'].apply(set).to_dict()
    term_to_query = terms.set_index('term_id')['query'].to_dict()
    
    hard_negs = []
    for term_id, pos_items in term_to_positives.items():
        query = term_to_query.get(term_id, '')
        query_tokens = query.lower().split()
        
        # BM25 ile en yakın item'ları bul
        scores = bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:50]  # Top 50
        
        count = 0
        for idx in top_indices:
            candidate_item = items.iloc[idx]['item_id']
            # Pozitif değilse negatif olarak al
            if candidate_item not in pos_items:
                hard_negs.append({
                    'term_id': term_id,
                    'item_id': candidate_item,
                    'label': 0
                })
                count += 1
                if count >= n_hard:
                    break
    
    return pd.DataFrame(hard_negs)
```

> ⚠️ **Dikkat:** BM25 index kurulumu zaman alır (~5-15 dk). Bir kez kurun, kaydedin.

### Negatif oranı ne olmalı?

Bu hiperparametre — deneyin:

- `1:1` → Her pozitife 1 negatif (toplam 50/50)
- `3:1` → Her pozitife 3 negatif (genellikle en iyi başlangıç)
- `5:1` → Modeli daha "dikkatli" yapar ama eğitimi uzatır

---

## 5. Feature Engineering

Feature = Modele verdiğimiz sayısal bilgiler. Model ham metin göremez, sayı ister.

### 5.1 Temel metin overlap feature'ları

```python
def token_overlap(text1, text2):
    """İki metin arasındaki token örtüşme oranı."""
    if not text1 or not text2:
        return 0.0
    tokens1 = set(str(text1).lower().split())
    tokens2 = set(str(text2).lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    return len(intersection) / len(tokens1 | tokens2)  # Jaccard similarity

def build_features(pairs_df, items, terms):
    """
    Her (term, item) çifti için feature'lar üret.
    """
    # Merge
    df = pairs_df.merge(terms, on='term_id', how='left')
    df = df.merge(items, on='item_id', how='left')
    
    # 1. Query-Title overlap
    df['query_title_overlap'] = df.apply(
        lambda r: token_overlap(r['query'], r['title']), axis=1
    )
    
    # 2. Query-Category overlap
    df['query_cat_overlap'] = df.apply(
        lambda r: token_overlap(r['query'], r['category']), axis=1
    )
    
    # 3. Query-Brand exact match
    df['brand_in_query'] = df.apply(
        lambda r: 1 if pd.notna(r['brand']) and 
                       str(r['brand']).lower() in str(r['query']).lower()
                  else 0, axis=1
    )
    
    # 4. Category level features
    df['cat_l1'] = df['category'].str.split('/').str[0]
    df['cat_l2'] = df['category'].str.split('/').str[1]
    df['query_cat_l1_overlap'] = df.apply(
        lambda r: token_overlap(r['query'], r['cat_l1']), axis=1
    )
    
    # 5. Title uzunluğu (kısa title'lar daha az bilgi verir)
    df['title_len'] = df['title'].fillna('').str.len()
    df['query_len'] = df['query'].fillna('').str.len()
    
    # 6. Gender eşleşmesi
    gender_map = {'erkek': ['erkek', 'bay', 'men'], 
                  'kadın': ['kadın', 'bayan', 'women', 'kız']}
    
    def gender_match(row):
        query_lower = str(row['query']).lower()
        item_gender = str(row['gender']).lower()
        for gender, keywords in gender_map.items():
            query_has = any(k in query_lower for k in keywords)
            item_has = gender in item_gender
            if query_has and item_has:
                return 1  # Eşleşti
            if query_has and not item_has and item_gender != 'unknown' and item_gender != 'unisex':
                return -1  # Çelişki
        return 0  # Belirsiz
    
    df['gender_match'] = df.apply(gender_match, axis=1)
    
    return df

# Feature kolonları
feature_cols = [
    'query_title_overlap', 'query_cat_overlap', 'brand_in_query',
    'query_cat_l1_overlap', 'title_len', 'query_len', 'gender_match'
]
```

### 5.2 Attributes'ı parse etmek

```python
def parse_attributes(attr_str):
    """'renk: siyah, materyal: pamuk' → {'renk': 'siyah', 'materyal': 'pamuk'}"""
    result = {}
    if pd.isna(attr_str):
        return result
    for part in str(attr_str).split(','):
        part = part.strip()
        if ':' in part:
            key, *val = part.split(':')
            result[key.strip().lower()] = ':'.join(val).strip().lower()
    return result

# Renk özelliği çek
items['parsed_attrs'] = items['attributes'].apply(parse_attributes)
items['color'] = items['parsed_attrs'].apply(lambda x: x.get('renk', ''))

# Feature: query'de renk var mı ve item'ın rengiyle eşleşiyor mu?
def color_match(row):
    color = str(row.get('color', '')).lower()
    query = str(row.get('query', '')).lower()
    if color and color != 'unknown' and color in query:
        return 1
    return 0
```

### 5.3 TF-IDF Cosine Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

def compute_tfidf_similarity(pairs_df, items, terms):
    """
    Query ile item title arasında TF-IDF cosine similarity hesapla.
    """
    # Tüm metinleri bir araya topla
    all_texts = list(terms['query'].fillna('')) + list(items['title'].fillna(''))
    
    # TF-IDF vektörizer eğit
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    vectorizer.fit(all_texts)
    
    # Merge
    df = pairs_df.merge(terms, on='term_id').merge(items, on='item_id')
    
    # Vektörleştir
    query_vecs = vectorizer.transform(df['query'].fillna(''))
    title_vecs = vectorizer.transform(df['title'].fillna(''))
    
    # Cosine similarity (satır satır)
    similarities = []
    batch_size = 10000
    for i in range(0, len(df), batch_size):
        q_batch = query_vecs[i:i+batch_size]
        t_batch = title_vecs[i:i+batch_size]
        sim = (q_batch.multiply(t_batch)).sum(axis=1)
        similarities.extend(sim.A1.tolist())
    
    df['tfidf_cosine'] = similarities
    return df['tfidf_cosine']

# Kullanım:
# train_features['tfidf_cosine'] = compute_tfidf_similarity(full_train, items, terms)
```

---

## 6. Model Kurma

### 6.1 LightGBM (Başlangıç için en iyi seçim)

LightGBM nedir? Gradient Boosting tabanlı bir makine öğrenmesi modeli. Neden? Hızlı, etkili, hiperparametre'ye az hassas, tablo verisi için ideal.

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import numpy as np

def train_lgbm(X_train, y_train, groups, feature_cols, n_splits=5):
    """
    5-fold cross-validation ile LightGBM eğit.
    """
    oof_preds = np.zeros(len(X_train))  # Out-of-fold tahminler
    models = []
    
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train, groups=groups)
    ):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr[feature_cols], label=y_tr)
        dval = lgb.Dataset(X_val[feature_cols], label=y_val)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1,
            'random_state': 42,
        }
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        val_pred = model.predict(X_val[feature_cols])
        oof_preds[val_idx] = val_pred
        models.append(model)
        
        # Threshold 0.5 ile F1
        f1 = f1_score(y_val, (val_pred > 0.5).astype(int), average='macro')
        print(f"Fold {fold+1} Macro-F1: {f1:.4f}")
    
    return models, oof_preds

# Kullanım
# models, oof_preds = train_lgbm(train_features, train_labels, feature_cols)
```

### 6.2 Threshold Optimizasyonu

```python
def find_best_threshold(y_true, y_pred_proba):
    """
    Macro-F1'i maksimize eden threshold'u bul.
    Önemli: 0.5 her zaman optimal değil!
    """
    best_threshold = 0.5
    best_f1 = 0
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for thresh in thresholds:
        preds = (y_pred_proba > thresh).astype(int)
        f1 = f1_score(y_true, preds, average='macro')
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"En iyi threshold: {best_threshold:.2f}")
    print(f"En iyi Macro-F1: {best_f1:.4f}")
    return best_threshold

# Kullanım
# best_thresh = find_best_threshold(train_labels, oof_preds)
```

### 6.3 Feature Importance — Hangi feature'lar işe yarıyor?

```python
import matplotlib.pyplot as plt

def plot_feature_importance(models, feature_cols):
    importance = np.zeros(len(feature_cols))
    for model in models:
        importance += model.feature_importance(importance_type='gain')
    importance /= len(models)
    
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['feature'][:15], feat_imp['importance'][:15])
    plt.title('Feature Importance (Top 15)')
    plt.tight_layout()
    plt.show()
    
    return feat_imp
```

> 💡 **Ne öğrenirsiniz?** Hangi feature'ların skoru artırdığını görürsünüz. Önemsiz feature'ları atar, önemlilere odaklanırsınız.

---

## 7. Değerlendirme — Macro-F1 Nedir?

### F1 Score nedir?

```
Precision = Doğru pozitif / Tüm pozitif tahminler
Recall    = Doğru pozitif / Gerçekte tüm pozitifler
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

### Macro-F1 nedir?

Her sınıf (0 ve 1) için F1 ayrı hesaplanır, sonra ortalaması alınır.

```python
from sklearn.metrics import f1_score, classification_report

y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1]

macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"Macro-F1: {macro_f1:.4f}")

# Detaylı rapor
print(classification_report(y_true, y_pred))
```

### Neden önemli?

Eğer sadece "hep 1 tahmin et" yaparsanız recall=1 ama precision=düşük. Macro-F1 sizi hem 0'ları hem 1'leri doğru tahmin etmeye zorlar.

### Dikkat: Class imbalance!

Submission çiftlerinde 0'lar ve 1'ler eşit değil — genellikle 0 çok fazla. Bu yüzden negatif oranı önemli. Çok fazla negatif koyarsanız model "hep 0 de" öğrenir.

---

## 8. Embedding Modelleri

### Embedding nedir?

Bir metni sayı dizisine (vektöre) çevirme işlemi. "siyah spor ayakkabı" → [0.23, -0.15, 0.89, ...] (384 boyutlu vektör)

İki benzer metin birbirine yakın vektörlere sahip olur. Cosine similarity ile benzerlikleri ölçebilirsiniz.

### Türkçe için hangi model?

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Kaggle'da çalışan, Türkçe destekli model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# Alternatif (daha büyük, daha iyi ama yavaş):
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Item metinleri oluştur
def create_item_text(row):
    parts = []
    if pd.notna(row['title']): parts.append(row['title'])
    if pd.notna(row['category']): parts.append(row['category'].replace('/', ' '))
    if pd.notna(row['brand']): parts.append(row['brand'])
    return ' '.join(parts)

items['item_text'] = items.apply(create_item_text, axis=1)

# Embedding hesapla (bir kez yap, kaydet)
print("Item embedding'leri hesaplanıyor...")
item_embeddings = model.encode(
    items['item_text'].tolist(),
    batch_size=512,
    show_progress_bar=True,
    device='cuda'  # GPU varsa
)

# Kaydet (bir daha hesaplamak zorunda kalmayın)
np.save('item_embeddings.npy', item_embeddings)
# Yüklemek için: item_embeddings = np.load('item_embeddings.npy')
```

### Cosine Similarity feature'ı

```python
from sklearn.metrics.pairwise import cosine_similarity

# Term embedding'leri
query_embeddings = model.encode(terms['query'].tolist(), batch_size=512)

# Her çift için similarity hesapla
def get_embedding_similarity(pairs_df, item_embeddings, query_embeddings, 
                              items, terms):
    item_id_to_idx = {id_: i for i, id_ in enumerate(items['item_id'])}
    term_id_to_idx = {id_: i for i, id_ in enumerate(terms['term_id'])}
    
    similarities = []
    for _, row in pairs_df.iterrows():
        item_idx = item_id_to_idx.get(row['item_id'])
        term_idx = term_id_to_idx.get(row['term_id'])
        
        if item_idx is None or term_idx is None:
            similarities.append(0.0)
            continue
        
        item_vec = item_embeddings[item_idx].reshape(1, -1)
        query_vec = query_embeddings[term_idx].reshape(1, -1)
        sim = cosine_similarity(item_vec, query_vec)[0][0]
        similarities.append(float(sim))
    
    return similarities
```

> ⚠️ **Dikkat:** Embedding hesaplama pahalı! 963K item için birkaç saat sürebilir. Bir kez hesaplayın, `npy` dosyası olarak kaydedin.

---

## 9. Submission Süreci

### Submission nasıl yapılır?

```python
from pipeline.inference import load_threshold

# 1. Submission pairs'e feature ekle
sub_features = build_features(submission, items, terms)
# + embedding similarity ekle
# + TF-IDF cosine ekle

# 2. Model ile tahmin yap
sub_preds_proba = np.mean([
    model.predict(sub_features[feature_cols]) for model in models
], axis=0)

# 3. En iyi threshold ile binary'e çevir
best_thresh = load_threshold()  # grouped OOF eğitim artifact'ından
sub_preds = (sub_preds_proba > best_thresh).astype(int)

# 4. Submission dosyası oluştur
submission_df = pd.DataFrame({
    'id': submission['id'],
    'prediction': sub_preds
})

print(f"Tahmin dağılımı: {sub_preds.mean():.2%} pozitif")
submission_df.to_csv('submission.csv', index=False)
```

### Submission öncesi kontroller

```python
# 1. Doğru format mı?
sample_sub = pd.read_csv('sample_submission.csv')
assert list(submission_df.columns) == list(sample_sub.columns), "Kolon adları yanlış!"

# 2. Tüm ID'ler var mı?
assert len(submission_df) == len(submission), "Satır sayısı yanlış!"
assert submission_df['id'].equals(submission['id']), "ID'ler eşleşmiyor!"

# 3. Değerler geçerli mi?
assert submission_df['prediction'].isin([0, 1]).all(), "0 veya 1 dışı değer var!"

print("✓ Submission hazır!")
```

---

## 10. Ekip İş Bölümü

### Kim ne yapacak?

#### Kişi A — Feature Lead

- `01_eda.ipynb` → veri keşfi
- Token overlap, category parsing feature'ları
- Threshold optimizasyonu
- Feature importance analizi

#### Kişi B — Data Engineer

- Negatif örnekleme pipeline'ı (random + BM25 hard)
- Attributes parse etme (renk, materyal, boyut)
- Veri birleştirme ve kaydetme

#### Kişi C — ML Model

- LightGBM / XGBoost eğitimi
- Cross-validation kurulumu
- Ensemble stratejisi

#### Kişi D — Deep Learning

- Embedding modeli kurma ve kaydetme
- (Zaman kalırsa) CrossEncoder fine-tuning

### Günlük sync — 15 dakika

Her gün aynı saatte (örn. akşam 21:00) toplanın:

1. Dün ne yaptım? (1-2 cümle)
2. Bugün ne yapacağım?
3. Bir şey beni engelliyor mu?

### Git workflow

```bash
# Her kişi kendi branch'ında:
git checkout -b feature/negative-mining

# Çalışınca main'e merge:
git add .
git commit -m "BM25 hard negative mining eklendi"
git push origin feature/negative-mining
# → Pull request aç
```

---

## 11. Nereden Öğrenilir?

### Pandas (veri işleme)

- Resmi docs: https://pandas.pydata.org/docs/getting_started/intro_tutorials/
- 10 dakika pandas: arama kutusuna "10 minutes to pandas" yazın
- Türkçe YouTube: "pandas tutorial türkçe" araması

### Scikit-learn (ML temel)

- https://scikit-learn.org/stable/getting_started.html
- Özellikle: Classification, Feature extraction bölümleri

### LightGBM

- https://lightgbm.readthedocs.io/en/stable/Quick-Start.html
- Kaggle'da onlarca örnek notebook var

### Sentence Transformers (Embedding)

- https://www.sbert.net/docs/quickstart.html
- Türkçe destekli modeller listesi: https://www.sbert.net/docs/pretrained_models.html

### Kaggle'ı öğrenmek

- https://www.kaggle.com/learn/intro-to-machine-learning (ücretsiz kurs, ~3 saat)
- https://www.kaggle.com/learn/intermediate-machine-learning
- Competition'daki diğer public notebook'ları okuyun

### Macro-F1 anlama

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- "macro average f1 score" araması yapın

---

## Hızlı Referans — Sık Karşılaşılan Hatalar

|Hata|Nedeni|Çözüm|
|---|---|---|
|`KeyError: 'item_id'`|Merge sonrası kolon kayboldu|`.merge()` sonrası `.columns` kontrol et|
|Çok düşük F1 (<%50)|Negatif oranı çok yüksek|Negatif/pozitif oranını düşür|
|Çok yavaş kod|Satır satır döngü|`.apply()` veya vektörizasyon kullan|
|GPU bellek hatası|Batch size çok büyük|`batch_size=128` ile dene|
|Submission format hatası|Kolon adı yanlış|`sample_submission.csv` ile karşılaştır|

---

## Hatırlatma: Macro-F1 Formülü

```
Sınıf 0 için: F1_0 = 2 × P0 × R0 / (P0 + R0)
Sınıf 1 için: F1_1 = 2 × P1 × R1 / (P1 + R1)
Macro-F1    = (F1_0 + F1_1) / 2
```

Her iki sınıfı da iyi tahmin etmek zorundayısınız. Sadece "hep 1 tahmin et" veya "hep 0 tahmin et" işe yaramaz.

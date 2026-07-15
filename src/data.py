"""
src/data.py
===========
G.G.A Takımı — Veri Yükleme ve Birleştirme (Pipeline) Modülü

Bu modül, büyük boyutlu CSV dosyalarını (terms.csv, items.csv, training_pairs.csv)
bilgisayarın RAM'ini (belleğini) taşırmayacak şekilde "bellek dostu" (memory efficient) 
olarak yüklemek ve birleştirmek için kullanılır.
"""

import os
import pandas as pd


TERM_COLUMNS = ["term_id", "query"]
ITEM_COLUMNS = [
    "item_id",
    "title",
    "category",
    "brand",
    "gender",
    "age_group",
    "attributes",
]
TRAIN_PAIR_COLUMNS = ["id", "term_id", "item_id", "label"]
SUBMISSION_PAIR_COLUMNS = ["id", "term_id", "item_id"]


def _validate_frame(df, expected_columns, id_column, source):
    actual_columns = df.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            f"{source} columns must be {expected_columns}, got {actual_columns}"
        )
    if df.empty:
        raise ValueError(f"{source} must not be empty")
    if df[id_column].isna().any():
        raise ValueError(f"{source} contains null {id_column} values")
    if df[id_column].duplicated().any():
        raise ValueError(f"{source} contains duplicate {id_column} values")
    return df


def _ensure_real_csv(file_path):
    """CSV dosyasının gerçek veri dosyası olduğunu doğrular.

    Bazı repo'larda büyük dosyalar Git LFS pointer dosyası olarak saklanır.
    Bu durumda pandas okumaya çalışsa da hatalı sonuç verir. Bu fonksiyon,
    böyle bir durum tespit edilirse kullanıcıya net bir rehber verir.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")

    with open(file_path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{file_path} gerçek CSV değil; Git LFS pointer dosyası olarak saklanmış. "
            "Lütfen `git lfs install` ve `git lfs pull` çalıştırın veya veri dosyalarını "
            "gerçek CSV formatında repo'ya çekin."
        )


def load_terms(file_path):
    """
    Sorgu terimlerini (terms.csv) belleği optimize ederek yükler.
    
    Parametreler:
        file_path (str): terms.csv dosyasının bulunduğu tam yol.
        
    Döndürür:
        pd.DataFrame: Yüklenen veri seti.
    """
    _ensure_real_csv(file_path)
    
    # Veri tiplerini baştan belirliyoruz.
    # Normalde pandas metinleri 'object' olarak yükler ve bu çok bellek harcar.
    # 'string' veri tipi pandas'ın yeni ve daha optimize metin veri tipidir.
    dtypes = {
        'term_id': 'string',
        'query': 'string'
    }
    
    # CSV'yi okurken veri tiplerini zorunlu kılıyoruz
    df = pd.read_csv(file_path, dtype=dtypes)
    _validate_frame(df, TERM_COLUMNS, "term_id", os.path.basename(file_path))
    if df["query"].isna().any():
        raise ValueError(f"{os.path.basename(file_path)} contains null query values")
    return df


def load_items(file_path):
    """
    Ürün kataloğunu (items.csv) belleği optimize ederek yükler.
    Özellikle az sayıda benzersiz değer içeren kolonları (gender, age_group) 
    'category' tipine çevirerek inanılmaz bir bellek tasarrufu sağlar.
    
    Parametreler:
        file_path (str): items.csv dosyasının bulunduğu yol.
    """
    _ensure_real_csv(file_path)
    
    # 'category' tipi: Eğer bir sütunda çok fazla tekrar eden değer varsa 
    # (örneğin cinsiyet: sadece kadın, erkek, unisex, unknown) bunları metin olarak
    # saklamak yerine arkada sayısal kodlarla tutar ve RAM kullanımını çok düşürür.
    dtypes = {
        'item_id': 'string',
        'title': 'string',
        'category': 'category',  # Yüzlerce kategori olsa da string'den daha iyidir
        'brand': 'string',
        'gender': 'category',    # Sadece 4 farklı değer alıyor
        'age_group': 'category', # Sadece birkaç farklı değer alıyor
        'attributes': 'string'
    }
    df = pd.read_csv(file_path, dtype=dtypes)
    _validate_frame(df, ITEM_COLUMNS, "item_id", os.path.basename(file_path))
    if df["title"].isna().any() or df["category"].isna().any():
        raise ValueError(
            f"{os.path.basename(file_path)} contains null title or category values"
        )
    
    # brand (marka) alanında bazen boş (NaN) değerler olabiliyor.
    # Bunları modelleme sırasında hata almamak için boş string ('') ile dolduruyoruz.
    df['brand'] = df['brand'].fillna('')
    return df


def merge_pairs(pairs_path, terms_df, items_df, is_train=True):
    """
    Eşleşme çiftlerini (training_pairs veya test_pairs) sorgular (terms) ve 
    ürünler (items) ile birleştirerek modelin anlayacağı devasa bir tablo oluşturur.
    
    Parametreler:
        pairs_path (str): Çiftlerin bulunduğu CSV dosyasının yolu.
        terms_df (DataFrame): load_terms() ile yüklenmiş sorgular.
        items_df (DataFrame): load_items() ile yüklenmiş ürünler.
        is_train (bool): Eğer bu eğitim verisi ise 'label' kolonu içerir.
    """
    _ensure_real_csv(pairs_path)
    
    # Önce baz ID'leri string olarak tanımlıyoruz
    dtypes = {
        'id': 'string',
        'term_id': 'string',
        'item_id': 'string'
    }
    
    # Eğer bu eğitim verisi ise işin içine 'label' (0 veya 1) girer.
    if is_train:
        # Pandas varsayılan olarak sayıları 64-bit int (int64) yapar.
        # Bizim etiketlerimiz sadece 0 ve 1 olduğu için 8-bit int (int8) yeterlidir.
        # Bu küçük değişiklik milyonlarca satırlık veride GB'larca RAM kurtarır!
        dtypes['label'] = 'int8' 
        
    pairs_df = pd.read_csv(pairs_path, dtype=dtypes)
    expected_columns = TRAIN_PAIR_COLUMNS if is_train else SUBMISSION_PAIR_COLUMNS
    if pairs_df.columns.tolist() != expected_columns:
        raise ValueError(
            f"{os.path.basename(pairs_path)} columns must be {expected_columns}, "
            f"got {pairs_df.columns.tolist()}"
        )
    if pairs_df.empty or pairs_df[expected_columns].isna().any().any():
        raise ValueError(f"{os.path.basename(pairs_path)} contains no rows or null values")
    if pairs_df["id"].duplicated().any():
        raise ValueError(f"{os.path.basename(pairs_path)} contains duplicate id values")
    if pairs_df.duplicated(["term_id", "item_id"]).any():
        raise ValueError(
            f"{os.path.basename(pairs_path)} contains duplicate term-item pairs"
        )
    if is_train and not pairs_df["label"].isin([0, 1]).all():
        raise ValueError(f"{os.path.basename(pairs_path)} contains non-binary labels")
    
    # Adım 1: training_pairs içindeki 'term_id' ile terms.csv içindeki 'term_id'yi eşleştir.
    # 'left' merge: training_pairs'deki tüm satırları koru, terms_df'den eşleşenleri yanına ekle.
    # validate='many_to_one': pairs'te bir term_id çok kez geçebilir ama terms_df'de
    # TEK olmalı. terms_df'de tekrar varsa (satır patlaması riski) sessizce geçmek
    # yerine burada hata fırlatılır.
    merged = pd.merge(pairs_df, terms_df, on='term_id', how='left', validate='many_to_one')

    # Adım 2: Aynı işlemi ürünler için yap. 'item_id' kolonunu kullanarak ürün detaylarını yanına çek.
    merged = pd.merge(merged, items_df, on='item_id', how='left', validate='many_to_one')

    if merged[["query", "title"]].isna().any().any():
        raise ValueError(
            f"{os.path.basename(pairs_path)} contains unresolved term_id or item_id values"
        )
    
    # Geriye tüm bilgilerin (query, title, brand, category, label) yan yana olduğu tablo döner
    return merged

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

def load_terms(file_path):
    """
    Sorgu terimlerini (terms.csv) belleği optimize ederek yükler.
    
    Parametreler:
        file_path (str): terms.csv dosyasının bulunduğu tam yol.
        
    Döndürür:
        pd.DataFrame: Yüklenen veri seti.
    """
    # Dosyanın var olup olmadığını kontrol et, yoksa hata fırlat
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sorgu dosyası bulunamadı: {file_path}")
        
    # Veri tiplerini baştan belirliyoruz.
    # Normalde pandas metinleri 'object' olarak yükler ve bu çok bellek harcar.
    # 'string' veri tipi pandas'ın yeni ve daha optimize metin veri tipidir.
    dtypes = {
        'term_id': 'string',
        'query': 'string'
    }
    
    # CSV'yi okurken veri tiplerini zorunlu kılıyoruz
    df = pd.read_csv(file_path, dtype=dtypes)
    return df

def load_items(file_path):
    """
    Ürün kataloğunu (items.csv) belleği optimize ederek yükler.
    Özellikle az sayıda benzersiz değer içeren kolonları (gender, age_group) 
    'category' tipine çevirerek inanılmaz bir bellek tasarrufu sağlar.
    
    Parametreler:
        file_path (str): items.csv dosyasının bulunduğu yol.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ürün dosyası bulunamadı: {file_path}")
        
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
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Eşleşme dosyası bulunamadı: {pairs_path}")
        
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
    
    # Adım 1: training_pairs içindeki 'term_id' ile terms.csv içindeki 'term_id'yi eşleştir.
    # 'left' merge: training_pairs'deki tüm satırları koru, terms_df'den eşleşenleri yanına ekle.
    merged = pd.merge(pairs_df, terms_df, on='term_id', how='left')
    
    # Adım 2: Aynı işlemi ürünler için yap. 'item_id' kolonunu kullanarak ürün detaylarını yanına çek.
    merged = pd.merge(merged, items_df, on='item_id', how='left')
    
    # Geriye tüm bilgilerin (query, title, brand, category, label) yan yana olduğu tablo döner
    return merged

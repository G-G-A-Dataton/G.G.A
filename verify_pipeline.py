"""
verify_pipeline.py
==================
G.G.A Takımı — Veri Hattı (Pipeline) Doğrulama Scripti

Bu script, src/data.py içinde yazdığımız veri yükleme ve birleştirme (merge)
fonksiyonlarının doğru çalışıp çalışmadığını hızlıca test etmek için yazılmıştır.
"""

import sys
import os

# Proje kök dizinini Python yoluna ekle.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.data import load_terms, load_items, merge_pairs

# Veri setlerinin bulunduğu klasörün yolu.
data_dir = os.path.join(PROJECT_ROOT, "datasets")
terms_path = os.path.join(data_dir, "terms.csv")
items_path = os.path.join(data_dir, "items.csv")
train_pairs_path = os.path.join(data_dir, "training_pairs.csv")

# 1. ADIM: Dosyalar gerçekten var mı diye kontrol et
print("Checking files existence...")
print("terms.csv:", os.path.exists(terms_path))
print("items.csv:", os.path.exists(items_path))
print("training_pairs.csv:", os.path.exists(train_pairs_path))

# 2. ADIM: Sorgu terimlerini (terms.csv) yükle ve test et
print("\nTesting load_terms...")
terms_df = load_terms(terms_path)
print("Terms Loaded! Shape:", terms_df.shape)  # Kaç satır/sütun var?
print("dtypes:\n", terms_df.dtypes)            # Veri tipleri (string vs) doğru ayarlanmış mı?

# 3. ADIM: Ürünleri (items.csv) yükle ve test et
print("\nTesting load_items...")
items_df = load_items(items_path)
print("Items Loaded! Shape:", items_df.shape)
print("dtypes:\n", items_df.dtypes)            # 'category' optimizasyonları çalışmış mı?

# 4. ADIM: Her şeyi birleştir ve nihai devasa tabloyu oluştur
print("\nTesting merge_pairs...")
merged_df = merge_pairs(train_pairs_path, terms_df, items_df, is_train=True)
print("Merged Train Shape:", merged_df.shape)  # Birleştirme sonrası satır sayısı aynı kalmalı
print("Merged Train Head:\n", merged_df.head(2)) # İlk iki satıra göz atalım
print("Null values in merged:\n", merged_df.isnull().sum()) # Eşleşmeyen/boş kalan veri var mı?

print("\nPipeline verified successfully!")


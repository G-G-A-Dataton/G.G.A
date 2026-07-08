"""
notebooks/01_veri_kalite_mert.py
================================
G.G.A Takımı — Veri Kalite Kontrolü (Gün 2 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (2 Temmuz görevi)

Neden veri kalitesi kontrol edilmeli?
  training_pairs.csv, terms.csv ve items.csv birleştirilip modele
  verilmeden önce sessiz veri sorunları (tekrar eden çift, geçersiz
  label, referans tablolarda eksik/tekrarlı id, kataloğun eksik
  alanları) fark edilmeden modele girerse yanıltıcı sonuç üretir.

Bu script şunları kontrol eder:
  K1. training_pairs.csv'de tekrar eden (term_id, item_id) çifti var mı
  K2. label kolonu gerçekten sadece {0, 1} mi
  K3/K4. pairs'teki term_id/item_id, terms.csv/items.csv'de karşılığı var mı
  K5. terms/items kendi id'lerinde tekrar var mı (merge'de satır patlaması riski)
  K6. items kolonlarında "unknown" oranı ne kadar
"""

from pathlib import Path

import pandas as pd

# Dosya Yolu
DATA = Path(__file__).resolve().parents[1] / "datasets"



# Dosyayı yükleme

terms = pd.read_csv(DATA / "terms.csv", dtype=str)

# Atribut kolonu büyük olduğu için sadece gerekli kolonları yükledik.
items = pd.read_csv(
    DATA / "items.csv",
    usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    dtype=str,
)

pairs = pd.read_csv(DATA / "training_pairs.csv", dtype=str)
# label'ı dtype=str yüzünden "1" (metin) okuduk; sayıya çeviriyoruz.
pairs["label"] = pairs["label"].astype("int8")

print(f"terms : {len(terms):,} satir")
print(f"items : {len(items):,} satir")
print(f"pairs : {len(pairs):,} satir")

# training_pairs.csv dosyasında tekrarlanan (term_id, item_id) çifti var mı?
dup_pairs = pairs.duplicated(subset=["term_id", "item_id"]).sum()
print(f"K1 | Tekrarlanan (term_id, item_id) çifti : {dup_pairs}")

# training_pairs.csv dosyasındaki label kolonunda gerçekten sadece 0 ve 1 var mı?
print(f"K2 | label'daki benzersiz değerler : {sorted(pairs['label'].unique())}")

# training_pairs.csv dosyasındaki her term_id, terms.csv dosyasında var mı?
term_ids = set(terms["term_id"])
missing_terms = (~pairs["term_id"].isin(term_ids)).sum()
print(f"K3 | terms.csv'de bulunamayan term_id     : {missing_terms}")

# training_pairs.csv dosyasındaki her item_id, items.csv dosyasında var mı?
item_ids = set(items["item_id"])
missing_items = (~pairs["item_id"].isin(item_ids)).sum()
print(f"K4 | items.csv'de bulunamayan item_id     : {missing_items}")

# training_pairs.csv dosyasındaki her term_id, terms.csv dosyasında tekrarlanmıyor mu?
dup_term_ids = terms["term_id"].duplicated().sum()
dup_item_ids = items["item_id"].duplicated().sum()
print(f"K5 | terms içinde tekrarlanan term_id     : {dup_term_ids}")
print(f"K5 | items içinde tekrarlanan item_id     : {dup_item_ids}")

# Kolon bazında "unknown" değerlerin sayısı
for col in items.columns:
    print()
    print(f"K6 | {col} : {(items[col] == 'unknown').sum()}")
    print(f"K6 | {col} : {(items[col] == 'unknown').sum() / len(items) * 100:.2f}%")  

print()
print(f"Kaç farkli term eğitimde kullanılmış     : {pairs['term_id'].nunique():,} / {len(terms):,}")
print(f"Kaç farkli item eğitimde kullanılmış     : {pairs['item_id'].nunique():,} / {len(items):,}")
print(f"Term başina ortalama pozitif çift        : {len(pairs) / pairs['term_id'].nunique():.1f}")
print()
print("Eksik değer sayıları (items):")
print(items.isna().sum().to_string())

print(items[items["brand"].isna()])




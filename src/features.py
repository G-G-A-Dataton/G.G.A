"""
src/features.py
===============
G.G.A Takımı — Feature Engineering (Özellik Mühendisliği) Modülü

Ahmet Emin Işın tarafından hazırlanmıştır. (3 Temmuz görevi)

Bu modül, her (sorgu, ürün) çifti için modelin öğrenebileceği sayısal
özellikler (feature) üretir. Metin karşılaştırmalarına dayalı basit ama
güçlü overlap (örtüşme) yöntemleri kullanır.

Üretilen feature'lar:
  - query_title_overlap    : Sorgu ile ürün başlığı arasındaki kelime örtüşmesi
  - query_category_overlap : Sorgu ile ürün kategorisi arasındaki kelime örtüşmesi
  - query_brand_match      : Sorguda marka adı tam olarak geçiyor mu?
  - query_cat_l1_overlap   : Sorgu ile L1 kategori arasındaki örtüşme
  - title_len              : Ürün başlığının uzunluğu (kısa başlıklar daha az bilgi içerir)
  - query_len              : Sorgu metninin uzunluğu
  - gender_match           : Sorgu ile ürün cinsiyeti uyumlu mu?
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temel Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text):
    """
    Metni küçük harfe çevirip boşluklardan kelimelere ayırır.

    Örnek: "Adidas Erkek Spor Ayakkabı" -> {"adidas", "erkek", "spor", "ayakkabı"}
    """
    if not text or not isinstance(text, str):
        return set()
    return set(text.lower().split())


def jaccard_overlap(text1, text2):
    """
    İki metin arasındaki Jaccard benzerliğini hesaplar.

    Jaccard = (Kesişim) / (Birleşim)
    Yani: her iki metinde de geçen kelimeler / en az birinde geçen kelimeler

    Değer aralığı: 0.0 (hiç örtüşme yok) ile 1.0 (tamamen aynı) arası.

    Örnek:
      text1 = "siyah erkek spor"  -> {"siyah", "erkek", "spor"}
      text2 = "erkek spor bot"    -> {"erkek", "spor", "bot"}
      kesişim = {"erkek", "spor"} = 2
      birleşim = {"siyah", "erkek", "spor", "bot"} = 4
      jaccard = 2/4 = 0.5
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    # Her iki metin de boşsa benzerlik sıfır
    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2   # Her iki sette de olanlar
    union        = tokens1 | tokens2   # En az birinde olanlar
    return len(intersection) / len(union)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tekil Feature Hesaplayıcılar
# ─────────────────────────────────────────────────────────────────────────────

def compute_query_title_overlap(query, title):
    """
    Sorgu ile ürün başlığı arasındaki Jaccard benzerliğini döndürür.

    Bu genellikle en güçlü sinyaldir çünkü kullanıcı ne arıyorsa
    ürün başlığında da o kelimeler geçiyorsa çok alakalı demektir.

    Örnek:
      query = "nike erkek spor ayakkabı"
      title = "Nike Air Max Erkek Spor Ayakkabı Siyah"
      → Yüksek örtüşme → Büyük ihtimalle label=1
    """
    return jaccard_overlap(query, title)


def compute_query_category_overlap(query, category):
    """
    Sorgu ile ürün kategorisi arasındaki Jaccard benzerliğini döndürür.

    Kategori genellikle hiyerarşik yapıdadır: "ayakkabı/spor ayakkabı/sneaker"
    Bu fonksiyon tüm hiyerarşiyi tek bir metin gibi ele alır.

    Örnek:
      query    = "spor ayakkabı"
      category = "ayakkabı/spor ayakkabı/sneaker"
      → "spor", "ayakkabı" her ikisinde de var → yüksek örtüşme
    """
    # Kategori patikasındaki "/" ayırıcısını boşlukla değiştirerek
    # hiyerarşiyi düz metin haline getiriyoruz
    if isinstance(category, str):
        category = category.replace("/", " ")
    return jaccard_overlap(query, category)


def compute_query_brand_match(query, brand):
    """
    Sorgu metninde ürünün markası tam olarak geçiyor mu? (0 veya 1)

    Marka eşleşmesi çok güçlü bir sinyaldir. Kullanıcı doğrudan marka
    adı yazıyorsa o markanın ürününü aradığı çok nettir.

    Örnek:
      query = "adidas erkek spor ayakkabı" ve brand = "adidas" → 1
      query = "erkek spor ayakkabı"        ve brand = "adidas" → 0
    """
    if not isinstance(query, str) or not isinstance(brand, str):
        return 0
    # Boş marka bilgisi olan ürünler eşleşemez
    if not brand.strip():
        return 0
    # Marka adı sorgu içinde tam kelime olarak geçiyor mu?
    return 1 if brand.lower() in query.lower() else 0


def compute_query_cat_l1_overlap(query, category):
    """
    Sorgu ile kategorinin yalnızca en üst seviyesi (L1) arasındaki örtüşmeyi hesaplar.

    Örnek:
      category = "ayakkabı/spor ayakkabı/sneaker" → cat_l1 = "ayakkabı"
      query    = "kadın ayakkabı"
      → "ayakkabı" her ikisinde de var → yüksek sinyal
    """
    if not isinstance(category, str):
        return 0.0
    # Kategorinin sadece ilk bölümünü alıyoruz (en genel seviye)
    cat_l1 = category.split("/")[0].strip()
    return jaccard_overlap(query, cat_l1)


def compute_gender_match(query, gender):
    """
    Sorgu metni ile ürünün cinsiyet bilgisini karşılaştırır.

    Döndürülen değerler:
       1  → Uyumlu (örn. sorguda "kadın" var, ürün de kadın)
      -1  → Çelişkili (örn. sorguda "erkek" var, ürün kadın — büyük ihtimalle label=0)
       0  → Belirsiz (sorguda cinsiyet yok ya da ürün "unknown"/"unisex")

    Bu feature özellikle negatif örnekleri ayırt etmede güçlüdür.
    """
    if not isinstance(query, str):
        return 0

    query_lower  = query.lower()
    gender_lower = str(gender).lower() if pd.notna(gender) else "unknown"

    # Sorgu metninde hangi cinsiyet sinyali var?
    query_has_kadin = any(k in query_lower for k in ["kadın", "bayan", "women", "kız", "girl"])
    query_has_erkek = any(k in query_lower for k in ["erkek", "bay", "men", "boy"])

    # Ürünün etiketi ne?
    item_kadin  = "kadın" in gender_lower
    item_erkek  = "erkek" in gender_lower
    item_belli  = gender_lower not in ("unknown", "unisex")

    # Değerlendirme mantığı
    if query_has_kadin and item_kadin:
        return 1   # Sorgu kadın, ürün kadın → uyumlu
    if query_has_erkek and item_erkek:
        return 1   # Sorgu erkek, ürün erkek → uyumlu
    if query_has_kadin and item_erkek and item_belli:
        return -1  # Sorgu kadın, ürün erkek → çelişki!
    if query_has_erkek and item_kadin and item_belli:
        return -1  # Sorgu erkek, ürün kadın → çelişki!
    return 0       # Belirsiz durum


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ana Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df):
    """
    Birleştirilmiş (merge edilmiş) bir DataFrame'e tüm feature'ları hesaplayıp ekler.

    Girdi DataFrame'in şu kolonları içermesi gerekir:
      - query    : Arama sorgusu metni
      - title    : Ürün başlığı
      - category : Ürün kategorisi (hiyerarşik: "l1/l2/l3")
      - brand    : Marka adı
      - gender   : Ürün cinsiyeti

    Parametreler
    ----------
    df : pd.DataFrame
        merge_pairs() ile üretilmiş birleştirilmiş veri seti.

    Döndürür
    -------
    pd.DataFrame
        Orijinal DataFrame'e yeni feature kolonları eklenmiş hali.
    """
    # Değiştirmemek için kopya alıyoruz
    out = df.copy()

    print("[features] query_title_overlap hesaplanıyor...")
    out["query_title_overlap"] = out.apply(
        lambda r: compute_query_title_overlap(r["query"], r["title"]), axis=1
    )

    print("[features] query_category_overlap hesaplanıyor...")
    out["query_category_overlap"] = out.apply(
        lambda r: compute_query_category_overlap(r["query"], r["category"]), axis=1
    )

    print("[features] query_brand_match hesaplanıyor...")
    out["query_brand_match"] = out.apply(
        lambda r: compute_query_brand_match(r["query"], r["brand"]), axis=1
    )

    print("[features] query_cat_l1_overlap hesaplanıyor...")
    out["query_cat_l1_overlap"] = out.apply(
        lambda r: compute_query_cat_l1_overlap(r["query"], r["category"]), axis=1
    )

    print("[features] title_len ve query_len hesaplanıyor...")
    # Kısa ürün başlıkları (örn. sadece model numarası) anlamlı sinyal vermez
    out["title_len"] = out["title"].fillna("").str.len()
    out["query_len"] = out["query"].fillna("").str.len()

    print("[features] gender_match hesaplanıyor...")
    out["gender_match"] = out.apply(
        lambda r: compute_gender_match(r["query"], r.get("gender", "unknown")), axis=1
    )

    print("[features] Tum feature'lar hesaplandi.")
    return out


# Bu modülde üretilen feature kolonlarının tam listesi
# Modeli eğitirken X = df[FEATURE_COLS] olarak kullanılır
FEATURE_COLS = [
    "query_title_overlap",
    "query_category_overlap",
    "query_brand_match",
    "query_cat_l1_overlap",
    "title_len",
    "query_len",
    "gender_match",
]


if __name__ == "__main__":
    # Hızlı birim testi: dummy verilerle feature'ların doğru çalıştığını kontrol eder
    print("=== FEATURES.PY BIRIM TESTI ===")

    test_data = pd.DataFrame([
        # (query,              title,                         category,                brand,   gender, beklenen_label)
        ("adidas erkek koşu",  "Adidas Erkek Koşu Ayakkabısı", "ayakkabı/spor/koşu",   "adidas","erkek", 1),
        ("kadın çanta",        "Erkek Deri Cüzdan",             "aksesuar/cüzdan",       "fossil","erkek", 0),
        ("nike spor ayakkabı", "Nike Air Max",                   "ayakkabı/spor/sneaker", "nike",  "unisex",1),
        ("laptop çantası",     "Siyah Laptop Sırt Çantası",      "aksesuar/çanta/laptop", "",      "unknown",1),
    ], columns=["query", "title", "category", "brand", "gender", "label"])

    result = build_features(test_data)
    print("\nHesaplanan feature'lar:")
    print(result[FEATURE_COLS + ["label"]].to_string())

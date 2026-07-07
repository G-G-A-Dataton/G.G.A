"""
src/features.py
===============
G.G.A Takımı — Feature Engineering (Özellik Mühendisliği) Modülü

Ahmet Emin Işın tarafından hazırlanmıştır.
3 Temmuz: Temel overlap ve cinsiyet feature'ları
4 Temmuz: Yaş grubu uyumu ve demografik çelişki skoru eklendi

Bu modül, her (sorgu, ürün) çifti için modelin öğrenebileceği sayısal
özellikler (feature) üretir. Metin karşılaştırmalarına dayalı basit ama
güçlü overlap (örtüşme) yöntemleri kullanır.

Üretilen feature'lar:
  --- Metin Benzerliği ---
  - query_title_overlap    : Sorgu ile ürün başlığı arasındaki kelime örtüşmesi
  - query_category_overlap : Sorgu ile ürün kategorisinin tamamı arasındaki kelime örtüşmesi
  - query_brand_match      : Sorguda marka adı tam olarak geçiyor mu?
  - query_cat_l1_overlap   : Sorgu ile L1 (en genel) kategori arasındaki örtüşme
  - title_len              : Ürün başlığının uzunluğu
  - query_len              : Sorgu metninin uzunluğu
  --- Kategori Seviyeleri (6 Temmuz) ---
  - query_cat_l2_overlap   : Sorgu ile L2 (orta) kategori arasındaki örtüşme
  - query_cat_l3_overlap   : Sorgu ile L3 (en spesifik) kategori arasındaki örtüşme
  - cat_depth              : Kategori hiyerarşisinin derinliği (kaç seviye var?)
  --- Demografik Özellikler (4 Temmuz) ---
  - gender_match           : Sorgu ile ürün cinsiyeti uyumlu mu? (1=uyumlu, -1=çelişki, 0=belirsiz)
  - age_group_match        : Sorguda yaş grubu sinyali var mı ve ürünle uyuşuyor mu?
  - demographic_conflict   : Cinsiyet veya yaş grubu çelişkisi var mı? (0/1 binary)
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


def compute_age_group_match(query, age_group):
    """
    Sorgu metni ile ürünün yaş grubu bilgisini karşılaştırır.

    Sorguda yaş grubu sinyali (bebek, çocuk, genç, yetişkin) aranır
    ve ürünün age_group etiketiyle karşılaştırılır.

    Döndürülen değerler:
       1  → Uyumlu (örn. sorguda 'bebek' var, ürün de bebek kategorisinde)
      -1  → Çelişkili (örn. sorguda 'bebek' var, ürün 'yetişkin')
       0  → Belirsiz (sorguda yaş sinyali yok ya da ürün 'unknown')
    """
    if not isinstance(query, str):
        return 0

    query_lower    = query.lower()
    age_group_lower = str(age_group).lower() if pd.notna(age_group) else "unknown"

    # Sorgudaki yaş grubu sinyalleri
    # Her anahtar kelime hangi kategoriye işaret ediyor?
    query_has_bebek  = any(k in query_lower for k in ["bebek", "baby", "infant"])
    query_has_cocuk  = any(k in query_lower for k in ["çocuk", "cocuk", "kids", "kid", "child"])
    query_has_genc   = any(k in query_lower for k in ["genç", "genc", "teen", "junior"])
    query_has_yetis  = any(k in query_lower for k in ["yetişkin", "yetiskin", "adult"])

    # Ürünün yaş grubu etiketi
    item_bebek  = "bebek" in age_group_lower
    item_cocuk  = "çocuk" in age_group_lower or "cocuk" in age_group_lower
    item_genc   = "genç" in age_group_lower   or "genc" in age_group_lower
    item_yetis  = "yetişkin" in age_group_lower or "yetiskin" in age_group_lower
    item_belli  = age_group_lower not in ("unknown",)

    # Uyumlu mu, çelişkili mi?
    if query_has_bebek:
        if item_bebek: return 1
        if item_belli and not item_bebek: return -1
    if query_has_cocuk:
        if item_cocuk: return 1
        if item_belli and not item_cocuk and not item_bebek: return -1
    if query_has_genc:
        if item_genc: return 1
        if item_belli and not item_genc: return -1
    if query_has_yetis:
        if item_yetis: return 1
        if item_belli and not item_yetis: return -1

    return 0  # Belirsiz


def compute_demographic_conflict(gender_match_val, age_group_match_val):
    """
    Cinsiyet VEYA yaş grubu çelişkisi var mı? (0: yok, 1: var)

    Bu binary feature modelin 'kesinlikle alakasız' durumları hızlıca
    öğrenmesi için tasarlanmıştır. Çelişki varsa label=0 olma ihtimali
    çok yüksektir.

    Örnek:
      Sorgu: 'erkek çocuk ayakkabı'  + Ürün gender='kadın' → conflict=1
      Sorgu: 'bebek kıyafeti'         + Ürün age_group='yetişkin' → conflict=1
    """
    return 1 if (gender_match_val == -1 or age_group_match_val == -1) else 0


# ───────────────────────────────────────────────────────────────────────────────
# 2b. Kategori Seviye Feature'ları (6 Temmuz)
# ───────────────────────────────────────────────────────────────────────────────

def split_category_levels(category):
    """
    Kategori hiyerarşisini seviyelerine ayırır.

    Trendyol kategorileri '/' ile ayrılan 1-4 seviyeli hiyerarşiler içerir.
      Örnek: "ayakkabı/spor ayakkabı/sneaker"
        L1 (en genel)   : "ayakkabı"
        L2 (orta)       : "spor ayakkabı"
        L3 (en spesifik): "sneaker"

    Seviye yoksa boş string döndürür (NaN değil — fonksiyonlar bunu handle eder).
    """
    if not isinstance(category, str):
        return "", "", "", 0
    parts = [p.strip() for p in category.split("/")]
    depth = len(parts)  # Kaç seviye olduğunu say
    l1 = parts[0] if depth >= 1 else ""
    l2 = parts[1] if depth >= 2 else ""
    l3 = parts[2] if depth >= 3 else ""
    return l1, l2, l3, depth


def compute_query_cat_l2_overlap(query, category):
    """
    Sorgu ile kategorinin ikinci seviyesi (L2) arasındaki Jaccard benzerliğini hesaplar.

    L1 ("ayakkabı") çok genel, L3 ("sneaker") çok spesifik olabilir.
    L2 ("spor ayakkabı") genellikle en anlamlı orta noktadır.

    Örnek:
      category = "ayakkabı/spor ayakkabı/sneaker"  →  L2 = "spor ayakkabı"
      query    = "erkek spor ayakkabı"
      → "spor", "ayakkabı" her iki tarafta da var → yüksek L2 örtüşmesi
    """
    _, l2, _, _ = split_category_levels(category)
    return jaccard_overlap(query, l2)


def compute_query_cat_l3_overlap(query, category):
    """
    Sorgu ile kategorinin üçüncü seviyesi (L3) arasındaki Jaccard benzerliğini hesaplar.

    L3 en spesifik alt-kategoridir. Kullanıcı "sneaker" yazan bir sorgu yaptıysa
    ve ürün de sneaker kategorisindeyse bu çok güçlü bir sinyal.
    Ancak çoğu sorguda bu kadar detay olmaz, dolayısıyla çoğunlukla 0.0 döner.

    Örnek:
      category = "ayakkabı/spor ayakkabı/sneaker"  →  L3 = "sneaker"
      query    = "erkek sneaker"
      → "sneaker" her iki tarafta da var → yüksek L3 örtüşmesi
    """
    _, _, l3, _ = split_category_levels(category)
    return jaccard_overlap(query, l3)


def compute_cat_depth(category):
    """
    Kategori hiyerarşisinin derinliğini (seviye sayısını) döndürür.

    Bu feature dolaylı bir bilgi taşır: derinliği yüksek kategoriler
    genellikle daha spesifik ürünlerdir (daha az belirsizlik).

    Örnekler:
      "ayakkabı"                       → depth=1 (sadece l1)
      "ayakkabı/spor ayakkabı"          → depth=2
      "ayakkabı/spor ayakkabı/sneaker"  → depth=3
    """
    _, _, _, depth = split_category_levels(category)
    return depth

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

    # ── 4 Temmuz: Demografik feature'lar ──────────────────────────────────────
    print("[features] age_group_match hesaplanıyor...")
    out["age_group_match"] = out.apply(
        lambda r: compute_age_group_match(r["query"], r.get("age_group", "unknown")), axis=1
    )

    print("[features] demographic_conflict hesaplanıyor...")
    # gender_match ve age_group_match hesaplandıktan SONRA çalıştırılmalı
    out["demographic_conflict"] = out.apply(
        lambda r: compute_demographic_conflict(r["gender_match"], r["age_group_match"]), axis=1
    )

    # ── 6 Temmuz: Kategori Seviye Feature'ları ────────────────────────────────────
    print("[features] query_cat_l2_overlap hesaplanıyor...")
    out["query_cat_l2_overlap"] = out.apply(
        lambda r: compute_query_cat_l2_overlap(r["query"], r["category"]), axis=1
    )

    print("[features] query_cat_l3_overlap hesaplanıyor...")
    out["query_cat_l3_overlap"] = out.apply(
        lambda r: compute_query_cat_l3_overlap(r["query"], r["category"]), axis=1
    )

    print("[features] cat_depth hesaplanıyor...")
    # Kategori derinliği çok hızlı — vektörel hesaplanabilir
    out["cat_depth"] = out["category"].apply(compute_cat_depth)

    print("[features] Tum feature'lar hesaplandi.")
    return out


# Bu modülde üretilen feature kolonlarının tam listesi
# Modeli eğitirken X = df[FEATURE_COLS] olarak kullanılır
FEATURE_COLS = [
    # Metin benzerliği feature'ları (3 Temmuz)
    "query_title_overlap",
    "query_category_overlap",
    "query_brand_match",
    "query_cat_l1_overlap",
    "title_len",
    "query_len",
    # Demografik feature'lar (4 Temmuz)
    "gender_match",
    "age_group_match",
    "demographic_conflict",
    # Kategori seviye feature'ları (6 Temmuz)
    "query_cat_l2_overlap",
    "query_cat_l3_overlap",
    "cat_depth",
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

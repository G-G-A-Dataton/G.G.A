"""
src/features.py
===============
G.G.A Takımı — Feature Engineering (Özellik Mühendisliği) Modülü

Ahmet Emin Işın tarafından hazırlanmıştır.
3 Temmuz: Temel overlap ve cinsiyet feature'ları
4 Temmuz: Yaş grubu uyumu ve demografik çelişki skoru eklendi
6 Temmuz: Kategori seviye feature'ları (L2/L3/depth)
8 Temmuz: Attributes feature'ları eklendi (renk, beden, materyal)

Bu modül, her (sorgu, ürün) çifti için modelin öğrenebileceği sayısal
özellikler (feature) üretir. Metin karşılaştırmalarına dayalı basit ama
güçlü overlap (örtüşme) yöntemleri kullanır.

Üretilen feature'lar:
  --- Metin Benzerliği ---
  - query_title_overlap    : Sorgu ile ürün başlığı arasındaki Jaccard örtüşmesi
  - query_title_coverage   : Sorgu tokenlerinin başlık tarafından kapsanma oranı
  - query_title_precision  : Başlık tokenlerinin sorguyla örtüşme oranı
  - query_title_phrase     : Sorgu başlıkta tam ifade olarak geçiyor mu?
  - query_category_overlap : Sorgu ile ürün kategorisinin tamamı arasındaki kelime örtüşmesi
  - query_category_coverage: Sorgunun kategori tarafından kapsanma oranı
  - query_model_token_match: Harf-rakam içeren model kodu eşleşmesi
  - query_model_token_conflict: Sorgu ve başlıktaki model kodları çelişiyor mu?
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
  --- Attributes Feature'ları (8 Temmuz) ---
  - query_color_match      : Sorgudaki renk bilgisi ürünün rengiyle uyuşuyor mu? (-1/0/1)
  - query_size_match       : Sorgudaki beden/numara bilgisi ürünün bedeniyle uyuşuyor mu? (-1/0/1)
  - query_material_match   : Sorgudaki materyal bilgisi ürünün materyaliyle uyuşuyor mu? (-1/0/1)
"""

import re

import pandas as pd

from src.attributes import add_attribute_features, ATTRIBUTE_FEATURE_COLS
from src.text_utils import contains_phrase, find_phrase_values, normalize_for_matching


_QUERY_GENDER_KEYWORDS = {
    "kadın": "kadın", "bayan": "kadın", "women": "kadın",
    "kız": "kadın", "girl": "kadın", "erkek": "erkek",
    "bay": "erkek", "men": "erkek", "boy": "erkek",
}
_ITEM_GENDER_KEYWORDS = {"kadın": "kadın", "erkek": "erkek"}
_AGE_KEYWORDS = {
    "bebek": "bebek", "baby": "bebek", "infant": "bebek",
    "çocuk": "çocuk", "cocuk": "çocuk", "kids": "çocuk",
    "kid": "çocuk", "child": "çocuk", "genç": "genç",
    "genc": "genç", "teen": "genç", "junior": "genç",
    "yetişkin": "yetişkin", "yetiskin": "yetişkin", "adult": "yetişkin",
}

FEATURE_SCHEMA_VERSION = 3

_MODEL_TOKEN_RE = re.compile(r"^(?=.*\d)[a-z0-9]{3,}$")


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
    return set(normalize_for_matching(text).split())


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

    return _jaccard_token_sets(tokens1, tokens2)


def _jaccard_token_sets(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def _coverage_token_sets(query_tokens, document_tokens):
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def _precision_token_sets(query_tokens, document_tokens):
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(document_tokens)


def _model_tokens(tokens):
    return {token for token in tokens if _MODEL_TOKEN_RE.match(token)}


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
    return int(contains_phrase(query, brand))


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

    gender_lower = str(gender).lower() if pd.notna(gender) else "unknown"

    # Sorgu metninde hangi cinsiyet sinyali var?
    query_groups = find_phrase_values(query, _QUERY_GENDER_KEYWORDS)
    query_has_kadin = "kadın" in query_groups
    query_has_erkek = "erkek" in query_groups

    # Ürünün etiketi ne?
    item_groups = find_phrase_values(gender_lower, _ITEM_GENDER_KEYWORDS)
    item_kadin = "kadın" in item_groups
    item_erkek = "erkek" in item_groups

    if query_has_kadin == query_has_erkek or item_kadin == item_erkek:
        return 0

    # Değerlendirme mantığı
    return 1 if query_has_kadin == item_kadin else -1


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

    age_group_lower = str(age_group).lower() if pd.notna(age_group) else "unknown"

    # Sorgudaki yaş grubu sinyalleri
    # Her anahtar kelime hangi kategoriye işaret ediyor?
    query_groups = find_phrase_values(query, _AGE_KEYWORDS)

    # Ürünün yaş grubu etiketi
    item_groups = find_phrase_values(age_group_lower, _AGE_KEYWORDS)

    if not query_groups or not item_groups:
        return 0
    return 1 if query_groups & item_groups else -1


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
    parts = [part.strip() for part in category.split("/") if part.strip()]
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

def build_features(df, verbose=True):
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
    required = {"query", "title", "category", "brand"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Feature input is missing required columns: {missing}")
    if "gender" not in out.columns:
        out["gender"] = "unknown"
    if "age_group" not in out.columns:
        out["age_group"] = "unknown"

    queries = out["query"].tolist()
    titles = out["title"].tolist()
    categories = out["category"].tolist()
    brands = out["brand"].tolist()

    query_token_cache = {}
    query_tokens = []
    for query in queries:
        key = query if isinstance(query, str) else None
        if key not in query_token_cache:
            query_token_cache[key] = tokenize(query)
        query_tokens.append(query_token_cache[key])

    title_token_cache = {}
    title_tokens = []
    for title in titles:
        key = title if isinstance(title, str) else None
        if key not in title_token_cache:
            title_token_cache[key] = tokenize(title)
        title_tokens.append(title_token_cache[key])

    category_cache = {}
    category_values = []
    for category in categories:
        key = category if isinstance(category, str) else None
        if key not in category_cache:
            levels = split_category_levels(category)
            flat = category.replace("/", " ") if isinstance(category, str) else ""
            category_cache[key] = (
                tokenize(flat), tokenize(levels[0]), tokenize(levels[1]),
                tokenize(levels[2]), levels[3],
            )
        category_values.append(category_cache[key])

    if verbose:
        print("[features] lexical features hesaplanıyor...")
    out["query_title_overlap"] = [
        _jaccard_token_sets(query_set, title_set)
        for query_set, title_set in zip(query_tokens, title_tokens)
    ]
    out["query_title_coverage"] = [
        _coverage_token_sets(query_set, title_set)
        for query_set, title_set in zip(query_tokens, title_tokens)
    ]
    out["query_title_precision"] = [
        _precision_token_sets(query_set, title_set)
        for query_set, title_set in zip(query_tokens, title_tokens)
    ]
    out["query_title_phrase"] = [
        int(bool(normalize_for_matching(query)) and contains_phrase(title, query))
        for query, title in zip(queries, titles)
    ]

    out["query_category_overlap"] = [
        _jaccard_token_sets(tokens, category_value[0])
        for tokens, category_value in zip(query_tokens, category_values)
    ]
    out["query_category_coverage"] = [
        _coverage_token_sets(tokens, category_value[0])
        for tokens, category_value in zip(query_tokens, category_values)
    ]

    query_model_tokens = [_model_tokens(tokens) for tokens in query_tokens]
    title_model_tokens = [_model_tokens(tokens) for tokens in title_tokens]
    out["query_model_token_match"] = [
        int(bool(query_models & title_models))
        for query_models, title_models in zip(query_model_tokens, title_model_tokens)
    ]
    out["query_model_token_conflict"] = [
        int(bool(query_models) and bool(title_models) and not query_models & title_models)
        for query_models, title_models in zip(query_model_tokens, title_model_tokens)
    ]

    out["query_brand_match"] = [
        compute_query_brand_match(query, brand)
        for query, brand in zip(queries, brands)
    ]

    out["query_cat_l1_overlap"] = [
        _jaccard_token_sets(tokens, category_value[1])
        for tokens, category_value in zip(query_tokens, category_values)
    ]

    # Character and token lengths capture different title verbosity effects.
    out["title_len"] = out["title"].fillna("").str.len()
    out["query_len"] = out["query"].fillna("").str.len()
    out["title_token_count"] = [len(tokens) for tokens in title_tokens]
    out["query_token_count"] = [len(tokens) for tokens in query_tokens]

    if verbose:
        print("[features] demographic features hesaplanıyor...")
    out["gender_match"] = [
        compute_gender_match(query, gender)
        for query, gender in zip(queries, out["gender"])
    ]

    # ── 4 Temmuz: Demografik feature'lar ──────────────────────────────────────
    out["age_group_match"] = [
        compute_age_group_match(query, age_group)
        for query, age_group in zip(queries, out["age_group"])
    ]

    # gender_match ve age_group_match hesaplandıktan SONRA çalıştırılmalı
    out["demographic_conflict"] = (
        (out["gender_match"] == -1) | (out["age_group_match"] == -1)
    ).astype("int8")

    # ── 6 Temmuz: Kategori Seviye Feature'ları ────────────────────────────────────
    # query_cat_l1_overlap tek başına yetersiz kalıyordu:
    # "ayakkabı" çok geneldi — spor, topuklu, terlik hepsini kapsıyor.
    # L2 ve L3 ile daha spesifik eşleşme yapılabilir:
    #   query="spor ayakkabı"  + cat_l2="spor ayakkabı" → güçlü sinyal
    #   query="sneaker"        + cat_l3="sneaker"       → çok güçlü sinyal
    if verbose:
        print("[features] category hierarchy features hesaplanıyor...")
    out["query_cat_l2_overlap"] = [
        _jaccard_token_sets(tokens, category_value[2])
        for tokens, category_value in zip(query_tokens, category_values)
    ]

    out["query_cat_l3_overlap"] = [
        _jaccard_token_sets(tokens, category_value[3])
        for tokens, category_value in zip(query_tokens, category_values)
    ]

    # cat_depth: "ayakkabı" = 1, "ayakkabı/spor" = 2, "ayakkabı/spor/sneaker" = 3
    # Derin kategoriler daha spesifik ürünlere işaret eder — model bunu öğrenebilir
    out["cat_depth"] = [value[4] for value in category_values]

    # ── 8 Temmuz: Attributes Feature'ları ────────────────────────────────────
    # items.csv'deki 'attributes' kolonu renk, beden, materyal gibi yapısal bilgiler içerir.
    # Bu bilgiler title'da yer almayabilir ama sorguda geçiyor olabilir.
    # Sprint 1 raporunda tespit edilmişti: kısa title'lar (örn. "530") TF-IDF için anlamsız.
    # Attributes bu boşluğu kapatır: "siyah 42 deri ayakkabı" sorgusunda tüm üç feature çalışır.
    if verbose:
        print("[features] attribute features hesaplanıyor...")
    out = add_attribute_features(out, verbose=verbose)

    if verbose:
        print("[features] Tum feature'lar hesaplandi.")
    return out


# Bu modülde üretilen feature kolonlarının tam listesi
# Modeli eğitirken X = df[FEATURE_COLS] olarak kullanılır
FEATURE_COLS = [
    # Metin benzerliği feature'ları (3 Temmuz)
    "query_title_overlap",
    "query_title_coverage",
    "query_title_precision",
    "query_title_phrase",
    "query_category_overlap",
    "query_category_coverage",
    "query_model_token_match",
    "query_model_token_conflict",
    "query_brand_match",
    "query_cat_l1_overlap",
    "title_len",
    "query_len",
    "title_token_count",
    "query_token_count",
    # Demografik feature'lar (4 Temmuz)
    "gender_match",
    "age_group_match",
    "demographic_conflict",
    # Kategori seviye feature'ları (6 Temmuz)
    "query_cat_l2_overlap",
    "query_cat_l3_overlap",
    "cat_depth",
    # Attributes feature'ları (8 Temmuz)
    *ATTRIBUTE_FEATURE_COLS,  # query_color_match, query_size_match, query_material_match
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

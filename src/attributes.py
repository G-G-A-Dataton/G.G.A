"""
src/attributes.py
=================
G.G.A Takımı — Attributes Parse ve Feature Üretimi (8 Temmuz Görevi)

Muhammed Köseoğlu ve Ahmet Emin Işın tarafından hazırlanmıştır.
8 Temmuz: Renk, materyal ve beden parse modülü

Neden attributes parse?
  items.csv'deki 'attributes' kolonu çoğunlukla virgülle ayrılmış
  "anahtar: değer" metni, bazı kaynaklarda ise dict/JSON metni içerir.
  Örnek: "renk: siyah, materyal: deri, numara: 42"
  Bu bilgiler başlıkta (title) yer almayabilir ama kullanıcının sorgusunda geçebilir.
  Örnek: "siyah deri erkek ayakkabı 42 numara" → renk, materyal ve beden eşleşmesi
  güçlü bir pozitif sinyal verir.

  Sprint 1 raporunda tespit edilmişti:
  - Kısa title'lar (örn. sadece "530" model kodu) TF-IDF için anlamsız.
  - Attributes ve category bilgisini kullanmak bu boşluğu kapatır.

Üretilen feature'lar:
  - query_color_match    : Sorguda renk bilgisi var ve ürünün rengiyle uyuşuyor mu?
  - query_size_match     : Sorguda beden/numara var ve ürünün bedeniyle uyuşuyor mu?
  - query_material_match : Sorguda materyal var ve ürünün materyaliyle uyuşuyor mu?
"""

import ast
import json
import re

from src.text_utils import find_phrase_values, first_phrase_value


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attributes JSON Parsing
# ─────────────────────────────────────────────────────────────────────────────

_FLAT_ATTRIBUTE_PATTERN = re.compile(
    r"(?:^|,\s*)([^,:]+?)\s*:\s*(.*?)(?=,\s*[^,:]+?\s*:|$)"
)


def _normalize_attribute_dict(attributes):
    return {
        str(key).casefold().strip(): str(value).casefold().strip()
        for key, value in attributes.items()
        if str(key).strip()
    }


def parse_attributes(attributes_str):
    """
    items.csv'deki 'attributes' kolonunu dict'e çevirir.

    Attributes kolonu birkaç farklı formatta gelebilir:
      - "{'Renk': 'Siyah', 'Materyal': 'Deri'}"  (Python dict string)
      - '{"Renk": "Siyah"}'                       (JSON string)
      - "renk: siyah, materyal: deri"              (katalogdaki düz format)
      - NaN / boş string                           (bilgi yok)

    Tüm bu durumları güvenli şekilde ele alır — hata fırlatmaz.

    Döndürür
    -------
    dict
        Anahtar-değer çiftleri. Parse edilemezse boş dict.
    """
    if not isinstance(attributes_str, str) or not attributes_str.strip():
        return {}

    value = attributes_str.strip()
    if value.startswith("{") and value.endswith("}"):
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(value)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(result, dict):
                return _normalize_attribute_dict(result)

    pairs = _FLAT_ATTRIBUTE_PATTERN.findall(value)
    return _normalize_attribute_dict(dict(pairs)) if pairs else {}


def _get_parsed_attribute_value(parsed, *keys):
    for key in keys:
        value = parsed.get(key.casefold().strip())
        if value is not None:
            return value
    return ""


def get_attribute_value(attributes_str, *keys):
    """
    Attributes string'inden belirli bir anahtara karşılık gelen değeri döndürür.

    Birden fazla olası anahtar kabul eder (Türkçe/İngilizce varyasyonlar için).

    Örnek:
      get_attribute_value(attr, "renk", "color", "colour")
      → "siyah"

    Döndürür
    -------
    str
        Bulunan değer (küçük harf). Bulunamazsa "".
    """
    return _get_parsed_attribute_value(parse_attributes(attributes_str), *keys)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Renk Parse
# ─────────────────────────────────────────────────────────────────────────────

# Türkçe ve İngilizce temel renkler + yaygın varyasyonlar
_COLORS_TR = {
    "siyah": ["siyah", "black"],
    "beyaz": ["beyaz", "white", "krem", "ekru", "kırık beyaz"],
    "kırmızı": ["kırmızı", "kirmizi", "red", "bordo", "şarap", "sarap"],
    "mavi": ["mavi", "blue", "lacivert", "indigo", "navy", "deniz mavisi"],
    "yeşil": ["yeşil", "yesil", "green", "haki", "zeytin", "mint"],
    "sarı": ["sarı", "sari", "yellow", "gold", "altın"],
    "turuncu": ["turuncu", "orange"],
    "pembe": ["pembe", "pink", "fuşya", "fusya"],
    "mor": ["mor", "purple", "violet", "lila", "leylak"],
    "gri": ["gri", "gray", "grey", "antrasit", "füme", "fume", "gümüş", "gumus"],
    "kahverengi": ["kahverengi", "kahve", "brown", "taba", "tan", "camel", "bej", "beige"],
    "bej": ["bej", "beige", "kum", "krem"],
}

# Sorguda renk tespiti için kullanılacak düzleştirilmiş set
_ALL_COLOR_KEYWORDS = {}
for canonical, variants in _COLORS_TR.items():
    for v in variants:
        _ALL_COLOR_KEYWORDS[v] = canonical


def _normalize_color(raw):
    if not raw:
        return ""
    raw_normalized = raw.casefold()
    return first_phrase_value(
        raw_normalized, _ALL_COLOR_KEYWORDS, default=raw_normalized
    )


def _parse_color_from_dict(parsed):
    raw = _get_parsed_attribute_value(
        parsed, "renk", "color", "colour", "ana renk", "primary color"
    )
    return _normalize_color(raw)


def parse_color(attributes_str):
    """
    Attributes string'inden renk bilgisini çıkarır.

    Döndürür
    -------
    str
        Normalize edilmiş renk adı (örn. "siyah"). Bulunamazsa "".
    """
    return _parse_color_from_dict(parse_attributes(attributes_str))


def extract_query_colors(query):
    """
    Sorgu metninde geçen renklerin kümesini döndürür.

    Örnek: "siyah beyaz erkek spor ayakkabı" → {"siyah", "beyaz"}
    """
    if not isinstance(query, str):
        return set()
    return find_phrase_values(query, _ALL_COLOR_KEYWORDS)


def compute_color_match(query, attributes_str):
    """
    Sorgu metnindeki renk bilgisi ürünün rengiyle uyuşuyor mu?

    Döndürülen değerler:
       1  → Sorgu ve ürün aynı rengi içeriyor (güçlü pozitif sinyal)
      -1  → Sorguda renk var ama ürünün rengi farklı (negatif sinyal)
       0  → Sorguda renk yok veya ürünün rengi bilinmiyor (belirsiz)

    Örnek:
      query = "siyah erkek spor ayakkabı"
      attributes = "{'Renk': 'Siyah'}"
      → canonical("siyah") == canonical("siyah") → 1

      query = "siyah erkek spor ayakkabı"
      attributes = "{'Renk': 'Beyaz'}"
      → canonical("siyah") != canonical("beyaz") → -1
    """
    query_colors = extract_query_colors(query)

    # Sorguda renk yok → belirsiz
    if not query_colors:
        return 0

    item_color = parse_color(attributes_str)

    # Ürünün rengi bilinmiyor → belirsiz (ürün siyah olabilir ama attributes boş)
    if not item_color:
        return 0

    # Renk uyuşuyor mu?
    if item_color in query_colors:
        return 1

    # Renk çelişiyor
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Beden / Numara Parse
# ─────────────────────────────────────────────────────────────────────────────

# Yaygın beden ifadelerini tanımak için regex pattern'ları
_SIZE_PATTERNS = [
    r"\b(xs|s|m|l|xl|xxl|xxxl|2xl|3xl)\b",              # Standart harf bedenleri
    r"\b(\d{1,2}[.,]?\d?)\s*(numara|no|beden|size)\b",   # "42 numara", "40 beden"
    r"\b(numara|beden|size)\s*:?\s*(\d{1,2}[.,]?\d?)\b", # "numara: 42"
    r"\b(\d{2,3})\s*cm\b",                               # "170 cm" (boy beden için)
]

_SIZE_LETTERS = {"xs", "s", "m", "l", "xl", "xxl", "xxxl", "2xl", "3xl"}


def _normalize_size(raw):
    if not raw:
        return ""
    raw_normalized = raw.casefold().strip()
    if raw_normalized in _SIZE_LETTERS:
        return raw_normalized
    match = re.search(r"(\d{1,3}[.,]?\d?)", raw_normalized)
    return match.group(1).replace(",", ".") if match else raw_normalized


def _parse_size_from_dict(parsed):
    raw = _get_parsed_attribute_value(
        parsed,
        "beden", "numara", "size", "no", "ayakkabı numarası",
        "beden/numara", "elbise bedeni", "pantolon bedeni",
    )
    return _normalize_size(raw)


def parse_size(attributes_str):
    """
    Attributes string'inden beden/numara bilgisini çıkarır.

    Döndürür
    -------
    str
        Normalize edilmiş beden (örn. "42", "m", "xl"). Bulunamazsa "".
    """
    return _parse_size_from_dict(parse_attributes(attributes_str))


def extract_query_sizes(query):
    """
    Sorgu metnindeki beden/numara bilgilerini çıkarır.

    Örnek: "42 numara erkek ayakkabı" → {"42"}
    Örnek: "l beden erkek tişört" → {"l"}
    """
    if not isinstance(query, str):
        return set()

    query_lower = query.lower()
    found = set()

    # Harf bedenleri (önce kısa olanlar arama yapılınca sorun çıkarır, kelime sınırı önemli)
    for pat in _SIZE_PATTERNS:
        for match in re.finditer(pat, query_lower):
            groups = [g for g in match.groups() if g]
            if groups:
                # "numara"/"beden" gibi keyword'ler grupta varsa son sayısal grubu al
                for g in groups:
                    g_clean = g.strip().replace(",", ".")
                    if g_clean not in ("numara", "beden", "size", "no"):
                        found.add(g_clean)

    return found


def compute_size_match(query, attributes_str):
    """
    Sorgu metnindeki beden bilgisi ürünün bedeniyle uyuşuyor mu?

    Döndürülen değerler:
       1  → Sorgu ve ürün aynı bedeni içeriyor
      -1  → Sorguda beden var ama ürünün bedeni farklı
       0  → Sorguda beden yok veya ürünün bedeni bilinmiyor

    Örnek:
      query = "42 numara siyah spor ayakkabı"
      attributes = "{'Numara': '42'}"
      → "42" == "42" → 1
    """
    query_sizes = extract_query_sizes(query)

    if not query_sizes:
        return 0

    item_size = parse_size(attributes_str)

    if not item_size:
        return 0

    # Herhangi bir sorgu bedeni ürün bedeniyle eşleşiyor mu?
    if item_size in query_sizes:
        return 1

    return -1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Materyal Parse
# ─────────────────────────────────────────────────────────────────────────────

# Yaygın materyal/kumaş terimleri
_MATERIALS = {
    "deri": ["deri", "leather", "nubuk", "süet", "suede", "nappa"],
    "kumaş": ["kumaş", "fabric", "tekstil", "dokuma"],
    "pamuk": ["pamuk", "pamuklu", "cotton", "viskon", "viscose"],
    "polyester": ["polyester", "polyamid", "naylon", "nylon", "sentetik"],
    "yün": ["yün", "wool", "kaşmir", "cashmere", "angora"],
    "keten": ["keten", "linen"],
    "ipek": ["ipek", "silk", "saten", "satin"],
    "spandex": ["spandex", "elastan", "elastane", "likra", "lycra"],
}

_ALL_MATERIAL_KEYWORDS = {}
for canonical, variants in _MATERIALS.items():
    for v in variants:
        _ALL_MATERIAL_KEYWORDS[v] = canonical


def _normalize_material(raw):
    if not raw:
        return ""
    raw_normalized = raw.casefold()
    return first_phrase_value(
        raw_normalized, _ALL_MATERIAL_KEYWORDS, default=raw_normalized
    )


def _parse_material_from_dict(parsed):
    raw = _get_parsed_attribute_value(
        parsed,
        "materyal", "kumaş", "material", "fabric", "ana madde", "dış malzeme",
    )
    return _normalize_material(raw)


def parse_material(attributes_str):
    """
    Attributes string'inden materyal/kumaş bilgisini çıkarır.

    Döndürür
    -------
    str
        Normalize edilmiş materyal (örn. "deri", "pamuk"). Bulunamazsa "".
    """
    return _parse_material_from_dict(parse_attributes(attributes_str))


def extract_query_materials(query):
    """
    Sorgu metninde geçen materyal bilgilerini çıkarır.

    Örnek: "deri erkek cüzdan" → {"deri"}
    """
    if not isinstance(query, str):
        return set()
    return find_phrase_values(query, _ALL_MATERIAL_KEYWORDS)


def compute_material_match(query, attributes_str):
    """
    Sorgu metnindeki materyal bilgisi ürünün materyaliyle uyuşuyor mu?

    Döndürülen değerler:
       1  → Sorgu ve ürün aynı materyali içeriyor
      -1  → Sorguda materyal var ama ürününki farklı
       0  → Sorguda materyal yok veya ürünün materyali bilinmiyor
    """
    query_materials = extract_query_materials(query)

    if not query_materials:
        return 0

    item_material = parse_material(attributes_str)

    if not item_material:
        return 0

    if item_material in query_materials:
        return 1

    return -1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ana Feature Builder (features.py ile entegrasyon için)
# ─────────────────────────────────────────────────────────────────────────────

def add_attribute_features(df):
    """
    Birleştirilmiş DataFrame'e attributes tabanlı feature'ları ekler.

    DataFrame'in şu kolonları içermesi gerekir:
      - query      : Arama sorgusu
      - attributes : Ürün attributes metni (items.csv'den gelir)

    Eklenecek feature kolonları:
      - query_color_match    : -1 / 0 / 1
      - query_size_match     : -1 / 0 / 1
      - query_material_match : -1 / 0 / 1

    Parametreler
    ----------
    df : pd.DataFrame
        merge_pairs() ile üretilmiş birleştirilmiş veri seti.

    Döndürür
    -------
    pd.DataFrame
        Orijinal DataFrame'e 3 yeni kolon eklenmiş hali.
    """
    out = df.copy()

    # 'attributes' kolonu yoksa (bazı eski scriptlerde eksik olabilir) boş string kullan
    if "attributes" not in out.columns:
        out["attributes"] = ""

    queries = out["query"].tolist()

    query_cache = {}
    query_values = []
    for query in queries:
        key = query if isinstance(query, str) else None
        if key not in query_cache:
            query_cache[key] = (
                extract_query_colors(query),
                extract_query_sizes(query),
                extract_query_materials(query),
            )
        query_values.append(query_cache[key])

    item_ids = out["item_id"] if "item_id" in out.columns else out["attributes"]
    item_cache = {}
    item_values = []
    for item_id, attributes in zip(item_ids, out["attributes"]):
        key = item_id if isinstance(item_id, str) else (
            attributes if isinstance(attributes, str) else None
        )
        if key not in item_cache:
            parsed = parse_attributes(attributes)
            item_cache[key] = (
                _parse_color_from_dict(parsed),
                _parse_size_from_dict(parsed),
                _parse_material_from_dict(parsed),
            )
        item_values.append(item_cache[key])

    def match(query_values, item_value):
        if not query_values or not item_value:
            return 0
        return 1 if item_value in query_values else -1

    print("[attributes] query_color_match hesaplanıyor...")
    out["query_color_match"] = [
        match(query_value[0], item_value[0])
        for query_value, item_value in zip(query_values, item_values)
    ]

    print("[attributes] query_size_match hesaplanıyor...")
    out["query_size_match"] = [
        match(query_value[1], item_value[1])
        for query_value, item_value in zip(query_values, item_values)
    ]

    print("[attributes] query_material_match hesaplanıyor...")
    out["query_material_match"] = [
        match(query_value[2], item_value[2])
        for query_value, item_value in zip(query_values, item_values)
    ]

    print("[attributes] Attribute feature'ları hesaplandı.")
    return out


# Bu modülde üretilen feature kolonlarının tam listesi
ATTRIBUTE_FEATURE_COLS = [
    "query_color_match",
    "query_size_match",
    "query_material_match",
]


if __name__ == "__main__":
    # ── Hızlı birim testi ──────────────────────────────────────────────────
    print("=== ATTRIBUTES.PY BİRİM TESTİ ===\n")

    test_cases = [
        # (query, attributes, beklenen_color, beklenen_size, beklenen_material)
        (
            "siyah 42 numara deri erkek ayakkabı",
            "{'Renk': 'Siyah', 'Numara': '42', 'Materyal': 'Deri'}",
            1, 1, 1,    # her şey uyuşuyor
        ),
        (
            "beyaz spor ayakkabı",
            "{'Renk': 'Siyah', 'Numara': '42'}",
            -1, 0, 0,   # renk çelişiyor, beden ve materyal belirsiz
        ),
        (
            "erkek spor ayakkabı",
            "{'Renk': 'Siyah', 'Numara': '42'}",
            0, 0, 0,    # sorguda renk/beden/materyal yok → hepsi belirsiz
        ),
        (
            "xl beden pamuk tişört",
            "{'Beden': 'XL', 'Materyal': 'Pamuk'}",
            0, 1, 1,    # renk belirsiz, beden ve materyal uyuşuyor
        ),
        (
            "m beden keten gömlek",
            "{'Beden': 'XL', 'Materyal': 'Pamuk'}",
            0, -1, -1,  # beden çelişiyor, materyal çelişiyor
        ),
    ]

    print(f"{'Sorgu':<35} {'color':>6} {'size':>6} {'mat':>6}  {'Sonuç'}")
    print("-" * 70)
    passed = 0
    for query, attrs, exp_c, exp_s, exp_m in test_cases:
        c = compute_color_match(query, attrs)
        s = compute_size_match(query, attrs)
        m = compute_material_match(query, attrs)
        ok = (c == exp_c and s == exp_s and m == exp_m)
        status = "[OK]" if ok else f"[FAIL] (beklenen: {exp_c}/{exp_s}/{exp_m})"
        print(f"  {query:<33} {c:>6} {s:>6} {m:>6}  {status}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(test_cases)} test geçti.")

    # DataFrame entegrasyon testi
    print("\n--- DataFrame Entegrasyon Testi ---")
    import pandas as pd
    test_df = pd.DataFrame([
        {"query": "siyah 42 deri erkek ayakkabı", "attributes": "{'Renk': 'Siyah', 'Numara': '42', 'Materyal': 'Deri'}"},
        {"query": "beyaz l beden pamuk tişört",   "attributes": "{'Renk': 'Beyaz', 'Beden': 'L', 'Materyal': 'Pamuk'}"},
        {"query": "erkek spor ayakkabı",           "attributes": ""},
    ])
    result = add_attribute_features(test_df)
    print(result[["query"] + ATTRIBUTE_FEATURE_COLS].to_string(index=False))

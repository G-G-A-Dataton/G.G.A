"""
src/item_text.py
================
G.G.A Takımı — Item Text Standardizasyonu (9 Temmuz Görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Bu modül her ürün için tek ve tutarlı bir metin temsili üretir.
Bu birleşik metin şu kullanımlar için hazırlanır:
  1. TF-IDF vectorizer eğitimi (tüm tokenleri kapsayan geniş corpus)
  2. Sentence-transformer embedding girişi (10 Temmuz)
  3. BM25 index oluşturma (Mert'in modülü ile entegrasyon)

Neden standardize?
  items.csv'de bilgi dağınık:
    title     = "Adidas Erkek Kosu Ayakkabisi"     (kısa, bazen model kodu)
    category  = "ayakkabi/spor ayakkabisi/koscu"    (/ ile ayrılmış hiyerarşi)
    brand     = "adidas"                            (marka)
    attributes= "renk: siyah, numara: 42"              (düz anahtar/değer)
  Bunlar birleştirilirse: zengin ve kapsamlı bir metin elde edilir.

Üretilen metin formatı:
  "<title> <brand> <category_flat> <attr_key1> <attr_val1> <attr_key2>..."
  Örnek:
  "Adidas Erkek Kosu Ayakkabisi adidas ayakkabi spor ayakkabisi koscu renk siyah numara 42"
"""

import re
import pandas as pd

from src.attributes import parse_attributes


# ─────────────────────────────────────────────────────────────────────────────
# 1. Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def flatten_category(category):
    """
    "ayakkabi/spor ayakkabisi/koscu" → "ayakkabi spor ayakkabisi koscu"

    Kategori hiyerarşisindeki '/' ayırıcısını boşlukla değiştirir.
    TF-IDF ve embedding modelleri hiyerarşiyi anlamaz; düz metin gerekir.
    """
    if not isinstance(category, str):
        return ""
    return category.replace("/", " ").strip()


def parse_attributes_flat(attributes_str):
    """
    Attributes düz metnini veya JSON/dict metnini tek metin haline getirir.

    Giriş : "{'Renk': 'Siyah', 'Numara': '42'}"
    Çıkış : "renk siyah numara 42"

    Bu format TF-IDF ve embedding için uygundur:
    - Anahtar kelimeler (renk, numara) bağlam sağlar
    - Değer kelimeleri (siyah, 42) sinyal taşır

    Hata toleranslı: bozuk format durumunda boş string döner.
    """
    if not isinstance(attributes_str, str) or not attributes_str.strip():
        return ""

    d = parse_attributes(attributes_str)
    if not d:
        return ""

    # Her anahtar-değer çiftini birleştir: "renk siyah"
    parts = []
    for k, v in d.items():
        if k and v:
            # Anahtarı ve değeri küçük harfe çevir, temizle
            k_clean = re.sub(r"[^a-zA-ZÇçĞğIıÖöŞşÜü0-9 ]", "", str(k)).lower().strip()
            v_clean = re.sub(r"[^a-zA-ZÇçĞğIıÖöŞşÜü0-9 ]", "", str(v)).lower().strip()
            if k_clean:
                parts.append(k_clean)
            if v_clean:
                parts.append(v_clean)

    return " ".join(parts)


def clean_text(text):
    """
    Metni temizler: fazla boşlukları kaldırır, küçük harfe çevirir.

    Bu fonksiyon tüm bileşenler birleştirildikten sonra çağrılır.
    Tutarlı bir format sağlar.
    """
    if not isinstance(text, str):
        return ""
    # Birden fazla boşluğu teke indir
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ana Standardizasyon Fonksiyonu
# ─────────────────────────────────────────────────────────────────────────────

def _combine_item_text(title, brand, category, attributes, include_attrs):
    title = title if isinstance(title, str) else ""
    brand = brand if isinstance(brand, str) else ""
    category = category if isinstance(category, str) else ""
    attributes = attributes if isinstance(attributes, str) else ""

    parts = [title] if title else []
    if brand and brand.casefold() not in title.casefold():
        parts.append(brand)
    category_flat = flatten_category(category)
    if category_flat:
        parts.append(category_flat)
    if include_attrs:
        attributes_flat = parse_attributes_flat(attributes)
        if attributes_flat:
            parts.append(attributes_flat)
    return clean_text(" ".join(parts))


def build_item_text(row, include_attrs=True):
    """
    Tek bir ürün satırından standart metin üretir.

    Bileşenler (öncelik sırasına göre):
      1. title    : En önemli — kullanıcı arama terimlerine en yakın
      2. brand    : Marka aramaları için kritik
      3. category : Kategori kelimeleri genel bağlamı sağlar
      4. attributes: Renk, materyal, beden — spesifik sinyaller

    Parametreler
    ----------
    row : pd.Series
        items.csv'den bir ürün satırı. title, brand, category, attributes
        kolonlarını içermeli.
    include_attrs : bool
        Attributes eklensin mi? (True: daha zengin ama yavaş)

    Döndürür
    -------
    str
        Birleştirilmiş ve temizlenmiş ürün metni.
    """
    return _combine_item_text(
        row.get("title", ""),
        row.get("brand", ""),
        row.get("category", ""),
        row.get("attributes", ""),
        include_attrs,
    )


def add_item_text_column(items_df, include_attrs=True, col_name="item_text"):
    """
    items_df DataFrame'ine 'item_text' kolonunu ekler.

    Bu kolon TF-IDF/embedding eğitimi ve BM25 indexlemesi için kullanılır.
    Büyük katalog (~963K ürün) için vektörize işlem yapar.

    Parametreler
    ----------
    items_df : pd.DataFrame
        items.csv'den yüklenmiş ürün verisi.
    include_attrs : bool
        Attributes eklensin mi?
    col_name : str
        Oluşturulacak kolon adı.

    Döndürür
    -------
    pd.DataFrame
        item_text kolonu eklenmiş items_df kopyası.
    """
    print(f"[item_text] {len(items_df):,} urun icin metin standardize ediliyor...")
    out = items_df.copy()
    for column in ("title", "brand", "category", "attributes"):
        if column not in out.columns:
            out[column] = ""
    out[col_name] = [
        _combine_item_text(title, brand, category, attributes, include_attrs)
        for title, brand, category, attributes in zip(
            out["title"], out["brand"], out["category"], out["attributes"]
        )
    ]
    n_empty = (out[col_name] == "").sum()
    print(f"[item_text] Tamamlandi. Bos metin sayisi: {n_empty}")
    return out


def build_item_texts(items_df, include_attrs=True):
    """Build standardized item texts without copying the source DataFrame."""
    required = {"title", "brand", "category"}
    missing = sorted(required - set(items_df.columns))
    if missing:
        raise ValueError(f"Item text input is missing required columns: {missing}")
    attributes = (
        items_df["attributes"]
        if "attributes" in items_df.columns
        else [""] * len(items_df)
    )
    return [
        _combine_item_text(title, brand, category, attribute, include_attrs)
        for title, brand, category, attribute in zip(
            items_df["title"], items_df["brand"], items_df["category"], attributes
        )
    ]


def build_query_text(row):
    """
    Sorgu (term) satırından standart metin üretir.

    Sorgular zaten kısa metin olduğundan ek işlem gerekmez.
    Sadece temizleme yapılır.

    Parametreler
    ----------
    row : pd.Series
        terms.csv'den bir sorgu satırı. query kolonunu içermeli.
    """
    query = str(row.get("query", "") or "")
    return clean_text(query)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Toplu Metin Üretimi (Embedding İçin)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_texts(terms_df, items_df, include_attrs=True):
    """
    Tüm sorgu ve ürün metinlerini standart formatta üretir.

    Bu fonksiyon şunlar için kullanılır:
      - TF-IDF vectorizer eğitimi (tüm corpus)
      - Sentence-transformer embedding eğitimi/üretimi
      - BM25 index oluşturma

    Parametreler
    ----------
    terms_df : pd.DataFrame
        Sorgu verisi.
    items_df : pd.DataFrame
        Ürün verisi.
    include_attrs : bool
        Ürün metnine attributes eklensin mi?

    Döndürür
    -------
    tuple (list, list)
        (query_texts, item_texts) — sıra terms/items DataFrame ile aynı
    """
    print("[item_text] Sorgu metinleri hazirlaniyor...")
    if "query" not in terms_df.columns:
        raise ValueError("terms_df must contain query")
    query_texts = [clean_text(query) for query in terms_df["query"]]

    print("[item_text] Urun metinleri hazirlaniyor...")
    item_texts = build_item_texts(items_df, include_attrs=include_attrs)

    return query_texts, item_texts


if __name__ == "__main__":
    # Hızlı birim testi
    print("=== ITEM_TEXT BIRIM TESTI ===")

    test_rows = [
        {
            "title"     : "Adidas Erkek Kosu Ayakkabisi",
            "brand"     : "Adidas",
            "category"  : "ayakkabi/spor ayakkabisi/koscu",
            "attributes": "{'Renk': 'Siyah', 'Numara': '42'}",
        },
        {
            "title"     : "Koton Kadin Pamuk Gomlek",
            "brand"     : "Koton",
            "category"  : "giyim/gomlek/gunluk",
            "attributes": "{'Renk': 'Beyaz', 'Materyal': 'Pamuk', 'Beden': 'M'}",
        },
        {
            "title"     : "530",   # Sadece model kodu — attributes olmadan anlamsız
            "brand"     : "New Balance",
            "category"  : "ayakkabi/spor ayakkabisi/sneaker",
            "attributes": "{'Renk': 'Bej', 'Numara': '41'}",
        },
        {
            "title"     : "Ürün",
            "brand"     : "",
            "category"  : "",
            "attributes": None,  # Boş attributes
        },
    ]

    for i, row in enumerate(test_rows, start=1):
        text = build_item_text(row, include_attrs=True)
        print(f"\n  Ornek {i}:")
        print(f"    title    : {row['title']}")
        print(f"    item_text: {text}")

"""
run_data_qa_report.py
======================
G.G.A Takimi — Veri Kalite Raporu (12 Temmuz Mert Gorevi)

Mustafa Mert Cevik tarafından hazırlanmıştır.

Bu script egitim seti ve urun katalogu uzerinde kapsamli veri kalitesi
kontrolu yapar ve sonuclari raporlar:

  1. Genel istatistikler (satir, kolon, eksik deger)
  2. Label dagilimi
  3. Tekrar eden (term_id, item_id) cifti var mi?
  4. Pozitif cifler gerçekten items.csv'de var mi?
  5. Attribute doluluk orani
  6. Kategori dagilimi
  7. Sızıntı kontrolu: training_pairs'teki item_id'ler submission'da var mi?

Calistirmak icin:
  python run_data_qa_report.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR  = os.path.join(PROJECT_ROOT, "datasets")
DOCS_DIR  = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)


def section(title):
    """Konsol baslik yazici."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_file(path, name, dtypes=None):
    """
    Dosyayi yukler ve temel bilgileri gosterir.

    Parametreler
    ----------
    path : str
    name : str
    dtypes : dict | None

    Dondurur
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        print(f"  [HATA] {name} bulunamadi: {path}")
        return None
    df = pd.read_csv(path, dtype=dtypes)
    print(f"  {name}: {len(df):,} satir, {len(df.columns)} kolon")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Veri Kalite Raporu (12 Temmuz Mert)")
    print("=" * 60)

    lines = [
        "# Veri Kalite Raporu (12 Temmuz)",
        "",
        "**Hazırlayan:** Mustafa Mert Çevik  ",
        "**Tarih:** 12 Temmuz 2026  ",
        "",
        "---",
        "",
    ]

    # ── 1. Dosyalari Yukle ──────────────────────────────────────────────────
    section("1. Dosya Yuklemesi")
    items_df   = check_file(os.path.join(DATA_DIR, "items.csv"),   "items.csv",
                            {"item_id": "string"})
    terms_df   = check_file(os.path.join(DATA_DIR, "terms.csv"),   "terms.csv",
                            {"term_id": "string"})
    train_df   = check_file(os.path.join(DATA_DIR, "training_pairs.csv"), "training_pairs.csv",
                            {"id": "string", "term_id": "string", "item_id": "string", "label": "int8"})
    sub_df     = check_file(os.path.join(DATA_DIR, "submission_pairs.csv"), "submission_pairs.csv",
                            {"id": "string", "term_id": "string", "item_id": "string"})

    lines += [
        "## 1. Dosya Boyutlari",
        "",
        "| Dosya | Satir | Kolon |",
        "|---|---|---|",
    ]
    for df, name in [(items_df, "items.csv"), (terms_df, "terms.csv"),
                      (train_df, "training_pairs.csv"), (sub_df, "submission_pairs.csv")]:
        if df is not None:
            lines.append(f"| {name} | {len(df):,} | {len(df.columns)} |")

    lines += ["", "---", ""]

    # ── 2. Eksik Deger Kontrolu ──────────────────────────────────────────────
    section("2. Eksik Deger Kontrolu")

    lines += ["## 2. Eksik Deger Analizi", "", "### items.csv", ""]
    if items_df is not None:
        missing = items_df.isnull().sum()
        missing_pct = (missing / len(items_df) * 100).round(1)
        lines.append("| Kolon | Eksik | Eksik % |")
        lines.append("|---|---|---|")
        for col in items_df.columns:
            m = missing[col]
            p = missing_pct[col]
            if m > 0:
                flag = " ⚠️" if p > 30 else ""
                lines.append(f"| {col} | {m:,} | {p}%{flag} |")
                print(f"  {col}: {m:,} eksik ({p}%)")
        lines.append("")

    lines += ["---", ""]

    # ── 3. Label Dagilimi ──────────────────────────────────────────────────────
    section("3. Label Dagilimi (training_pairs.csv)")

    lines += ["## 3. Label Dagilimi", ""]
    if train_df is not None:
        vc = train_df["label"].value_counts().sort_index()
        for lbl, cnt in vc.items():
            pct = 100 * cnt / len(train_df)
            print(f"  Label {lbl}: {cnt:,}  ({pct:.1f}%)")
        lines.append("| Label | Sayi | Oran |")
        lines.append("|---|---|---|")
        for lbl, cnt in vc.items():
            pct = 100 * cnt / len(train_df)
            lines.append(f"| {lbl} | {cnt:,} | {pct:.1f}% |")
        lines.append("")

        # Tum label'lar 0 veya 1 mi?
        invalid = train_df[~train_df["label"].isin([0, 1])]
        if len(invalid) > 0:
            msg = f"[HATA] {len(invalid)} gecersiz label degeri!"
            print(f"  {msg}")
            lines.append(f"> [!WARNING]\n> {msg}")
        else:
            print("  Tum label degerler 0 veya 1 - OK")
            lines.append("> [!NOTE]\n> Tum label degerler 0 veya 1 ✅")
        lines += ["", "---", ""]

    # ── 4. Tekrar Eden Cift Kontrolu ────────────────────────────────────────
    section("4. Tekrar Eden (term_id, item_id) Cifti")

    lines += ["## 4. Tekrar Eden Cift Kontrolu", ""]
    if train_df is not None:
        dupes = train_df.duplicated(subset=["term_id", "item_id"]).sum()
        print(f"  Tekrar eden (term_id, item_id) cifti: {dupes:,}")
        if dupes > 0:
            lines.append(f"> [!WARNING]\n> {dupes:,} tekrar eden (term_id, item_id) cifti bulundu!")
        else:
            lines.append("> [!NOTE]\n> Tekrar eden cift yok ✅")
        lines += ["", "---", ""]

    # ── 5. Referans Butunlugu ─────────────────────────────────────────────
    section("5. Referans Butunlugu")

    lines += ["## 5. Referans Butunlugu", ""]
    if train_df is not None and items_df is not None and terms_df is not None:
        item_ids_catalog  = set(items_df["item_id"].astype(str))
        term_ids_catalog  = set(terms_df["term_id"].astype(str))
        item_ids_train    = set(train_df["item_id"].astype(str))
        term_ids_train    = set(train_df["term_id"].astype(str))

        missing_items = item_ids_train - item_ids_catalog
        missing_terms = term_ids_train - term_ids_catalog

        for name, missing in [("item_id", missing_items), ("term_id", missing_terms)]:
            n = len(missing)
            print(f"  Training'de katalogda olmayan {name}: {n}")
            if n > 0:
                lines.append(f"> [!WARNING]\n> Training'de items.csv'de olmayan {name}: **{n:,}** adet")
            else:
                lines.append(f"> [!NOTE]\n> {name} referans butunlugu tam ✅")
            lines.append("")

        lines += ["---", ""]

    # ── 6. Attribute Doluluk Orani ───────────────────────────────────────
    section("6. Attribute Doluluk Orani")

    lines += ["## 6. Attribute Doluluk", ""]
    if items_df is not None and "attributes" in items_df.columns:
        total   = len(items_df)
        has_attr = items_df["attributes"].notna() & (items_df["attributes"] != "") & (items_df["attributes"] != "{}")
        n_attr  = has_attr.sum()
        pct     = 100 * n_attr / total
        print(f"  Attributes dolu: {n_attr:,} / {total:,}  ({pct:.1f}%)")
        lines.append(f"- Attribute dolu urun: **{n_attr:,}** / {total:,} (**{pct:.1f}%**)")
        lines.append(f"- Attribute bos urun: **{total - n_attr:,}** (**{100-pct:.1f}%**)")

        if pct < 70:
            lines.append("\n> [!WARNING]\n> Attribute doluluk orani %70'in altinda. Renk/materyal feature'lari bu urunlerde 0 kalacak.")
        lines += ["", "---", ""]

    # ── 7. Kategori Dagilimi (Top 10) ───────────────────────────────────
    section("7. Kategori Dagilimi")

    lines += ["## 7. Top 10 Kategori (L1)", ""]
    if items_df is not None and "category" in items_df.columns:
        items_df["cat_l1"] = items_df["category"].str.split("/").str[0].str.strip()
        top10 = items_df["cat_l1"].value_counts().head(10)
        lines.append("| Kategori | Urun Sayisi | Oran |")
        lines.append("|---|---|---|")
        for cat, cnt in top10.items():
            pct = 100 * cnt / len(items_df)
            print(f"  {cat}: {cnt:,}  ({pct:.1f}%)")
            lines.append(f"| {cat} | {cnt:,} | {pct:.1f}% |")
        lines += ["", "---", ""]

    # ── 8. Sizinti Kontrolu ──────────────────────────────────────────────
    section("8. Sizinti Kontrolu (Training vs Submission)")

    lines += ["## 8. Sizinti Kontrolu", ""]
    if train_df is not None and sub_df is not None:
        train_ids = set(zip(train_df["term_id"].astype(str), train_df["item_id"].astype(str)))
        sub_ids   = set(zip(sub_df["term_id"].astype(str),   sub_df["item_id"].astype(str)))
        overlap   = train_ids & sub_ids
        print(f"  Training-Submission (term_id, item_id) ortusmesi: {len(overlap):,}")
        if len(overlap) > 0:
            lines.append(f"> [!CAUTION]\n> Training ve submission arasinda **{len(overlap):,}** ortak (term_id, item_id) cifti! Veri sizintisi riski!")
        else:
            lines.append("> [!NOTE]\n> Training ve submission arasinda ortak cift yok ✅")
        lines += ["", "---", ""]

    # ── Sonuc Ozeti ─────────────────────────────────────────────────────────
    lines += [
        "## Sonuc",
        "",
        "Bu rapor elle guncellenmeden skript tarafindan otomatik uretilmistir.",
        f"*Uretim tarihi: 12 Temmuz 2026*",
    ]

    out_md = os.path.join(DOCS_DIR, "data_qa_raporu.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Rapor: {out_md}")
    print("=" * 60)

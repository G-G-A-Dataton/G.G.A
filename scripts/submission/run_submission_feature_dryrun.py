"""
run_submission_feature_dryrun.py
=================================
G.G.A Takimi — Submission Feature Dry-Run (13 Temmuz Mert Gorevi)

Mustafa Mert Cevik tarafından hazırlanmıştır.

Bu script submission_pairs.csv uzerinde feature uretim akisini
uc tan uca dener. Hata, eksik kolon, bellek sorunu gibi
problemleri onceden tespit etmek icin tasarlanmistir.

Dry-run mantigi:
  1. Kucuk bir alt kume (10K satir) ile tam akisi test et
  2. Belirlenen feature kolonu eksikliklerini raporla
  3. Feature dagilimlarini kontrol et
  4. Bellek kullanimi raporla

Calistirmak icin:
  python run_submission_feature_dryrun.py
  python run_submission_feature_dryrun.py --full  # Tam submission seti
"""

import os
import sys
import time
import argparse
import warnings
import psutil
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data     import load_terms, load_items
from src.features import build_features, FEATURE_COLS

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR   = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRYRUN_ROWS = 10_000  # Test alt kumesi boyutu


def mb_used():
    """Mevcut Python prosesinin RAM kullanimi (MB)."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 / 1024


def check_features(df_featured, expected_cols):
    """
    Beklenen feature kolonlarinin hepsinin mevcut olup olmadigini kontrol eder.
    Eksik ve sifir-varyanslilari raporlar.
    """
    issues = []
    for col in expected_cols:
        if col not in df_featured.columns:
            issues.append(f"[EKSIK KOLON] {col} bulunamadi!")
        elif df_featured[col].isnull().any():
            n_null = df_featured[col].isnull().sum()
            issues.append(f"[NaN VAR] {col}: {n_null} NaN deger")
        elif df_featured[col].std() == 0:
            issues.append(f"[SIFIR VARYANS] {col}: Tum degerler {df_featured[col].iloc[0]}")
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submission Feature Dry-Run")
    parser.add_argument("--full", action="store_true",
                        help="Tum submission setini isle (yavas, ~3.36M satir)")
    args = parser.parse_args()

    print("=" * 65)
    print("  G.G.A - Submission Feature Dry-Run (13 Temmuz Mert)")
    print(f"  Mod: {'TAM (~3.36M satir)' if args.full else f'DRY-RUN ({DRYRUN_ROWS:,} satir)'}")
    print("=" * 65)

    lines = [
        "# Submission Feature Dry-Run Raporu (13 Temmuz)",
        "",
        "**Hazırlayan:** Mustafa Mert Çevik  ",
        "**Tarih:** 13 Temmuz 2026  ",
        f"**Mod:** {'Tam' if args.full else f'Dry-Run ({DRYRUN_ROWS:,} satir)'}  ",
        "",
        "---",
        "",
    ]

    # 1. Veri yukle
    print("\n[1/4] Veri yukleniyor...")
    mem0 = mb_used()
    t0   = time.time()

    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))
    sub_df   = pd.read_csv(
        os.path.join(DATA_DIR, "submission_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string"}
    )

    if not args.full:
        # Dry-run: rastgele alt kume al
        sub_df = sub_df.sample(DRYRUN_ROWS, random_state=42).reset_index(drop=True)

    print(f"  {len(sub_df):,} submission cifti yuklendi  ({mb_used()-mem0:.0f} MB)")

    # 2. Join
    print("\n[2/4] Veri birlestiriliyor...")
    merged = sub_df.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df, on="item_id", how="left")

    # Kataloğda olmayan ID'ler
    missing_terms = merged["query"].isnull().sum()
    missing_items = merged["title"].isnull().sum()
    print(f"  Katalogsuz term_id: {missing_terms:,}")
    print(f"  Katalogsuz item_id: {missing_items:,}")

    lines += [
        "## 1. Join Sonucu",
        "",
        f"| Metrik | Deger |",
        f"|---|---|",
        f"| Submission satir | {len(sub_df):,} |",
        f"| Katalogsuz term_id | {missing_terms:,} |",
        f"| Katalogsuz item_id | {missing_items:,} |",
        "",
        "---",
        "",
    ]

    # 3. Feature uretimi
    print("\n[3/4] Feature'lar uretiliyor...")
    t_feat = time.time()
    featured = build_features(merged)
    feat_elapsed = time.time() - t_feat
    mem1 = mb_used()

    # Kontrol
    issues = check_features(featured, FEATURE_COLS)

    print(f"  Feature suresi : {feat_elapsed:.1f}s")
    print(f"  RAM kullanimi  : {mem1:.0f} MB")
    if issues:
        print(f"  [SORUN] {len(issues)} sorun tespit edildi:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  Tum feature'lar tamam!")

    # 4. Feature dagilim ozeti
    print("\n[4/4] Feature dagilim kontrolu...")
    dist_rows = []
    for col in FEATURE_COLS:
        if col in featured.columns:
            v = featured[col]
            dist_rows.append({
                "feature": col,
                "min"    : round(float(v.min()), 4),
                "max"    : round(float(v.max()), 4),
                "mean"   : round(float(v.mean()), 4),
                "null_n" : int(v.isnull().sum()),
            })

    dist_df = pd.DataFrame(dist_rows)
    total_elapsed = time.time() - t0
    rate = len(sub_df) / feat_elapsed

    lines += [
        "## 2. Feature Dagilimi",
        "",
        "| Feature | Min | Max | Mean | Null |",
        "|---|---|---|---|---|",
    ]
    for _, row in dist_df.iterrows():
        lines.append(f"| {row['feature']} | {row['min']} | {row['max']} | {row['mean']} | {row['null_n']} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Performans",
        "",
        f"| Metrik | Deger |",
        f"|---|---|",
        f"| Toplam sure | {total_elapsed:.1f}s |",
        f"| Feature uretim suresi | {feat_elapsed:.1f}s |",
        f"| Isleme hizi | {rate:,.0f} satir/s |",
        f"| RAM kullanimi | {mem1:.0f} MB |",
        "",
    ]

    if issues:
        lines += [
            "## 4. Sorunlar",
            "",
        ]
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("> [!WARNING]\n> Yukaridaki sorunlar cozulmeden submission uretilmemeli!")
    else:
        lines += [
            "> [!NOTE]",
            "> Tum feature'lar basariyla uretildi. Submission akisi hazir.",
        ]

    # Tahmini tam sure
    if not args.full:
        full_size = 3_360_000
        estimated = (full_size / len(sub_df)) * feat_elapsed
        lines += [
            "",
            f"> [!NOTE]",
            f"> Dry-run hizi: **{rate:,.0f} satir/s**. "
            f"Tam submission icin tahmini sure: **{estimated/60:.1f} dakika**.",
        ]

    out_md = os.path.join(DOCS_DIR, "submission_feature_dryrun.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 65)
    print(f"  Toplam sure       : {total_elapsed:.1f}s")
    print(f"  Isleme hizi       : {rate:,.0f} satir/s")
    print(f"  Sorun sayisi      : {len(issues)}")
    if not args.full:
        full_size = 3_360_000
        print(f"  Tahmini tam sure  : {(full_size/len(sub_df))*feat_elapsed/60:.1f} dakika")
    print(f"  Rapor             : {out_md}")
    print("=" * 65)

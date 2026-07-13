"""
notebooks/06_bm25_karsilastirma_tam_omerfaruk.py
================================================
G.G.A Takımı — BM25 Hard Negative Üretimi + Tam Strateji Karşılaştırması (7-8 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu script şunları yapar:
  1. BM25 hard negative'leri src/bm25_hard_negative.py ile üretir
  2. Üretilen negatifleri outputs/negative_bm25_v1.csv olarak kaydeder
  3. run_hard_neg_comparison.py --bm25 ile tam karşılaştırmayı çalıştırır
  4. Sonuçlar EXP-006 olarak experiment_log.md'ye kopyalanmaya hazır hale getirilir

Çalıştırmak için:
  python notebooks/06_bm25_karsilastirma_tam_omerfaruk.py

Bağımlılıklar:
  - src/bm25_hard_negative.py  (Mert'in BM25 modülü — 7-8 Temmuz)
  - src/negative_sampling.py   (verify_no_leakage için)
  - run_hard_neg_comparison.py (karşılaştırma scripti — Ömer Faruk)
"""

import os
import sys
import subprocess
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data               import load_terms, load_items
from src.bm25_hard_negative import generate_bm25_hard_negatives
from src.features           import FEATURE_COLS
from src.negative_sampling  import verify_no_leakage

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BM25_CSV = os.path.join(OUTPUT_DIR, "negative_bm25_v1.csv")

# ─── BM25 parametreleri (7-8 Temmuz görevi) ───────────────────────────────────
# top_n=50: Her sorgu için en benzer 50 ürüne bak, pozitif olanları eliyor →
#           kalanlardan ratio kadar hard negative seç.
# ratio=3 : Pozitif başına 3 hard negative (random ile aynı oran — adil karşılaştırma).
TOP_N      = 50
RATIO      = 3
SAMPLE_N   = 5_000  # Hızlı deney için kaç benzersiz sorgu? (None = tümü, yavaş)
RANDOM_SEED = 42

print("=" * 60)
print("  G.G.A — BM25 Hard Negative + Tam Karşılaştırma")
print("  7-8 Temmuz Görevi — Ömer Faruk Kara")
print("=" * 60)

# ─── 1. Veri Yükle ────────────────────────────────────────────────────────────
print("\n[1/4] Veri yükleniyor...")
terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
train_raw = pd.read_csv(
    os.path.join(DATA_DIR, "training_pairs.csv"),
    dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
)
print(f"  Sorgular: {len(terms_df):,} | Ürünler: {len(items_df):,} | Pozitif: {len(train_raw):,}")

# Hız için küçük örnek — tam veri için SAMPLE_N = None yapılabilir
if SAMPLE_N:
    sample_terms = train_raw["term_id"].drop_duplicates().sample(
        n=min(SAMPLE_N, train_raw["term_id"].nunique()),
        random_state=RANDOM_SEED
    )
    sample_train = train_raw[train_raw["term_id"].isin(sample_terms)].copy()
    print(f"  Hızlı deney modu: {len(sample_terms):,} benzersiz sorgu "
          f"({len(sample_train):,} pozitif çift) kullanılıyor.")
    print(f"  [NOT] Tüm veri için SAMPLE_N = None yapın (daha uzun sürer).")
else:
    sample_train = train_raw.copy()
    print(f"  Tam veri modu: {train_raw['term_id'].nunique():,} benzersiz sorgu.")
sample_term_count = sample_train["term_id"].nunique()

# ─── 2. BM25 Hard Negative Üret ───────────────────────────────────────────────
print(f"\n[2/4] BM25 hard negative üretiliyor (top_n={TOP_N}, ratio={RATIO})...")
t0 = time.time()
hard_negatives = generate_bm25_hard_negatives(
    train_df=sample_train,
    terms_df=terms_df,
    items_df=items_df,
    top_n=TOP_N,
    ratio=RATIO,
    verbose=True,
    positive_reference_df=train_raw,
)
elapsed = time.time() - t0
print(f"\n  ✅ Üretim tamamlandı: {len(hard_negatives):,} hard negative ({elapsed:.1f}s)")

# ─── 3. Sızıntı Kontrolü ──────────────────────────────────────────────────────
print("\n[3/4] Sızıntı kontrolü yapılıyor...")
ok = verify_no_leakage(hard_negatives, train_raw)
if not ok:
    print("  [HATA] Sızıntı var! Devam edilemiyor.")
    sys.exit(1)
print("  ✅ Sızıntı yok.")

# ─── 4. CSV Kaydet + Karşılaştırmayı Başlat ───────────────────────────────────
hard_negatives.to_csv(BM25_CSV, index=False)
print(f"\n[4/4] BM25 negatifleri kaydedildi: {BM25_CSV}")

print("\n" + "=" * 60)
print("  Hard Negative vs Random Negative Karşılaştırması Başlıyor")
print("=" * 60)

comparison_script = os.path.join(
    PROJECT_ROOT, "scripts", "data", "run_hard_neg_comparison.py"
)
cmd = [sys.executable, comparison_script, "--bm25", BM25_CSV]
cmd.extend(["--sample-terms", str(SAMPLE_N)] if SAMPLE_N else ["--all-terms"])
print(f"  Komut: {' '.join(cmd)}\n")

# Karşılaştırma scriptini aynı process içinde çalıştır
result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode != 0:
    print(f"\n  [UYARI] Karşılaştırma scripti sıfır dışı kod ile çıktı: {result.returncode}")
else:
    print("\n  ✅ Karşılaştırma tamamlandı.")
    comparison_csv = os.path.join(OUTPUT_DIR, "hard_neg_comparison.csv")
    if os.path.exists(comparison_csv):
        print("\n  Sonuç tablosu:")
        df_result = pd.read_csv(comparison_csv)
        print(df_result[[
            "strategy", "cross_fitted_macro_f1", "fold_macro_f1_std",
            "deploy_threshold"
        ]].to_string(index=False))

        print("\n" + "=" * 60)
        print("  EXP-006 için experiment_log.md'ye eklenecek satır:")
        print("=" * 60)
        if len(df_result) >= 2:
            bm25_row = df_result[df_result["strategy"] == "bm25"].iloc[0]
            rand_row = df_result[df_result["strategy"] == "random"].iloc[0]
            diff = bm25_row["cross_fitted_macro_f1"] - rand_row["cross_fitted_macro_f1"]
            print(f"  | EXP-006 | 8 Tem | Ömer Faruk | LightGBM | BM25 Hard 3:1 / {sample_term_count:,} sorgu "
                  f"| {len(FEATURE_COLS)} temel | {bm25_row['cross_fitted_macro_f1']:.4f} ± "
                  f"{bm25_row['fold_macro_f1_std']:.4f} | — | "
                  f"BM25 vs Random fark: {diff:+.4f} |")
        else:
            print("  Karşılaştırma için iki strateji de olmalı.")

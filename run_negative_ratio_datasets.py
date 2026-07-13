#!/usr/bin/env python
"""
run_negative_ratio_datasets.py
===============================
G.G.A Takımı — Negatif Oranı Deney Veri Setleri (Gün 10 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (10 Temmuz görevi)

Amaç:
  1:1, 2:1, 3:1, 5:1 negatif oranları için TAM ölçekli (250K pozitifin
  tamamı üzerinde) eğitim veri setleri üretmek ve diske kaydetmek.

  `src/negative_sampling.generate_ratio_experiments` hızlı deney modu
  için varsayılan olarak küçük bir örneklem alır (sample_size=5000).
  Bu script aynı fonksiyonu sample_size=None ile çağırarak TÜM pozitif
  çiftler üzerinde çalıştırır — sonuç, modelleme ekibinin (Ömer) farklı
  oranları gerçek ölçekte karşılaştırabileceği kalıcı dosyalardır.

Çalıştırmak için:
  python run_negative_ratio_datasets.py
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items
from src.negative_sampling import generate_ratio_experiments, verify_no_leakage

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
RATIOS = [1, 2, 3, 5]
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  G.G.A — Negatif Orani Deney Veri Setleri (Gun 10)")
print("=" * 70)

print("\nVeriler yukleniyor...")
items_df = load_items(os.path.join(DATA_DIR, "items.csv"))
train_df = pd.read_csv(
    os.path.join(DATA_DIR, "training_pairs.csv"),
    dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
)
print(f"Pozitif cift sayisi: {len(train_df):,}")
print(f"Katalog urun sayisi: {len(items_df):,}")

experiments = generate_ratio_experiments(
    train_df, items_df, ratios=RATIOS, sample_size=None, random_state=SEED,
)

print("\nSizinti kontrolu ve kayit...")
summary_rows = []
for ratio, dataset in experiments.items():
    neg = dataset[dataset["label"] == 0]
    leakage_ok = verify_no_leakage(neg, train_df)

    out_path = os.path.join(OUTPUT_DIR, f"train_random{ratio}_seed{SEED}.csv")
    dataset.to_csv(out_path, index=False)

    pos_n = int((dataset["label"] == 1).sum())
    neg_n = int((dataset["label"] == 0).sum())
    print(f"  Kaydedildi: {out_path} ({len(dataset):,} satir)")

    summary_rows.append({
        "ratio": ratio,
        "total_rows": len(dataset),
        "positive": pos_n,
        "negative": neg_n,
        "positive_share": round(pos_n / len(dataset), 4),
        "leakage_ok": leakage_ok,
        "output_path": out_path,
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, f"ratio_experiments_summary_seed{SEED}.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\nOzet kaydedildi: {summary_path}")
print(summary_df.to_string(index=False))

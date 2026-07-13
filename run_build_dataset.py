#!/usr/bin/env python
"""
run_build_dataset.py
=====================
G.G.A Takımı — Parametreli Veri Pipeline (Gün 11 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (11 Temmuz görevi)

Amaç:
  Gün 1-10'da ayrı ayrı yazılan negatif örnekleme stratejilerini
  (random — negative_sampling.py, BM25 hard — bm25_hard_negative.py,
  hard+random karışım — negative_mix.py) TEK parametreli komut satırı
  aracında birleştirmek. Yeni bir deney için modül kodu değiştirmeye
  gerek kalmaz, sadece parametre verilir:

    python run_build_dataset.py --mode random --ratio 3 --seed 42
    python run_build_dataset.py --mode hard   --ratio 3 --top-n 50 --seed 42
    python run_build_dataset.py --mode mix    --ratio 3 --top-n 50 --seed 42

  Çıktı dosya adı verilmezse plan'daki isimlendirme kuralına uygun
  otomatik üretilir (bkz. TAKIM_GOREV_PLANI.md bölüm 12):
  outputs/train_{mode}{ratio}_seed{seed}.csv

  --sample-size ile hızlı deney modu: sadece o kadar pozitif örnek
  (rastgele seçilmiş sorgular üzerinden) kullanılır. Verilmezse tüm
  veri (250K pozitif) işlenir.
"""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_terms, load_items
from src.negative_sampling import build_training_set, verify_no_leakage
from src.bm25_hard_negative import generate_bm25_hard_negatives
from src.negative_mix import build_mixed_training_set


def parse_args():
    p = argparse.ArgumentParser(
        description="G.G.A parametreli negatif ornekleme / veri pipeline'i"
    )
    p.add_argument("--mode", choices=["random", "hard", "mix"], default="random",
                    help="Negatif ornekleme stratejisi (varsayilan: random)")
    p.add_argument("--ratio", type=int, default=3,
                    help="Pozitif basina hedeflenen negatif sayisi (varsayilan: 3)")
    p.add_argument("--top-n", type=int, default=50,
                    help="BM25 aday havuzu buyuklugu (sadece hard/mix modu, varsayilan: 50)")
    p.add_argument("--max-df-ratio", type=float, default=0.15,
                    help="BM25 indeksi icin yaygin kelime filtresi (varsayilan: 0.15)")
    p.add_argument("--seed", type=int, default=42,
                    help="Tekrar uretilebilirlik icin random seed (varsayilan: 42)")
    p.add_argument("--sample-size", type=int, default=None,
                    help="Hizli deney icin kullanilacak pozitif sorgu sayisi (varsayilan: tum veri)")
    p.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "datasets"),
                    help="Veri seti klasoru (varsayilan: ./datasets)")
    p.add_argument("--output", default=None,
                    help="Cikti CSV yolu (verilmezse otomatik: outputs/train_{mode}{ratio}_seed{seed}.csv)")
    p.add_argument("--quiet", action="store_true",
                    help="Alt modullerin ilerleme ciktilarini bastir")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_path = args.output or os.path.join(
        output_dir, f"train_{args.mode}{args.ratio}_seed{args.seed}.csv"
    )

    print("=" * 70)
    print("  G.G.A - Parametreli Veri Pipeline (Gun 11)")
    print("=" * 70)
    print(f"  mode={args.mode}  ratio={args.ratio}  top_n={args.top_n}  "
          f"seed={args.seed}  sample_size={args.sample_size or 'tum veri'}")
    print(f"  output={out_path}")
    print("-" * 70)

    print("\n[1/3] Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(args.data_dir, "terms.csv"))
    items_df = load_items(os.path.join(args.data_dir, "items.csv"))
    train_df = pd.read_csv(
        os.path.join(args.data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )

    if args.sample_size:
        unique_terms = train_df["term_id"].drop_duplicates()
        sample_terms = unique_terms.sample(
            n=min(args.sample_size, len(unique_terms)), random_state=args.seed
        )
        train_df = train_df[train_df["term_id"].isin(sample_terms)].reset_index(drop=True)

    print(f"  Pozitif cift sayisi: {len(train_df):,}")
    print(f"  Katalog urun sayisi: {len(items_df):,}")
    print(f"  Sorgu sayisi: {len(terms_df):,}")

    print(f"\n[2/3] Negatif ornekleme (mode={args.mode})...")
    if args.mode == "random":
        full_df = build_training_set(
            train_df, items_df, ratio=args.ratio,
            random_state=args.seed, verbose=verbose,
        )
    elif args.mode == "hard":
        negatives_df = generate_bm25_hard_negatives(
            train_df, terms_df, items_df,
            top_n=args.top_n, ratio=args.ratio,
            max_df_ratio=args.max_df_ratio, verbose=verbose,
        )
        positives_df = train_df[["term_id", "item_id"]].copy()
        positives_df["label"] = 1
        full_df = pd.concat([positives_df, negatives_df], ignore_index=True)
        full_df = full_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:  # mix
        full_df = build_mixed_training_set(
            train_df, terms_df, items_df,
            ratio=args.ratio, top_n=args.top_n, max_df_ratio=args.max_df_ratio,
            random_state=args.seed, verbose=verbose,
        )

    print("\n[3/3] Sizinti kontrolu ve kayit...")
    neg_only = full_df[full_df["label"] == 0]
    leakage_ok = verify_no_leakage(neg_only, train_df)
    if not leakage_ok:
        print("  [HATA] Sizinti tespit edildi! Dosya yine de kaydediliyor, incelenmeli.")

    full_df.to_csv(out_path, index=False)

    pos_n = int((full_df["label"] == 1).sum())
    neg_n = int((full_df["label"] == 0).sum())
    print("-" * 70)
    print(f"  Kaydedildi: {out_path}")
    print(f"  Toplam: {len(full_df):,} ({pos_n:,} pozitif, {neg_n:,} negatif, "
          f"pozitif oran {pos_n / len(full_df):.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
src/neg_ratio_datasets.py
==========================
G.G.A Takimi — Negatif Oran Deney Veri Seti Uretici (10 Temmuz Mert Gorevi)

Mustafa Mert Cevik tarafından hazırlanmıştır.

Bu modul farkli negatif/pozitif oranlarda egitim setleri uretir.
Her veri seti diske kaydedilir ve deney matrisinde kullanilmak uzere hazirlanir.

Desteklenen oranlar: 1:1, 2:1, 3:1, 5:1
(Numerator = negatif, Denominator = pozitif)

Cikti dizini:
  datasets/neg_ratio/train_ratio_1_1.csv
  datasets/neg_ratio/train_ratio_2_1.csv
  datasets/neg_ratio/train_ratio_3_1.csv
  datasets/neg_ratio/train_ratio_5_1.csv
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_items
from src.negative_sampling import build_training_set

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
RATIO_DIR  = os.path.join(DATA_DIR, "neg_ratio")
os.makedirs(RATIO_DIR, exist_ok=True)

# Denenecek negatif oranlar
NEG_RATIOS  = [1, 2, 3, 5]
SAMPLE_POS  = 5_000  # Her oran icin kac pozitif kullanilacak
RANDOM_SEED = 42


def produce_ratio_datasets(
    train_raw,
    items_df,
    ratios=None,
    sample_pos=SAMPLE_POS,
    random_state=RANDOM_SEED,
    output_dir=RATIO_DIR,
    verbose=True,
):
    """
    Verilen oranlar icin egitim setlerini uretir ve CSV olarak kaydeder.

    Her oran icin:
      - Ayni pozitif ornekleri kullanir (kontrolluluk icin seed sabit)
      - Farkli sayida negatif ekler
      - Cikti: datasets/neg_ratio/train_ratio_{neg}_{pos}.csv

    Parametreler
    ----------
    train_raw : pd.DataFrame
        Ham pozitif ciftler (training_pairs.csv).
    items_df : pd.DataFrame
        Urun katalogu.
    ratios : list of int
        Negatif oranlar. Varsayilan: [1, 2, 3, 5]
    sample_pos : int
        Her oranda kullanilacak pozitif ornek sayisi.
    random_state : int
        Tekrar uretebilirlik icin seed.
    output_dir : str
        CSV dosyalarinin kaydedilecegi dizin.
    verbose : bool

    Dondurur
    -------
    dict  — {ratio: pd.DataFrame}
        Her oran icin uretilen veri seti.
    """
    if ratios is None:
        ratios = NEG_RATIOS

    # Sabit pozitif set — karsilastirma adaleti icin tum oranlarda ayni
    pos_sample = train_raw.sample(sample_pos, random_state=random_state)

    datasets = {}
    for ratio in ratios:
        out_path = os.path.join(output_dir, f"train_ratio_{ratio}_1.csv")

        # Zaten uretilmisse atla (checkpoint mantigi)
        if os.path.exists(out_path):
            if verbose:
                print(f"  [{ratio}:1] Zaten mevcut: {out_path}")
            datasets[ratio] = pd.read_csv(out_path)
            continue

        if verbose:
            print(f"  [{ratio}:1] Uretiliyor...", end=" ", flush=True)

        df = build_training_set(
            pos_sample, items_df,
            ratio=ratio,
            random_state=random_state,
            verbose=False,
        )

        # Metadata sutunlari ekle
        df["neg_ratio"]   = ratio
        df["sample_seed"] = random_state

        df.to_csv(out_path, index=False)
        datasets[ratio] = df

        pos_n = (df["label"] == 1).sum()
        neg_n = (df["label"] == 0).sum()
        if verbose:
            print(f"{len(df):,} satir  (pos={pos_n:,}, neg={neg_n:,})  -> {out_path}")

    return datasets


if __name__ == "__main__":
    print("=" * 60)
    print("  G.G.A - Negatif Oran Veri Seti Uretimi (10 Temmuz Mert)")
    print(f"  Oranlar: {NEG_RATIOS}  |  Pozitif: {SAMPLE_POS:,}")
    print("=" * 60)

    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))

    produce_ratio_datasets(train_raw, items_df, verbose=True)

    print("\n  Cikti dizini:", RATIO_DIR)
    print("=" * 60)

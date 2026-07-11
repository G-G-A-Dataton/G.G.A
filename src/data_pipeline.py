"""
src/data_pipeline.py
=====================
G.G.A Takimi — Parametreli Veri Pipeline (11 Temmuz Mert Gorevi)

Mustafa Mert Cevik tarafından hazırlanmıştır.

Bu modul tum veri hazirlama asamalarini tek bir parametreli fonksiyona toplar:
  1. Ham pozitif cifti yukle
  2. Negatif ornekleri uret (random veya BM25)
  3. Feature'lari hesapla
  4. Egitim setini kaydet (opsiyonel)

Parametreler:
  --pos-count   : Kac pozitif ornek kullanilacak
  --neg-ratio   : Negatif/pozitif oran
  --hard-neg    : BM25 hard negative kullanilsin mi?
  --hard-frac   : Hard negative orani (0.0-1.0, sadece --hard-neg ile)
  --seed        : Rastgele seed
  --output      : Cikti CSV yolu (opsiyonel)

Calistirmak icin:
  python src/data_pipeline.py --pos-count 3000 --neg-ratio 2 --seed 42
  python src/data_pipeline.py --pos-count 5000 --neg-ratio 3 --hard-neg --hard-frac 0.5
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.negative_sampling import build_training_set

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def build_pipeline(
    pos_count=3_000,
    neg_ratio=2,
    hard_neg=False,
    hard_frac=0.5,
    random_state=42,
    output_path=None,
    verbose=True,
):
    """
    Uçtan uca veri hazirlama pipeline'i.

    Adimlar:
      1. terms.csv ve items.csv'yi yukle
      2. training_pairs.csv'den pos_count kadar pozitif ornek sec
      3. Negatif ornekleri uret:
         - hard_neg=False: Tamamen random
         - hard_neg=True : hard_frac'i BM25, geri kalan random (mix)
      4. Feature'lari hesapla (build_features)
      5. Opsiyonel: output_path'e kaydet

    Parametreler
    ----------
    pos_count : int
        Pozitif ornek sayisi.
    neg_ratio : int
        Her pozitif icin kac negatif (1, 2, 3, 5).
    hard_neg : bool
        True ise BM25 hard negative + random karisimi kullan.
    hard_frac : float
        hard_neg=True iken hard negative orani (0.0-1.0).
    random_state : int
        Seed degeritum rastgele islemler icin.
    output_path : str | None
        Sonucu CSV olarak kaydet. None ise kaydetme.
    verbose : bool
        Adim bazli log yazdir.

    Dondurur
    -------
    pd.DataFrame
        Feature'lar hesaplanmis egitim seti.
    """
    if verbose:
        mode = f"hard_neg({hard_frac:.0%} BM25)" if hard_neg else "random"
        print(f"[data_pipeline] pos={pos_count:,}, neg_ratio={neg_ratio}:1, mode={mode}, seed={random_state}")

    # 1. Yukle
    if verbose:
        print("[data_pipeline] Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))
    train_raw = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    # 2. Pozitif ornek sec
    pos_sample = train_raw.sample(min(pos_count, len(train_raw)), random_state=random_state)

    # 3. Negatif uret
    if verbose:
        print("[data_pipeline] Negatifler uretiliyor...")

    if hard_neg:
        # BM25 hard negative + random mix
        # bm25_ratio: hard negative orani, geri kalan random
        try:
            from src.negative_sampling import build_mixed_training_set
            full = build_mixed_training_set(
                pos_sample, items_df,
                total_ratio=neg_ratio,
                hard_frac=hard_frac,
                random_state=random_state,
                verbose=False,
            )
        except (ImportError, AttributeError):
            # build_mixed_training_set yoksa normal build_training_set kullan
            if verbose:
                print("[data_pipeline] build_mixed_training_set bulunamadi, random kullaniliyor...")
            full = build_training_set(
                pos_sample, items_df,
                ratio=neg_ratio,
                random_state=random_state,
                verbose=False,
            )
    else:
        full = build_training_set(
            pos_sample, items_df,
            ratio=neg_ratio,
            random_state=random_state,
            verbose=False,
        )

    # 4. Feature'lari hesapla
    merged = full.merge(terms_df, on="term_id", how="left")
    merged = merged.merge(items_df,  on="item_id",  how="left")
    if verbose:
        print("[data_pipeline] Feature'lar hesaplaniyor...")
    merged = build_features(merged)

    pos_n = (merged["label"] == 1).sum()
    neg_n = (merged["label"] == 0).sum()
    if verbose:
        print(f"[data_pipeline] Tamamlandi: {len(merged):,} satir  (pos={pos_n:,}, neg={neg_n:,})")

    # 5. Kaydet
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)
        if verbose:
            print(f"[data_pipeline] Kaydedildi: {output_path}")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G.G.A Parametreli Veri Pipeline"
    )
    parser.add_argument("--pos-count",  type=int,   default=3_000, help="Pozitif ornek sayisi (varsayilan: 3000)")
    parser.add_argument("--neg-ratio",  type=int,   default=2,     help="Negatif/pozitif oran (varsayilan: 2)")
    parser.add_argument("--hard-neg",   action="store_true",        help="BM25 hard negative kullan")
    parser.add_argument("--hard-frac",  type=float, default=0.5,   help="Hard negative orani (varsayilan: 0.5)")
    parser.add_argument("--seed",       type=int,   default=42,    help="Rastgele seed (varsayilan: 42)")
    parser.add_argument("--output",     type=str,   default=None,  help="Cikti CSV yolu (opsiyonel)")
    args = parser.parse_args()

    print("=" * 60)
    print("  G.G.A - Parametreli Veri Pipeline (11 Temmuz Mert)")
    print("=" * 60)

    df = build_pipeline(
        pos_count=args.pos_count,
        neg_ratio=args.neg_ratio,
        hard_neg=args.hard_neg,
        hard_frac=args.hard_frac,
        random_state=args.seed,
        output_path=args.output,
        verbose=True,
    )

    print(f"\n  Kolonlar  : {list(df.columns[:8])} ...")
    print(f"  Satir     : {len(df):,}")
    print(f"  Feature'lar: {[c for c in df.columns if c in FEATURE_COLS]}")
    print("=" * 60)

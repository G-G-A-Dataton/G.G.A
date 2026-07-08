"""
notebooks/02_negative_sampling_mert.py
=======================================
G.G.A Takımı — Random Negative Üretimi (Gün 3 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (3 Temmuz görevi)

Neden negatif örneklemeye ihtiyaç var?
  training_pairs.csv'de SADECE pozitif (label=1) çiftler var. Ama bir
  model 0 ve 1'i ayırt etmeyi öğrenmek zorunda; bu yüzden kendimiz "bu
  çift alakasız" (label=0) örnekleri üretiyoruz.

Bu script şunları sağlar:
  1. 1:1, 3:1, 5:1 oranlarında negatif örnek üretimi
     (üretim mantığı src/negative_sampling.py'deki
     generate_random_negatives'te yaşıyor, burada tekrar yazılmıyor)
  2. Üretilen her oran için pozitif+negatif birleşik eğitim setini
     karıştırıp train_random{oran}_seed{SEED}.parquet olarak kaydeder
  3. Tekrar üretilebilirlik için sabit seed (42) kullanımı
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # src.negative_sampling'i import edebilmek için

from src.negative_sampling import generate_random_negatives  # noqa: E402

DATA = ROOT / "datasets"
OUT = ROOT / "artifacts" / "data"

SEED = 42            # tekrar üretilebilirlik: raporlara bu sayı yazılacak
ORANLAR = [1, 3, 5]  # pozitif başına negatif sayısı (1:1, 3:1, 5:1)

pairs = pd.read_csv(DATA / "training_pairs.csv", dtype=str)
pairs["label"] = pairs["label"].astype("int8")
items = pd.read_csv(DATA / "items.csv", usecols=["item_id"], dtype=str)
item_havuzu = items["item_id"].to_numpy()

print(f"pozitif çift: {len(pairs):,} | ürün havuzu: {len(item_havuzu):,}")


def negatif_uret(oran: int, seed: int) -> pd.DataFrame:
    """Her pozitif çift için `oran` adet negatif çift üretir.

    Üretim mantığı src/negative_sampling.py'deki generate_random_negatives'te
    yaşıyor; bu sarmalayıcı sadece dosyaya kayıt için gereken "id" kolonunu
    ekliyor.
    """
    negatifler = generate_random_negatives(
        pairs, items, ratio=oran, random_state=seed, verbose=True,
    )
    negatifler["label"] = negatifler["label"].astype("int8")
    # Negatiflere de takip için benzersiz id veriyoruz (pozitiflerde TRN_ vardı).
    negatifler.insert(0, "id", [f"NEG_{i:07d}" for i in range(len(negatifler))])
    return negatifler


# ---------------------------------------------------------------
# 2) HER ORAN İÇİN ÜRET, POZİTİFLERLE BİRLEŞTİR, KAYDET
# ---------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
for oran in ORANLAR:
    print(f"\n=== oran 1:{oran} ===")
    negatifler = negatif_uret(oran, SEED)

    train = pd.concat(
        [pairs[["id", "term_id", "item_id", "label"]], negatifler],
        ignore_index=True,
    )
    # Karıştırma da seed'li: pozitifler üstte negatifler altta kalmasın.
    # (Sıralı bırakırsak bazı validasyon bölmeleri yanlı olabilir.)
    train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    yol = OUT / f"train_random{oran}_seed{SEED}.parquet"
    train.to_parquet(yol, index=False)
    print(f"  -> {yol.name}: {len(train):,} satır "
          f"({(train['label']==1).sum():,} pozitif / {(train['label']==0).sum():,} negatif)")


from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = ROOT / "artifacts" / "data"

SEED = 42            # tekrar üretilebilirlik: raporlara bu sayı yazılacak
ORANLAR = [1, 3, 5]  # pozitif başına negatif sayısı (1:1, 3:1, 5:1)

pairs = pd.read_csv(DATA / "training_pairs.csv", dtype=str)
pairs["label"] = pairs["label"].astype("int8")
items = pd.read_csv(DATA / "items.csv", usecols=["item_id"], dtype=str)
item_havuzu = items["item_id"].to_numpy()

pozitif_anahtarlar = set(pairs["term_id"] + "|" + pairs["item_id"])
print(f"pozitif çift: {len(pairs):,} | ürün havuzu: {len(item_havuzu):,}")


def negatif_uret(oran: int, seed: int) -> pd.DataFrame:
    """Her pozitif çift için `oran` adet negatif çift üretir."""
    # default_rng: seed'li, izole rastgelelik kaynağı. Aynı seed → aynı sayılar.
    rng = np.random.default_rng(seed)

    # np.repeat: her pozitif çiftin term'ini `oran` kez tekrarla.
    # Örn. oran=3 ise TERM_A'nın 14 pozitifi varsa listede 42 kez geçer →
    # G3 garantisi: term'lerin ağırlığı pozitiflerdeki ile aynı kalır.
    uretilecek_termler = np.repeat(pairs["term_id"].to_numpy(), oran)
    hedef = len(uretilecek_termler)

    kabul_edilenler: list[pd.DataFrame] = []
    kabul_anahtarlari: set[str] = set()

    # Reddetmeli örnekleme (rejection sampling) döngüsü:
    # aday üret → geçersizleri ele → elenenlerin term'leri için tekrar dene.
    tur = 0
    while len(uretilecek_termler) > 0:
        tur += 1
        adaylar = pd.DataFrame({
            "term_id": uretilecek_termler,
            "item_id": rng.choice(item_havuzu, size=len(uretilecek_termler)),
        })
        anahtar = adaylar["term_id"] + "|" + adaylar["item_id"]

        # Üç eleme kuralı: (~ = değil, & = ve)
        gecerli = (
            ~anahtar.isin(pozitif_anahtarlar)   # G1: pozitifle çakışma
            & ~anahtar.isin(kabul_anahtarlari)  # G2: önceki turlarla tekrar
            & ~anahtar.duplicated()             # G2: bu tur içinde tekrar
        )
        kabul_edilenler.append(adaylar[gecerli])
        kabul_anahtarlari.update(anahtar[gecerli])

        elenen = (~gecerli).sum()
        print(f"  oran {oran} | tur {tur}: {gecerli.sum():,} kabul, {elenen:,} elendi")

        # Elenen adayların term'leri yeniden denenecek → her term tam
        # hakkını alana kadar döngü sürer (havuz dev, 2-3 turda biter).
        uretilecek_termler = adaylar.loc[~gecerli, "term_id"].to_numpy()

    negatifler = pd.concat(kabul_edilenler, ignore_index=True)
    assert len(negatifler) == hedef  # G3 sağlandı mı?
    negatifler["label"] = np.int8(0)
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
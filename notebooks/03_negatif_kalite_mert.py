"""
notebooks/03_negatif_kalite_mert.py
====================================
G.G.A Takımı — Negatif Kalite Kontrolü (Gün 4 görevi)

Mustafa Mert Çevik tarafından hazırlanmıştır. (4 Temmuz görevi)

Neden bağımsız bir doğrulama gerekiyor?
  02'nin ürettiği negatif örnekler, kendi assert'lerine güvenilerek
  doğru kabul edilemez — üretim kodundaki bir hata kendi kontrolünü de
  yanıltabilir. Bu yüzden train_random*.parquet dosyaları diskten
  açılıp ham veriyle (training_pairs.csv, items.csv) sıfırdan,
  bağımsız olarak karşılaştırılır.

Bu script şunları kontrol eder:
  N1. Hiçbir negatif gerçek bir pozitifle çakışmıyor
  N2. Negatifler kendi içinde tekrarsız
  N3. Satır sayıları beklenen orana uyuyor (oran x pozitif sayısı)
  N4. Oran her term bazında da korunmuş (sadece toplamda değil)
  N5. Negatiflerdeki tüm id'ler gerçek (katalogda/pozitiflerde var)
  N6. Pozitifler dosyaya kayıpsız/bozulmasız girmiş

Çalıştırma:  python notebooks/03_negatif_kalite_mert.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
ART = ROOT / "artifacts" / "data"

SEED = 42
ORANLAR = [1, 3, 5]

# ---------------------------------------------------------------
# Referans: ham pozitifler ve katalog (kontrolün "doğruluk kaynağı")
# ---------------------------------------------------------------
pairs = pd.read_csv(DATA / "training_pairs.csv", dtype=str)
items = pd.read_csv(DATA / "items.csv", usecols=["item_id"], dtype=str)

pozitif_anahtarlar = set(pairs["term_id"] + "|" + pairs["item_id"])
katalog = set(items["item_id"])

# Her term'in kaç pozitifi var? (groupby = "term_id'ye göre grupla,
# her grubun satırını say"). N4'te oran kontrolü için lazım.
pozitif_sayilari = pairs.groupby("term_id").size()


# kosul hem Python bool'u hem numpy bool'u olabilir: pandas/numpy
# karşılaştırmaları (==, .all(), .sum()==0) np.bool_ döndürür.
# Sözleşmeyi gerçeğe uydurup dönüşte saf bool'a çeviriyoruz.
def kontrol(ad: str, kosul: bool | np.bool_, detay: str = "") -> bool:
    isaret = "✓" if kosul else "✗ HATA"
    print(f"  {isaret} | {ad}" + (f" ({detay})" if detay else ""))
    return bool(kosul)


hepsi_gecti = True
for oran in ORANLAR:
    yol = ART / f"train_random{oran}_seed{SEED}.parquet"
    df = pd.read_parquet(yol)
    poz = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    neg_anahtar = neg["term_id"] + "|" + neg["item_id"]
    print(f"\n=== {yol.name} ===")

    sonuclar = [
        # N1 — EN KRİTİK: hiçbir negatif gerçek bir pozitifle çakışmıyor.
        kontrol("N1 negatif-pozitif çakışması yok",
                neg_anahtar.isin(pozitif_anahtarlar).sum() == 0),

        # N2 — negatifler kendi içinde tekrarsız.
        kontrol("N2 negatiflerde tekrar yok",
                neg_anahtar.duplicated().sum() == 0),

        # N3 — sayılar beklenen: 250K pozitif, oran x 250K negatif.
        kontrol("N3 satır sayıları doğru",
                len(poz) == len(pairs) and len(neg) == oran * len(pairs),
                f"{len(poz):,} poz / {len(neg):,} neg"),

        # N4 — oran TERM BAZINDA da korunmuş mu? Toplam doğru olsa bile
        # bir term fazla, diğeri eksik almış olabilirdi.
        # reindex: iki sayım tablosunu aynı term sırasına getirir ki
        # karşılaştırma satır satır doğru eşleşsin.
        kontrol("N4 her term'de oran korunmuş",
                (neg.groupby("term_id").size()
                    .reindex(pozitif_sayilari.index, fill_value=0)
                 == oran * pozitif_sayilari).all()),

        # N5 — negatiflerdeki ID'ler gerçek: item katalogda, term pozitiflerde.
        kontrol("N5 tüm ID'ler geçerli",
                neg["item_id"].isin(katalog).all()
                and neg["term_id"].isin(set(pairs["term_id"])).all()),

        # N6 — pozitifler dosyaya kayıpsız/bozulmasız girmiş:
        # dosyadaki pozitif çift kümesi == ham pozitif çift kümesi.
        kontrol("N6 pozitifler birebir korunmuş",
                set(poz["term_id"] + "|" + poz["item_id"]) == pozitif_anahtarlar),
    ]
    hepsi_gecti &= all(sonuclar)

    # İçerik parmak izi: satırların İÇERİĞİNDEN üretilen özet sayı.
    # 02'yi aynı seed'le tekrar çalıştırıp bunu yeniden yazdırırsan aynı
    # sayı çıkmalı → tekrar üretilebilirliğin kanıtı. Rapora yazılacak.
    parmak_izi = pd.util.hash_pandas_object(df, index=False).sum()
    print(f"  içerik parmak izi: {parmak_izi}")

print()
print("SONUÇ:", "TÜM KONTROLLER GEÇTİ ✓" if hepsi_gecti else "HATA VAR ✗")
raise SystemExit(0 if hepsi_gecti else 1)

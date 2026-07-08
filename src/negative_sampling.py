"""
src/negative_sampling.py
========================
G.G.A Takımı — Negatif Örnek Üretimi (Random Negative Mining)

Mustafa Mert Çevik tarafından hazırlanmıştır. (3 Temmuz görevi)

Neden negatif örneklemeye ihtiyaç var?
  training_pairs.csv'de SADECE pozitif (label=1) çiftler var.
  Ama bir model 0 ve 1'i ayırt etmeyi öğrenmek zorunda.
  Bu yüzden kendimiz "bu çift alakasız" (label=0) örnekleri üretiyoruz.

Bu modül şunları sağlar:
  1. Random Negative: Rastgele bir ürünü negatif olarak seç (basit ama etkili başlangıç)
  2. Farklı negatif oranlarında (1:1, 3:1, 5:1) örnek dataset üretimi
  3. Üretilen negatif örneklerin asla pozitiflerle çakışmadığının garantisi
  4. Tekrar üretilebilirlik için sabit seed (42) kullanımı
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temel Random Negative Üretici
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_negatives(
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
    ratio: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Her pozitif (term_id, item_id) çifti için rastgele 'ratio' kadar negatif üretir.

    Yöntem:
      - Her sorgu (term_id) için pozitif olduğu bilinen ürünleri not al
      - Kataloğun geri kalanından rastgele ürün seç → label=0

    Önemli güvence: Seçilen ürün hiçbir zaman o sorgunun pozitif ürünü olmaz.
    Bu olursa yanlış negatif (false negative) üretilmiş olur!

    Parametreler
    ----------
    train_df : pd.DataFrame
        Pozitif çiftler (term_id, item_id). training_pairs.csv'den gelir.
    items_df : pd.DataFrame
        Tüm ürün kataloğu. Rastgele seçim için item_id listesi alınır.
    ratio : int, default=1
        Her pozitif çift için üretilecek negatif sayısı.
        ratio=1 → 1 pozitife 1 negatif (dengeli veri seti)
        ratio=3 → 1 pozitife 3 negatif (genellikle en iyi başlangıç)
        ratio=5 → 1 pozitife 5 negatif (model daha "dikkatli" olur ama yavaş eğitim)
    random_state : int, default=42
        Rastgele sayı üreteci tohumu. Aynı seed → aynı çıktı (tekrar üretilebilirlik).
    verbose : bool, default=True
        İlerleme bilgisi yazdır.

    Döndürür
    -------
    pd.DataFrame
        Kolonlar: term_id, item_id, label (hepsi 0)
    """
    # NOT (performans): Önceki sürüm her term için `[x for x in all_item_ids
    # if x not in pos_set]` ile TÜM katalogu Python döngüsünde tarıyordu.
    # Bu, term sayısı x katalog boyutu kadar işlem demek (gerçek veride
    # ~50K term x ~963K item ≈ 48 milyar adım) — pratikte bitmiyor.
    # Bunun yerine "reddetmeli örnekleme" (rejection sampling) kullanıyoruz:
    # tüm ihtiyacı tek seferde vektörize üret, geçersizleri ele, sadece
    # elenenler için tekrar dene. Sonuç aynı (term başına tam `ratio` adet,
    # asla pozitifle çakışmayan negatif) ama birkaç turda ve tamamen
    # pandas/numpy vektör işlemleriyle biter.
    rng = np.random.default_rng(random_state)
    all_item_ids = items_df["item_id"].to_numpy()

    pozitif_anahtarlar = set(train_df["term_id"] + "|" + train_df["item_id"])

    # Her pozitif çiftin term'ini 'ratio' kez tekrarla → term ağırlığı
    # pozitiflerdeki ile aynı kalır (bir term'in 10 pozitifi varsa 10*ratio
    # negatif üretilir).
    uretilecek_termler = np.repeat(train_df["term_id"].to_numpy(), ratio)
    hedef = len(uretilecek_termler)

    kabul_edilenler = []
    kabul_anahtarlari = set()
    tur = 0

    while len(uretilecek_termler) > 0:
        tur += 1
        adaylar = pd.DataFrame({
            "term_id": uretilecek_termler,
            "item_id": rng.choice(all_item_ids, size=len(uretilecek_termler)),
        })
        anahtar = adaylar["term_id"] + "|" + adaylar["item_id"]

        gecerli = (
            ~anahtar.isin(pozitif_anahtarlar)   # pozitifle çakışmıyor
            & ~anahtar.isin(kabul_anahtarlari)  # önceki turlarla tekrar değil
            & ~anahtar.duplicated()             # bu tur içinde tekrar değil
        )
        kabul_edilenler.append(adaylar[gecerli])
        kabul_anahtarlari.update(anahtar[gecerli])

        if verbose:
            print(f"  [negative_sampling] tur {tur}: {gecerli.sum():,} kabul, "
                  f"{(~gecerli).sum():,} elendi (kalan hedef: {len(uretilecek_termler) - gecerli.sum():,})")

        uretilecek_termler = adaylar.loc[~gecerli, "term_id"].to_numpy()

    negatives = pd.concat(kabul_edilenler, ignore_index=True)
    assert len(negatives) == hedef, "Uretilen negatif sayisi beklenenden farkli!"
    negatives["label"] = 0

    if verbose:
        print(f"  [negative_sampling] Toplam {len(negatives):,} negatif ornek uretildi.")

    return negatives[["term_id", "item_id", "label"]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tam Eğitim Seti Oluşturucu (Pozitif + Negatif)
# ─────────────────────────────────────────────────────────────────────────────

def build_training_set(
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
    ratio: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pozitif çiftlere negatif örnekleri ekleyerek tam eğitim seti oluşturur.

    Parametreler
    ----------
    train_df : pd.DataFrame
        Orijinal pozitif çiftler (training_pairs.csv).
    items_df : pd.DataFrame
        Ürün kataloğu.
    ratio : int, default=3
        Her pozitife kaç negatif üretileceği.
    random_state : int, default=42
        Tekrar üretilebilirlik için seed.

    Döndürür
    -------
    pd.DataFrame
        Karıştırılmış pozitif + negatif çiftler. Kolonlar: term_id, item_id, label
    """
    if verbose:
        pos_count = len(train_df)
        print(f"[build_training_set] Pozitif ornekler: {pos_count:,}")
        print(f"[build_training_set] Ratio {ratio}:1 ile negatif uretiliyor...")

    # Negatif örnekleri üret
    negatives_df = generate_random_negatives(
        train_df, items_df, ratio=ratio,
        random_state=random_state, verbose=verbose
    )

    # Pozitif çiftlere label=1 ekle (eğer yoksa)
    positives_df = train_df[["term_id", "item_id"]].copy()
    positives_df["label"] = 1

    # Pozitif ve negatif örnekleri birleştir
    full_df = pd.concat([positives_df, negatives_df], ignore_index=True)

    # Rastgele karıştır (modelin sıralı veri görmemesi için önemli)
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if verbose:
        neg_count = len(negatives_df)
        print(f"[build_training_set] Sonuc: {pos_count:,} pozitif + {neg_count:,} negatif = {len(full_df):,} toplam")
        print(f"[build_training_set] Pozitif oran: {pos_count/len(full_df):.1%}")

    return full_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Çoklu Oran Deneyi Üreticisi
# ─────────────────────────────────────────────────────────────────────────────

def generate_ratio_experiments(
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
    ratios: list = [1, 3, 5],
    sample_size: Optional[int] = 5000,
    random_state: int = 42,
) -> dict:
    """
    Farklı negatif oranları (1:1, 3:1, 5:1) için küçük örnek veri setleri üretir.

    3 Temmuz görevine göre: "1:1, 3:1, 5:1 oranları için küçük örnek dataset çıkar"
    Bu fonksiyon her oranı deneyebilmek için birden fazla DataFrame döndürür.

    Parametreler
    ----------
    train_df : pd.DataFrame
        Pozitif çiftler.
    items_df : pd.DataFrame
        Ürün kataloğu.
    ratios : list of int, default=[1, 3, 5]
        Denenecek negatif oranları.
    sample_size : int or None, default=5000
        Her deneyde kullanılacak pozitif örnek sayısı (hızlı deney için).
        None verilirse tüm pozitifler kullanılır.
    random_state : int, default=42
        Seed.

    Döndürür
    -------
    dict
        { ratio: pd.DataFrame } şeklinde her oran için bir veri seti.
    """
    # Hızlı deney için pozitiflerden küçük bir örnek al
    if sample_size and len(train_df) > sample_size:
        sample_train = train_df.sample(n=sample_size, random_state=random_state)
        print(f"[ratio_experiments] {sample_size:,} pozitif ornekle calisiyor (hizli deney modu).")
    else:
        sample_train = train_df
        print(f"[ratio_experiments] Tum {len(train_df):,} pozitif ornekle calisiyor.")

    results = {}
    for ratio in ratios:
        print(f"\n--- Ratio {ratio}:1 ---")
        dataset = build_training_set(
            sample_train, items_df, ratio=ratio,
            random_state=random_state, verbose=True
        )
        results[ratio] = dataset

    # Özet tablosu
    print("\n=== ORAN DENEY OZETI ===")
    print(f"{'Oran':<10} {'Toplam':<12} {'Pozitif':<12} {'Negatif':<12} {'Poz. Oran'}")
    print("-" * 55)
    for ratio, df in results.items():
        pos = (df["label"] == 1).sum()
        neg = (df["label"] == 0).sum()
        print(f"{ratio}:1{'':<7} {len(df):<12,} {pos:<12,} {neg:<12,} {pos/len(df):.1%}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tekrar Üretilebilirlik Kontrolü
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(negatives_df, positives_df):
    """
    Üretilen negatif örneklerin asla pozitif çiftlerle çakışmadığını doğrular.

    Eğer bir (term_id, item_id) çifti hem pozitif hem negatif olarak işaretlenmişse
    bu 'veri sızıntısı' (data leakage) demektir ve modeli bozar.

    Parametreler
    ----------
    negatives_df : pd.DataFrame
        Üretilen negatif örnekler (term_id, item_id, label=0).
    positives_df : pd.DataFrame
        Orijinal pozitif çiftler (term_id, item_id).

    Döndürür
    -------
    bool
        True: Sızıntı yok. False: Sızıntı var!
    """
    # Her iki taraftaki çiftleri (term_id, item_id) tuple seti olarak al
    pos_pairs = set(zip(positives_df["term_id"], positives_df["item_id"]))
    neg_pairs = set(zip(negatives_df["term_id"], negatives_df["item_id"]))

    # Kesişim = hem pozitif hem negatif olarak işaretlenmiş çiftler
    overlap = pos_pairs & neg_pairs

    if overlap:
        print(f"[UYARI] {len(overlap)} cift hem pozitif hem negatif! Sızıntı var!")
        return False
    else:
        print(f"[OK] Sizinti kontrolu gecti. 0 cakisan cift bulundu.")
        return True


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data import load_terms, load_items

    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    print("Veriler yukleniyor...")
    items_df = load_items(os.path.join(data_dir, "items.csv"))
    train_df = pd.read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )

    print(f"Pozitif cift sayisi: {len(train_df):,}")
    print(f"Katalog urun sayisi: {len(items_df):,}")

    # Küçük örnek üzerinde 3 farklı oranı dene
    experiments = generate_ratio_experiments(
        train_df, items_df, ratios=[1, 3, 5], sample_size=1000
    )

    # Sızıntı kontrolü
    print("\nSizinti kontrolu yapiliyor...")
    for ratio, dataset in experiments.items():
        neg = dataset[dataset["label"] == 0]
        verify_no_leakage(neg, train_df)

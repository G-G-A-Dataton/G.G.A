
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = ROOT / "artifacts" / "data"

terms = pd.read_csv(DATA / "terms.csv", dtype=str)
items = pd.read_csv(
    DATA / "items.csv",
    usecols=["item_id", "title", "category", "brand", "gender", "age_group"],
    dtype=str,
)
pairs = pd.read_csv(DATA / "training_pairs.csv", dtype=str)
pairs["label"] = pairs["label"].astype("int8")




# how="left"  : soldaki tablonun (pairs) TÜM satırları korunur.
#               Eşleşme bulunamazsa satır kaybolmaz, boşluk (NaN) kalır.
#               Böylece kayıp "sessiz" değil, ölçülebilir olur.
# validate    : "many_to_one" = soldaki tabloda aynı ID çok kez
#               geçebilir ama sağdaki tabloda TEK olmalı. Değilse
#               pandas hata fırlatır → satır patlamasına karşı sigorta.
merged = pairs.merge(terms, on="term_id", how="left", validate="many_to_one")
merged = merged.merge(items, on="item_id", how="left", validate="many_to_one")


assert len(merged) == len(pairs), "Satır sayısı değişti!"

print(f"Satır sayısı        : {len(merged):,} (değişmedi ✓)")
print(f"query'si boş satır  : {merged['query'].isna().sum()}")
print(f"title'ı boş satır   : {merged['title'].isna().sum()}")
print()


ornekler = merged.sample(5, random_state=42)
for _, satir in ornekler.iterrows():
    print(f"ARAMA : {satir['query']}")
    print(f"ÜRÜN  : {satir['title']}")
    print(f"        [{satir['category']} | {satir['brand']} | {satir['gender']} | {satir['age_group']}]")
    print()


OUT.mkdir(parents=True, exist_ok=True)
cikti = OUT / "train_positives_merged.parquet"
merged.to_parquet(cikti, index=False)
print(f"Kaydedildi: {cikti} ({cikti.stat().st_size / 1e6:.1f} MB)")
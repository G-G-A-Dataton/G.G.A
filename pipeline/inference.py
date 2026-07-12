"""
scripts/submission/run_pipeline.py
==================================
G.G.A Takımı — Uçtan Uca Feature Üretim ve Prediction Pipeline (14 Temmuz Görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Bu script, ham veriden nihai Kaggle submission dosyasına kadar olan tüm
akışı tek bir komutla çalıştırır.

Kullanım:
  python scripts/submission/run_pipeline.py --mode predict
  python scripts/submission/run_pipeline.py --mode predict --sample 10000
"""

import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data              import load_terms, load_items
from src.features          import build_features, FEATURE_COLS
from src.tfidf_features    import add_tfidf_features, load_vectorizer
from src.validate_submission import validate_submission

# Loglama Ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "outputs", "pipeline.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("G.G.A.Pipeline")

DATA_DIR    = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATHS = [os.path.join(OUTPUT_DIR, f"lgbm_v2_fold_{i}.txt") for i in range(1, 6)]
VEC_PATH    = os.path.join(OUTPUT_DIR, "tfidf_vectorizer_v2.pkl")
SUB_OUTPUT  = os.path.join(OUTPUT_DIR, "submission_v2.csv")


def parse_args():
    p = argparse.ArgumentParser(description="G.G.A Uçtan Uca Prediction Pipeline")
    p.add_argument("--mode", choices=["predict"], default="predict", help="Çalışma modu")
    p.add_argument("--sample", type=int, default=None, help="Hızlı test için test setinden alınacak örnek satır sayısı")
    p.add_argument("--batch-size", type=int, default=100000, help="Bellek tasarrufu için batch boyutu")
    return p.parse_args()


def check_dependencies():
    """Gerekli model ve vectorizer dosyalarının varlığını kontrol eder."""
    logger.info("Dosya bağımlılıkları kontrol ediliyor...")
    missing = []
    for m in MODEL_PATHS:
        if not os.path.exists(m):
            missing.append(m)
    if not os.path.exists(VEC_PATH):
        missing.append(VEC_PATH)
        
    if missing:
        logger.error("Aşağıdaki model bağımlılıkları eksik:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("Lütfen önce modelleri eğitin: python scripts/training/run_model_shortlist.py")
        sys.exit(1)
    logger.info("Tüm model bağımlılıkları doğrulandı.")


def run_prediction_pipeline(args):
    t_start = time.time()
    logger.info("G.G.A Uçtan Uca Prediction Pipeline Başlatıldı.")

    # 1. Bağımlılık Kontrolü
    check_dependencies()

    # 2. Veri Yükleme
    logger.info("Adım 1/5: Ham veri setleri yükleniyor (terms.csv, items.csv)...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))

    sub_path = os.path.join(DATA_DIR, "submission_pairs.csv")
    logger.info(f"Adım 2/5: Arama çiftleri yükleniyor: {sub_path}")
    sub_df = pd.read_csv(sub_path, dtype={"id": "string", "term_id": "string", "item_id": "string"})

    if args.sample:
        logger.warning(f"Hızlı test modu aktif: {args.sample:,} satır örneklenecek.")
        sub_df = sub_df.sample(args.sample, random_state=42).reset_index(drop=True)

    # 3. Model ve Vectorizer Yükleme
    logger.info("Adım 3/5: Modeller ve TF-IDF Vectorizer yükleniyor...")
    models = [lgb.Booster(model_file=p) for p in MODEL_PATHS]
    vectorizer = load_vectorizer(VEC_PATH)

    # 4. Batch Tahmin ve Özellik Üretimi
    logger.info("Adım 4/5: Batch'ler halinde özellik üretimi ve inference başlıyor...")
    n_rows = len(sub_df)
    batch_size = args.batch_size
    all_preds = []

    feature_cols = FEATURE_COLS + ["tfidf_cosine"]

    for i in range(0, n_rows, batch_size):
        t_b0 = time.time()
        batch_df = sub_df.iloc[i : i + batch_size].copy()
        
        # Join işlemleri
        batch_merged = batch_df.merge(terms_df, on="term_id", how="left")
        batch_merged = batch_merged.merge(items_df, on="item_id", how="left")
        
        # Özellik üretimi
        batch_features = build_features(batch_merged)
        batch_features = add_tfidf_features(batch_features, vectorizer)
        
        X_batch = batch_features[feature_cols]
        
        # 5-Fold model tahmini (ortalama alarak)
        batch_pred = np.zeros(len(X_batch))
        for model in models:
            batch_pred += model.predict(X_batch) / len(models)
            
        all_preds.append(batch_pred)
        t_b1 = time.time()
        
        processed = min(i + batch_size, n_rows)
        logger.info(f"  İşlenen: {processed:,} / {n_rows:,} ({processed/n_rows:.1%}) | Hız: {len(X_batch)/(t_b1-t_b0):.0f} satır/sn")

    preds_arr = np.concatenate(all_preds)

    # 5. Threshold ve Çıktı Üretimi
    logger.info("Adım 5/5: Karar eşiği uygulanıyor ve çıktı dosyası oluşturuluyor...")
    threshold = 0.35  # En iyi threshold değeri
    sub_df["label"] = (preds_arr >= threshold).astype(int)

    logger.info(f"Sonuçlar kaydediliyor: {SUB_OUTPUT}")
    sub_df[["id", "label"]].to_csv(SUB_OUTPUT, index=False)

    # Submission QA kontrolü
    logger.info("Submission dosyası format doğrulaması yapılıyor...")
    validate_submission(sub_df, sub_df)

    t_end = time.time()
    logger.info(f"Pipeline başarıyla tamamlandı! Toplam süre: {t_end - t_start:.1f} saniye.")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "predict":
        run_prediction_pipeline(args)

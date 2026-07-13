"""
src/submission.py
=================
G.G.A Takımı — Submission CSV Üretme Modülü (5 Temmuz Görevi)

Ömer Faruk Kara tarafından hazırlanmıştır.

Bu modül eğitilmiş LightGBM modellerini kullanarak submission_pairs.csv
için tahmin üretir ve Kaggle formatında CSV dosyası oluşturur.

Kullanım:
  from src.submission import generate_submission
  generate_submission(models, vectorizer, terms_df, items_df, feature_cols)
"""

import os
import numpy as np
import pandas as pd
from src.features       import build_features
from src.tfidf_features import add_tfidf_features
from src.validate_submission import validate_submission


def prepare_submission_features(
    submission_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    vectorizer=None,
    feature_cols: list = None,
    batch_size: int = 50_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Submission çiftleri için feature'ları hesaplar.

    Büyük submission seti (3.36M satır) için feature hesaplamasını batch'ler;
    döndürülen birleşik DataFrame yine tüm feature setini bellekte tutar.

    Parametreler
    ----------
    submission_df : pd.DataFrame
        submission_pairs.csv içeriği (id, term_id, item_id kolonları).
    terms_df, items_df : pd.DataFrame
        Sorgu ve ürün verileri.
    vectorizer : TfidfVectorizer or None
        TF-IDF vectorizer (None ise TF-IDF feature hesaplanmaz).
    feature_cols : list
        Kullanılacak feature kolon adları.
    batch_size : int
        Her batch'te işlenecek satır sayısı.
    verbose : bool
        İlerleme bilgisi yazdır.

    Döndürür
    -------
    pd.DataFrame
        Feature'ları hesaplanmış submission seti.
    """
    n = len(submission_df)
    if verbose:
        print(f"[submission] {n:,} satir icin feature hesaplaniyor...")

    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = submission_df.iloc[start:end].copy()

        # Sorgu ve ürün bilgilerini birleştir
        batch = batch.merge(terms_df, on="term_id", how="left")
        batch = batch.merge(items_df,  on="item_id",  how="left")

        # Temel feature'ları hesapla
        batch = build_features(batch)

        # TF-IDF cosine feature ekle (varsa)
        if vectorizer is not None:
            batch = add_tfidf_features(batch, vectorizer, batch_size=batch_size)

        batches.append(batch)

        if verbose:
            print(f"  ... {end:,}/{n:,} satir islendi")

    result = pd.concat(batches, ignore_index=True)
    if verbose:
        print(f"[submission] Feature hesaplama tamamlandi.")
    return result


def generate_submission(
    models: list,
    terms_df: pd.DataFrame,
    items_df: pd.DataFrame,
    feature_cols: list,
    submission_pairs_path: str,
    output_path: str,
    vectorizer=None,
    threshold: float = 0.5,
    batch_size: int = 50_000,
    verbose: bool = True,
    sample_submission_path: str = None,
) -> pd.DataFrame:
    """
    Eğitilmiş model(ler) ile Kaggle formatında submission CSV üretir.

    Birden fazla model varsa ensemble (ortalama) kullanılır.
    Bu sayede tek modelden daha kararlı (robust) tahminler elde edilir.

    Parametreler
    ----------
    models : list
        Eğitilmiş LightGBM model nesneleri (5-fold'dan 5 model).
    terms_df, items_df : pd.DataFrame
        Sorgu ve ürün verileri.
    feature_cols : list
        Modelin kullandığı feature kolon adları.
    submission_pairs_path : str
        submission_pairs.csv dosyasının yolu.
    output_path : str
        Üretilecek submission.csv'nin kaydedileceği yol.
    vectorizer : TfidfVectorizer or None
        TF-IDF vectorizer (kullanılıyorsa).
    threshold : float, default=0.5
        Olasılığı binary tahmine çevirmek için eşik değeri.
        find_best_threshold() ile belirlenen değer kullanılmalıdır.
    batch_size : int
        Her batch'te işlenecek satır sayısı.
    sample_submission_path : str, optional
        Satır sayısı ve ID sırası doğrulaması için referans submission yolu.

    Döndürür
    -------
    pd.DataFrame
        Üretilen submission DataFrame'i (id, prediction kolonları).
    """
    if verbose:
        print("=" * 60)
        print("  G.G.A — Submission Uretimi")
        print("=" * 60)
    if not models:
        raise ValueError("At least one model is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    # Submission çiftlerini oku
    if verbose:
        print(f"\n[1/4] Submission ciftleri yukleniyor: {submission_pairs_path}")
    sub_df = pd.read_csv(
        submission_pairs_path,
        dtype={"id": "string", "term_id": "string", "item_id": "string"}
    )
    if verbose:
        print(f"  Tahmin edilecek cift sayisi: {len(sub_df):,}")

    if verbose:
        print(f"\n[2/4] Batch feature ve {len(models)}-model ensemble tahmini...")
    prediction_chunks = []
    for start in range(0, len(sub_df), batch_size):
        end = min(start + batch_size, len(sub_df))
        batch = sub_df.iloc[start:end].copy()
        batch = batch.merge(
            terms_df, on="term_id", how="left", validate="many_to_one"
        ).merge(items_df, on="item_id", how="left", validate="many_to_one")
        if batch[["query", "title"]].isna().any().any():
            raise ValueError(f"Unresolved term_id or item_id in rows {start}:{end}")
        batch = build_features(batch)
        if vectorizer is not None:
            batch = add_tfidf_features(batch, vectorizer, batch_size=batch_size)
        X_batch = batch[feature_cols]
        probabilities = np.vstack(
            [model.predict(X_batch) for model in models]
        ).mean(axis=0)
        prediction_chunks.append((probabilities >= threshold).astype(np.int8))
        if verbose:
            print(f"  ... {end:,}/{len(sub_df):,} satir islendi")

    predictions = np.concatenate(prediction_chunks)
    pos_rate = predictions.mean()
    if verbose:
        print(f"  Threshold         : {threshold}")
        print(f"  Pozitif oran      : {pos_rate:.2%}  ({predictions.sum():,} adet '1')")

    # Submission DataFrame'ini oluştur
    # The source order is preserved across batch prediction.
    submission_out = pd.DataFrame({
        "id"        : sub_df["id"].values,
        "prediction": predictions
    })

    # Dosyaya kaydet
    if verbose:
        print(f"\n[4/4] Submission dosyasi kaydediliyor: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary_output = output_path + ".tmp"
    submission_out.to_csv(temporary_output, index=False)

    # Format doğrulaması
    if verbose:
        print("\nFormat dogrulamasi yapiliyor...")
    if sample_submission_path is None:
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(submission_pairs_path)),
            "sample_submission.csv",
        )
        sample_submission_path = candidate if os.path.exists(candidate) else None
    if not validate_submission(
        temporary_output,
        sample_submission_path=sample_submission_path,
        expected_rows=len(sub_df),
        verbose=verbose,
    ):
        os.remove(temporary_output)
        raise RuntimeError("Generated submission failed validation")
    os.replace(temporary_output, output_path)

    if verbose:
        print(f"\n[HAZIR] Submission: {output_path}")

    return submission_out


def run_submission_pipeline(threshold: float = None):
    """
    Tam submission pipeline'ını baştan sona çalıştırır.

    Canonical manifest-doğrulamalı inference pipeline'ını çalıştırır.
    """
    from pipeline.inference import main

    arguments = ["--mode", "predict"]
    if threshold is not None:
        arguments.extend(["--threshold", str(threshold)])
    return main(arguments)


if __name__ == "__main__":
    run_submission_pipeline()

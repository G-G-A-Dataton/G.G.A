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

    Büyük submission seti (3.36M satır) için batch işleme kullanılır.
    Tüm seti bir anda belleğe almak yerine parça parça işlenir.

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

    Döndürür
    -------
    pd.DataFrame
        Üretilen submission DataFrame'i (id, prediction kolonları).
    """
    if verbose:
        print("=" * 60)
        print("  G.G.A — Submission Uretimi")
        print("=" * 60)

    # Submission çiftlerini oku
    if verbose:
        print(f"\n[1/4] Submission ciftleri yukleniyor: {submission_pairs_path}")
    sub_df = pd.read_csv(
        submission_pairs_path,
        dtype={"id": "string", "term_id": "string", "item_id": "string"}
    )
    if verbose:
        print(f"  Tahmin edilecek cift sayisi: {len(sub_df):,}")

    # Feature'ları hesapla
    if verbose:
        print("\n[2/4] Feature'lar hesaplaniyor...")
    sub_features = prepare_submission_features(
        sub_df, terms_df, items_df,
        vectorizer=vectorizer,
        feature_cols=feature_cols,
        batch_size=batch_size,
        verbose=verbose
    )

    # Model(ler) ile tahmin yap — ensemble (ortalama)
    if verbose:
        print(f"\n[3/4] {len(models)} model ile tahmin yapiliyor (ensemble)...")
    X_sub = sub_features[feature_cols]  # Model, sadece bu kolonları görecek — sıralama önemli!

    # Her fold'un modeli tahmin üretiyor; listeyi oluşturuyoruz
    proba_list = [model.predict(X_sub) for model in models]

    # Ensemble: tüm fold modellerinin olasılık tahminlerini ortalıyoruz.
    # Ortalama, tek bir modelin aşırı uyumu (overfit) durumunu düzeltiyor.
    avg_proba = np.mean(proba_list, axis=0)

    # Olasılık (0.0-1.0) → Binary tahmin (0 veya 1)
    # avg_proba >= threshold ise "alakalı (1)", aksi hâlde "alakasız (0)"
    predictions = (avg_proba >= threshold).astype(int)
    pos_rate = predictions.mean()
    if verbose:
        print(f"  Threshold         : {threshold}")
        print(f"  Pozitif oran      : {pos_rate:.2%}  ({predictions.sum():,} adet '1')")

    # Submission DataFrame'ini oluştur
    # id kolonu: sub_features["id"] → merge sonrası orijinal ID'ler korunuyor mu kontrol et!
    # .values ile numpy dizisine al — index karışıklığını önler
    submission_out = pd.DataFrame({
        "id"        : sub_features["id"].values,
        "prediction": predictions
    })

    # Dosyaya kaydet
    if verbose:
        print(f"\n[4/4] Submission dosyasi kaydediliyor: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_out.to_csv(output_path, index=False)

    # Format doğrulaması
    if verbose:
        print("\nFormat dogrulamasi yapiliyor...")
    validate_submission(output_path, verbose=verbose)

    if verbose:
        print(f"\n[HAZIR] Submission: {output_path}")

    return submission_out


def run_submission_pipeline(threshold: float = 0.5):
    """
    Tam submission pipeline'ını baştan sona çalıştırır.

    Eğitilmiş bir model yoksa önce run_baseline.py veya
    run_baseline_tfidf.py çalıştırılmalıdır.
    """
    import pickle
    import lightgbm as lgb
    from src.data import load_terms, load_items

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR     = os.path.join(PROJECT_ROOT, "datasets")
    OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

    print("[submission] Veriler yukleniyor...")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df  = load_items(os.path.join(DATA_DIR, "items.csv"))

    # Kaydedilmiş modeli yükle
    model_path = os.path.join(OUTPUT_DIR, "lgbm_model.txt")
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadi: {model_path}")
        print("Once run_baseline.py veya run_baseline_tfidf.py calistirin.")
        return

    model = lgb.Booster(model_file=model_path)
    models = [model]

    # TF-IDF vectorizer
    vec_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    vectorizer = None
    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        print(f"[submission] TF-IDF vectorizer yuklendi: {vec_path}")

    from src.features import FEATURE_COLS
    feature_cols = FEATURE_COLS + (["tfidf_cosine"] if vectorizer else [])

    generate_submission(
        models=models,
        terms_df=terms_df,
        items_df=items_df,
        feature_cols=feature_cols,
        submission_pairs_path=os.path.join(DATA_DIR, "submission_pairs.csv"),
        output_path=os.path.join(OUTPUT_DIR, "submission_v1.csv"),
        vectorizer=vectorizer,
        threshold=threshold,
        verbose=True
    )


if __name__ == "__main__":
    run_submission_pipeline(threshold=0.45)

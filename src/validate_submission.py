"""
src/validate_submission.py
==========================
G.G.A Takımı — Kaggle Submission Format Doğrulama

Muhammed Köseoğlu tarafından hazırlanmıştır.

Submission yüklenmeden önce zorunlu kontroller:
  1. Kolon isimleri: sadece 'id' ve 'prediction'
  2. Satır sayısı: sample_submission.csv ile aynı
  3. ID sırası: sample_submission.csv ile birebir aynı
  4. Tahmin değerleri: sadece 0 veya 1 (binary, NaN yok)
"""

import os
import pandas as pd
from pandas.api.types import is_integer_dtype

EXPECTED_ROWS = 3_359_679  # sample_submission.csv satır sayısı (header hariç)


def validate_submission(
    submission_path,
    sample_submission_path=None,
    expected_rows=None,
    verbose=True,
):
    """
    Kaggle submission dosyasını resmi formata göre doğrular.

    Parametreler
    ----------
    submission_path : str
        Doğrulanacak submission CSV dosyasının yolu.
    sample_submission_path : str, optional
        sample_submission.csv yolu. ID sırası kontrolü için gerekli.
        Verilmezse ID sırası kontrolü atlanır.
    verbose : bool, default=True
        Kontrol sonuçlarını ekrana yazdır.

    Döndürür
    -------
    bool
        Tüm kontroller geçildiyse True, herhangi biri başarısız ise False.

    Hatalar
    -------
    Kontrol başarısız olduğunda hata detayı yazdırılır ancak exception fırlatılmaz.
    """

    errors = []
    sample_df = None

    if verbose:
        print("=" * 55)
        print("  SUBMISSION KALİTE KONTROL")
        print(f"  Dosya: {os.path.basename(submission_path)}")
        print("=" * 55)

    # ─── Dosyayı oku ────────────────────────────────────────
    try:
        sub_df = pd.read_csv(submission_path, dtype={"id": "string"})
    except Exception as e:
        errors.append(f"Dosya okunamadı: {e}")
        _report(errors, verbose)
        return False

    if sample_submission_path:
        try:
            sample_df = pd.read_csv(
                sample_submission_path,
                usecols=["id"],
                dtype={"id": "string"},
            )
        except Exception as e:
            errors.append(f"Sample submission okunamadı: {e}")

    if expected_rows is None:
        expected_rows = len(sample_df) if sample_df is not None else EXPECTED_ROWS

    # ─── Kontrol 1: Kolon isimleri ──────────────────────────
    expected_cols = ["id", "prediction"]
    actual_cols = sub_df.columns.tolist()

    if actual_cols != expected_cols:
        errors.append(f"Yanlış kolonlar/sıra: {actual_cols} — Beklenen: {expected_cols}")
    else:
        _ok("[1] Kolon isimleri doğru: 'id', 'prediction'", verbose)

    # ─── Kontrol 2: Satır sayısı ────────────────────────────
    actual_rows = len(sub_df)
    if actual_rows != expected_rows:
        errors.append(
            f"Yanlış satır sayısı: {actual_rows:,} — Beklenen: {expected_rows:,}"
        )
    else:
        _ok(f"[2] Satır sayısı doğru: {actual_rows:,}", verbose)

    # ─── Kontrol 3: Binary prediction değerleri ─────────────
    if "prediction" in sub_df.columns:
        null_count = sub_df["prediction"].isnull().sum()
        if null_count > 0:
            errors.append(f"'prediction' kolonunda {null_count} NaN değeri var!")

        if not is_integer_dtype(sub_df["prediction"].dtype):
            errors.append(
                "'prediction' kolonu integer olmalı; "
                f"bulunan dtype: {sub_df['prediction'].dtype}"
            )

        unique_vals = set(sub_df["prediction"].dropna().unique())
        if not unique_vals or not unique_vals.issubset({0, 1}):
            errors.append(
                "'prediction' yalnızca 0 veya 1 değerlerini içermeli. "
                f"Bulunan: {unique_vals}"
            )
        else:
            pred_counts = sub_df["prediction"].value_counts().to_dict()
            _ok(
                f"[3] Tahmin değerleri binary. 0: {pred_counts.get(0, 0):,}  |  1: {pred_counts.get(1, 0):,}",
                verbose,
            )

    # ─── Kontrol 4: ID sırası (sample_submission ile karşılaştır) ───
    if sample_df is not None and "id" in sub_df.columns:
        if not sub_df["id"].reset_index(drop=True).equals(
            sample_df["id"].reset_index(drop=True)
        ):
            errors.append("ID sırası sample_submission.csv ile uyuşmuyor!")
        else:
            _ok("[4] ID sırası sample_submission.csv ile birebir eşleşiyor.", verbose)

    if "id" in sub_df.columns:
        if sub_df["id"].isna().any():
            errors.append("'id' kolonunda eksik değer var!")
        if sub_df["id"].duplicated().any():
            errors.append("'id' kolonunda tekrar eden değer var!")

    # ─── Kontrol 5: Index kolonu olmadığından emin ol ───────
    unnamed_cols = [c for c in sub_df.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        errors.append(
            f"'Unnamed' index kolonu tespit edildi: {unnamed_cols}. "
            f"Kaydetirken index=False kullanın."
        )
    else:
        _ok("[5] Index kolonu yok. Temiz.", verbose)

    _report(errors, verbose)
    return len(errors) == 0


def _ok(msg, verbose):
    if verbose:
        print(f"  [OK] {msg}")


def _report(errors, verbose):
    if verbose:
        print("")
        if errors:
            print(f"  [HATA] {len(errors)} HATA BULUNDU:")
            for e in errors:
                print(f"     -> {e}")
        else:
            print("  [HAZIR] Tum kontroller basariyla gecildi. Submission hazir!")
        print("=" * 55)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanım: python -m src.validate_submission <submission.csv> [sample_submission.csv]")
        sys.exit(1)

    sub_path    = sys.argv[1]
    sample_path = sys.argv[2] if len(sys.argv) > 2 else None
    ok = validate_submission(sub_path, sample_path)
    sys.exit(0 if ok else 1)

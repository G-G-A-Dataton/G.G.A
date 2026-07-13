"""
src/validate_submission.py
==========================
G.G.A Takımı - Kaggle Submission Format Doğrulama

Submission yüklenmeden önce zorunlu kontroller:
  1. Kolon isimleri: sadece 'id' ve 'prediction'
  2. Satır sayısı: sample_submission.csv ile aynı
  3. ID sırası: sample_submission.csv ile birebir aynı
  4. Tahmin değerleri: sadece 0 veya 1 (binary, NaN yok)
"""

import argparse
import os

import pandas as pd
from pandas.api.types import is_integer_dtype


EXPECTED_ROWS = 3_359_679


def validate_submission(
    submission_path,
    sample_submission_path=None,
    expected_rows=None,
    verbose=True,
    chunk_size=250_000,
):
    """Validate the exact binary submission contract with bounded memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    errors = []
    if verbose:
        print("=" * 55)
        print("  SUBMISSION KALITE KONTROL")
        print(f"  Dosya: {os.path.basename(submission_path)}")
        print("=" * 55)

    try:
        sub_reader = pd.read_csv(
            submission_path,
            dtype={"id": "string"},
            chunksize=chunk_size,
        )
    except Exception as exc:
        errors.append(f"Dosya okunamadi: {exc}")
        _report(errors, verbose)
        return False

    sample_reader = None
    if sample_submission_path:
        try:
            sample_reader = pd.read_csv(
                sample_submission_path,
                usecols=["id"],
                dtype={"id": "string"},
                chunksize=chunk_size,
                nrows=expected_rows,
            )
        except Exception as exc:
            errors.append(f"Sample submission okunamadi: {exc}")

    expected_cols = ["id", "prediction"]
    actual_cols = None
    actual_rows = 0
    sample_rows = 0
    prediction_nulls = 0
    prediction_is_integer = True
    prediction_values = set()
    id_has_null = False
    id_has_duplicate = False
    id_order_matches = True
    # Duplicate IDs are invalid even when a sample file is supplied. Keep the
    # global set so duplicates spanning CSV chunk boundaries are also caught.
    seen_ids = set()
    sample_iterator = iter(sample_reader) if sample_reader is not None else None

    try:
        for sub_chunk in sub_reader:
            if actual_cols is None:
                actual_cols = sub_chunk.columns.tolist()
            actual_rows += len(sub_chunk)

            if "prediction" in sub_chunk.columns:
                predictions = sub_chunk["prediction"]
                prediction_nulls += int(predictions.isnull().sum())
                prediction_is_integer &= is_integer_dtype(predictions.dtype)
                prediction_values.update(predictions.dropna().unique().tolist())

            if "id" in sub_chunk.columns:
                ids = sub_chunk["id"]
                id_has_null |= bool(ids.isna().any())
                id_has_duplicate |= bool(ids.duplicated().any())
                non_null_ids = ids.dropna().tolist()
                if any(item_id in seen_ids for item_id in non_null_ids):
                    id_has_duplicate = True
                seen_ids.update(non_null_ids)
                if sample_iterator is not None:
                    sample_chunk = next(sample_iterator, None)
                    if sample_chunk is None:
                        id_order_matches = False
                    else:
                        sample_rows += len(sample_chunk)
                        id_order_matches &= ids.reset_index(drop=True).equals(
                            sample_chunk["id"].reset_index(drop=True)
                        )

        if sample_iterator is not None:
            remaining_sample = next(sample_iterator, None)
            if remaining_sample is not None:
                sample_rows += len(remaining_sample)
                sample_rows += sum(len(chunk) for chunk in sample_iterator)
                id_order_matches = False
    except Exception as exc:
        errors.append(f"Dosya dogrulama sirasinda okunamadi: {exc}")
    finally:
        sub_reader.close()
        if sample_reader is not None:
            sample_reader.close()

    actual_cols = actual_cols or []
    if actual_cols != expected_cols:
        errors.append(f"Yanlis kolonlar/sira: {actual_cols} - Beklenen: {expected_cols}")
    else:
        _ok("[1] Kolon isimleri dogru: 'id', 'prediction'", verbose)

    if expected_rows is None:
        expected_rows = sample_rows if sample_reader is not None else EXPECTED_ROWS
    if actual_rows != expected_rows:
        errors.append(
            f"Yanlis satir sayisi: {actual_rows:,} - Beklenen: {expected_rows:,}"
        )
    else:
        _ok(f"[2] Satir sayisi dogru: {actual_rows:,}", verbose)

    if prediction_nulls:
        errors.append(f"'prediction' kolonunda {prediction_nulls} NaN degeri var")
    if not prediction_is_integer:
        errors.append("'prediction' kolonu integer olmali")
    if not prediction_values or not prediction_values.issubset({0, 1}):
        errors.append(
            "'prediction' yalnizca 0 veya 1 degerlerini icermeli. "
            f"Bulunan: {prediction_values}"
        )
    else:
        _ok("[3] Tahmin degerleri binary.", verbose)

    if sample_reader is not None:
        if not id_order_matches or sample_rows != actual_rows:
            errors.append("ID sirasi sample_submission.csv ile uyusmuyor")
        else:
            _ok("[4] ID sirasi sample_submission.csv ile birebir eslesiyor.", verbose)
    if id_has_null:
        errors.append("'id' kolonunda eksik deger var")
    if id_has_duplicate:
        errors.append("'id' kolonunda tekrar eden deger var")

    unnamed_cols = [column for column in actual_cols if "Unnamed" in str(column)]
    if unnamed_cols:
        errors.append(
            f"'Unnamed' index kolonu tespit edildi: {unnamed_cols}. "
            "Kaydederken index=False kullanin."
        )
    else:
        _ok("[5] Index kolonu yok. Temiz.", verbose)

    _report(errors, verbose)
    return not errors


def _ok(message, verbose):
    if verbose:
        print(f"  [OK] {message}")


def _report(errors, verbose):
    if verbose:
        print("")
        if errors:
            print(f"  [HATA] {len(errors)} HATA BULUNDU:")
            for error in errors:
                print(f"     -> {error}")
        else:
            print("  [HAZIR] Tum kontroller basariyla gecildi. Submission hazir!")
        print("=" * 55)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate a binary submission CSV")
    parser.add_argument("submission", help="Submission CSV to validate")
    parser.add_argument(
        "sample_submission",
        nargs="?",
        default=None,
        help="Reference sample submission for row count and ID order",
    )
    args = parser.parse_args(argv)
    return 0 if validate_submission(args.submission, args.sample_submission) else 1


if __name__ == "__main__":
    raise SystemExit(main())

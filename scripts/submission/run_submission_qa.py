"""
run_submission_qa.py
====================
G.G.A Takımı — Submission QA & Final Akış Bağlantısı (5 Temmuz Görevi)

Muhammed Köseoğlu tarafından hazırlanmıştır.

Bu script, üretilen submission dosyasını validate_submission.py modülü
üzerinden kapsamlı şekilde doğrular ve insan tarafından okunabilir
bir QA raporu üretir.

Kaggle'a yüklemeden önce MUTLAKA çalıştırılmalıdır.

Çalıştırmak için:
  python scripts/submission/run_submission_qa.py outputs/submission_v2.csv
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.validate_submission import validate_submission

DATA_DIR   = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")


def run_full_qa(submission_path: str) -> bool:
    """
    Submission dosyasını tüm kalite kontrol adımlarından geçirir.

    Kontrol Listesi:
      [1] Dosya okunabiliyor mu?
      [2] Kolonlar doğru mu? (id, prediction)
      [3] Satır sayısı sample_submission ile eşleşiyor mu?
      [4] Tahmin değerleri sadece 0 veya 1 mi?
      [5] ID sırası sample_submission ile aynı mı?
      [6] Unnamed index kolonu yok mu?
      [7] Dosya boyutu raporlanıyor mu?

    Parametreler
    ----------
    submission_path : str
        Doğrulanacak submission CSV dosyasının yolu.

    Döndürür
    -------
    bool
        Tüm kontroller geçildi mi?
    """
    print("=" * 60)
    print("  G.G.A — SUBMISSION QA KONTROL LISTESI")
    print(f"  Dosya: {os.path.basename(submission_path)}")
    print("=" * 60)

    # ─── Temel format kontrolleri (validate_submission.py'den) ────────────
    format_ok = validate_submission(
        submission_path,
        sample_submission_path=SAMPLE_SUB_PATH,
        expected_rows=None,
        verbose=True
    )

    # ─── Ek kontroller ─────────────────────────────────────────────────────
    extra_errors = []

    try:
        # File size and class balance are diagnostics, not acceptance rules.
        file_size_mb = os.path.getsize(submission_path) / (1024 * 1024)
        print(f"  [OK] Dosya boyutu: {file_size_mb:.1f} MB")

    except Exception as e:
        extra_errors.append(f"Ek kontroller sirasinda hata: {e}")

    # ─── Nihai Karar ───────────────────────────────────────────────────────
    all_ok = format_ok and len(extra_errors) == 0

    print()
    if extra_errors:
        print(f"  [HATA] {len(extra_errors)} ek sorun bulundu:")
        for err in extra_errors:
            print(f"    -> {err}")
    
    if all_ok:
        print("  [HAZIR] Tum QA kontrolleri gecildi!")
        print("  Kaggle'a yuklemek icin hazir.")
    else:
        print("  [ENGEL] Sorunlar giderilmeden submission yapilmamalidir!")

    print("=" * 60)
    return all_ok


def print_submission_stats(submission_path: str) -> None:
    """
    Submission dosyasının istatistiklerini özetler.
    Kaggle'a yüklemeden önce kontrol için kullanılır.
    """
    total_rows = 0
    positive_rows = 0
    first_rows = None
    for chunk in pd.read_csv(submission_path, chunksize=250_000):
        if first_rows is None:
            first_rows = chunk.head(3)
        total_rows += len(chunk)
        positive_rows += int((chunk["prediction"] == 1).sum())
    positive_rate = positive_rows / total_rows if total_rows else 0.0

    print("\n  SUBMISSION ISTATISTIKLERI")
    print(f"  Toplam satir      : {total_rows:,}")
    print(f"  Pozitif (1)       : {positive_rows:,}  ({positive_rate:.2%})")
    print(f"  Negatif (0)       : {total_rows - positive_rows:,}  ({1 - positive_rate:.2%})")
    print(f"  Dosya boyutu      : {os.path.getsize(submission_path) / 1024 / 1024:.2f} MB")
    print(f"  Ilk 3 satir:")
    print(first_rows.to_string(index=False))


if __name__ == "__main__":
    # Komut satırından dosya yolu alınabilir:
    #   python scripts/submission/run_submission_qa.py outputs/submission_v2.csv
    if len(sys.argv) > 1:
        sub_path = sys.argv[1]
    else:
        # Varsayılan yol
        sub_path = os.path.join(OUTPUT_DIR, "submission_v2.csv")

    if not os.path.exists(sub_path):
        print(f"[HATA] Submission dosyasi bulunamadi: {sub_path}")
        print("Once RUNBOOK.md icindeki production training ve inference adimlarini calistirin.")
        print("Veya: python scripts/submission/run_submission_qa.py <dosya_yolu>")
        sys.exit(1)

    # QA çalıştır
    ok = run_full_qa(sub_path)

    # İstatistikleri göster
    if ok:
        print_submission_stats(sub_path)

    sys.exit(0 if ok else 1)

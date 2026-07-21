"""
scripts/analysis/run_retrieval_eval.py
=======================================
G.G.A Takımı — Retrieval Evaluation Raporu

Mevcut OOF ve test tahminleri üzerinde retrieval metriklerini hesaplar.
Çıktı: outputs/retrieval_eval.csv + docs/retrieval_eval.md

Kullanım:
  python scripts/analysis/run_retrieval_eval.py
  python scripts/analysis/run_retrieval_eval.py --artifact-dir outputs/ensemble_artifacts
  python scripts/analysis/run_retrieval_eval.py --use-golden-testset
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval_metrics import build_eval_dataframe, evaluation_report


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "ensemble_artifacts")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Retrieval evaluation report")
    parser.add_argument(
        "--artifact-dir", default=ARTIFACT_DIR,
        help="OOF artifact dizini (oof_lgbm.npy, oof_xgb.npy, y_true.npy, fold_ids.npy)",
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100],
        help="Değerlendirilecek K değerleri",
    )
    parser.add_argument(
        "--use-golden-testset", action="store_true",
        help="Golden test set varsa onu kullan (datasets/golden_testset_v1.parquet)",
    )
    parser.add_argument(
        "--output-csv", default=os.path.join(OUTPUT_DIR, "retrieval_eval.csv"),
    )
    parser.add_argument(
        "--output-md", default=os.path.join(DOCS_DIR, "retrieval_eval.md"),
    )
    return parser.parse_args(argv)


def _load_oof_artifacts(artifact_dir: str) -> dict:
    """OOF artifact'larını yükle ve temel doğrulama yap."""
    files = {
        "oof_lgbm": os.path.join(artifact_dir, "oof_lgbm.npy"),
        "oof_xgb": os.path.join(artifact_dir, "oof_xgb.npy"),
        "y_true": os.path.join(artifact_dir, "y_true.npy"),
        "fold_ids": os.path.join(artifact_dir, "fold_ids.npy"),
    }
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Eksik OOF artifact'ları: {missing}\n"
            f"Önce 'python scripts/run_production.py --stage train' çalıştırın."
        )

    data = {k: np.load(p) for k, p in files.items()}
    n = len(data["y_true"])
    for key in ["oof_lgbm", "oof_xgb", "fold_ids"]:
        if len(data[key]) != n:
            raise ValueError(f"{key} uzunluğu y_true ile eşleşmiyor: {len(data[key])} != {n}")

    return data


def _build_eval_df_from_metadata(
    artifact_dir: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """test_metadata.csv veya OOF metadata'sından eval DataFrame oluştur."""
    # OOF için training metadata kullan
    # submission için test_metadata.csv
    meta_path = os.path.join(OUTPUT_DIR, "test_metadata.csv")
    oof_meta_path = os.path.join(artifact_dir, "oof_metadata.csv")

    for path in [oof_meta_path, meta_path]:
        if os.path.exists(path):
            # Sadece term_id ve item_id sütunlarını yükle (büyük dosya)
            meta = pd.read_csv(
                path,
                usecols=["term_id", "item_id"],
                dtype={"term_id": "string", "item_id": "string"},
                nrows=len(scores),  # Yalnızca ihtiyaç duyulan satırlar
            )
            if len(meta) == len(scores):
                return build_eval_dataframe(meta, scores, labels)

    raise FileNotFoundError(
        "Metadata dosyası bulunamadı. "
        "OOF değerlendirme için metadata gerekli.\n"
        f"Aranan: {oof_meta_path}, {meta_path}"
    )


def _run_oof_evaluation(args, artifact_dir: str) -> dict[str, dict]:
    """OOF artifact'larından retrieval metriklerini hesapla."""
    print("\n[1/2] OOF artifact'ları yükleniyor...")
    data = _load_oof_artifacts(artifact_dir)

    y_true = data["y_true"]
    oof_lgbm = data["oof_lgbm"]
    oof_xgb = data["oof_xgb"]

    print(f"  OOF satır sayısı  : {len(y_true):,}")
    print(f"  Pozitif oranı     : {y_true.mean():.3f}")

    results = {}

    # LGBM OOF değerlendirmesi
    print("\n[2/2] Retrieval metrikleri hesaplanıyor...")
    for model_name, scores in [("lgbm", oof_lgbm), ("xgb", oof_xgb)]:
        print(f"\n  Model: {model_name.upper()}")
        try:
            eval_df = _build_eval_df_from_metadata(artifact_dir, scores, y_true)
            report = evaluation_report(eval_df, ks=args.ks, verbose=True)
            results[model_name] = report
        except FileNotFoundError as exc:
            print(f"  ⚠️  {exc}")
            print("  Retrieval eval için metadata gerekli. Atlanıyor.")

    return results


def _run_golden_testset_evaluation(args) -> dict | None:
    """Golden test set varsa retrieval metriklerini hesapla."""
    golden_path = os.path.join(PROJECT_ROOT, "datasets", "golden_testset_v1.parquet")
    if not os.path.exists(golden_path):
        print(f"  ⚠️  Golden test set bulunamadı: {golden_path}")
        print("  Oluşturmak için: python scripts/data/run_golden_testset_build.py")
        return None

    print(f"\n  Golden test set yükleniyor: {golden_path}")
    try:
        df = pd.read_parquet(golden_path)
        print(f"  Satır sayısı: {len(df):,} ({df['term_id'].nunique():,} sorgu)")

        # Golden test set'te skor sütunu yok → label'ı skor olarak kullan
        # (Bu sadece veri seti kalitesini kontrol eder)
        if "score" not in df.columns:
            # Placeholder: BM25 rank'ı ters çevir (düşük rank = yüksek skor)
            df["score"] = df["bm25_rank"].fillna(0).apply(
                lambda r: 1.0 / (1.0 + r) if r > 0 else 0.5
            )

        report = evaluation_report(df, ks=args.ks, verbose=True)
        return report
    except Exception as exc:
        print(f"  Golden test set değerlendirmesi başarısız: {exc}")
        return None


def _save_results(results: dict[str, dict], output_csv: str, output_md: str) -> None:
    """Sonuçları CSV ve Markdown olarak kaydet."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    # CSV formatına dönüştür
    rows = []
    for model_name, report in results.items():
        for metric_name, value in report.items():
            rows.append({
                "model": model_name,
                "metric": metric_name,
                "value": value,
            })

    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(output_csv, index=False)
    print(f"\n  Sonuçlar kaydedildi: {output_csv}")

    # Markdown raporu
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Retrieval Evaluation Raporu\n",
        f"**Tarih:** {now}  \n",
        f"**Kaynak:** {output_csv}\n\n---\n",
    ]

    for model_name, report in results.items():
        lines.append(f"## {model_name.upper()}\n")
        lines.append(f"| K | Recall@K | Precision@K | NDCG@K |\n")
        lines.append(f"|---|---|---|---|\n")
        ks = sorted(set(
            int(k.split("@")[1]) for k in report if "@" in k and k.startswith("recall")
        ))
        for k in ks:
            recall = report.get(f"recall@{k}", float("nan"))
            prec = report.get(f"precision@{k}", float("nan"))
            ndcg = report.get(f"ndcg@{k}", float("nan"))
            lines.append(f"| {k} | {recall:.4f} | {prec:.4f} | {ndcg:.4f} |\n")
        lines.append(f"\n**MRR:** {report.get('mrr', float('nan')):.4f}  \n")
        lines.append(f"**Sorgu sayısı:** {int(report.get('n_queries', 0)):,}\n\n")

    with open(output_md, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Rapor kaydedildi  : {output_md}")


def main(argv=None):
    args = parse_args(argv)
    artifact_dir = os.path.abspath(args.artifact_dir)

    print("=" * 60)
    print("  G.G.A — Retrieval Evaluation")
    print("=" * 60)
    print(f"  Artifact dir : {artifact_dir}")
    print(f"  K değerleri  : {args.ks}")

    all_results: dict[str, dict] = {}

    # OOF değerlendirmesi
    try:
        oof_results = _run_oof_evaluation(args, artifact_dir)
        all_results.update(oof_results)
    except FileNotFoundError as exc:
        print(f"\n⚠️  OOF değerlendirmesi atlandı: {exc}")

    # Golden test set değerlendirmesi (opsiyonel)
    if args.use_golden_testset:
        golden_result = _run_golden_testset_evaluation(args)
        if golden_result is not None:
            all_results["golden_testset"] = golden_result

    if not all_results:
        print("\n❌ Değerlendirilebilecek veri bulunamadı.")
        print("   Önce 'python scripts/run_production.py --stage train' çalıştırın.")
        sys.exit(1)

    # Sonuçları kaydet
    _save_results(all_results, args.output_csv, args.output_md)
    print("\n✅ Retrieval evaluation tamamlandı.")


if __name__ == "__main__":
    main()

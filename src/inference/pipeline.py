"""
src/inference/pipeline.py
==========================
G.G.A Takımı — Config-Driven Production Inference Pipeline

Hardcoded path'leri kaldırır, konfigürasyon ve manifest verilerinden
dinamik olarak model, vektörizer ve eşik değerlerini yükler.

Kullanım:
  >>> from src.inference.pipeline import InferencePipeline
  >>> pipeline = InferencePipeline.from_config("configs/inference.yaml")
  >>> pipeline.run_submission(output_path="outputs/submission_v2.csv")
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.training.run_train_full_v2 import sha256_file
from src.calibration import PlattCalibrator
from src.data import load_items, load_terms
from src.modeling import MODEL_FEATURE_COLS
from src.out_of_core_features import (
    build_base_feature_store,
    build_context_feature_store,
    load_feature_batch,
    remove_feature_stores,
)
from src.tfidf_features import load_vectorizer
from src.validate_submission import validate_submission


class InferencePipeline:
    """
    Production-grade Inference Pipeline.

    Config-driven architecture using single source of truth for model artifacts.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, "datasets")
        self.output_dir = os.path.join(self.project_root, "outputs")

        self.manifest_path = os.path.join(self.project_root, self.config.get("manifest_path", "outputs/model_manifest_v2.json"))
        self.batch_size = int(self.config.get("batch_size", 100_000))
        self.expected_rows = int(self.config.get("expected_submission_rows", 3_359_679))

        self.models: List[lgb.Booster] = []
        self.vectorizer = None
        self.threshold: float = 0.5
        self.calibrator: Optional[PlattCalibrator] = None

    @classmethod
    def from_config(cls, config_path: str = "configs/inference.yaml") -> InferencePipeline:
        """Config dosyasından InferencePipeline oluşturur."""
        abs_path = os.path.abspath(config_path)
        if not os.path.exists(abs_path):
            abs_path = os.path.join(PROJECT_ROOT, config_path)

        if os.path.exists(abs_path):
            if abs_path.endswith(".json"):
                with open(abs_path, encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                try:
                    import yaml
                    with open(abs_path, encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                except ImportError:
                    cfg = {}
        else:
            cfg = {}

        return cls(cfg)

    def load_artifacts(self) -> None:
        """Manifesto ve model dosyalarını yükler ve SHA-256 doğrulaması yapar."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Inference manifest bulunamadı: {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        # Artifact kontrat kontrolleri
        if manifest.get("training_mode") != "full":
            raise ValueError("InferencePipeline yalnızca 'full' training mode artifact'ları kabul eder.")

        art_dir = os.path.dirname(self.manifest_path)

        # Model yükleme
        self.models = []
        for model_info in manifest.get("artifact_manifest", []):
            fname = model_info["filename"]
            if fname.startswith("lgbm_") and fname.endswith(".txt"):
                mpath = os.path.join(art_dir, fname)
                if sha256_file(mpath) != model_info["sha256"]:
                    raise ValueError(f"Model dosyası SHA-256 doğrulama hatası: {fname}")
                self.models.append(lgb.Booster(model_file=mpath))

        if not self.models:
            raise RuntimeError("Yüklenecek geçerli LightGBM model bulunamadı.")

        # Vektörizer yükleme
        vec_name = manifest.get("support_files", {}).get("vectorizer") or "tfidf_vectorizer_v2.pkl"
        vec_path = os.path.join(art_dir, vec_name)
        if os.path.exists(vec_path):
            self.vectorizer = load_vectorizer(vec_path)

        # Threshold yükleme
        thresh_name = manifest.get("support_files", {}).get("best_threshold") or "best_threshold_v2.txt"
        thresh_path = os.path.join(art_dir, thresh_name)
        if os.path.exists(thresh_path):
            with open(thresh_path, encoding="utf-8") as f:
                self.threshold = float(f.read().strip())

    def predict_batch(self, X_batch: pd.DataFrame) -> np.ndarray:
        """Toplu öznitelik matrisi üzerinde model ensemble tahmini yapar."""
        if not self.models:
            self.load_artifacts()

        preds = [model.predict(X_batch) for model in self.models]
        mean_preds = np.mean(preds, axis=0)

        if self.calibrator is not None:
            mean_preds = self.calibrator.predict_proba(mean_preds)

        return mean_preds

    def run_submission(
        self,
        output_path: Optional[str] = None,
        submission_pairs_path: Optional[str] = None,
    ) -> str:
        """Tüm submission çiftleri üzerinde batch çıkarım çalıştırır ve CSV yazar."""
        if not self.models:
            self.load_artifacts()

        sub_pairs_path = submission_pairs_path or os.path.join(self.data_dir, "submission_pairs.csv")
        out_path = output_path or os.path.join(self.output_dir, "submission_v2.csv")
        tmp_out_path = out_path + ".tmp"

        print(f"[InferencePipeline] Veriler yükleniyor: {sub_pairs_path}")
        terms = load_terms(os.path.join(self.data_dir, "terms.csv"))
        items = load_items(os.path.join(self.data_dir, "items.csv"))

        # Toplam satır sayısı
        test_rows = self.expected_rows
        if os.path.exists(sub_pairs_path):
            with open(sub_pairs_path, "r", encoding="utf-8") as f:
                test_rows = max(1, sum(1 for _ in f) - 1)

        store_prefix = os.path.join(self.output_dir, f".inference_store_{os.getpid()}")
        store_paths = []

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

        try:
            print("[InferencePipeline] Base öznitelik deposu oluşturuluyor...")
            base_path, codes_path = build_base_feature_store(
                sub_pairs_path, terms, items, self.vectorizer,
                row_count=test_rows, batch_size=self.batch_size, output_prefix=store_prefix
            )
            store_paths.extend([base_path, codes_path])

            print("[InferencePipeline] Context öznitelik deposu oluşturuluyor...")
            context_path = build_context_feature_store(base_path, codes_path, store_prefix)
            store_paths.append(context_path)

            base_store = np.load(base_path, mmap_mode="r")
            context_store = np.load(context_path, mmap_mode="r")

            print(f"[InferencePipeline] Çıkarım başladı (Threshold={self.threshold:.4f})...")
            with open(sub_pairs_path, "r", encoding="utf-8") as f_in, open(tmp_out_path, "w", encoding="utf-8", newline="") as f_out:
                header = f_in.readline()  # Skip input header
                f_out.write("id,prediction\n")

                for start in range(0, test_rows, self.batch_size):
                    end = min(start + self.batch_size, test_rows)
                    X_batch = load_feature_batch(base_store, context_store, start, end)
                    probs = self.predict_batch(X_batch)
                    preds = (probs >= self.threshold).astype(int)

                    # ID'leri oku ve yaz
                    chunk_df = pd.read_csv(sub_pairs_path, skiprows=start + 1, nrows=end - start, usecols=[0], names=["id"])
                    for id_val, pred_val in zip(chunk_df["id"], preds):
                        f_out.write(f"{id_val},{pred_val}\n")

            print("[InferencePipeline] Çıkarım çıktısı doğrulanıyor...")
            validate_submission(tmp_out_path, expected_rows=test_rows)
            os.replace(tmp_out_path, out_path)
            print(f"[+] Submission yayınlandı: {out_path}")
            return out_path

        finally:
            remove_feature_stores(*store_paths)

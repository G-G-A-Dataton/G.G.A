"""
src/experiment_tracker.py
==========================
G.G.A Takımı — MLflow Experiment Tracker

MLflow tabanlı deney takip sarmalayıcısı.
MLflow yüklü değilse veya tracking URI ayarlı değilse JSON dosyasına
fallback yapar — offline yarışma ortamında da çalışır.

Kullanım:
  >>> tracker = ExperimentTracker.from_config("configs/mlflow.yaml")
  >>> tracker.start_run(run_name="shortlist_train_v3")
  >>> tracker.log_params({"n_splits": 5, "feature_schema_version": 3})
  >>> tracker.log_metrics({"cross_fitted_macro_f1": 0.9712, "deploy_threshold": 0.43})
  >>> tracker.log_artifact("outputs/ensemble_artifacts/oof_manifest.json")
  >>> tracker.end_run()
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Yardımcı: YAML config yükleme (PyYAML opsiyonel, json fallback)
# ---------------------------------------------------------------------------

def _load_yaml_or_json(path: str) -> dict:
    """YAML veya JSON config yükle. PyYAML yoksa JSON dene."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {path}")

    if path_obj.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            with open(path_obj, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # PyYAML yüklü değil — JSON fallback için hata ver
            raise ImportError(
                "configs/mlflow.yaml okumak için PyYAML gerekli: pip install pyyaml"
            )
    else:
        with open(path_obj, encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Yardımcı: MLflow lazy import
# ---------------------------------------------------------------------------

def _try_import_mlflow():
    """MLflow import'u dene, başarısız olursa None döndür."""
    try:
        import mlflow  # type: ignore
        return mlflow
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Ana Sınıf
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    MLflow experiment tracker with JSON file fallback.

    Öncelik sırası:
    1. MLflow (tracking_uri tanımlıysa ve mlflow yüklüyse)
    2. JSON log dosyası (outputs/experiment_log.json)

    Attributes
    ----------
    tracking_uri : str
        MLflow tracking server URI veya local dizin.
    experiment_name : str
        MLflow experiment adı.
    run_tags : dict
        Her run'a eklenen sabit tag'lar.
    fallback_path : str
        MLflow yoksa JSON'un yazılacağı dosya.
    """

    def __init__(
        self,
        tracking_uri: str = "mlflow/mlruns",
        experiment_name: str = "gga_product_matching",
        run_tags: Optional[Dict[str, str]] = None,
        fallback_path: str = "outputs/experiment_log.json",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_tags = run_tags or {}
        self.fallback_path = fallback_path

        self._mlflow = _try_import_mlflow()
        self._use_mlflow = self._mlflow is not None
        self._active_run = None
        self._run_data: dict = {}
        self._run_name: Optional[str] = None

        if self._use_mlflow:
            try:
                self._mlflow.set_tracking_uri(tracking_uri)
                self._mlflow.set_experiment(experiment_name)
            except Exception as exc:
                print(f"[ExperimentTracker] MLflow kurulum hatası: {exc}. JSON fallback aktif.")
                self._use_mlflow = False

    @classmethod
    def from_config(cls, config_path: str = "configs/mlflow.yaml") -> "ExperimentTracker":
        """
        YAML/JSON config dosyasından ExperimentTracker oluştur.

        configs/mlflow.yaml formatı:
          tracking_uri: "mlflow/mlruns"
          experiment_name: "gga_product_matching"
          run_tags:
            project: "G.G.A"
            competition: "TEKNOFEST2026"
        """
        try:
            cfg = _load_yaml_or_json(config_path)
        except FileNotFoundError:
            print(f"[ExperimentTracker] Config bulunamadı: {config_path}. Varsayılanlar kullanılıyor.")
            cfg = {}
        except ImportError as exc:
            print(f"[ExperimentTracker] {exc}. Varsayılanlar kullanılıyor.")
            cfg = {}

        return cls(
            tracking_uri=cfg.get("tracking_uri", "mlflow/mlruns"),
            experiment_name=cfg.get("experiment_name", "gga_product_matching"),
            run_tags=cfg.get("run_tags", {}),
        )

    # ------------------------------------------------------------------
    # Run Yönetimi
    # ------------------------------------------------------------------

    def start_run(self, run_name: Optional[str] = None) -> "ExperimentTracker":
        """
        Yeni bir deney run'ı başlat.

        Parameters
        ----------
        run_name : str, optional
            Run için okunabilir ad (örn. "shortlist_train_v3").
        """
        self._run_name = run_name or f"run_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        self._run_data = {
            "run_name": self._run_name,
            "experiment_name": self.experiment_name,
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "tags": dict(self.run_tags),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

        if self._use_mlflow:
            try:
                tags = {**self.run_tags, "mlflow.runName": self._run_name}
                self._active_run = self._mlflow.start_run(run_name=self._run_name, tags=tags)
            except Exception as exc:
                print(f"[ExperimentTracker] MLflow run başlatılamadı: {exc}. JSON fallback.")
                self._use_mlflow = False
                self._active_run = None

        return self

    def end_run(self) -> None:
        """Aktif run'ı kapat ve JSON fallback varsa kaydet."""
        self._run_data["ended_at"] = datetime.now(tz=timezone.utc).isoformat()

        if self._use_mlflow and self._active_run is not None:
            try:
                self._mlflow.end_run()
            except Exception as exc:
                print(f"[ExperimentTracker] MLflow run kapatılamadı: {exc}")

        # Her zaman JSON'a da kaydet (ek güvence)
        self._write_json_log()
        self._active_run = None

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_run()
        return False  # Exception'ı yutma

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Deney parametrelerini logla (her run için sabit değerler).

        Örnek: {"n_splits": 5, "feature_schema_version": 3, "negative_ratio": 3}
        """
        self._run_data["params"].update(params)

        if self._use_mlflow and self._active_run is not None:
            try:
                # MLflow sadece str, int, float kabul ediyor
                safe_params = {k: str(v) for k, v in params.items()}
                self._mlflow.log_params(safe_params)
            except Exception as exc:
                print(f"[ExperimentTracker] log_params hatası: {exc}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Metrik değerlerini logla.

        Örnek: {"cross_fitted_macro_f1": 0.9712, "deploy_threshold": 0.43}
        """
        key = f"step_{step}" if step is not None else "final"
        self._run_data["metrics"].setdefault(key, {}).update(metrics)

        if self._use_mlflow and self._active_run is not None:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as exc:
                print(f"[ExperimentTracker] log_metrics hatası: {exc}")

    def log_artifact(self, local_path: str, artifact_dir: Optional[str] = None) -> None:
        """
        Dosya artifact'ı logla (JSON manifest, CSV, model dosyası vb.).

        Parameters
        ----------
        local_path : str
            Loglanacak dosyanın yerel yolu.
        artifact_dir : str, optional
            MLflow içindeki hedef klasör adı.
        """
        if not os.path.exists(local_path):
            print(f"[ExperimentTracker] Uyarı: artifact dosyası bulunamadı: {local_path}")
            return

        self._run_data["artifacts"].append(local_path)

        if self._use_mlflow and self._active_run is not None:
            try:
                self._mlflow.log_artifact(local_path, artifact_path=artifact_dir)
            except Exception as exc:
                print(f"[ExperimentTracker] log_artifact hatası: {exc}")

    def log_dict(self, data: dict, artifact_filename: str) -> None:
        """
        Dict'i geçici JSON'a yaz ve artifact olarak logla.
        Büyük config/manifest dict'leri için kullanışlı.
        """
        tmp_path = f"outputs/.tracker_tmp_{artifact_filename}"
        os.makedirs("outputs", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        self.log_artifact(tmp_path)
        # Artifact dizine kopyalandıktan sonra geçiciyi sil
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    def set_tag(self, key: str, value: str) -> None:
        """Run tag ekle."""
        self._run_data["tags"][key] = value
        if self._use_mlflow and self._active_run is not None:
            try:
                self._mlflow.set_tag(key, value)
            except Exception as exc:
                print(f"[ExperimentTracker] set_tag hatası: {exc}")

    # ------------------------------------------------------------------
    # Durum Sorgulama
    # ------------------------------------------------------------------

    @property
    def is_mlflow_active(self) -> bool:
        """MLflow aktif mi, yoksa JSON fallback mı kullanılıyor?"""
        return self._use_mlflow and self._active_run is not None

    @property
    def run_id(self) -> Optional[str]:
        """Aktif MLflow run ID'si (MLflow aktifse)."""
        if self._use_mlflow and self._active_run is not None:
            try:
                return self._active_run.info.run_id
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # JSON Fallback
    # ------------------------------------------------------------------

    def _write_json_log(self) -> None:
        """Run verilerini JSON log dosyasına ekle."""
        os.makedirs(os.path.dirname(self.fallback_path) or ".", exist_ok=True)

        existing = []
        if os.path.exists(self.fallback_path):
            try:
                with open(self.fallback_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.append(self._run_data)

        try:
            with open(self.fallback_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
        except OSError as exc:
            print(f"[ExperimentTracker] JSON log yazılamadı: {exc}")


# ---------------------------------------------------------------------------
# Convenience: context manager olmadan tek satırda kayıt
# ---------------------------------------------------------------------------

def log_run(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifact_paths: Optional[list] = None,
    run_name: Optional[str] = None,
    config_path: str = "configs/mlflow.yaml",
) -> None:
    """
    Tek fonksiyon çağrısı ile bir deney run'ı kaydet.

    Örnek:
    >>> log_run(
    ...     params={"n_splits": 5, "selected_model": "weighted_blend"},
    ...     metrics={"cross_fitted_macro_f1": 0.9712},
    ...     artifact_paths=["outputs/ensemble_artifacts/oof_manifest.json"],
    ...     run_name="shortlist_v3_full_data",
    ... )
    """
    tracker = ExperimentTracker.from_config(config_path)
    with tracker.start_run(run_name=run_name):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        for path in (artifact_paths or []):
            tracker.log_artifact(path)

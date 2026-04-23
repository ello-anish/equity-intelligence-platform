"""
quant/ml/tracking.py — MLflow experiment tracking shim.

Wrapper around MLflow that degrades gracefully when mlflow isn't installed.
Usage:

    from quant.ml.tracking import tracker
    with tracker.run("fold-0", params={"seed": 42}) as run:
        run.log_metric("ic", 0.05)
        run.log_artifact("artifacts/shap_global.csv")
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MLFLOW_DIR = Path("mlruns")


class _NullRun:
    def log_param(self, *a, **kw): pass
    def log_params(self, *a, **kw): pass
    def log_metric(self, *a, **kw): pass
    def log_metrics(self, *a, **kw): pass
    def log_artifact(self, *a, **kw): pass
    def log_dict(self, *a, **kw): pass
    def set_tag(self, *a, **kw): pass


class _Tracker:
    def __init__(self):
        self.mlflow = None
        self.enabled = False

    def enable(self, experiment_name: str = "equity-intelligence", tracking_uri: Optional[str] = None) -> bool:
        try:
            import mlflow
            self.mlflow = mlflow
            if tracking_uri is None:
                MLFLOW_DIR.mkdir(exist_ok=True)
                tracking_uri = f"file:{MLFLOW_DIR.resolve().as_posix()}"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.enabled = True
            logger.info("MLflow enabled → %s / experiment=%s", tracking_uri, experiment_name)
            return True
        except Exception as e:
            logger.warning("MLflow unavailable (%s); tracking disabled", e)
            self.enabled = False
            return False

    @contextlib.contextmanager
    def run(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        if not self.enabled:
            null = _NullRun()
            yield null
            return
        mlflow = self.mlflow
        with mlflow.start_run(run_name=name) as active:
            try:
                if params:
                    mlflow.log_params({k: (v if _loggable(v) else str(v)) for k, v in params.items()})
                if tags:
                    for k, v in tags.items():
                        mlflow.set_tag(k, v)

                class _Run:
                    def log_param(self, k, v): mlflow.log_param(k, v)
                    def log_params(self, d): mlflow.log_params(d)
                    def log_metric(self, k, v, step=None):
                        try:
                            mlflow.log_metric(k, float(v), step=step)
                        except Exception:
                            pass
                    def log_metrics(self, d, step=None):
                        mlflow.log_metrics({k: float(v) for k, v in d.items() if _is_number(v)}, step=step)
                    def log_artifact(self, path):
                        try:
                            mlflow.log_artifact(path)
                        except Exception:
                            pass
                    def log_dict(self, d, filename):
                        try:
                            mlflow.log_dict(d, filename)
                        except Exception:
                            pass
                    def set_tag(self, k, v): mlflow.set_tag(k, v)

                yield _Run()
            finally:
                pass


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def _loggable(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool))


tracker = _Tracker()

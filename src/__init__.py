"""
Credit Risk Prediction Pipeline with ANFIS
A machine learning pipeline for predicting credit card default risk
"""

import logging
import os
import sys
from pathlib import Path

from . import config
from .anfis_network import ANFISNetwork
from .data_preprocessing import DataPreprocessor
from .evaluation import ModelEvaluator
from .explainability import SHAPExplainer
from .feature_selection import FeatureSelector
from .models import ModelTrainer

__version__ = "1.0.0"
__author__ = "Master's Project"


_DOCKER_STDOUT_STREAM = None  # opened once per process, never closed


def _docker_stdout_stream():
    """Return PID 1's stdout if running inside Docker, else None.

    Inside a Jupyter kernel, `sys.stdout` is the ipykernel capture stream — it
    does *not* reach the container's stdout, so `docker logs` / Dozzle never
    see logger output. PID 1 is the jupyter-lab process whose stdout *is*
    captured by Docker, and /proc/1/fd/1 reopens that descriptor.

    The stream is cached at module scope: re-opening on every call would leak
    file descriptors, and letting the old file object be garbage-collected can
    close a descriptor number that ZMQ (ipykernel) has meanwhile reused for
    one of its sockets, surfacing as `ZMQError: Socket operation on non-socket`
    and killing the kernel's reply channel.
    """
    global _DOCKER_STDOUT_STREAM
    if _DOCKER_STDOUT_STREAM is not None:
        return _DOCKER_STDOUT_STREAM
    if not Path("/.dockerenv").exists():
        return None
    try:
        _DOCKER_STDOUT_STREAM = open("/proc/1/fd/1", "w", buffering=1)  # noqa: SIM115
    except (OSError, PermissionError):
        _DOCKER_STDOUT_STREAM = None
    return _DOCKER_STDOUT_STREAM


def setup_logging(stream=None):
    """Configure root logger to stream to stdout for Docker/Dozzle capture.

    Idempotent: repeated calls (e.g. re-importing `src` from notebook cells)
    replace the handler but reuse the cached Docker stdout stream, so no
    stale file descriptor ever gets closed by the GC.

    Args:
        stream: Optional writable stream. Explicit override wins over the
                Docker auto-detection.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    if stream is None:
        stream = _docker_stdout_stream()
    target_stream = stream if stream is not None else sys.stdout
    handler = logging.StreamHandler(target_stream)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)


def setup_mlflow():
    """Configure MLflow tracking URI + experiment.

    Tracking URI comes from MLFLOW_TRACKING_URI (default http://mlflow:5000
    inside the Docker compose network), with a file:./mlruns fallback if the
    server is unreachable so offline / unit-test runs still produce a
    local store instead of crashing on import.
    """
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "anfis-credit-risk")

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "MLflow server unreachable at %s (%s); falling back to file:./mlruns",
            tracking_uri,
            exc,
        )
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)


setup_logging()
setup_mlflow()

__all__ = [
    "ANFISNetwork",
    "DataPreprocessor",
    "FeatureSelector",
    "ModelTrainer",
    "ModelEvaluator",
    "SHAPExplainer",
    "config",
    "setup_logging",
    "setup_mlflow",
]

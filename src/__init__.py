"""
Credit Risk Prediction Pipeline with ANFIS
A machine learning pipeline for predicting credit card default risk
"""

import logging
import os
import sys

from . import config
from .anfis_network import ANFISNetwork
from .data_preprocessing import DataPreprocessor
from .evaluation import ModelEvaluator
from .explainability import SHAPExplainer
from .feature_selection import FeatureSelector
from .models import ModelTrainer

__version__ = "1.0.0"
__author__ = "Master's Project"


def setup_logging(stream=None):
    """Configure root logger to stream to stdout for Docker/Dozzle capture.

    Args:
        stream: Optional writable stream. Defaults to sys.stdout.
                Pass an open file object (e.g. /proc/1/fd/1) to redirect
                logs to Docker's captured stdout when running inside Jupyter.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

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


setup_logging()

__all__ = [
    "ANFISNetwork",
    "DataPreprocessor",
    "FeatureSelector",
    "ModelTrainer",
    "ModelEvaluator",
    "SHAPExplainer",
    "config",
    "setup_logging",
]

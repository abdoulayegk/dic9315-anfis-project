"""
Credit Risk Prediction Pipeline with ANFIS
A machine learning pipeline for predicting credit card default risk
"""

__version__ = "1.0.0"
__author__ = "Master's Project"

# Explicit imports instead of wildcard to avoid namespace pollution
from .anfis_network import ANFISNetwork
from .config import (
    ANFIS_CONFIG,
    COLORS,
    CV_FOLDS,
    CV_FOLDS_FINAL,
    MODELS_DIR,
    N_FEATURES_ANFIS,
    OUTLIER_THRESHOLD,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    REPORTS_DIR,
    SMOTE_SAMPLING_STRATEGY,
    SVM_PARAMS,
    TRAIN_TEST_SPLIT,
    USE_FEATURE_SELECTION,
    USE_SMOTE,
    USE_WINSORIZATION,
)
from .data_preprocessing import DataPreprocessor
from .evaluation import ModelEvaluator
from .explainability import SHAPExplainer
from .feature_selection import FeatureSelector
from .models import ModelTrainer

__all__ = [
    # Configuration
    "RANDOM_SEED",
    "TRAIN_TEST_SPLIT",
    "CV_FOLDS",
    "CV_FOLDS_FINAL",
    "USE_SMOTE",
    "SMOTE_SAMPLING_STRATEGY",
    "USE_FEATURE_SELECTION",
    "N_FEATURES_ANFIS",
    "OUTLIER_THRESHOLD",
    "USE_WINSORIZATION",
    "RANDOM_FOREST_PARAMS",
    "SVM_PARAMS",
    "ANFIS_CONFIG",
    "OUTPUT_DIR",
    "MODELS_DIR",
    "PLOTS_DIR",
    "REPORTS_DIR",
    "COLORS",
    # Classes
    "DataPreprocessor",
    "FeatureSelector",
    "ModelTrainer",
    "ModelEvaluator",
    "SHAPExplainer",
    "ANFISNetwork",
]

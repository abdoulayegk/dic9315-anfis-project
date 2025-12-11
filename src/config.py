"""
Configuration file for credit risk prediction pipeline
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split configuration
TRAIN_TEST_SPLIT = 0.8
CV_FOLDS = 5
CV_FOLDS_FINAL = 10  # For final stability check

# Class imbalance handling
USE_SMOTE = False
SMOTE_SAMPLING_STRATEGY = 'auto'

# Feature selection for ANFIS
USE_FEATURE_SELECTION = True
N_FEATURES_ANFIS = 6  # Best 6 features for interpretability

# Outlier handling
OUTLIER_THRESHOLD = 3  # Z-score threshold
USE_WINSORIZATION = True

# Model hyperparameter search spaces
RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
}

SVM_PARAMS = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': ['balanced', None]
}

# ANFIS configuration
ANFIS_CONFIG = {
    'n_membership_functions': 3,  # MFs per input
    'membership_type': 'gaussian',
    'max_epochs': 100,
    'learning_rate': 0.01,
    'batch_size': 32,
    'n_rules': 10  # Initial guess, will be refined by clustering
}

# Output paths
OUTPUT_DIR = 'results'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
REPORTS_DIR = 'reports'

# Visualization settings
COLORS = ["#7400ff", "#a788e4", "#d216d2", "#ffb500", "#36c9dd"]

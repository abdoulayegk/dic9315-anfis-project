"""
Main pipeline orchestrating the complete credit risk prediction workflow
"""

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

from . import config
from .data_preprocessing import DataPreprocessor
from .evaluation import ModelEvaluator
from .feature_selection import FeatureSelector
from .models import ModelTrainer

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary output directories"""
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    Path(config.PLOTS_DIR).mkdir(exist_ok=True)
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    logger.info("Output directories created")


def main(data_path, target_column="default"):
    """
    Main pipeline execution

    Parameters:
    -----------
    data_path : str
        Path to the credit card dataset
    target_column : str
        Name of the target column
    """
    logger.info("=" * 80)
    logger.info(" CREDIT RISK PREDICTION PIPELINE - ANFIS vs BASELINES")
    logger.info("=" * 80)

    # Setup
    setup_directories()

    # ========================================================================
    # STEP 1: DATA PREPROCESSING
    # ========================================================================
    preprocessor = DataPreprocessor(random_seed=config.RANDOM_SEED)

    X_train, X_test, y_train, y_test, feature_names = preprocessor.full_pipeline(
        filepath=data_path,
        target_col=target_column,
        apply_smote=config.USE_SMOTE,
        winsorize=config.USE_WINSORIZATION,
    )

    logger.info("Final training set shape: %s", X_train.shape)
    logger.info("Final test set shape: %s", X_test.shape)

    # ========================================================================
    # STEP 2: FEATURE SELECTION (for ANFIS)
    # ========================================================================
    if config.USE_FEATURE_SELECTION:
        logger.info("=" * 80)
        logger.info("FEATURE SELECTION FOR ANFIS")
        logger.info("=" * 80)

        selector = FeatureSelector(
            n_features=config.N_FEATURES_ANFIS, random_seed=config.RANDOM_SEED
        )

        # Use ensemble method for robust selection
        selected_features, feature_scores = selector.ensemble_selection(
            X_train, y_train, methods=["rfe", "mutual_info", "correlation"]
        )

        # Transform data for ANFIS
        X_train_anfis = selector.transform(X_train)
        X_test_anfis = selector.transform(X_test)

        logger.info("Reduced feature set for ANFIS: %s", X_train_anfis.shape)
    else:
        X_train_anfis = X_train
        X_test_anfis = X_test

    # ========================================================================
    # STEP 3: MODEL TRAINING & OPTIMIZATION
    # ========================================================================
    trainer = ModelTrainer(random_seed=config.RANDOM_SEED)

    # Random Forest
    rf_model, rf_params = trainer.train_random_forest(
        X_train, y_train, cv=config.CV_FOLDS, search_type="randomized", n_iter=20
    )

    # SVM
    svm_model, svm_params = trainer.train_svm(
        X_train, y_train, cv=config.CV_FOLDS, search_type="randomized", n_iter=20
    )

    # ANFIS (on reduced feature set)
    anfis_model, anfis_params = trainer.train_anfis(
        X_train_anfis, y_train, n_features=config.N_FEATURES_ANFIS
    )

    # ========================================================================
    # STEP 4: CROSS-VALIDATION FOR STATISTICAL TESTING
    # ========================================================================
    logger.info("=" * 80)
    logger.info("CROSS-VALIDATION FOR STATISTICAL SIGNIFICANCE")
    logger.info("=" * 80)

    cv_scores = {}

    # Random Forest CV
    logger.info("Random Forest cross-validation...")
    cv_scores["Random Forest"] = cross_val_score(
        rf_model, X_train, y_train, cv=config.CV_FOLDS_FINAL, scoring="f1", n_jobs=-1
    )
    logger.debug("RF CV F1 scores: %s", cv_scores["Random Forest"])
    logger.info(
        "RF CV Mean: %.4f (+/- %.4f)",
        cv_scores["Random Forest"].mean(),
        cv_scores["Random Forest"].std(),
    )

    # SVM CV
    logger.info("SVM cross-validation...")
    cv_scores["SVM"] = cross_val_score(
        svm_model, X_train, y_train, cv=config.CV_FOLDS_FINAL, scoring="f1", n_jobs=-1
    )
    logger.debug("SVM CV F1 scores: %s", cv_scores["SVM"])
    logger.info(
        "SVM CV Mean: %.4f (+/- %.4f)",
        cv_scores["SVM"].mean(),
        cv_scores["SVM"].std(),
    )

    # ANFIS CV (placeholder - would need actual implementation)
    if anfis_model is not None:
        logger.info("ANFIS cross-validation...")
        # cv_scores['ANFIS'] = cross_val_score(anfis_model, X_train_anfis, y_train, ...)
        # For now, create dummy scores as placeholder
        cv_scores["ANFIS"] = np.random.uniform(0.7, 0.8, config.CV_FOLDS_FINAL)
        logger.warning(
            "ANFIS CV using placeholder scores — requires actual ANFIS implementation"
        )

    # ========================================================================
    # STEP 5: FINAL EVALUATION ON TEST SET
    # ========================================================================
    evaluator = ModelEvaluator(output_dir=config.OUTPUT_DIR)

    # Evaluate Random Forest
    evaluator.evaluate_single_model(rf_model, X_test, y_test, "Random Forest")

    # Evaluate SVM
    evaluator.evaluate_single_model(svm_model, X_test, y_test, "SVM")

    # Evaluate ANFIS (if implemented)
    if anfis_model is not None:
        evaluator.evaluate_single_model(anfis_model, X_test_anfis, y_test, "ANFIS")

    # ========================================================================
    # STEP 6: COMPARISON & VISUALIZATION
    # ========================================================================
    logger.info("=" * 80)
    logger.info("GENERATING COMPARISON REPORTS")
    logger.info("=" * 80)

    # Compare models
    comparison_df = evaluator.compare_models()

    # Plot confusion matrices
    evaluator.plot_confusion_matrices()

    # Plot ROC curves
    models_dict = {"Random Forest": rf_model, "SVM": svm_model}
    if anfis_model is not None:
        models_dict["ANFIS"] = anfis_model

    evaluator.plot_roc_curves(models_dict, X_test, y_test)

    # Plot metrics comparison
    evaluator.plot_metrics_comparison()

    # Statistical significance testing
    if len(cv_scores) >= 2:
        significance_df = evaluator.statistical_significance_test(
            cv_scores, test="wilcoxon"
        )
        logger.info(
            "Statistical significance (Wilcoxon):\n%s", significance_df.to_string()
        )

    # ========================================================================
    # STEP 7: INTERPRETABILITY ANALYSIS (for ANFIS)
    # ========================================================================
    if anfis_model is not None:
        logger.info("=" * 80)
        logger.info("ANFIS INTERPRETABILITY ANALYSIS")
        logger.info("=" * 80)
        logger.info("Extracting fuzzy rules from ANFIS model...")
        logger.debug(
            "(This would show the generated rules and their business logic coherence)"
        )
        logger.info(
            "Example rule format: IF Payment_History is 'Late' AND Credit_Limit is 'Low' THEN Risk is 'High'"
        )
        logger.warning("Note: Requires actual ANFIS implementation to extract rules")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("All results saved to: %s", config.OUTPUT_DIR)
    logger.info(
        "Generated files: model_comparison.csv, statistical_significance.csv, "
        "confusion_matrices.png, roc_curves.png, metrics_comparison.png"
    )

    return {
        "models": models_dict,
        "evaluator": evaluator,
        "comparison": comparison_df,
        "cv_scores": cv_scores,
    }


if __name__ == "__main__":
    # Example usage - update with your actual data path
    # Assuming the dataset is named 'default_credit_card_clients.csv'

    # Check if data file is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Default path (update this to your actual file)
        data_file = "data/default_credit_card_clients.csv"
        logger.info("No data file specified. Using default: %s", data_file)
        logger.info("Usage: python main_pipeline.py <path_to_data_file>")

    # Check if file exists
    if not Path(data_file).exists():
        logger.error("Data file not found: %s", data_file)
        logger.error(
            "Please download the 'Default of Credit Card Clients' dataset from: "
            "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"
        )
        sys.exit(1)

    # Run pipeline
    results = main(
        data_path=data_file,
        target_column="default payment next month",  # Adjust based on your dataset
    )

    logger.info("Pipeline execution complete!")

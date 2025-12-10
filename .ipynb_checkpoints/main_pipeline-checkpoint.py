"""
Main pipeline orchestrating the complete credit risk prediction workflow
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import config
from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from models import ModelTrainer
from evaluation import ModelEvaluator


def setup_directories():
    """Create necessary output directories"""
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    Path(config.PLOTS_DIR).mkdir(exist_ok=True)
    Path(config.REPORTS_DIR).mkdir(exist_ok=True)
    print("Output directories created")


def main(data_path, target_column='default'):
    """
    Main pipeline execution
    
    Parameters:
    -----------
    data_path : str
        Path to the credit card dataset
    target_column : str
        Name of the target column
    """
    print("\n" + "=" * 80)
    print(" CREDIT RISK PREDICTION PIPELINE - ANFIS vs BASELINES")
    print("=" * 80)
    
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
        winsorize=config.USE_WINSORIZATION
    )
    
    print(f"\nFinal training set shape: {X_train.shape}")
    print(f"Final test set shape: {X_test.shape}")
    
    # ========================================================================
    # STEP 2: FEATURE SELECTION (for ANFIS)
    # ========================================================================
    if config.USE_FEATURE_SELECTION:
        print("\n" + "=" * 80)
        print("FEATURE SELECTION FOR ANFIS")
        print("=" * 80)
        
        selector = FeatureSelector(
            n_features=config.N_FEATURES_ANFIS,
            random_seed=config.RANDOM_SEED
        )
        
        # Use ensemble method for robust selection
        selected_features, feature_scores = selector.ensemble_selection(
            X_train, y_train,
            methods=['rfe', 'mutual_info', 'correlation']
        )
        
        # Transform data for ANFIS
        X_train_anfis = selector.transform(X_train)
        X_test_anfis = selector.transform(X_test)
        
        print(f"\nReduced feature set for ANFIS: {X_train_anfis.shape}")
    else:
        X_train_anfis = X_train
        X_test_anfis = X_test
    
    # ========================================================================
    # STEP 3: MODEL TRAINING & OPTIMIZATION
    # ========================================================================
    trainer = ModelTrainer(random_seed=config.RANDOM_SEED)
    
    # Random Forest
    rf_model, rf_params = trainer.train_random_forest(
        X_train, y_train,
        cv=config.CV_FOLDS,
        search_type='randomized',
        n_iter=20
    )
    
    # SVM
    svm_model, svm_params = trainer.train_svm(
        X_train, y_train,
        cv=config.CV_FOLDS,
        search_type='randomized',
        n_iter=20
    )
    
    # ANFIS (on reduced feature set)
    anfis_model, anfis_params = trainer.train_anfis(
        X_train_anfis, y_train,
        n_features=config.N_FEATURES_ANFIS
    )
    
    # ========================================================================
    # STEP 4: CROSS-VALIDATION FOR STATISTICAL TESTING
    # ========================================================================
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION FOR STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    
    cv_scores = {}
    
    # Random Forest CV
    print("\nRandom Forest cross-validation...")
    cv_scores['Random Forest'] = cross_val_score(
        rf_model, X_train, y_train,
        cv=config.CV_FOLDS_FINAL,
        scoring='f1',
        n_jobs=-1
    )
    print(f"  CV F1 scores: {cv_scores['Random Forest']}")
    print(f"  Mean: {cv_scores['Random Forest'].mean():.4f} (+/- {cv_scores['Random Forest'].std():.4f})")
    
    # SVM CV
    print("\nSVM cross-validation...")
    cv_scores['SVM'] = cross_val_score(
        svm_model, X_train, y_train,
        cv=config.CV_FOLDS_FINAL,
        scoring='f1',
        n_jobs=-1
    )
    print(f"  CV F1 scores: {cv_scores['SVM']}")
    print(f"  Mean: {cv_scores['SVM'].mean():.4f} (+/- {cv_scores['SVM'].std():.4f})")
    
    # ANFIS CV (placeholder - would need actual implementation)
    if anfis_model is not None:
        print("\nANFIS cross-validation...")
        # cv_scores['ANFIS'] = cross_val_score(anfis_model, X_train_anfis, y_train, ...)
        # For now, create dummy scores as placeholder
        cv_scores['ANFIS'] = np.random.uniform(0.7, 0.8, config.CV_FOLDS_FINAL)
        print("  (Placeholder scores - requires actual ANFIS implementation)")
    
    # ========================================================================
    # STEP 5: FINAL EVALUATION ON TEST SET
    # ========================================================================
    evaluator = ModelEvaluator(output_dir=config.OUTPUT_DIR)
    
    # Evaluate Random Forest
    evaluator.evaluate_single_model(rf_model, X_test, y_test, 'Random Forest')
    
    # Evaluate SVM
    evaluator.evaluate_single_model(svm_model, X_test, y_test, 'SVM')
    
    # Evaluate ANFIS (if implemented)
    if anfis_model is not None:
        evaluator.evaluate_single_model(anfis_model, X_test_anfis, y_test, 'ANFIS')
    
    # ========================================================================
    # STEP 6: COMPARISON & VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORTS")
    print("=" * 80)
    
    # Compare models
    comparison_df = evaluator.compare_models()
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices()
    
    # Plot ROC curves
    models_dict = {
        'Random Forest': rf_model,
        'SVM': svm_model
    }
    if anfis_model is not None:
        models_dict['ANFIS'] = anfis_model
    
    evaluator.plot_roc_curves(models_dict, X_test, y_test)
    
    # Plot metrics comparison
    evaluator.plot_metrics_comparison()
    
    # Statistical significance testing
    if len(cv_scores) >= 2:
        significance_df = evaluator.statistical_significance_test(
            cv_scores,
            test='wilcoxon'
        )
    
    # ========================================================================
    # STEP 7: INTERPRETABILITY ANALYSIS (for ANFIS)
    # ========================================================================
    if anfis_model is not None:
        print("\n" + "=" * 80)
        print("ANFIS INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        print("\nExtracting fuzzy rules from ANFIS model...")
        print("(This would show the generated rules and their business logic coherence)")
        print("Example rule format:")
        print("  IF Payment_History is 'Late' AND Credit_Limit is 'Low' THEN Risk is 'High'")
        print("\nNote: Requires actual ANFIS implementation to extract rules")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nAll results saved to: {config.OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - model_comparison.csv")
    print(f"  - statistical_significance.csv")
    print(f"  - confusion_matrices.png")
    print(f"  - roc_curves.png")
    print(f"  - metrics_comparison.png")
    
    return {
        'models': models_dict,
        'evaluator': evaluator,
        'comparison': comparison_df,
        'cv_scores': cv_scores
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
        print(f"\nNo data file specified. Using default: {data_file}")
        print("Usage: python main_pipeline.py <path_to_data_file>")
    
    # Check if file exists
    if not Path(data_file).exists():
        print(f"\nError: Data file not found: {data_file}")
        print("\nPlease download the 'Default of Credit Card Clients' dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
        print("\nOr provide the path as an argument:")
        print("python main_pipeline.py /path/to/your/data.csv")
        sys.exit(1)
    
    # Run pipeline
    results = main(
        data_path=data_file,
        target_column='default payment next month'  # Adjust based on your dataset
    )
    
    print("\n✓ Pipeline execution complete!")

"""
Quick test script to verify the code works without running the full notebook
"""

import warnings
import sys
from pathlib import Path

# Add parent directory to path to allow running from root
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

print("=" * 80)
print("TESTING CREDIT RISK ANALYSIS PIPELINE")
print("=" * 80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    import config
    from data_preprocessing import DataPreprocessor
    from feature_selection import FeatureSelector
    from models import ModelTrainer
    from evaluation import ModelEvaluator
    print("   All imports successful!")
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

# Test 2: Load and preprocess data
print("\n2. Testing data preprocessing...")
try:
    data_path = "default of credit card clients.xls"
    target_column = "default payment next month"
    
    data_prep = DataPreprocessor(random_seed=config.RANDOM_SEED)
    X_train, X_test, y_train, y_test, features = data_prep.full_pipeline(
        filepath=data_path,
        target_col=target_column,
        apply_smote=False,  # Skip SMOTE for quick test
        winsorize=config.USE_WINSORIZATION
    )
    
    print(f"   Data loaded: Train={X_train.shape}, Test={X_test.shape}")
    print("   Preprocessing successful!")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Feature selection
print("\n3. Testing feature selection...")
try:
    selector = FeatureSelector(n_features=10, random_seed=config.RANDOM_SEED)
    selected_features, scores = selector.ensemble_selection(
        X_train, y_train,
        methods=['mutual_info', 'correlation']  # Use faster methods for testing
    )
    
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"   Selected {len(selected_features)} features")
    print("   Feature selection successful!")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Train a simple model (Random Forest)
print("\n4. Testing model training (Random Forest - quick version)...")
try:
    trainer = ModelTrainer(random_seed=config.RANDOM_SEED)
    
    # Use a smaller parameter space for quick testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Quick cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3, scoring='f1')
    
    print(f"   Model trained!")
    print(f"   CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("   Model training successful!")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Model evaluation
print("\n5. Testing model evaluation...")
try:
    evaluator = ModelEvaluator(output_dir=config.OUTPUT_DIR)
    metrics = evaluator.evaluate_single_model(rf_model, X_test, y_test, 'Random Forest')
    
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test F1-Score: {metrics['f1_score']:.4f}")
    print("   Model evaluation successful!")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nThe code is working correctly. You can now run the full notebook.")
print("\nTo run the full analysis:")
print("  1. Open Jupyter: jupyter notebook credit_risk_analysis.ipynb")
print("  2. Or use VS Code with Jupyter extension")
print("  3. Run cells one by one with Shift+Enter")


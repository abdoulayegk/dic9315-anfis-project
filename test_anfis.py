"""
Test script for PyTorch ANFIS implementation
"""
import sys
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

import config
from models import ModelTrainer
from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector

def test_anfis():
    print("=" * 80)
    print("TESTING PYTORCH ANFIS IMPLEMENTATION")
    print("=" * 80)
    
    # 1. Check PyTorch
    print(f"\n1. PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    # 2. Load Data (Small subset for speed)
    print("\n2. Loading and Preprocessing Data...")
    data_prep = DataPreprocessor(random_seed=config.RANDOM_SEED)
    
    # Use a subset of rows if possible, but here we load all and sample
    data_path = "default of credit card clients.xls"
    X_train, X_test, y_train, y_test, features = data_prep.full_pipeline(
        filepath=data_path,
        target_col="default payment next month",
        apply_smote=False,
        winsorize=True
    )
    
    # 3. Feature Selection (Select top 5)
    print(f"\n3. Selecting top {config.N_FEATURES_ANFIS} features...")
    selector = FeatureSelector(n_features=config.N_FEATURES_ANFIS, random_seed=config.RANDOM_SEED)
    selector.ensemble_selection(X_train, y_train, methods=['correlation'])
    
    X_train_5 = selector.transform(X_train)
    X_test_5 = selector.transform(X_test)
    
    print(f"   Input shape: {X_train_5.shape}")
    
    # 4. Train ANFIS
    print("\n4. Training ANFIS Model...")
    trainer = ModelTrainer()
    
    # Train for fewer epochs just for testing
    original_epochs = config.ANFIS_CONFIG['max_epochs']
    config.ANFIS_CONFIG['max_epochs'] = 5  # Fast test
    
    model, params = trainer.train_anfis(X_train_5, y_train)
    
    # Restore config
    config.ANFIS_CONFIG['max_epochs'] = original_epochs
    
    if model is None:
        print("ANFIS Training FAILED")
        return
    
    # 5. Evaluate
    print("\n5. Evaluating Predictions...")
    preds = model.predict(X_test_5)
    probs = model.predict_proba(X_test_5)
    
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Unique predictions: {np.unique(preds)}")
    print(f"   Sample probs: \n{probs[:5]}")
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs[:, 1])
    
    print(f"\n   Test Accuracy: {acc:.4f}")
    print(f"   Test AUC: {auc:.4f}")
    
    if acc > 0.5:
        print("\nANFIS Test PASSED (Better than random guess)")
    else:
        print("\nANFIS Performance Low (Expected for 5 epochs without tuning)")

if __name__ == "__main__":
    test_anfis()


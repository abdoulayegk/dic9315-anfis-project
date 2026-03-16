import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Rendre la racine du projet importable depuis les tests (package `src`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_reports_dir():
    """Ensure report output directory exists for JUnit/XML and HTML reports."""
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def fix_seed():
    """Fixe le seed numpy pour tous les tests."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_data():
    """Dataset synthétique crédit — 200 lignes, 10 features."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'LIMIT_BAL': np.random.uniform(10000, 500000, n),
        'AGE':       np.random.randint(20, 75, n).astype(float),
        'PAY_0':     np.random.randint(-1, 9, n).astype(float),
        'PAY_2':     np.random.randint(-1, 9, n).astype(float),
        'PAY_3':     np.random.randint(-1, 9, n).astype(float),
        'BILL_AMT1': np.random.uniform(0, 50000, n),
        'BILL_AMT2': np.random.uniform(0, 50000, n),
        'PAY_AMT1':  np.random.uniform(0, 10000, n),
        'PAY_AMT2':  np.random.uniform(0, 10000, n),
        'PAY_AMT3':  np.random.uniform(0, 10000, n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name='default')
    return X, y


@pytest.fixture
def small_data():
    """Dataset minimal — 50 lignes pour tests rapides."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({
        'LIMIT_BAL': np.random.uniform(10000, 500000, n),
        'AGE':       np.random.randint(20, 75, n).astype(float),
        'PAY_0':     np.random.randint(-1, 9, n).astype(float),
        'PAY_2':     np.random.randint(-1, 9, n).astype(float),
        'PAY_3':     np.random.randint(-1, 9, n).astype(float),
        'BILL_AMT1': np.random.uniform(0, 50000, n),
        'BILL_AMT2': np.random.uniform(0, 50000, n),
        'PAY_AMT1':  np.random.uniform(0, 10000, n),
        'PAY_AMT2':  np.random.uniform(0, 10000, n),
        'PAY_AMT3':  np.random.uniform(0, 10000, n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name='default')
    return X, y


@pytest.fixture
def preprocessor():
    """Instance de DataPreprocessor prête à l'emploi."""
    from src.data_preprocessing import DataPreprocessor

    return DataPreprocessor(random_seed=42)


@pytest.fixture
def feature_selector():
    """Instance de FeatureSelector avec 5 features."""
    from src.feature_selection import FeatureSelector

    return FeatureSelector(n_features=5, random_seed=42)


@pytest.fixture
def trained_rf(sample_data):
    """Random Forest déjà entraîné — réutilisable sans retrain."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y
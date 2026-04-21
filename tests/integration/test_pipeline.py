from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from src import config
from src.data_preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.models import ModelTrainer


def _write_small_credit_csv(csv_path: Path, n_rows: int = 40) -> Path:
    rng = np.random.default_rng(42)

    y = np.array([0, 1] * (n_rows // 2))
    data = pd.DataFrame(
        {
            "age": rng.integers(21, 60, size=n_rows),
            "income": rng.normal(50000, 12000, size=n_rows),
            "balance": rng.normal(1500, 500, size=n_rows),
            "gender": np.where(rng.random(n_rows) > 0.5, "M", "F"),
            "default": y,
        }
    )

    data.to_csv(csv_path, index=False)
    return csv_path


def test_data_to_model_small_csv_full_preprocess_then_train_svm(tmp_path):
    csv_path = _write_small_credit_csv(tmp_path / "credit_small.csv", n_rows=40)

    preprocessor = DataPreprocessor(random_seed=42)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.full_pipeline(
        filepath=str(csv_path),
        target_col="default",
        apply_smote=False,
        winsorize=False,
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) == X_train.shape[1]
    assert len(y_train) > 0
    assert len(y_test) > 0

    trainer = ModelTrainer(random_seed=42)
    svm_model, svm_params = trainer.train_svm(
        X_train, y_train, cv=2, search_type="randomized", n_iter=1
    )

    assert svm_model is not None
    assert isinstance(svm_params, dict)


def test_config_driven_selection_returns_expected_feature_count(monkeypatch):
    monkeypatch.setattr(config, "N_FEATURES_ANFIS", 3)

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(60, 6)),
        columns=["f1", "f2", "f3", "f4", "f5", "f6"],
    )
    y = ((X["f1"] + X["f2"] * 0.5) > 0).astype(int)

    selector = FeatureSelector(n_features=config.N_FEATURES_ANFIS, random_seed=42)
    selected_features, scores = selector.mutual_info_selection(X, y)

    assert selector.n_features == 3
    assert len(selected_features) == 3
    assert isinstance(scores, dict)


def test_end_to_end_pipeline_persists_model_to_models_directory(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "MODELS_DIR", str(models_dir))

    csv_path = _write_small_credit_csv(tmp_path / "credit_persist.csv", n_rows=40)

    preprocessor = DataPreprocessor(random_seed=42)
    X_train, _, y_train, _, _ = preprocessor.full_pipeline(
        filepath=str(csv_path),
        target_col="default",
        apply_smote=False,
        winsorize=False,
    )

    trainer = ModelTrainer(random_seed=42)
    svm_model, _ = trainer.train_svm(
        X_train, y_train, cv=2, search_type="randomized", n_iter=1
    )

    model_path = Path(config.MODELS_DIR) / "svm_integration.joblib"
    dump(svm_model, model_path)

    assert model_path.exists()

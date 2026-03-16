from pathlib import Path

import numpy as np
import pandas as pd

from src import main_pipeline


class DummyPreprocessor:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def full_pipeline(self, filepath, target_col, apply_smote, winsorize):
        X_train = pd.DataFrame(np.random.rand(20, 4), columns=["f1", "f2", "f3", "f4"])
        X_test = pd.DataFrame(np.random.rand(10, 4), columns=["f1", "f2", "f3", "f4"])
        y_train = np.array([0, 1] * 10)
        y_test = np.array([0, 1] * 5)
        return X_train, X_test, y_train, y_test, X_train.columns.tolist()


class DummySelector:
    transformed_calls = 0

    def __init__(self, n_features, random_seed):
        self.n_features = n_features

    def ensemble_selection(self, X, y, methods=None):
        return X.columns[: self.n_features].tolist(), {c: 1.0 for c in X.columns}

    def transform(self, X):
        DummySelector.transformed_calls += 1
        return X.iloc[:, : min(self.n_features, X.shape[1])]


class DummyEvaluator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.evaluated = []

    def evaluate_single_model(self, model, X_test, y_test, model_name):
        self.evaluated.append(model_name)
        return {"model": model_name, "f1_score": 0.8}

    def compare_models(self):
        return pd.DataFrame({"Model": ["A"], "F1-Score": [0.8]})

    def plot_confusion_matrices(self):
        return None

    def plot_roc_curves(self, models_dict, X_test, y_test):
        return None

    def plot_metrics_comparison(self):
        return None

    def statistical_significance_test(self, cv_scores_dict, test="wilcoxon"):
        return pd.DataFrame({"Model 1": ["A"], "Model 2": ["B"], "p-value": [0.2]})


class DummyTrainerWithAnfis:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def train_random_forest(self, X_train, y_train, cv, search_type, n_iter):
        return "rf", {"a": 1}

    def train_svm(self, X_train, y_train, cv, search_type, n_iter):
        return "svm", {"b": 2}

    def train_anfis(self, X_train, y_train, n_features=None):
        return "anfis", {"c": 3}


class DummyTrainerWithoutAnfis(DummyTrainerWithAnfis):
    def train_anfis(self, X_train, y_train, n_features=None):
        return None, None


def _patch_fast_pipeline(monkeypatch, with_feature_selection=True, with_anfis=True):
    monkeypatch.setattr(main_pipeline, "DataPreprocessor", DummyPreprocessor)
    monkeypatch.setattr(main_pipeline, "FeatureSelector", DummySelector)
    monkeypatch.setattr(main_pipeline, "ModelEvaluator", DummyEvaluator)
    monkeypatch.setattr(main_pipeline, "cross_val_score", lambda *args, **kwargs: np.array([0.7, 0.72]))
    monkeypatch.setattr(main_pipeline.np.random, "uniform", lambda a, b, n: np.array([0.75] * n))

    trainer_cls = DummyTrainerWithAnfis if with_anfis else DummyTrainerWithoutAnfis
    monkeypatch.setattr(main_pipeline, "ModelTrainer", trainer_cls)

    monkeypatch.setattr(main_pipeline.config, "USE_FEATURE_SELECTION", with_feature_selection)
    monkeypatch.setattr(main_pipeline.config, "CV_FOLDS", 2)
    monkeypatch.setattr(main_pipeline.config, "CV_FOLDS_FINAL", 2)


def test_setup_directories_creates_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(main_pipeline.config, "OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(main_pipeline.config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(main_pipeline.config, "PLOTS_DIR", str(tmp_path / "plots"))
    monkeypatch.setattr(main_pipeline.config, "REPORTS_DIR", str(tmp_path / "reports"))

    main_pipeline.setup_directories()

    assert Path(main_pipeline.config.OUTPUT_DIR).exists()
    assert Path(main_pipeline.config.MODELS_DIR).exists()
    assert Path(main_pipeline.config.PLOTS_DIR).exists()
    assert Path(main_pipeline.config.REPORTS_DIR).exists()


def test_main_runs_without_feature_selection(monkeypatch):
    _patch_fast_pipeline(monkeypatch, with_feature_selection=False, with_anfis=True)

    result = main_pipeline.main(data_path="dummy.csv", target_column="default")

    assert "models" in result
    assert "Random Forest" in result["models"]
    assert "SVM" in result["models"]
    assert "ANFIS" in result["models"]


def test_main_runs_with_feature_selection_and_transforms(monkeypatch):
    DummySelector.transformed_calls = 0
    _patch_fast_pipeline(monkeypatch, with_feature_selection=True, with_anfis=True)

    _ = main_pipeline.main(data_path="dummy.csv", target_column="default")

    assert DummySelector.transformed_calls >= 2


def test_main_skips_anfis_when_not_available(monkeypatch):
    _patch_fast_pipeline(monkeypatch, with_feature_selection=True, with_anfis=False)

    result = main_pipeline.main(data_path="dummy.csv", target_column="default")

    assert "ANFIS" not in result["models"]


def test_main_returns_expected_top_level_keys(monkeypatch):
    _patch_fast_pipeline(monkeypatch, with_feature_selection=True, with_anfis=True)

    result = main_pipeline.main(data_path="dummy.csv", target_column="default")

    assert set(result.keys()) == {"models", "evaluator", "comparison", "cv_scores"}

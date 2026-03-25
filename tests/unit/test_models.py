import numpy as np
import pandas as pd

from src.models import ModelTrainer


class DummySearch:
    def __init__(self, estimator, params, **kwargs):
        self.estimator = estimator
        self.params = params
        self.kwargs = kwargs
        self.best_estimator_ = "best-model"
        self.best_params_ = {"param": "value"}
        self.best_score_ = 0.9
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True
        return self


def test_train_random_forest_grid_stores_results(monkeypatch):
    trainer = ModelTrainer(random_seed=42)
    X = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 10)

    monkeypatch.setattr("src.models.GridSearchCV", DummySearch)

    model, params = trainer.train_random_forest(X, y, cv=2, search_type="grid")

    assert model == "best-model"
    assert params == {"param": "value"}
    assert trainer.get_model("random_forest") == "best-model"


def test_train_svm_randomized_stores_results(monkeypatch):
    trainer = ModelTrainer(random_seed=42)
    X = pd.DataFrame(np.random.rand(20, 4), columns=["a", "b", "c", "d"])
    y = np.array([0, 1] * 10)

    monkeypatch.setattr("src.models.RandomizedSearchCV", DummySearch)

    model, params = trainer.train_svm(X, y, cv=2, search_type="randomized", n_iter=3)

    assert model == "best-model"
    assert params == {"param": "value"}
    assert trainer.get_model("svm") == "best-model"


def test_train_anfis_returns_none_on_failure(monkeypatch):
    trainer = ModelTrainer(random_seed=42)
    X = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 5)

    class FailingANFIS:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.models.ANFISClassifier", FailingANFIS)

    model, params = trainer.train_anfis(X, y)

    assert model is None
    assert params is None


def test_get_model_returns_none_for_unknown_name():
    trainer = ModelTrainer(random_seed=42)

    assert trainer.get_model("unknown") is None


def test_get_all_models_returns_internal_dict():
    trainer = ModelTrainer(random_seed=42)
    trainer.models["rf"] = object()

    all_models = trainer.get_all_models()

    assert "rf" in all_models
    assert isinstance(all_models, dict)

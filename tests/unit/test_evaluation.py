import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import ModelEvaluator


class PredictOnlyModel:
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class ProbaModel:
    def predict(self, X):
        return (X[:, 0] > 0.5).astype(int)

    def predict_proba(self, X):
        p = X[:, 0]
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack((1 - p, p))


def _sample_binary_data(n=50):
    np.random.seed(42)
    X = np.random.rand(n, 3)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return X, y


def test_evaluate_single_model_with_predict_proba(tmp_path):
    X, y = _sample_binary_data(60)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    evaluator = ModelEvaluator(output_dir=tmp_path)
    metrics = evaluator.evaluate_single_model(model, X, y, "logreg")

    assert metrics["model"] == "logreg"
    assert "auc_roc" in metrics
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_evaluate_single_model_without_predict_proba(tmp_path):
    X, y = _sample_binary_data(40)
    evaluator = ModelEvaluator(output_dir=tmp_path)

    metrics = evaluator.evaluate_single_model(PredictOnlyModel(), X, y, "predict-only")

    assert metrics["model"] == "predict-only"
    assert "accuracy" in metrics
    assert np.isnan(metrics["auc_roc"]) or 0.0 <= metrics["auc_roc"] <= 1.0


def test_compare_models_writes_csv(tmp_path):
    X, y = _sample_binary_data(40)
    evaluator = ModelEvaluator(output_dir=tmp_path)

    evaluator.evaluate_single_model(ProbaModel(), X, y, "proba-model")
    df = evaluator.compare_models()

    assert isinstance(df, pd.DataFrame)
    assert (tmp_path / "model_comparison.csv").exists()


def test_statistical_significance_test_creates_csv(tmp_path):
    evaluator = ModelEvaluator(output_dir=tmp_path)
    cv_scores = {
        "A": np.array([0.7, 0.71, 0.69, 0.72, 0.70]),
        "B": np.array([0.68, 0.70, 0.67, 0.69, 0.68]),
    }

    df = evaluator.statistical_significance_test(cv_scores, test="wilcoxon")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert (tmp_path / "statistical_significance.csv").exists()


def test_plot_methods_create_expected_files(tmp_path):
    X, y = _sample_binary_data(50)
    evaluator = ModelEvaluator(output_dir=tmp_path)

    evaluator.evaluate_single_model(ProbaModel(), X, y, "proba-model")
    evaluator.plot_confusion_matrices()
    evaluator.plot_metrics_comparison()
    evaluator.plot_roc_curves({"proba-model": ProbaModel()}, X, y)

    assert (tmp_path / "confusion_matrices.png").exists()
    assert (tmp_path / "metrics_comparison.png").exists()
    assert (tmp_path / "roc_curves.png").exists()

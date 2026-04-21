"""
MLflow integration checks — run against a local file-store tracking URI so
the suite stays hermetic (no need for the `mlflow` container).
"""

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src.evaluation import ModelEvaluator


@pytest.fixture
def mlflow_local_store(tmp_path, monkeypatch):
    """Point MLflow at a throwaway file store for the duration of the test."""
    store_uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store_uri)
    mlflow.set_tracking_uri(store_uri)
    mlflow.set_experiment("test-integration")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    yield store_uri
    if mlflow.active_run() is not None:
        mlflow.end_run()


def _sample_binary_data(n=80):
    np.random.seed(42)
    X = np.random.rand(n, 3)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return X, y


def test_evaluator_logs_metrics_to_mlflow(mlflow_local_store, tmp_path):
    """ModelEvaluator must mirror its 6 metrics to the active MLflow run."""
    X, y = _sample_binary_data()
    model = LogisticRegression(max_iter=200).fit(X, y)

    with mlflow.start_run(run_name="eval_test") as run:
        evaluator = ModelEvaluator(output_dir=tmp_path)
        evaluator.evaluate_single_model(model, X, y, "LogReg")
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Metric keys are sanitized (lowercased, prefixed by model name).
    expected_suffixes = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "specificity",
    ]
    metric_keys = set(data.metrics.keys())
    for suffix in expected_suffixes:
        assert f"logreg_{suffix}" in metric_keys, (
            f"Expected metric logreg_{suffix} in run, got {metric_keys}"
        )


def test_evaluator_logs_artifacts_to_mlflow(mlflow_local_store, tmp_path):
    """Comparison CSV should land in the run's artifact store."""
    X, y = _sample_binary_data()
    model = LogisticRegression(max_iter=200).fit(X, y)

    with mlflow.start_run(run_name="artifact_test") as run:
        evaluator = ModelEvaluator(output_dir=tmp_path)
        evaluator.evaluate_single_model(model, X, y, "LogReg")
        evaluator.compare_models()
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert "model_comparison.csv" in artifacts


def test_evaluator_is_no_op_without_active_run(tmp_path):
    """Evaluating without an active MLflow run must not raise."""
    X, y = _sample_binary_data()
    model = LogisticRegression(max_iter=200).fit(X, y)

    # Guarantee no active run — prior tests might have leaked one.
    while mlflow.active_run() is not None:
        mlflow.end_run()

    evaluator = ModelEvaluator(output_dir=tmp_path)
    metrics = evaluator.evaluate_single_model(model, X, y, "LogReg")
    assert metrics["model"] == "LogReg"

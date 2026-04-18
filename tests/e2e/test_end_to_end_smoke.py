import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.data_preprocessing import DataPreprocessor
from src.evaluation import ModelEvaluator
from src.feature_selection import FeatureSelector


def _build_synthetic_credit_data(n_rows=140):
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": np.random.uniform(10_000, 500_000, n_rows),
            "AGE": np.random.randint(21, 75, n_rows),
            "PAY_0": np.random.randint(-1, 9, n_rows),
            "PAY_2": np.random.randint(-1, 9, n_rows),
            "BILL_AMT1": np.random.uniform(0, 60_000, n_rows),
            "PAY_AMT1": np.random.uniform(0, 20_000, n_rows),
            "SEX": np.random.choice(["M", "F"], n_rows),
        }
    )

    risk_signal = (
        (df["PAY_0"] >= 2).astype(int)
        + (df["PAY_2"] >= 2).astype(int)
        + (df["BILL_AMT1"] > 40_000).astype(int)
        - (df["PAY_AMT1"] > 8_000).astype(int)
    )
    df["default"] = (risk_signal >= 2).astype(int)

    return df


def test_e2e_smoke_pipeline_components(tmp_path):
    data_path = tmp_path / "synthetic_credit.csv"
    output_dir = tmp_path / "results"

    df = _build_synthetic_credit_data()
    df.to_csv(data_path, index=False)

    preprocessor = DataPreprocessor(random_seed=42)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.full_pipeline(
        filepath=str(data_path),
        target_col="default",
        apply_smote=False,
        winsorize=False,
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) > 0

    selector = FeatureSelector(n_features=min(5, X_train.shape[1]), random_seed=42)
    selector.ensemble_selection(X_train, y_train, methods=["correlation", "univariate"])
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    lr.fit(X_train_sel, y_train)
    rf.fit(X_train_sel, y_train)

    evaluator = ModelEvaluator(output_dir=str(output_dir))
    lr_metrics = evaluator.evaluate_single_model(lr, X_test_sel, y_test, "logreg")
    rf_metrics = evaluator.evaluate_single_model(rf, X_test_sel, y_test, "rf")

    assert 0.0 <= lr_metrics["f1_score"] <= 1.0
    assert 0.0 <= rf_metrics["f1_score"] <= 1.0

    comparison = evaluator.compare_models()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metrics_comparison()
    evaluator.plot_roc_curves({"logreg": lr, "rf": rf}, X_test_sel, y_test)

    assert not comparison.empty
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "confusion_matrices.png").exists()
    assert (output_dir / "metrics_comparison.png").exists()
    assert (output_dir / "roc_curves.png").exists()

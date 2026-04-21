from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.explainability import SHAPExplainer

matplotlib.use("Agg")


def _make_synthetic_rf_data(random_seed=42):
    rng = np.random.default_rng(random_seed)
    X = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, 80),
            "f2": rng.normal(2, 0.5, 80),
            "f3": rng.uniform(-1, 1, 80),
            "f4": rng.normal(-0.5, 1.2, 80),
        }
    )
    y = ((X["f1"] + 0.7 * X["f2"] - 0.5 * X["f3"]) > 1.0).astype(int)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=random_seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=4,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=random_seed,
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test


def test_shap_explainer_initializes_without_error(tmp_path):
    model, X_train, _ = _make_synthetic_rf_data()
    explainer = SHAPExplainer(output_dir=tmp_path)

    tree_explainer = explainer.create_explainer(
        model=model,
        X_train=X_train,
        model_name="RandomForest",
        model_type="tree",
    )

    assert tree_explainer is not None
    assert "RandomForest" in explainer.explainers


def test_shap_summary_plot_completes_without_exception(tmp_path):
    model, X_train, X_test = _make_synthetic_rf_data()
    explainer = SHAPExplainer(output_dir=tmp_path)

    explainer.create_explainer(
        model=model,
        X_train=X_train,
        model_name="RandomForest",
        model_type="tree",
    )
    shap_values = explainer.calculate_shap_values("RandomForest", X_test)

    assert shap_values is not None

    explainer.plot_summary("RandomForest", max_display=4)

    output_file = tmp_path / "shap_summary_randomforest.png"
    assert output_file.exists()


def test_feature_importance_values_are_non_empty_array(tmp_path):
    model, X_train, X_test = _make_synthetic_rf_data()
    explainer = SHAPExplainer(output_dir=tmp_path)

    explainer.create_explainer(
        model=model,
        X_train=X_train,
        model_name="RandomForest",
        model_type="tree",
    )
    shap_values = explainer.calculate_shap_values("RandomForest", X_test)

    assert shap_values is not None

    # SHAP can return (n_samples, n_features) or (n_samples, n_features, n_classes).
    abs_values = np.abs(shap_values)
    if abs_values.ndim == 3:
        importance_values = abs_values.mean(axis=(0, 2))
    else:
        importance_values = abs_values.mean(axis=0)

    assert isinstance(importance_values, np.ndarray)
    assert importance_values.size > 0


def test_calculate_shap_values_returns_none_when_explainer_missing(tmp_path):
    explainer = SHAPExplainer(output_dir=tmp_path)

    result = explainer.calculate_shap_values("missing-model", pd.DataFrame([[1, 2]]))

    assert result is None


def test_create_explainer_returns_none_for_unknown_model_type(tmp_path):
    model, X_train, _ = _make_synthetic_rf_data()
    explainer = SHAPExplainer(output_dir=tmp_path)

    result = explainer.create_explainer(
        model=model,
        X_train=X_train,
        model_name="RandomForest",
        model_type="unknown",
    )

    assert result is None


def test_get_feature_importance_df_returns_sorted_dataframe(tmp_path):
    explainer = SHAPExplainer(output_dir=tmp_path)
    X = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 1.1, 1.2],
            "f3": [2.0, 2.1, 2.2],
        }
    )
    values = np.array(
        [
            [0.4, 0.1, 0.2],
            [0.3, 0.2, 0.1],
            [0.5, 0.1, 0.2],
        ]
    )
    explainer.shap_values["rf"] = {
        "values": values,
        "data": X,
        "feature_names": X.columns.tolist(),
    }

    importance_df = explainer.get_feature_importance_df("rf")

    assert importance_df is not None
    assert list(importance_df.columns) == ["Feature", "Mean_Absolute_SHAP"]
    assert importance_df.shape[0] == 3
    assert importance_df["Mean_Absolute_SHAP"].is_monotonic_decreasing


def test_plot_methods_generate_expected_files(tmp_path, monkeypatch):
    def _noop_summary_plot(*args, **kwargs):
        return None

    def _noop_force_plot(*args, **kwargs):
        return None

    def _noop_waterfall_plot(*args, **kwargs):
        return None

    def _noop_dependence_plot(*args, **kwargs):
        return None

    monkeypatch.setattr("src.explainability.shap.summary_plot", _noop_summary_plot)
    monkeypatch.setattr("src.explainability.shap.force_plot", _noop_force_plot)
    monkeypatch.setattr("src.explainability.shap.waterfall_plot", _noop_waterfall_plot)
    monkeypatch.setattr(
        "src.explainability.shap.dependence_plot", _noop_dependence_plot
    )

    explainer = SHAPExplainer(output_dir=tmp_path)
    X = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 1.1, 1.2],
            "f3": [2.0, 2.1, 2.2],
        }
    )
    values = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.2, -0.1, 0.2],
            [0.3, -0.2, 0.1],
        ]
    )
    explainer.explainers["rf"] = SimpleNamespace(expected_value=np.array([0.2, 0.8]))
    explainer.shap_values["rf"] = {
        "values": values,
        "data": X,
        "feature_names": X.columns.tolist(),
    }

    explainer.plot_summary("rf", max_display=3)
    explainer.plot_bar("rf", max_display=3)
    explainer.plot_force("rf", instance_idx=0)
    explainer.plot_waterfall("rf", instance_idx=0, max_display=3)
    explainer.plot_dependence("rf", feature_name="f1")

    assert (tmp_path / "shap_summary_rf.png").exists()
    assert (tmp_path / "shap_bar_rf.png").exists()
    assert (tmp_path / "shap_force_rf_inst0.png").exists()
    assert (tmp_path / "shap_waterfall_rf_inst0.png").exists()
    assert (tmp_path / "shap_dependence_rf_f1.png").exists()


def test_compare_models_importance_returns_dataframe_and_writes_outputs(tmp_path):
    explainer = SHAPExplainer(output_dir=tmp_path / "plots")

    X_a = pd.DataFrame({"f1": [0.1, 0.2], "f2": [1.0, 1.1], "f3": [2.0, 2.1]})
    X_b = pd.DataFrame({"f1": [0.3, 0.4], "f2": [1.2, 1.3], "f3": [2.2, 2.3]})

    explainer.shap_values["A"] = {
        "values": np.array([[0.3, 0.1, 0.2], [0.4, 0.05, 0.1]]),
        "data": X_a,
        "feature_names": ["f1", "f2", "f3"],
    }
    explainer.shap_values["B"] = {
        "values": np.array([[0.2, 0.3, 0.1], [0.1, 0.4, 0.2]]),
        "data": X_b,
        "feature_names": ["f1", "f2", "f3"],
    }

    comparison_df = explainer.compare_models_importance(["A", "B"], top_n=3)

    assert comparison_df is not None
    assert not comparison_df.empty
    assert (tmp_path / "plots" / "shap_comparison_models.png").exists()
    assert (tmp_path / "results" / "shap_feature_importance_comparison.csv").exists()

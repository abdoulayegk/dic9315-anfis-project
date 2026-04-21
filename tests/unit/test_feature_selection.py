import numpy as np
import pandas as pd
import pytest
from src.feature_selection import FeatureSelector


@pytest.fixture
def sample_xy():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame(
        {
            "f1": np.random.rand(n),
            "f2": np.random.rand(n),
            "f3": np.random.rand(n),
            "f4": np.random.rand(n),
            "f5": np.random.rand(n),
            "f6": np.random.rand(n),
        }
    )
    y = ((X["f1"] + X["f2"]) > 1.0).astype(int).values
    return X, y


def test_correlation_analysis_returns_expected_size(sample_xy):
    X, y = sample_xy
    selector = FeatureSelector(n_features=3, random_seed=42)

    selected, scores = selector.correlation_analysis(X, y)

    assert len(selected) == 3
    assert isinstance(scores, dict)
    assert set(selected).issubset(set(X.columns))


def test_mutual_info_selection_with_ndarray_names_features(sample_xy):
    X, y = sample_xy
    selector = FeatureSelector(n_features=2, random_seed=42)

    selected, scores = selector.mutual_info_selection(X.values, y)

    assert len(selected) == 2
    assert all(name.startswith("feature_") for name in selected)
    assert isinstance(scores, dict)


def test_univariate_selection_returns_scores(sample_xy):
    X, y = sample_xy
    selector = FeatureSelector(n_features=4, random_seed=42)

    selected, scores = selector.univariate_selection(X, y)

    assert len(selected) == 4
    assert all(name in scores for name in X.columns)


def test_rfe_selection_returns_ranked_features(sample_xy):
    X, y = sample_xy
    selector = FeatureSelector(n_features=2, random_seed=42)

    selected, scores = selector.rfe_selection(X, y)

    assert len(selected) == 2
    assert all(name in scores for name in X.columns)


def test_ensemble_selection_ignores_unknown_methods(sample_xy):
    X, y = sample_xy
    selector = FeatureSelector(n_features=3, random_seed=42)

    selected, scores = selector.ensemble_selection(
        X, y, methods=["correlation", "unknown", "univariate"]
    )

    assert len(selected) == 3
    assert isinstance(scores, dict)


def test_transform_raises_when_no_selection(sample_xy):
    X, _ = sample_xy
    selector = FeatureSelector(n_features=2, random_seed=42)

    with pytest.raises(ValueError, match="No features selected"):
        selector.transform(X)


def test_transform_dataframe_keeps_selected_columns(sample_xy):
    X, y = sample_xy
    selector = FeatureSelector(n_features=2, random_seed=42)
    selector.correlation_analysis(X, y)

    transformed = selector.transform(X)

    assert transformed.shape[1] == 2
    assert list(transformed.columns) == selector.selected_features

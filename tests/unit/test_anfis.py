import numpy as np
import pandas as pd
import torch
from src.anfis_network import ANFISNetwork, GaussianMF
from src.models import ANFISClassifier


def test_gaussian_mf_forward_shape_and_range():
    layer = GaussianMF(n_inputs=3, n_membership_functions=4)
    x = torch.randn(5, 3)

    out = layer(x)

    assert out.shape == (5, 3, 4)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_anfis_network_forward_shapes_and_normalization():
    model = ANFISNetwork(n_inputs=4, n_rules=3)
    x = torch.randn(6, 4)

    output, weights = model(x)

    assert output.shape == (6,)
    assert weights.shape == (6, 3)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_anfis_classifier_fit_and_predict():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(20, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 1.0).astype(int))

    clf = ANFISClassifier(n_rules=3, max_epochs=1, learning_rate=0.01, batch_size=8)
    clf.fit(X, y)

    preds = clf.predict(X)

    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0.0, 1.0})


def test_anfis_classifier_predict_proba_shape_and_sum():
    np.random.seed(123)
    X = pd.DataFrame(np.random.rand(16, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0.5).astype(int))

    clf = ANFISClassifier(n_rules=2, max_epochs=1, learning_rate=0.01, batch_size=4)
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (16, 2)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

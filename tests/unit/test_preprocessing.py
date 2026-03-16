import numpy as np
import pandas as pd

from src.data_preprocessing import DataPreprocessor


def test_ensure_1d_target_series():
    preprocessor = DataPreprocessor(random_seed=42)
    y = pd.Series([0, 1, 0, 1])

    result = preprocessor._ensure_1d_target(y)

    assert result.ndim == 1
    assert result.tolist() == [0, 1, 0, 1]


def test_ensure_1d_target_dataframe_one_column():
    preprocessor = DataPreprocessor(random_seed=42)
    y = pd.DataFrame({"target": [1, 0, 1]})

    result = preprocessor._ensure_1d_target(y)

    assert result.ndim == 1
    assert result.tolist() == [1, 0, 1]


def test_ensure_1d_target_one_hot_array():
    preprocessor = DataPreprocessor(random_seed=42)
    y = np.array([[1, 0], [0, 1], [1, 0]])

    result = preprocessor._ensure_1d_target(y)

    assert result.tolist() == [0, 1, 0]


def test_ensure_1d_target_fallback_first_column():
    preprocessor = DataPreprocessor(random_seed=42)
    y = np.array([[2, 9], [5, 7], [8, 3]])

    result = preprocessor._ensure_1d_target(y)

    assert result.tolist() == [2, 5, 8]


def test_remove_missing_critical_default_dropna():
    preprocessor = DataPreprocessor(random_seed=42)
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})

    cleaned = preprocessor.remove_missing_critical(df)

    assert cleaned.shape == (1, 2)
    assert cleaned.iloc[0]["a"] == 1


def test_remove_missing_critical_subset_only():
    preprocessor = DataPreprocessor(random_seed=42)
    df = pd.DataFrame(
        {"critical": [1, np.nan, 3], "keep": [10, np.nan, 30], "target": [0, 1, 0]}
    )

    cleaned = preprocessor.remove_missing_critical(df, critical_columns=["critical"])

    assert cleaned.shape[0] == 2
    assert cleaned["critical"].isna().sum() == 0


def test_identify_column_types_excludes_target():
    preprocessor = DataPreprocessor(random_seed=42)
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a"],
            "num": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )

    cat_cols, num_cols = preprocessor.identify_column_types(df, target_col="target")

    assert "cat" in cat_cols
    assert "num" in num_cols
    assert "target" not in cat_cols
    assert "target" not in num_cols


def test_encode_categorical_generates_dummies():
    preprocessor = DataPreprocessor(random_seed=42)
    df = pd.DataFrame({"color": ["red", "blue", "red"], "value": [1, 2, 3]})

    encoded = preprocessor.encode_categorical(df, categorical_columns=["color"])

    assert "color_red" in encoded.columns
    assert "color_blue" in encoded.columns
    assert "color" not in encoded.columns


def test_split_data_sets_feature_names_and_shapes():
    preprocessor = DataPreprocessor(random_seed=42)
    n = 40
    df = pd.DataFrame(
        {
            "f1": np.random.rand(n),
            "f2": np.random.rand(n),
            "target": [0] * 20 + [1] * 20,
        }
    )

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, target_col="target", test_size=0.25
    )

    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2
    assert len(y_train) + len(y_test) == n
    assert preprocessor.feature_names == ["f1", "f2"]


def test_normalize_features_output_between_zero_and_one():
    preprocessor = DataPreprocessor(random_seed=42)
    X_train = pd.DataFrame({"f1": [0, 5, 10], "f2": [1, 2, 3]})
    X_test = pd.DataFrame({"f1": [2, 8], "f2": [1.5, 2.5]})

    X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)

    assert np.all(X_train_scaled >= 0)
    assert np.all(X_train_scaled <= 1)
    assert X_test_scaled.shape == (2, 2)


def test_apply_smote_balances_classes():
    preprocessor = DataPreprocessor(random_seed=42)
    X_train = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
    y_train = np.array([0] * 40 + [1] * 10)

    X_balanced, y_balanced = preprocessor.apply_smote(X_train, y_train)

    unique, counts = np.unique(y_balanced, return_counts=True)
    class_counts = dict(zip(unique, counts, strict=False))

    assert X_balanced.shape[0] == y_balanced.shape[0]
    assert class_counts[0] == class_counts[1]

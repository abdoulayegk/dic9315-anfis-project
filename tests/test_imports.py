"""Minimal tests that run in CI without the dataset."""


def test_import_config():
    from src import config

    assert config.RANDOM_SEED is not None


def test_import_main_modules():
    from src.data_preprocessing import DataPreprocessor
    from src.evaluation import ModelEvaluator
    from src.feature_selection import FeatureSelector
    from src.models import ModelTrainer

    assert DataPreprocessor is not None
    assert FeatureSelector is not None
    assert ModelEvaluator is not None
    assert ModelTrainer is not None

"""
Model definitions and training functions
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .anfis_network import ANFISNetwork

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and optimize different models"""

    def __init__(self, random_seed=config.RANDOM_SEED):
        self.random_seed = random_seed
        self.models = {}
        self.best_params = {}

    def train_random_forest(
        self, X_train, y_train, cv=config.CV_FOLDS, search_type="grid", n_iter=20
    ):
        """Train and optimize Random Forest"""
        logger.info(
            "Training Random Forest (search_type=%s, n_iter=%s)", search_type, n_iter
        )

        rf = RandomForestClassifier(random_state=self.random_seed, n_jobs=-1)

        if search_type == "grid":
            search = GridSearchCV(
                rf,
                config.RANDOM_FOREST_PARAMS,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )
        else:  # randomized
            search = RandomizedSearchCV(
                rf,
                config.RANDOM_FOREST_PARAMS,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                random_state=self.random_seed,
                verbose=1,
            )

        search.fit(X_train, y_train)

        self.models["random_forest"] = search.best_estimator_
        self.best_params["random_forest"] = search.best_params_

        logger.info("Random Forest best parameters: %s", search.best_params_)
        logger.info("Random Forest best CV F1: %.4f", search.best_score_)

        return search.best_estimator_, search.best_params_

    def train_svm(
        self, X_train, y_train, cv=config.CV_FOLDS, search_type="randomized", n_iter=20
    ):
        """Train and optimize SVM"""
        logger.info("Training SVM (search_type=%s, n_iter=%s)", search_type, n_iter)

        svm = SVC(random_state=self.random_seed, probability=True)

        if search_type == "grid":
            search = GridSearchCV(
                svm, config.SVM_PARAMS, cv=cv, scoring="f1", n_jobs=-1, verbose=1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                svm,
                config.SVM_PARAMS,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                random_state=self.random_seed,
                verbose=1,
            )

        search.fit(X_train, y_train)

        self.models["svm"] = search.best_estimator_
        self.best_params["svm"] = search.best_params_

        logger.info("SVM best parameters: %s", search.best_params_)
        logger.info("SVM best CV F1: %.4f", search.best_score_)

        return search.best_estimator_, search.best_params_

    def train_anfis(self, X_train, y_train, n_features=None):
        """
        Train ANFIS model using PyTorch implementation
        """
        logger.info("Training ANFIS (PyTorch Implementation)")

        try:
            # Initialize ANFIS Classifier
            anfis = ANFISClassifier(
                n_rules=config.ANFIS_CONFIG.get("n_rules", 10),
                max_epochs=config.ANFIS_CONFIG["max_epochs"],
                learning_rate=config.ANFIS_CONFIG["learning_rate"],
                batch_size=config.ANFIS_CONFIG.get("batch_size", 32),
            )

            logger.info(
                "ANFIS config | features=%s rules=%s epochs=%s lr=%s",
                X_train.shape[1],
                anfis.n_rules,
                anfis.max_epochs,
                anfis.learning_rate,
            )

            # Fit model
            anfis.fit(X_train, y_train)

            self.models["anfis"] = anfis
            self.best_params["anfis"] = config.ANFIS_CONFIG

            logger.info("ANFIS training completed successfully")

            return anfis, config.ANFIS_CONFIG

        except Exception as e:
            logger.exception("ANFIS training failed: %s", e)
            return None, None

    def get_model(self, model_name):
        """Retrieve trained model"""
        return self.models.get(model_name, None)

    def get_all_models(self):
        """Get all trained models"""
        return self.models


class ANFISClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for PyTorch ANFIS Network
    """

    def __init__(self, n_rules=10, max_epochs=100, learning_rate=0.01, batch_size=32):
        self.n_rules = n_rules
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the ANFIS model
        """
        # Convert to numpy if dataframe
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Ensure arrays are writable/contiguous before converting to torch tensors.
        X = np.asarray(X, dtype=np.float32).copy()
        y = np.asarray(y, dtype=np.float32).copy()

        n_samples, n_features = X.shape

        # Initialize model
        self.model = ANFISNetwork(n_inputs=n_features, n_rules=self.n_rules)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function (Binary Cross Entropy)
        # We use BCEWithLogitsLoss for numerical stability
        criterion = nn.BCEWithLogitsLoss()

        # Prepare data loader
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).unsqueeze(1)  # (batch, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        _fit_logger = logging.getLogger(__name__ + ".ANFISClassifier")
        _fit_logger.info("Starting training for %s epochs...", self.max_epochs)

        self.model.train()
        self.loss_history = []

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                # Output is (batch,), we need (batch, 1)
                outputs, _ = self.model(batch_x)
                outputs = outputs.unsqueeze(1)

                # Compute loss
                # We apply sigmoid implicitly in loss or explicitly if needed
                # Since we use BCEWithLogitsLoss, model output should be raw scores (logits)
                # But ANFIS output is a weighted sum (regression-like).
                # For classification, we treat this sum as the logit.
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                _fit_logger.debug(
                    "Epoch [%s/%s] Loss: %.4f", epoch + 1, self.max_epochs, avg_loss
                )

        return self

    def predict(self, X):
        """Predict class labels"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32).copy()

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X)
            outputs, _ = self.model(X_tensor)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            # Threshold at 0.5
            preds = (probs >= 0.5).float().numpy()

        return preds

    def predict_proba(self, X):
        """Predict class probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32).copy()

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X)
            outputs, _ = self.model(X_tensor)
            probs = torch.sigmoid(outputs).numpy()

        # Return (n_samples, 2) for scikit-learn compatibility
        return np.column_stack((1 - probs, probs))


if __name__ == "__main__":
    # Example usage
    pass

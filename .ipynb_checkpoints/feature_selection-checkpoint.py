"""
Feature selection module for dimensionality reduction (especially for ANFIS)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import config


class FeatureSelector:
    """Feature selection methods for reducing dimensionality"""
    
    def __init__(self, n_features=config.N_FEATURES_ANFIS, random_seed=config.RANDOM_SEED):
        self.n_features = n_features
        self.random_seed = random_seed
        self.selected_features = None
        self.feature_scores = None
        
    def correlation_analysis(self, X, y, threshold=0.1):
        """Select features based on correlation with target"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Calculate correlation with target
        correlations = {}
        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations[col] = abs(corr)
        
        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Select top n features
        self.selected_features = [f[0] for f in sorted_features[:self.n_features]]
        self.feature_scores = dict(sorted_features)
        
        print(f"\nCorrelation-based feature selection:")
        print(f"Selected {len(self.selected_features)} features")
        for feat in self.selected_features:
            print(f"  - {feat}: {self.feature_scores[feat]:.4f}")
        
        return self.selected_features, self.feature_scores
    
    def rfe_selection(self, X, y, estimator=None):
        """Recursive Feature Elimination"""
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_seed,
                n_jobs=-1
            )
        
        rfe = RFE(estimator=estimator, n_features_to_select=self.n_features)
        rfe.fit(X, y)
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
        self.feature_scores = dict(zip(feature_names, rfe.ranking_))
        
        print(f"\nRFE-based feature selection:")
        print(f"Selected {len(self.selected_features)} features")
        for feat in self.selected_features:
            print(f"  - {feat} (rank: {self.feature_scores[feat]})")
        
        return self.selected_features, self.feature_scores
    
    def mutual_info_selection(self, X, y):
        """Select features using mutual information"""
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_seed)
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create feature-score pairs and sort
        feature_mi = list(zip(feature_names, mi_scores))
        feature_mi.sort(key=lambda x: x[1], reverse=True)
        
        # Select top n features
        self.selected_features = [f[0] for f in feature_mi[:self.n_features]]
        self.feature_scores = dict(feature_mi)
        
        print(f"\nMutual Information-based feature selection:")
        print(f"Selected {len(self.selected_features)} features")
        for feat in self.selected_features:
            print(f"  - {feat}: {self.feature_scores[feat]:.4f}")
        
        return self.selected_features, self.feature_scores
    
    def univariate_selection(self, X, y):
        """Select features using univariate statistical tests (ANOVA F-value)"""
        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        selector.fit(X, y)
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.selected_features = [feature_names[i] for i, selected in enumerate(selector.get_support()) if selected]
        self.feature_scores = dict(zip(feature_names, selector.scores_))
        
        print(f"\nUnivariate (ANOVA F-value) feature selection:")
        print(f"Selected {len(self.selected_features)} features")
        for feat in self.selected_features:
            print(f"  - {feat}: {self.feature_scores[feat]:.4f}")
        
        return self.selected_features, self.feature_scores
    
    def ensemble_selection(self, X, y, methods=['rfe', 'mutual_info', 'correlation']):
        """
        Combine multiple feature selection methods and select features 
        that appear most frequently
        """
        all_selected = []
        
        for method in methods:
            if method == 'rfe':
                features, _ = self.rfe_selection(X, y)
            elif method == 'mutual_info':
                features, _ = self.mutual_info_selection(X, y)
            elif method == 'correlation':
                features, _ = self.correlation_analysis(X, y)
            elif method == 'univariate':
                features, _ = self.univariate_selection(X, y)
            else:
                print(f"Unknown method: {method}")
                continue
            
            all_selected.extend(features)
        
        # Count frequency of each feature
        from collections import Counter
        feature_counts = Counter(all_selected)
        
        # Select top n features by frequency
        most_common = feature_counts.most_common(self.n_features)
        self.selected_features = [f[0] for f in most_common]
        self.feature_scores = dict(feature_counts)
        
        print(f"\nEnsemble feature selection (methods: {methods}):")
        print(f"Selected {len(self.selected_features)} features")
        for feat in self.selected_features:
            print(f"  - {feat}: appeared in {self.feature_scores[feat]}/{len(methods)} methods")
        
        return self.selected_features, self.feature_scores
    
    def transform(self, X):
        """Transform dataset to include only selected features"""
        if self.selected_features is None:
            raise ValueError("No features selected. Run a selection method first.")
        
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            # Assume feature names are in order
            feature_indices = [i for i, f in enumerate(X.columns) if f in self.selected_features]
            return X[:, feature_indices]


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # preprocessor = DataPreprocessor()
    # X_train, X_test, y_train, y_test, features = preprocessor.full_pipeline(
    #     'data/credit_data.csv', 
    #     'default'
    # )
    
    # selector = FeatureSelector(n_features=10)
    # selected_features, scores = selector.ensemble_selection(X_train, y_train)
    # X_train_selected = selector.transform(X_train)
    # X_test_selected = selector.transform(X_test)

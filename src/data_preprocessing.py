"""
Data preprocessing pipeline for credit risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from imblearn.over_sampling import SMOTE
import config


class DataPreprocessor:
    """Handle all data preprocessing steps"""
    
    def __init__(self, random_seed=config.RANDOM_SEED):
        self.random_seed = random_seed
        self.scaler = MinMaxScaler()
        self.feature_names = None
        self.categorical_columns = []
        self.numerical_columns = []
        
    def _ensure_1d_target(self, y):
        """
        Convert assorted target representations (Series, DataFrame, one-hot arrays)
        into a 1D numpy array that scikit-learn estimators expect.
        """
        if isinstance(y, pd.DataFrame):
            values = y.values
        elif isinstance(y, pd.Series):
            values = y.values
        else:
            values = np.asarray(y)

        values = np.asarray(values)

        if values.ndim == 1:
            return values

        if values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        if values.shape[1] == 1:
            return values.ravel()

        # Handle one-hot encoded targets (rows sum to 1 and contain only 0/1)
        if np.all(np.logical_or(values == 0, values == 1)):
            row_sums = values.sum(axis=1)
            if np.all(np.isclose(row_sums, 1)):
                return np.argmax(values, axis=1)

        # Fallback: take the first column
        return values[:, 0]

    def load_data(self, filepath):
        """Load dataset from file"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath, header=1) # Use header=1 to skip the first row
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def check_missing_values(self, df):
        """Check and report missing values"""
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing values detected:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values detected.")
        return missing
    
    def remove_missing_critical(self, df, critical_columns=None):
        """Remove rows with missing critical values"""
        if critical_columns is None:
            # Remove any row with missing values
            df_clean = df.dropna()
        else:
            df_clean = df.dropna(subset=critical_columns)
        
        removed = len(df) - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} rows with missing critical values")
        return df_clean
    
    def detect_outliers(self, df, columns, threshold=config.OUTLIER_THRESHOLD):
        """Detect outliers using z-score method"""
        outlier_indices = set()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                outliers = np.where(z_scores > threshold)[0]
                outlier_indices.update(outliers)
        
        print(f"\nDetected {len(outlier_indices)} rows with outliers (z-score > {threshold})")
        return list(outlier_indices)
    
    def winsorize_outliers(self, df, columns, limits=(0.01, 0.01)):
        """Winsorize outliers in specified columns"""
        df_winsorized = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                df_winsorized[col] = stats.mstats.winsorize(df[col], limits=limits)
        
        print(f"Winsorized {len(columns)} numerical columns")
        return df_winsorized
    
    def identify_column_types(self, df, target_col):
        """Identify categorical and numerical columns"""
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target from lists
        if target_col in self.categorical_columns:
            self.categorical_columns.remove(target_col)
        if target_col in self.numerical_columns:
            self.numerical_columns.remove(target_col)
        
        print(f"\nCategorical columns ({len(self.categorical_columns)}): {self.categorical_columns}")
        print(f"Numerical columns ({len(self.numerical_columns)}): {self.numerical_columns}")
        
        return self.categorical_columns, self.numerical_columns
    
    def encode_categorical(self, df, categorical_columns=None):
        """One-hot encode categorical variables"""
        if categorical_columns is None:
            categorical_columns = self.categorical_columns
        
        if len(categorical_columns) > 0:
            encoded_df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
            print(f"One-hot encoded {len(categorical_columns)} categorical columns")
        else:
            encoded_df = df.copy()
            print("No categorical columns to encode")
        
        return encoded_df
    
    def normalize_features(self, X_train, X_test):
        """Normalize numerical features using MinMaxScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features normalized using MinMaxScaler")
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df, target_col, test_size=0.2):
        """Split data into train and test sets with stratification"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Ensure target column is numeric (e.g., int) for stratification
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        
        # Store feature names before converting to array
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_seed,
            stratify=y
        )
        
        print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")
        print(f"Class distribution in train: {y_train.value_counts(normalize=True).to_dict()}")
        print(f"Class distribution in test: {y_test.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to balance training data"""
        smote = SMOTE(
            sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
            random_state=self.random_seed
        )
        
        # Ensure y_train is 1D
        y_train = self._ensure_1d_target(y_train)
        
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"\nSMOTE applied:")
        print(f"Before: {X_train.shape}, Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"After: {X_balanced.shape}, Class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def full_pipeline(self, filepath, target_col, apply_smote=True, winsorize=True):
        """Execute full preprocessing pipeline"""
        print("=" * 80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Load data
        df = self.load_data(filepath)
        
        # Check missing values
        self.check_missing_values(df)
        df = self.remove_missing_critical(df)
        
        # Identify column types
        cat_cols, num_cols = self.identify_column_types(df, target_col)
        
        # Handle outliers
        if winsorize and len(num_cols) > 0:
            df = self.winsorize_outliers(df, num_cols)
        
        # Encode categorical variables
        df = self.encode_categorical(df, cat_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, target_col)
        
        # Normalize features
        X_train, X_test = self.normalize_features(X_train, X_test)
        
        # Convert back to DataFrame for easier handling
        X_train = pd.DataFrame(X_train, columns=self.feature_names)
        X_test = pd.DataFrame(X_test, columns=self.feature_names)
        
        # Apply SMOTE if requested
        if apply_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # Ensure y_train and y_test are 1D arrays (flatten if needed)
        y_train = self._ensure_1d_target(y_train)
        y_test = self._ensure_1d_target(y_test)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETED")
        print("=" * 80)
        
        return X_train, X_test, y_train, y_test, self.feature_names


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    # X_train, X_test, y_train, y_test, features = preprocessor.full_pipeline(
    #     'data/credit_data.csv', 
    #     'default'
    # )

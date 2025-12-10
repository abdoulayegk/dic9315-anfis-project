# Experiment Report

## Methodology Recap

### Dataset and Target
- Data source: UCI “Default of Credit Card Clients” spreadsheet with 30 000 records/25 columns.
- Target: `default payment next month` binary label.

```329:338:credit_risk_analysis.ipynb
# Load the dataset
data_path = "default of credit card clients.xls"
target_column = "default payment next month"

# Read the data
df = pd.read_excel(data_path, header=1)  # Skip first row (metadata)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()
```

### Preprocessing Pipeline
- Winsorize numeric attributes, encode categoricals, stratify train/test, normalize with MinMax, optionally rebalance via SMOTE, and flatten targets for sklearn compatibility.

```195:240:data_preprocessing.py
def full_pipeline(self, filepath, target_col, apply_smote=True, winsorize=True):
    """Execute full preprocessing pipeline"""
    ...
    df = self.load_data(filepath)
    self.check_missing_values(df)
    df = self.remove_missing_critical(df)
    cat_cols, num_cols = self.identify_column_types(df, target_col)
    if winsorize and len(num_cols) > 0:
        df = self.winsorize_outliers(df, num_cols)
    df = self.encode_categorical(df, cat_cols)
    X_train, X_test, y_train, y_test = self.split_data(df, target_col)
    X_train, X_test = self.normalize_features(X_train, X_test)
    X_train = pd.DataFrame(X_train, columns=self.feature_names)
    X_test = pd.DataFrame(X_test, columns=self.feature_names)
    if apply_smote:
        X_train, y_train = self.apply_smote(X_train, y_train)
    y_train = self._ensure_1d_target(y_train)
    y_test = self._ensure_1d_target(y_test)
    ...
    return X_train, X_test, y_train, y_test, self.feature_names
```

### Feature Selection for ANFIS
- Ensemble voting across RFE, mutual information, and correlation ensures the six most repeatedly important predictors feed ANFIS.

```121:157:feature_selection.py
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
        ...
        all_selected.extend(features)
    feature_counts = Counter(all_selected)
    most_common = feature_counts.most_common(self.n_features)
    self.selected_features = [f[0] for f in most_common]
    ...
    return self.selected_features, self.scores
```

### Model Training
- Random Forest and SVM leverage F1-focused hyperparameter searches; ANFIS wraps the PyTorch network to standardize training and evaluation.

```29:140:models.py
def train_random_forest(...):
    rf = RandomForestClassifier(...)
    if search_type == 'grid':
        search = GridSearchCV(...)
    else:
        search = RandomizedSearchCV(...)
    search.fit(X_train, y_train)
    self.models['random_forest'] = search.best_estimator_
...
def train_svm(...):
    svm = SVC(..., probability=True)
    ...
def train_anfis(...):
    anfis = ANFISClassifier(
        n_rules=config.ANFIS_CONFIG.get('n_rules', 10),
        max_epochs=config.ANFIS_CONFIG['max_epochs'],
        learning_rate=config.ANFIS_CONFIG['learning_rate'],
        batch_size=config.ANFIS_CONFIG.get('batch_size', 32)
    )
    anfis.fit(X_train, y_train)
    self.models['anfis'] = anfis
```

## Model Outcomes

- Random Forest achieves the best F1/ROC balance, SVM is close, and ANFIS maximizes specificity while sacrificing recall despite slightly higher accuracy.

```1:4:results/model_comparison.csv
Model,Accuracy,Precision,Recall,F1-Score,AUC-ROC,Specificity
Random Forest,0.7877,0.518,0.5742,0.5447,0.7763,0.8483
SVM,0.7738,0.4902,0.5674,0.526,0.756,0.8324
ANFIS,0.8177,0.6594,0.3632,0.4684,0.7391,0.9467
```

- Confusion matrices confirm RF/SVM capture more positives, while ANFIS yields fewer false positives.

```1003:1077:credit_risk_analysis.ipynb
Metrics for Random Forest:
  Accuracy:  0.7877
  Precision: 0.5180
  Recall (Sensitivity): 0.5742
  Specificity: 0.8483
  F1-Score:  0.5447
  AUC-ROC:   0.7763

Confusion Matrix:
[[3964  709]
 [ 565  762]]
...
Metrics for ANFIS:
  Accuracy:  0.8177
  Precision: 0.6594
  Recall (Sensitivity): 0.3632
  Specificity: 0.9467
  F1-Score:  0.4684
  AUC-ROC:   0.7391

Confusion Matrix:
[[4424  249]
 [ 845  482]]
```

Artifacts `results/confusion_matrices.png`, `results/roc_curves.png`, and `results/metrics_comparison.png` visualize these differences.

## Validation Evidence

- Ten-fold CV confirms Random Forest’s tighter dispersion (μ ≈ 0.546, σ ≈ 0.016) versus SVM (μ ≈ 0.535, σ ≈ 0.025).

```949:981:credit_risk_analysis.ipynb
print("Cross-validating Random Forest...")
cv_scores['Random Forest'] = cross_val_score(
    rf_model, X_train, y_train,
    cv=config.CV_FOLDS_FINAL,
    scoring='f1',
    n_jobs=-1
)
print(f"Mean F1: {cv_scores['Random Forest'].mean():.4f} (+/- {cv_scores['Random Forest'].std():.4f})")

print("\nCross-validating SVM...")
cv_scores['SVM'] = cross_val_score(
    svm_model, X_train, y_train,
    cv=config.CV_FOLDS_FINAL,
    scoring='f1',
    n_jobs=-1
)
print(f"Mean F1: {cv_scores['SVM'].mean():.4f} (+/- {cv_scores['SVM'].std():.4f})")
```

- Wilcoxon signed-rank testing (p = 0.0195) indicates the RF vs SVM gap is statistically significant at α = 0.05.

```1:2:results/statistical_significance.csv
Model 1,Model 2,Mean Diff,Test,p-value,Significant (α=0.05)
Random Forest,SVM,0.01153383947441644,Wilcoxon,0.01953125,Yes
```

## ANFIS Behavior and Interpretability

- The ANFIS network fuzzifies each of the six selected inputs with Gaussian membership functions, computes rule firing strengths, normalizes them, and aggregates Takagi–Sugeno consequents—providing an interpretable rule base once exported.

```35:107:models/anfis_network.py
class ANFISNetwork(nn.Module):
    """
    Takagi-Sugeno ANFIS Network
    """
    def __init__(self, n_inputs, n_rules):
        ...
        self.fuzzification = GaussianMF(n_inputs, n_rules)
        self.consequent_weights = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.consequent_bias = nn.Parameter(torch.randn(n_rules))
    def forward(self, x):
        memberships = self.fuzzification(x)
        firing_strengths = torch.prod(memberships, dim=1)
        normalized_weights = firing_strengths / (firing_strengths.sum(dim=1, keepdim=True) + 1e-10)
        rule_outputs = (x.unsqueeze(1) * self.consequent_weights).sum(dim=2) + self.consequent_bias
        output = (normalized_weights * rule_outputs).sum(dim=1)
        return output, normalized_weights
```

- Trained on the reduced feature space, ANFIS becomes a conservative screener: high accuracy/specificity, but recall of 0.36, meaning it excels at avoiding false alarms while missing more defaulters than RF/SVM.

```1061:1077:credit_risk_analysis.ipynb
Metrics for ANFIS:
  Accuracy:  0.8177
  Precision: 0.6594
  Recall (Sensitivity): 0.3632
  Specificity: 0.9467
  F1-Score:  0.4684
  AUC-ROC:   0.7391

Confusion Matrix:
[[4424  249]
 [ 845  482]]
```

**Next steps:** export the tuned Random Forest for deployment, extend ANFIS rule extraction for stakeholder review, and broaden the statistical testing suite once ANFIS cross-validation runs are available.



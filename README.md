# Credit Risk Prediction Pipeline: ANFIS vs Baseline Models

This project implements a complete machine learning pipeline for credit risk prediction, comparing ANFIS (Adaptive Neuro-Fuzzy Inference System) against classical supervised learning baselines (Random Forest and SVM).

## 📋 Project Structure

```
.
├── config.py                  # Configuration parameters
├── data_preprocessing.py      # Data loading and preprocessing
├── feature_selection.py       # Feature selection for ANFIS
├── models.py                  # Model definitions and training
├── evaluation.py              # Evaluation metrics and visualization
├── main_pipeline.py           # Main orchestration script
├── credit_student_pipeline.py # Student/learning version
├── methodology_section.md     # Detailed methodology documentation
├── requirement.text           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

Install required packages:

```bash
pip install -r requirement.text
```

Or install individually:

```bash
pip install pandas numpy scikit-learn imbalanced-learn scikit-fuzzy matplotlib seaborn openpyxl xlrd
```

### Dataset

Download the "Default of Credit Card Clients" dataset from UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

Place the dataset in a `data/` directory or specify the path when running.

### Running the Pipeline

```bash
# With default settings
python main_pipeline.py path/to/your/dataset.csv

# Or modify config.py and run
python main_pipeline.py
```

## 📊 Pipeline Overview

### 1. Data Preprocessing (`data_preprocessing.py`)

- Missing value detection and handling
- Outlier detection using z-scores
- Winsorization of extreme values
- One-hot encoding of categorical variables
- MinMax normalization
- Train/test split with stratification
- SMOTE for class imbalance

### 2. Feature Selection (`feature_selection.py`)

- **Purpose**: Reduce dimensionality for ANFIS (curse of dimensionality)
- **Methods**:
  - Recursive Feature Elimination (RFE)
  - Mutual Information
  - Correlation Analysis
  - Univariate Statistical Tests (ANOVA F-value)
  - Ensemble Selection (combines multiple methods)
- **Output**: Top 5-10 most important features

### 3. Model Training (`models.py`)

#### Random Forest (Baseline)

- Hyperparameter optimization via RandomizedSearchCV
- Parameters: n_estimators, max_depth, min_samples_split, criterion
- Class weight balancing

#### Support Vector Machine

- RBF kernel for non-linear decision boundaries
- Hyperparameters: C, gamma
- Optimized for imbalanced data

#### ANFIS (Main Model)

- Takagi-Sugeno fuzzy inference
- Gaussian membership functions
- Hybrid learning (LSE + gradient descent)
- Subtractive clustering for rule generation
- **Note**: Requires custom implementation or external library

### 4. Evaluation (`evaluation.py`)

#### Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall/Sensitivity**: True positive rate (critical for credit risk)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Specificity**: True negative rate
- **Confusion Matrix**: Detailed error analysis

#### Statistical Significance Testing

- Wilcoxon signed-rank test
- Paired t-test
- Validates if performance differences are statistically significant

#### Visualizations

- Confusion matrices for all models
- ROC curves comparison
- Metrics bar chart comparison

### 5. Interpretability Analysis

- Extraction and analysis of ANFIS fuzzy rules
- Validation of rule coherence with financial domain knowledge
- Comparison of "white box" (ANFIS) vs "black box" (SVM) interpretability

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Random seed for reproducibility
RANDOM_SEED = 42

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.8

# Cross-validation folds
CV_FOLDS = 5
CV_FOLDS_FINAL = 10

# Feature selection for ANFIS
N_FEATURES_ANFIS = 10

# SMOTE configuration
USE_SMOTE = True

# Model hyperparameters
RANDOM_FOREST_PARAMS = {...}
SVM_PARAMS = {...}
ANFIS_CONFIG = {...}
```

## 📈 Results

All results are saved in the `results/` directory:

- `model_comparison.csv`: Performance metrics for all models
- `statistical_significance.csv`: Statistical test results
- `confusion_matrices.png`: Visual comparison of confusion matrices
- `roc_curves.png`: ROC curves for all models
- `metrics_comparison.png`: Bar chart of key metrics

## 🔬 Methodology Highlights

1. **Class Imbalance Handling**: SMOTE oversampling + class weights
2. **Feature Selection**: Ensemble method to select most relevant features for ANFIS
3. **Rigorous Evaluation**: Multiple metrics + statistical significance testing
4. **Reproducibility**: Fixed random seeds, versioned code, documented parameters
5. **Interpretability**: Fuzzy rule extraction and analysis

## 📝 Key Findings (Template)

After running the pipeline, document:

1. Which model achieved the best F1-score?
2. Is the difference statistically significant?
3. What is the recall for the minority class (defaulters)?
4. Are the ANFIS fuzzy rules interpretable and business-logical?

## 🔧 ANFIS Implementation Notes

The current `models.py` contains a placeholder for ANFIS. To fully implement:

### Option 1: Use existing library

```bash
pip install anfis
```

### Option 2: Implement custom Takagi-Sugeno ANFIS

Required components:

- Layer 1: Fuzzification (membership functions)
- Layer 2: Rule firing strengths
- Layer 3: Normalization
- Layer 4: Consequent parameters (linear functions)
- Layer 5: Defuzzification (weighted sum)

### Option 3: Use scikit-fuzzy

```python
import skfuzzy as fuzz
# Implement control system with fuzzy rules
```

## 📚 References

- UCI ML Repository: Default of Credit Card Clients Dataset
- Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference System
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique

## 🎓 Academic Context

This implementation follows the methodology described in `methodology_section.md` and is designed for a master's thesis project evaluating ANFIS for credit risk prediction.

### Strengths of this approach:

✅ Complete data preprocessing pipeline
✅ Addresses class imbalance
✅ Feature selection for ANFIS dimensionality
✅ Statistical significance testing
✅ Interpretability analysis
✅ Reproducible and well-documented

## 👤 Author

Master's Project - Credit Risk Prediction with ANFIS

## 📄 License

Academic use only

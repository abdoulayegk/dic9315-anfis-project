# Credit Risk Prediction Pipeline: ANFIS vs Baseline Models

This project implements a complete machine learning pipeline for credit risk prediction, comparing ANFIS (Adaptive Neuro-Fuzzy Inference System) against classical supervised learning baselines (Random Forest and SVM).

## Project Structure

```
.
├── src/                       # Source code directory...
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration parameters
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── feature_selection.py   # Feature selection for ANFIS
│   ├── models.py             # Model definitions and training
│   ├── evaluation.py         # Evaluation metrics and visualization
│   ├── explainability.py     # SHAP-based explainability analysis
│   ├── anfis_network.py      # ANFIS neural network implementation
│   ├── main_pipeline.py      # Main orchestration script
│   ├── credit_student_pipeline.py  # Student/learning version
│   ├── test_pipeline.py      # Pipeline tests
│   └── test_anfis.py         # ANFIS-specific tests
├── models/                    # Trained model storage
├── plots/                     # Generated visualizations
├── results/                   # Results and metrics
├── reports/                   # Analysis reports
├── docs/                      # Sphinx source (API reference)
├── methodology_section.md     # Detailed methodology documentation
├── requirements.txt          # Python dependencies
└── README.md                  # This file
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup with uv

```bash
# Install dependencies and dev tools (lint, format, type-check, pre-commit)
uv sync --extra dev

# Optional: install pre-commit hooks so checks run before each commit
pre-commit install
```

### Verify setup

```bash
# Lint and format check
ruff check src/
ruff format --check src/

# Type checking
mypy src/

# Security scan
bandit -r src/ -c pyproject.toml

# Run all checks (lint, format, mypy, bandit)
pre-commit run --all-files
```

### Documentation (Sphinx)

```bash
uv sync --extra dev
uv run make -C docs html
```

Then open `docs/build/html/index.html`. Pushes to `main` or `develop` deploy the built site to GitHub Pages. See [DEVELOPMENT.md](DEVELOPMENT.md) for details.

### Install with pip (alternative)

```bash
pip install -r requirements.txt
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
# From the root directory
python src/main_pipeline.py path/to/your/dataset.csv

# Or run tests to verify everything works
python src/test_pipeline.py

# Or run ANFIS-specific tests
python src/test_anfis.py

# Or modify src/config.py and run
python src/main_pipeline.py
```

### Running Automated Tests

```bash
# Full suite: unit + integration + e2e, with coverage and reports
pytest

# E2E smoke tests only
pytest tests/e2e -k smoke
```

### Bonus: Mutation Testing

```bash
mutmut run --paths-to-mutate src --tests-dir tests
mutmut results
```

### Running with Docker

#### Docker Images

This project provides optimized Docker images:

**Lightweight CPU-only Image** (`Dockerfile.ci`) — **Recommended**

- Size: ~1.85GB (optimized, no CUDA dependencies)
- Includes: Jupyter Lab, pytest, all analysis tools
- Default: Runs Jupyter Lab
- Use for: Development, testing, CI/CD, documentation
- Published to: `ghcr.io/your-repo/projet-equipe3:main` (GitHub Container Registry)

#### Run from GitHub Container Registry (Recommended)

**Quick Start — Run Jupyter Lab**

```bash
# Pull and run in one command (auto-pulls if not present)
docker run -p 8888:8888 ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main

# Or use specific commit SHA for reproducibility
docker run -p 8888:8888 ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:b0c0325
```

**Detailed Usage**

```bash
# Pull the specific image (by commit SHA)
docker pull ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:b0c0325

# Or pull the latest from main branch
docker pull ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main

# Or pull from develop branch
docker pull ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:develop

# Run Jupyter Lab (default)
docker run -p 8888:8888 ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main
# Access at: http://localhost:8888 (token in logs)

# Run tests (override default command)
docker run ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main pytest -v --cov=src

# Run with volume mounting for local data
docker run -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main

# Run in detached mode (background)
docker run -d -p 8888:8888 \
  --name anfis-lab \
  ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main

# View logs and get Jupyter token
docker logs anfis-lab

# Stop the container
docker stop anfis-lab
```

**Available Tags**

| Tag                 | Branch           | Use Case                       |
| ------------------- | ---------------- | ------------------------------ |
| `main`              | main             | Latest stable release          |
| `develop`           | develop          | Latest development version     |
| `b0c0325` (SHA)     | Any              | Specific commit (reproducible) |
| `v1.0.0` (released) | Release branches | Semantic version releases      |

**Example: Run with everything**

```bash
docker run -d -p 8888:8888 \
  --name anfis-lab \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/notebooks:/app/notebooks \
  ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main

# Get the Jupyter token
docker logs anfis-lab | grep token

# Access Jupyter at: http://localhost:8888
```

#### Build Locally

```bash
# Build lightweight CPU image
docker build -f Dockerfile.ci -t anfis:latest .

# Run Jupyter Lab (default)
docker run -p 8888:8888 anfis:latest

# Run tests
docker run anfis:latest pytest -v --cov=src

# Run tests with coverage reports mounted
docker run -v $(pwd)/reports:/app/reports anfis:latest pytest -v --cov=src
```

#### Access Jupyter Lab

Once the container is running, Jupyter Lab will be available at:

```
http://localhost:8888
```

The container will output a token-based URL in the logs. Use it to access the lab environment.

#### Stop the container

```bash
# Press Ctrl+C if running in foreground mode

# Or if running in detached mode
docker stop <container-id>
```

#### Run in Detached Mode

```bash
# Run Jupyter Lab in background
docker run -d -p 8888:8888 --name anfis-lab anfis:latest

# View logs and get the Jupyter token
docker logs anfis-lab

# Stop the container
docker stop anfis-lab
```

### Docker Optimization Strategy

**Size Comparison:**

- Full GPU image (original): 8.59GB (includes CUDA libraries)
- Lightweight CPU image: ~1.85GB (CPU-only, optimized)
- **Savings: 78% reduction** ✅

**Image Registry:**

- Published to: `ghcr.io/hayou-azizkd/mgl7760_projet_equipe3`
- Automatically pushed on every commit to main/develop/release/hotfix branches
- No authentication required (public image)

**What we optimized:**

1. **Multi-stage Docker build**: Removes build dependencies from final image
2. **CPU-only PyTorch**: Uses `torch` CPU wheels instead of CUDA
3. **Stripped unnecessary packages**: Just essentials for analysis and testing
4. **Smart .dockerignore**: Excludes tests, docs, git history from image

**Benefits:**

- ✅ Faster builds (60-70% quicker)
- ✅ Faster pulls from registry
- ✅ Lower storage costs (78% less)
- ✅ Suitable for CI/CD pipelines
- ✅ Still includes Jupyter Lab + full analysis stack

**CI/CD Integration:**

- Automatic builds on push to `main`, `develop`, `release/*`, `hotfix/*` branches
- Automatic push to GitHub Container Registry (GHCR)
- Tagged with branch name, commit SHA, and semantic versions
- GitHub Actions layer caching for fast rebuilds

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/ci.yml`):

**Stages:**

1. **Code Quality & Security** (runs natively)
   - Ruff linting and formatting
   - MyPy type checking
   - Bandit security scanning
   - Vulnerability audits

2. **Testing** (Docker-based, CPU-optimized)
   - Pytest with coverage reports
   - HTML and XML reports
   - Artifact uploads

3. **Mutation Testing** (Optional, on PRs)
   - Targeted mutation analysis
   - Code quality assessment

4. **SonarQube Analysis** (Optional, on PRs)
   - Code quality metrics
   - Security hotspots
   - Technical debt tracking

5. **Documentation** (on main/develop)
   - Sphinx builds
   - Auto-deploy to GitHub Pages

6. **Container Publishing**
   - Lightweight CPU image published to GHCR
   - Auto-tagged per branch/version
   - Layer caching enabled for speed

## Pipeline Overview

### 1. Data Preprocessing (`src/data_preprocessing.py`)

- Missing value detection and handling
- Outlier detection using z-scores
- Winsorization of extreme values
- One-hot encoding of categorical variables
- MinMax normalization
- Train/test split with stratification
- SMOTE for class imbalance

### 2. Feature Selection (`src/feature_selection.py`)

- **Purpose**: Reduce dimensionality for ANFIS (curse of dimensionality)
- **Methods**:
  - Recursive Feature Elimination (RFE)
  - Mutual Information
  - Correlation Analysis
  - Univariate Statistical Tests (ANOVA F-value)
  - Ensemble Selection (combines multiple methods)
- **Output**: Top 5-10 most important features

### 3. Model Training (`src/models.py`)

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

### 4. Evaluation (`src/evaluation.py`)

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

## Configuration

Edit `src/config.py` to customize:

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

## Results

All results are saved in the `results/` directory:

- `model_comparison.csv`: Performance metrics for all models
- `statistical_significance.csv`: Statistical test results
- `confusion_matrices.png`: Visual comparison of confusion matrices
- `roc_curves.png`: ROC curves for all models
- `metrics_comparison.png`: Bar chart of key metrics

## Methodology Highlights

1. **Class Imbalance Handling**: SMOTE oversampling + class weights
2. **Feature Selection**: Ensemble method to select most relevant features for ANFIS
3. **Rigorous Evaluation**: Multiple metrics + statistical significance testing
4. **Reproducibility**: Fixed random seeds, versioned code, documented parameters
5. **Interpretability**: Fuzzy rule extraction and analysis

## Key Findings (Template)

After running the pipeline, document:

1. Which model achieved the best F1-score?
2. Is the difference statistically significant?
3. What is the recall for the minority class (defaulters)?
4. Are the ANFIS fuzzy rules interpretable and business-logical?

## ANFIS Implementation Notes

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

## References

- UCI ML Repository: Default of Credit Card Clients Dataset
- Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference System
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique

## Academic Context

This implementation follows the methodology described in `methodology_section.md` and is designed for a master's thesis project evaluating ANFIS for credit risk prediction.

### Strengths of this approach:

- Complete data preprocessing pipeline
- Addresses class imbalance
- Feature selection for ANFIS dimensionality
- Statistical significance testing
- Interpretability analysis
- Reproducible and well-documented

## Author

Master's Project - Credit Risk Prediction with ANFIS

## License

Academic use only

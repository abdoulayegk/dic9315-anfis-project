# Prédiction du risque de crédit : ANFIS vs modèles de référence

Projet de maîtrise comparant ANFIS (Adaptive Neuro-Fuzzy Inference System) à Random Forest et SVM sur le dataset UCI _Default of Credit Card Clients_.

## Structure

```
src/
├── config.py               # Paramètres globaux
├── data_preprocessing.py   # Chargement et prétraitement
├── feature_selection.py    # Sélection de variables pour ANFIS
├── anfis_network.py        # Réseau ANFIS (PyTorch)
├── models.py               # Entraînement des modèles
├── evaluation.py           # Métriques et visualisations
├── explainability.py       # Analyse SHAP
└── main_pipeline.py        # Point d'entrée principal
tests/
├── unit/                   # Tests unitaires par module
├── integration/            # Tests d'intégration du pipeline
└── e2e/                    # Tests de bout en bout (smoke)
```

## Installation

Avec `uv` (recommandé) :

```bash
# Installe toutes les dépendances y compris les outils de développement (mypy, pre-commit, bandit, ruff, etc.)
uv sync --extra dev

# Configure les hooks de qualité avant chaque commit
pre-commit install
pre-commit run --all-files  # Vérifier que tout fonctionne
```

Avec `pip` :

```bash
python3 -m venv .venv && source .venv/bin/activate

# Installe les dépendances de base + dev
pip install -r requirements.txt
pip install -e ".[dev]"

# Configure les hooks de qualité avant chaque commit
pre-commit install
pre-commit run --all-files  # Vérifier que tout fonctionne
```

## Utilisation

```bash
# Lancer le pipeline complet
python src/main_pipeline.py data/default_credit_card_clients.csv

# Tests automatisés (unitaires + intégration + e2e)
pytest

# Tests e2e seulement
pytest tests/e2e -k smoke

# Tests de mutation (optionnel)
mutmut run --paths-to-mutate src --tests-dir tests
mutmut results
```

## Qualité du code

Les vérifications de qualité s'exécutent **automatiquement** via `pre-commit` avant chaque commit :

```bash
# Affiche l'état sans modifier
pre-commit run --all-files

# Forcé à s'exécuter (même sur fichiers non modifiés)
pre-commit run --all-files --verbose
```

Ou manuellement :

```bash
ruff check src/          # Lint (E, F, I, N, W, UP, B, C4)
ruff format --check src/ # Formatage (remplace Black)
mypy src/ --ignore-missing-imports  # Type checking
bandit -r src/ -c pyproject.toml    # Sécurité
```

**Outils pré-commit installés :**

- **Ruff** : lint + formatage rapide
- **MyPy** : vérification de types
- **Bandit** : audit de sécurité

## Documentation (Sphinx)

```bash
uv run make -C docs html
# Ouvrir docs/build/html/index.html
```

Les pushes sur `main` ou `develop` déploient automatiquement la doc sur GitHub Pages.

## Docker

Build local :

```bash
docker build -f Dockerfile.ci -t anfis:latest .
docker run -p 8888:8888 anfis:latest              # Jupyter Lab
docker run anfis:latest pytest -v --cov=src       # Tests
```

Image publiée sur GHCR à chaque push :

```bash
docker run -p 8888:8888 ghcr.io/hayou-azizkd/mgl7760_projet_equipe3:main
# Accès : http://localhost:8888 (token dans les logs)
```

Tags disponibles : `main`, `develop`, SHA de commit (ex. `b0c0325`).

## CI/CD (GitHub Actions)

Le workflow `.github/workflows/ci.yml` enchaîne :

1. **Qualité & sécurité** — Ruff, MyPy, Bandit, `uv audit`
2. **Tests** — pytest avec rapport de couverture (artefact `coverage.xml`)
3. **Tests de mutation** — mutmut sur `data_preprocessing.py` et `feature_selection.py` (sur PR et branches principales)
4. **SonarQube** — analyse de qualité et dette technique (sur PR et branches principales)
5. **Documentation** — build Sphinx + déploiement GitHub Pages
6. **Publication du package** — `uv build`, artefact `python-package`
7. **Publication de l'image** — push vers GHCR, tag par branche/SHA/version sémantique

## Pipeline de traitement

1. **Prétraitement** (`data_preprocessing.py`) : détection de valeurs manquantes, winsorisation des extrêmes, encodage one-hot, normalisation MinMax, split stratifié, SMOTE
2. **Sélection de variables** (`feature_selection.py`) : méthode d'ensemble — RFE, information mutuelle, corrélation, ANOVA F — retient les 10 variables les plus importantes pour réduire la dimensionnalité ANFIS
3. **Entraînement** (`models.py`) : Random Forest (RandomizedSearchCV), SVM (noyau RBF), ANFIS Takagi-Sugeno (PyTorch, fonctions d'appartenance gaussiennes)
4. **Évaluation** (`evaluation.py`) : accuracy, précision, rappel, F1, AUC-ROC, spécificité, matrices de confusion, test de Wilcoxon et t-test apparié
5. **Visualisations** : matrices de confusion, courbes ROC, graphique comparatif des métriques (enregistrés dans `results/`)
6. **Explicabilité** (`explainability.py`) : SHAP pour Random Forest et SVM, extraction de règles floues pour ANFIS
7. **Résumé** : résultats exportés dans `results/model_comparison.csv` et `results/statistical_significance.csv`

## Configuration

Les principaux paramètres sont dans `src/config.py` : `RANDOM_SEED`, `TRAIN_TEST_SPLIT`, `CV_FOLDS`, `N_FEATURES_ANFIS`, `USE_SMOTE`, espaces de recherche hyperparam pour RF/SVM/ANFIS.

## Données

Dataset UCI — _Default of Credit Card Clients_ :
https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

Placer le fichier dans `data/` ou passer le chemin en argument.

## Références

- Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference System
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique

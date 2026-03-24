# Development Setup Guide

This document explains how to set up the development environment with code quality tools.

## Prerequisites

- Python 3.12 installed
- Git repository cloned
- `uv` package manager (recommended) or `pip`

## Installation

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync --extra dev

# OR using pip
pip install -e ".[dev]"
```

### 2. Set up pre-commit hooks

```bash
pre-commit install
```

This will run quality checks automatically before each commit.

## Code Quality Tools

We use several tools to maintain code quality:

### Ruff - Fast Python linter and formatter (replaces Black)
```bash
ruff check src          # Check for linting issues
ruff check src --fix    # Fix auto-fixable linting issues
ruff format src         # Format code (replaces black)
ruff format --check src # Check if code is formatted
```

### MyPy - Type checking
```bash
mypy src                # Check types
```

### Bandit - Security linter
```bash
bandit -r src -c pyproject.toml  # Check for security issues
```

### pip-audit - Vulnerability scanner
```bash
pip-audit -r requirements.txt    # Check dependencies for vulnerabilities
```

## Running all checks

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Or run individual tools:
ruff check src --fix    # Lint and auto-fix
ruff format src         # Format code
mypy src               # Type check
bandit -r src -c pyproject.toml  # Security check
pip-audit -r requirements.txt    # Vulnerability check
```

## Test Suite and Reports

Run the full automated suite (unit + integration + e2e) with coverage and reports:

```bash
pytest
```

Current enforced coverage gate: `--cov-fail-under=52`.

Generated report artifacts:

- `reports/junit.xml` (JUnit XML)
- `reports/pytest_report.html` (HTML test report)
- `reports/coverage.xml` (coverage XML)
- `reports/coverage_html/index.html` (coverage HTML)

Run only end-to-end smoke tests:

```bash
pytest tests/e2e -k smoke
```

## Mutation Testing (Bonus)

Install dev dependencies first, then run mutation tests:

```bash
mutmut run --paths-to-mutate src --tests-dir tests
mutmut results
```

In CI, mutation testing runs on pull requests as an optional job.

Inspect one mutant in detail:

```bash
mutmut show <mutant_id>
```

## Sphinx documentation (API)

API documentation is generated with [Sphinx](https://www.sphinx-doc.org/) and `sphinx.ext.autodoc` from docstrings in `src/`.

```bash
uv sync --extra dev
uv run make -C docs html
```

Open `docs/build/html/index.html` in a browser. The `docs/build/` directory is generated and should not be committed.

On push to `main` or `develop`, GitHub Actions builds the same output and publishes it to GitHub Pages (branch `gh-pages`). Configure the site under **Settings → Pages** if needed (source: branch `gh-pages`, folder `/`).

## CI/CD

The GitHub Actions workflow will automatically run all these checks on:
- Every push to main/develop branches
- Every pull request

## Troubleshooting

### Pre-commit not running
Make sure you ran `pre-commit install` after cloning.

### Formatting issues
Ruff handles both linting and formatting. Run `ruff format src` to auto-format your code.

### Type errors from MyPy
MyPy errors won't fail the CI (set to continue-on-error) so you can learn gradually.
Add type hints to fix warnings:
```python
def my_function(x: int) -> str:
    return str(x)
```

### Security warnings from Bandit
Review the warnings - they might indicate real security issues.
Use `# nosec` comment to ignore false positives (with good reason).

### Dependency vulnerabilities
Update vulnerable packages when possible, or document why they can't be updated.
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
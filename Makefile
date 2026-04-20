.PHONY: fix check test

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/ --ignore-missing-imports
	uv run bandit -r src/ -c pyproject.toml

test:
	uv run pytest

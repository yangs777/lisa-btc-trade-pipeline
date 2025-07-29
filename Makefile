.PHONY: help install dev-install test lint format type-check security-check ci clean

help:
	@echo "Available commands:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev-install   Install development dependencies"
	@echo "  make test          Run tests with coverage"
	@echo "  make lint          Run linting (ruff)"
	@echo "  make format        Format code (black + isort)"
	@echo "  make type-check    Run type checking (mypy)"
	@echo "  make security-check Run security checks (bandit + safety)"
	@echo "  make ci            Run all CI checks"
	@echo "  make clean         Clean up generated files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev-install: install
	pip install pre-commit
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/ tests/

security-check:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

ci: lint type-check test security-check
	@echo "All CI checks passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
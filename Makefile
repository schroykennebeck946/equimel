PYTHON ?= python
PKG := src/equimel

.DEFAULT_GOAL := help

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "Targets:"
	@echo "  install   install the package and runtime dependencies"
	@echo "  dev       install with development extras"
	@echo "  lint      run ruff, black --check, isort --check"
	@echo "  type      run mypy --strict on the package"
	@echo "  test      run the full pytest suite"
	@echo "  smoke     run two federated micro-rounds on the smoke config"
	@echo "  docker    build the container image"
	@echo "  clean     remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	$(PYTHON) -m mypy --config-file pyproject.toml

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m equimel.fieldbook.chart --config configs/experiment/_smoke.gin

docker:
	docker build -t equimel:latest .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

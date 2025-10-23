.PHONY: help install install-dev test test-verbose coverage lint format clean build docs run-webcam run-tello

help:
	@echo "Available commands:"
	@echo "  make install       - Install package and dependencies"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make lint          - Run linters (flake8, pylint)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make build         - Build distribution packages"
	@echo "  make docs          - Generate documentation"
	@echo "  make run-webcam    - Run webcam demo"
	@echo "  make run-tello     - Run Tello demo"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:
	python -m unittest discover tests/

test-verbose:
	python -m unittest discover tests/ -v

coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=100 --exclude=venv,build,dist
	@echo "Running pylint..."
	pylint src/ --max-line-length=100 --disable=C0111,R0913,R0914

format:
	@echo "Running black..."
	black src/ tests/ --line-length=100
	@echo "Running isort..."
	isort src/ tests/

type-check:
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "Clean complete!"

build:
	python -m build

docs:
	@echo "Documentation is in README.md and other .md files"
	@echo "To generate Sphinx docs (if configured):"
	@echo "  cd docs && make html"

run-webcam:
	python webcam_demo.py

run-tello:
	@echo "Make sure you're connected to Tello WiFi!"
	python tello_demo.py

# Development helpers
setup-venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate    # Windows"

check-deps:
	pip list --outdated

update-deps:
	pip install --upgrade -r requirements.txt

# CI/CD helpers
ci-test:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pytest --cov=src --cov-report=xml
	flake8 src/ tests/ --max-line-length=100

# Git helpers
sync:
	git fetch upstream
	git checkout main
	git merge upstream/main

new-branch:
	@read -p "Enter branch name: " branch; \
	git checkout -b $$branch

# Docker (if needed in future)
docker-build:
	docker build -t tello-collision-avoidance .

docker-run:
	docker run -it --rm tello-collision-avoidance

# Install dependencies using uv
install:
	pip install uv
	uv sync

# Run tests with coverage
test:
	uv run python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli

# Format code using Black
format:
	uv run black mylib/ api/ cli/ *.py

# Lint code using Pylint
lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*?py mylib/*.py api/*.py cli/*.py

# --- Run the API locally ---
run:
	uv run uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload

# --- Build the Docker Image ---
build:
	docker build -t mlops-lab2 .

# Clean up cache files
clean:
	rm -rf __pycache__ .pytest_cache .coverage
	find . -name "*.pyc" -delete

# Do everything" command
all: install format lint test
.PHONY: setup test lint fmt doctor clean

setup:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

doctor:
	uv run diagsys doctor

clean:
	rm -rf .pytest_cache .ruff_cache build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

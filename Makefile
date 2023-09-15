PACKAGE = partition-0.1.0

.PHONY: format
format:
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .
	poetry run isort . && poetry run black .

.PHONY: test
test:
	poetry run python -m pytest

.PHONY: lint
lint:
	poetry run flake8
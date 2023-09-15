PACKAGE = sample_lib-0.1.0

.PHONY: build
build:
	poetry build
	tar zxvf dist/$(PACKAGE).tar.gz -C ./dist
	cp dist/$(PACKAGE)/setup.py setup.py
	rm -rf dist

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
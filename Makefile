
.DEFAULT_GOAL=lint

SHELL := /bin/bash
SOURCE_FOLDERS=penelope tests
PENELOPE_VERSION:=v$(shell grep "version \= " pyproject.toml | sed "s/version = //" | sed "s/\"//g")

init: tools
	@pip install --upgrade pip
	@pip install poetry --upgrade
	@poetry install

tools:
	@pip install --upgrade pip
	@pip install poetry --upgrade

build: tools requirements.txt
	@poetry build

tag:
	@echo $(PENELOPE_VERSION)
	@git push
	@git tag $(PENELOPE_VERSION) -a
	@git push origin --tags

test-coverage:
	-poetry run coverage --rcfile=.coveragerc run -m pytest
	-poetry run coveralls

test: clean
	@mkdir -p ./tests/output
	@poetry run pytest --verbose --durations=0 \
		--cov=penelope \
		--cov-report=term \
		--cov-report=xml \
		--cov-report=html \
		tests
	@rm -rf ./tests/output/*

pylint:
	@poetry run pylint $(SOURCE_FOLDERS)
	# @poetry run mypy --version
	# @poetry run mypy .

pylint2:
	@-find $(SOURCE_FOLDERS) -type f -name "*.py" | grep -v .ipynb_checkpoints | xargs poetry run pylint --disable=W0511 | sort | uniq

flake8:
	@poetry run flake8 --version
	@poetry run flake8

lint: pylint flake8

format: clean black isort

isort:
	@poetry run isort --profile black --float-to-top --line-length 120 --py 38 penelope

yapf: clean
	@poetry run yapf --version
	@poetry run yapf --in-place --recursive penelope

black:clean
	@poetry run black --line-length 120 --target-version py38 --skip-string-normalization $(SOURCE_FOLDERS)

tidy: black isort

clean:
	@rm -rf .pytest_cache build dist .eggs *.egg-info
	@rm -rf .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	# @rm -rf tests/output

clean_cache:
	@poetry cache clear pypi --all

update:
	@poetry update

install_graphtool:
	@sudo echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt buster main" >> /etc/apt/sources.list
	@sudo apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
	@sudo apt update && apt install python3-graph-tool

requirements.txt: poetry.lock
	@poetry export -f requirements.txt --output requirements.txt

.PHONY: init lint flake8 pylint pylint2 format yapf black clean test test-coverage update install_graphtool build isort tidy tag tools
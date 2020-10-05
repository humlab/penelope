
.DEFAULT_GOAL=lint

# init:
# 	@pip install --upgrade pip
# ifeq (, $(PIPENV_PATH))
# 	@pip install poetry --upgrade
# endif
# 	@export PIPENV_TIMEOUT=7200
# 	@pipenv install --dev

test-coverage:
	-poetry run coverage --rcfile=.coveragerc run -m pytest
	-coveralls

build:
	@poetry build

test: clean
	@poetry run pytest --verbose --durations=0 \
		--cov=penelope \
		--cov-report=term \
		--cov-report=xml \
		--cov-report=html \
		tests

lint:
	@poetry run pylint penelope tests | sort | uniq | grep -v "************* Module" > pylint.log
	@poetry run flake8 --version
	@poetry run flake8
	# @poetry run mypy --version
	# @poetry run mypy .

format: clean black isort

isort:
	@poetry run isort penelope

yapf: clean
	@poetry run yapf --version
	@poetry run yapf --in-place --recursive penelope

black:clean
	@poetry run black --version
	@poetry run black --line-length 120 --target-version py38 --skip-string-normalization penelope tests

clean:
	@rm -rf .pytest_cache build dist .eggs *.egg-info
	@rm -rf .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@rm -rf tests/output

update:
	#@export PIPENV_VENV_IN_PROJECT=true
	#@export PIPENV_TIMEOUT=7200
	@poetry update

install_graphtool:
	@sudo echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt buster main" >> /etc/apt/sources.list
	@sudo apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
	@sudo apt update && apt install python3-graph-tool

requirements.txt: poetry.lock
	@poetry export -f requirements.txt --output requirements.txt

.PHONY: init lint format yapf black clean test test-coverage update install_graphtool build isort

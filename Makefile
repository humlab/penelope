.DEFAULT_GOAL=lint
SHELL := /bin/bash
SOURCE_FOLDERS=penelope tests

init: tools
	@poetry install

.ONESHELL: guard_clean_working_repository
guard_clean_working_repository:
	@status="$$(git status --porcelain)"
	@if [[ "$$status" != "" ]]; then
		echo "error: changes exists, please commit or stash them: "
		echo "$$status"
		exit 65
	fi

version:
	@echo $(shell grep "^version \= " pyproject.toml | sed "s/version = //" | sed "s/\"//g")

tools:
	@pip install --upgrade pip --quiet
	@pip install poetry --upgrade --quiet

build: tools requirements.txt
	@poetry build

release: guard_clean_working_repository build bump.patch tag

bump.patch:
	@poetry run dephell project bump patch
	@git add pyproject.toml
	@git commit -m "Bump version patch"
	@git push

tag:
	@git push
	@git tag $(shell grep "^version \= " pyproject.toml | sed "s/version = //" | sed "s/\"//g") -a
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

pytest:
	@mkdir -p ./tests/output
	@poetry run pytest --quiet tests

pylint:
	@time poetry run pylint $(SOURCE_FOLDERS)
	# @poetry run mypy --version
	# @poetry run mypy .

pylint2:
	@-find $(SOURCE_FOLDERS) -type f -name "*.py" | \
		grep -v .ipynb_checkpoints | \
			poetry run xargs -I @@ bash -c '{ echo "@@" ; pylint "@@" ; }'

	# xargs poetry run pylint --disable=W0511 | sort | uniq

flake8:
	@poetry run flake8 --version
	@poetry run flake8

lint: pylint flake8

format: clean black isort

isort:
	@poetry run isort --profile black --float-to-top --line-length 120 --py 38 $(SOURCE_FOLDERS)

yapf: clean
	@poetry run yapf --version
	@poetry run yapf --in-place --recursive $(SOURCE_FOLDERS)

black: clean
	@poetry run black --version
	@poetry run black --line-length 120 --target-version py38 --skip-string-normalization $(SOURCE_FOLDERS)

tidy: black isort

ready: tools format tidy test flake8 pylint

clean:
	@rm -rf .pytest_cache build dist .eggs *.egg-info
	@rm -rf .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@rm -rf tests/output

clean_cache:
	@poetry cache clear pypi --all

penelope_data: nltk_data

nltk_data:
	@mkdir -p $(NLTK_DATA)
	@poetry run python -m nltk.downloader -d $(NLTK_DATA) stopwords punkt sentiwordnet

spacy_data:
	@poetry run python -m spacy download en

gh:
	@sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
	@sudo apt-add-repository https://cli.github.com/packages
	@sudo apt update && sudo apt install gh

update:
	@poetry update

install_graphtool:
	@echo "source code repository: https://git.skewed.de/count0/graph-tool"
	@sudo echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt buster main" >> /etc/apt/sources.list
	@sudo apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
	@sudo apt update && apt install python3-graph-tool

requirements.txt: poetry.lock
	@poetry export -f requirements.txt --output requirements.txt

check-gh: gh-exists
gh-exists: ; @which gh > /dev/null

.PHONY: check gh-exists
.PHONY: init lint release flake8 pylint pytest pylint2 ready format yapf black clean test test-coverage \
	update install_graphtool gh build isort tidy tag tools bump.patch penelope_data nltk_data
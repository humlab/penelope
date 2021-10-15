.DEFAULT_GOAL=lint
SHELL := /bin/bash
SOURCE_FOLDERS=penelope tests
PACKAGE_FOLDER=penelope
SPACY_MODEL=en_core_web_sm

RUN_TIMESTAMP := $(shell /bin/date "+%Y-%m-%d-%H%M%S")

fast-release: clean tidy build guard_clean_working_repository bump.patch tag publish

release: ready guard_clean_working_repository bump.patch tag  publish

#gh release create v0.2.35 --title "INIDUN release" --notes "Update that facilitates INIDUN release"

.PHONY: watch
watch:
	@fswatch -1 -r --latency 1 ./tests/notebook/co_occurrence | xargs -0 -n1 -I{} pytest ./tests/notebook/co_occurrence

ready: tools clean tidy full-test lint build

build: requirements.txt
	@poetry build

publish:
	@poetry publish

lint: tidy pylint flake8

tidy: black isort

test: output-dir
	@echo SKIPPING LONG RUNNING TESTS!
	@poetry run pytest -m "not long_running" --durations=0 tests
	@rm -rf ./tests/output/*

pytest: output-dir
	@poetry run pytest -m "not long_running" --durations=0 tests

test-coverage: output-dir
	@echo SKIPPING LONG RUNNING TESTS!
	@poetry run pytest -m "not long_running" --cov=$(PACKAGE_FOLDER) --cov-report=html tests
	@rm -rf ./tests/output/*

full-test: output-dir
	@poetry run pytest tests
	@rm -rf ./tests/output/*

long-test: output-dir
	@poetry run pytest -m "long_running" --durations=0 tests
	@rm -rf ./tests/output/*

full-test-coverage: output-dir
	@mkdir -p ./tests/output
	@poetry run pytest --cov=$(PACKAGE_FOLDER) --cov-report=html tests
	@rm -rf ./tests/output/*

output-dir:
	@mkdir -p ./tests/output

retest:
	@poetry run pytest --durations=0 --last-failed tests

init: tools
	@poetry install

info:
	@poetry run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'

TYPINGS_PACKAGES=spacy IPython bokeh ftfy gensim holoviews ipycytoscape ipyfilechooser ipywidgets itertoolz networkx nltk numpy pydotplus scipy sklearn smart_open statsmodels textacy tqdm
.PHONY: typings
.ONESHELL: typings
typings:
	@for package in $(TYPINGS_PACKAGES); do \
		poetry run pyright --createstub $$package ; \
	done

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

bump.patch: requirements.txt
	@poetry version patch
	@git add pyproject.toml requirements.txt
	@git commit -m "Bump version patch"
	@git push

tag:
	@poetry build
	@git push
	@git tag $(shell grep "^version \= " pyproject.toml | sed "s/version = //" | sed "s/\"//g") -a
	@git push origin --tags

# test-coverage:
# 	-poetry run coverage --rcfile=.coveragerc run -m pytest
# 	-poetry run coveralls

pylint:
	@time poetry run pylint $(SOURCE_FOLDERS)
	# @poetry run mypy --version
	# @poetry run mypy .

pylint-source:
	@time poetry run pylint penelope


show-size:
	@pip list --format freeze | \
		awk -F = {'print $$1'} | \
			xargs pip show | \
				grep -E 'Location:|Name:' | \
					cut -d ' ' -f 2 | \
						paste -d ' ' - - | \
							awk '{gsub("-","_",$$1); print $$2 "/" tolower($$1)}' | \
								xargs du -sh 2> /dev/null | \
									sort -h

pylint_diff:
	@time poetry run pylint -j 2 `git diff --name-only --diff-filter=d | grep -E '\.py$' | tr '\n' ' '`

# https://nerderati.com/speed-up-pylint-by-reducing-the-files-it-examines/
# delta_files=`git diff --name-only --diff-filter=d | grep -E '\.py$' | tr '\n' ' '`
# delta_files=`git diff --name-only --staged --diff-filter=d | grep -E '\.py$' | tr '\n' ' '`
.ONESHELL: pylint_diff_only
pylint_diff_only:
	@delta_files=$$(git status --porcelain | awk '{print $$2}' | grep -E '\.py$$' | tr '\n' ' ')
	@if [[ "$$delta_files" != "" ]]; then
		time poetry run pylint -j 2 $$delta_files
	fi

pylint_by_file:
	@-find $(SOURCE_FOLDERS) -type f -name "*.py" | \
		grep -v .ipynb_checkpoints | \
			poetry run xargs -I @@ bash -c '{ echo "@@" ; pylint "@@" ; }'

	# xargs poetry run pylint --disable=W0511 | sort | uniq

flake8:
	@poetry run flake8 --version
	@poetry run flake8

isort:
	@poetry run isort --profile black --float-to-top --line-length 120 --py 38 $(SOURCE_FOLDERS)

yapf: clean
	@poetry run yapf --version
	@poetry run yapf --in-place --recursive $(SOURCE_FOLDERS)

black: clean
	@poetry run black --version
	@poetry run black --line-length 120 --target-version py38 --skip-string-normalization $(SOURCE_FOLDERS)

clean:
	@rm -rf .pytest_cache build dist .eggs *.egg-info
	@rm -rf .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@rm -rf tests/output

clean_cache:
	@poetry cache clear pypi --all

data: nltk_data spacy_data

update:
	@poetry update

nltk_data:
	@mkdir -p $(NLTK_DATA)
	@poetry run python -m nltk.downloader -d $(NLTK_DATA) stopwords punkt sentiwordnet

spacy_data:
	@poetry run python -m spacy download $(SPACY_MODEL)
	@poetry run python -m spacy link $(SPACY_MODEL) en --force

gh:
	@sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
	@sudo apt-add-repository https://cli.github.com/packages
	@sudo apt update && sudo apt install gh

install_graphtool:
	@echo "source code repository: https://git.skewed.de/count0/graph-tool"
	@sudo echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt buster main" >> /etc/apt/sources.list
	@sudo apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
	@sudo apt update && apt install python3-graph-tool

requirements.txt: poetry.lock
	@poetry export --without-hashes -f requirements.txt --output requirements.txt

check-gh: gh-exists
gh-exists: ; @which gh > /dev/null

profile-co_occurrence-pyinstrument:
	@mkdir -p ./profile-reports
	@poetry run python -m pyinstrument -r html -o ./profile-reports/$(RUN_TIMESTAMP)_workflow-pyinstrument.html ./tests/profiling/profile-workflow-pyinstrument.py

profile-co_occurrence-cprofile:
	@mkdir -p ./profile-reports
	@poetry run python ./tests/profiling/profile-workflow-cprofile.py &> ./profile-reports/$(RUN_TIMESTAMP)_workflow-pyinstrument.txt

profile-compute-keyness-pyinstrument:
	@mkdir -p ./profile-reports
	@poetry run python -m pyinstrument -r html -o ./profile-reports/$(RUN_TIMESTAMP)_keyness-pyinstrument.html ./tests/profiling/profile-compute-keyness.py

profile-compute-keyness-cprofile:
	@mkdir -p ./profile-reports
	@poetry run python -m cProfile ./tests/profiling/profile-compute-keyness.py &> ./profile-reports/$(RUN_TIMESTAMP)_workflow-cprofile.txt
	# @poetry run python -m cProfile -o ./profile-reports/$(RUN_TIMESTAMP)_keyness-cprofile.cprof ./tests/profiling/profile-compute-keyness.py
	# @poetry run pyprof2calltree -k -i ./profile-reports/$(RUN_TIMESTAMP)_keyness-cprofile.cprof

.PHONY: install-mkl install-gfortran install-mkl-basekit reinstall-numpy-scipy numpy-site-config
install-mkl: install-gfortran install-mkl-basekit reinstall-numpy-scipy numpy-site-config
	echo "installed: MKL (Math Kernel Library)"

TARGET_MKL_BASE_KIT := https://registrationcenter-download.intel.com/akdlm/irc_nas/17977/l_BaseKit_p_2021.3.0.3219.sh
install-mkl-basekit:
	wget -q -O /tmp/mkl-installer.sh $(TARGET_MKL_BASE_KIT)
	sudo bash /tmp/mkl-installer.sh

install-gfortran:
	sudo apt-get update
	sudo apt-get install gfortran

# Template https://github.com/numpy/numpy/blob/main/site.cfg.example
NUMPY_CONFIG := $(HOME)/.numpy-site.cfg
numpy-site-config: $(HOME)/.numpy-site.cfg
	echo "[mkl]" > $(NUMPY_CONFIG)
	"library_dirs = /opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64" >> $(NUMPY_CONFIG)
	"include_dirs = /opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/include" >> $(NUMPY_CONFIG)
	"libraries = mkl_rt" >> $(NUMPY_CONFIG)

reinstall-numpy-scipy:
	@pip install numpy scipy --no-binary numpy,scipy --force-reinstall

.PHONY: stubs
stubs:
	@stubgen penelope/corpus/dtm/vectorized_corpus.py --output ./typings

.PHONY: help check init version
.PHONY: lint flake8 pylint pylint_by_file yapf black isort tidy pylint_diff_only
.PHONY: test retest test-coverage pytest
.PHONY: ready build tag bump.patch release fast-release
.PHONY: clean clean_cache update
.PHONY: install_graphtool gh check-gh gh-exists tools
.PHONY: data spacy_data nltk_data
.PHONY: profile-co_occurrence

venus:
	# @tar czvf ./tmp/VENUS.$(RUN_TIMESTAMP).tar.gz ./tests/test_data/VENUS
	@poetry run python -c 'from tests.pipeline.fixtures import create_test_data_bundles; create_test_data_bundles()'

help:
	@echo "Higher level recepies: "
	@echo " make ready            Makes ready for release (tools tidy test flake8 pylint)"
	@echo " make build            Updates tools, requirement.txt and build dist/wheel"
	@echo " make release          Bumps version (patch), pushes to origin and creates a tag on origin"
	@echo " make fast-release     Same as release but without lint and test"
	@echo " make test             Runs tests with code coverage"
	@echo " make retest           Runs failed tests with code coverage"
	@echo " make lint             Runs pylint and flake8"
	@echo " make tidy             Runs black and isort"
	@echo " make clean            Removes temporary files, caches, build files"
	@echo " make data             Downloads NLTK and SpaCy data"
	@echo "  "
	@echo "Lower level recepies: "
	@echo " make init             Install development tools and dependencies (dev recepie)"
	@echo " make tag              bump.patch + creates a tag on origin"
	@echo " make bump.patch       Bumps version (patch), pushes to origin"
	@echo " make pytest           Runs teets without code coverage"
	@echo " make pylint           Runs pylint"
	@echo " make pylint_diff_only Runs pylint on changed files only (git status --porcelain)"
	@echo " make pytest2          Runs pylint on a per-file basis"
	@echo " make flake8           Runs flake8 (black, flake8-pytest-style, mccabe, naming, pycodestyle, pyflakes)"
	@echo " make isort            Runs isort"
	@echo " make yapf             Runs yapf"
	@echo " make black            Runs black"
	@echo " make gh               Installs Github CLI"
	@echo " make update           Updates dependencies"

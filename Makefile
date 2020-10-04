
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

test:
	@poetry run pytest -v --durations=0
	# --failed-first --maxfail=1

lint:
	-poetry run pylint penelope tests | sort | uniq | grep -v "************* Module" > pylint.log

format:
	poetry run yapf --in-place --recursive penelope

black:
	poetry run black --line-length 120 --target-version py38 --skip-string-normalization penelope

clean:
	@rm -rf .pytest_cache
	@find -name __pycache__ | xargs rm -r
	@rm -rf penelope/test/output

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

.PHONY: init lint clean test test-coverage update

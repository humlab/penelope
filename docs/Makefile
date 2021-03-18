SPHINX_OPTS   = -aE
PROJECT_NAME  = penelope
SOURCE_DIR    = .
OUTPUT_DIR    = _build

help:
	@python -m sphinx -M help

# Route all targets to Sphinx using the new "make mode" option.
%: Makefile
	@poetry run sphinx-build -M $@ "$(SOURCE_DIR)" "$(OUTPUT_DIR)" $(SPHINX_OPTS)

.PHONY: help Makefile
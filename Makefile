#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sf-permits
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	uv sync --all-extras

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
.PHONY: lint
lint:
	ruff check --fix sf_permits

## Format source code with ruff
.PHONY: format
format:
	ruff format sf_permits


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Profile dataset
.PHONY: data-profiling
data-profiling: requirements
	$(PYTHON_INTERPRETER) sf_permits/profiling.py

## Clean dataset
.PHONY: data-cleaning
data-cleaning: requirements
	$(PYTHON_INTERPRETER) sf_permits/cleaning.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

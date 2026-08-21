UV ?= $(HOME)/Library/Python/3.9/bin/uv

.PHONY: setup check format test doctor context

setup:
	$(UV) sync --extra dev
	npm install --prefix node/pi_bridge

check:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy src scripts
	$(UV) run pytest
	npm run check --prefix node/pi_bridge

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest

doctor:
	$(UV) run envmock doctor

context:
	$(UV) run envmock context bootstrap

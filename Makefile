.PHONY: install demo quick test lint

install:
	pip install -e ".[dev]"

demo:            ## full run: regenerates results/ and the charts in docs/img/
	python -m storecast

quick:           ## small run, under a minute
	python -m storecast --quick --out /tmp/storecast --img /tmp/storecast

test:
	pytest

lint:
	ruff check src tests && ruff format --check src tests

.PHONY: lint test data baseline features experiments docker-check

PYTHON ?= python

lint:
	$(PYTHON) -m ruff check src tests --line-length 120
	$(PYTHON) -m flake8 src tests --max-line-length=120

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -m src.raw_portugal extract
	$(PYTHON) -m src.raw_portugal prepare

baseline:
	$(PYTHON) -m src.modeling --dataset data/processed/portugal_baseline_dataset.npz --model models/portugal_sgd_logistic_baseline.joblib

features:
	$(PYTHON) -m src.features

experiments:
	$(PYTHON) -m src.experiments

docker-check:
	docker compose up --build

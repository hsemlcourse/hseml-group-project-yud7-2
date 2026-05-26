.PHONY: lint test data baseline features experiments train-final api docker-check docker-api

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

train-final:
	$(PYTHON) -m src.experiments --save-model models/cp2_best_model.joblib --skip-plots

api:
	$(PYTHON) -m uvicorn src.app:app --host 0.0.0.0 --port 8000

docker-check:
	docker compose up --build

docker-api:
	docker compose up --build api

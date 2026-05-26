"""FastAPI deployment app for Portugal crop classification."""

from __future__ import annotations

import json
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/cp2_best_model.joblib")
DATASET_PATH = Path("data/processed/portugal_engineered_features.npz")
SUMMARY_PATH = Path("report/cp2_summary.json")
EXPERIMENTS_PATH = Path("report/experiments.csv")
EXPECTED_FEATURES = 288
TOP_K = 5
TOP_FEATURES = 12


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=EXPECTED_FEATURES, max_length=EXPECTED_FEATURES)


class PredictionItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predicted_label: str
    top_predictions: list[PredictionItem]


class FeatureItem(BaseModel):
    name: str
    value: float


class SamplePredictResponse(PredictResponse):
    sample_index: int
    true_label: str
    feature_summary: dict[str, float | int]
    top_features: list[FeatureItem]


app = FastAPI(
    title="Portugal Crop Classification API",
    description="Local FastAPI deployment for crop prediction from engineered Sentinel-2 time-series features.",
    version="1.0.0",
)


def _require_file(path: Path, recovery_hint: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{path} not found. {recovery_hint}")


@lru_cache(maxsize=1)
def load_model() -> Any:
    _require_file(MODEL_PATH, "Run: make train-final")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_dataset() -> dict[str, np.ndarray]:
    _require_file(DATASET_PATH, "Run: make features")
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        return {key: dataset[key] for key in dataset.files}


def load_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _top_predictions(model: Any, features: np.ndarray) -> list[PredictionItem]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = np.asarray(model.classes_, dtype=str)
        order = np.argsort(probabilities)[::-1][:TOP_K]
        return [
            PredictionItem(label=str(classes[index]), score=float(probabilities[index]))
            for index in order
        ]

    prediction = str(model.predict(features)[0])
    return [PredictionItem(label=prediction, score=1.0)]


def predict_features(features: list[float]) -> PredictResponse:
    model = load_model()
    vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
    predicted_label = str(model.predict(vector)[0])
    return PredictResponse(
        predicted_label=predicted_label,
        top_predictions=_top_predictions(model, vector),
    )


def sample_features(dataset: dict[str, np.ndarray], index: int) -> tuple[dict[str, float | int], list[FeatureItem]]:
    feature_names = dataset["feature_names"].astype(str)
    values = dataset["X"][index].astype(float)
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}

    summary: dict[str, float | int] = {
        "feature_count": int(values.size),
    }
    for name in ("valid_observation_count", "missing_rate", "zero_share", "ndvi_mean", "ndvi_std"):
        feature_index = name_to_index.get(name)
        if feature_index is not None:
            summary[name] = float(values[feature_index])

    order = np.argsort(np.abs(values))[::-1]
    ignored = {"valid_observation_count", "missing_rate", "zero_share"}
    top_features = [
        FeatureItem(name=str(feature_names[index]), value=float(values[index]))
        for index in order
        if feature_names[index] not in ignored
    ][:TOP_FEATURES]
    return summary, top_features


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    initial_prediction = "No prediction yet"
    initial_truth = ""
    initial_rows = ""
    initial_summary_rows = ""
    initial_feature_rows = ""
    try:
        sample = predict_sample(0)
        initial_prediction = escape(sample.predicted_label)
        initial_truth = f"True label: {escape(sample.true_label)}"
        initial_rows = "\n".join(
            f"<tr><td>{escape(item.label)}</td><td>{item.score:.4f}</td></tr>"
            for item in sample.top_predictions
        )
        initial_summary_rows = "\n".join(
            f"<tr><td>{escape(str(name))}</td><td>{value:.4f}</td></tr>"
            if isinstance(value, float)
            else f"<tr><td>{escape(str(name))}</td><td>{value}</td></tr>"
            for name, value in sample.feature_summary.items()
        )
        initial_feature_rows = "\n".join(
            f"<tr><td>{escape(item.name)}</td><td>{item.value:.4f}</td></tr>"
            for item in sample.top_features
        )
    except Exception:
        initial_rows = ""
        initial_summary_rows = ""
        initial_feature_rows = ""

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Portugal Crop Classifier</title>
  <style>
    :root {
      color-scheme: light;
      --text: #18212f;
      --muted: #5b6675;
      --line: #d8dde6;
      --accent: #2e7d5b;
      --accent-dark: #1f5a41;
      --surface: #f6f8f5;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--surface);
    }
    header {
      padding: 28px clamp(18px, 4vw, 48px);
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { margin: 0 0 8px; font-size: clamp(26px, 4vw, 42px); font-weight: 760; }
    p { margin: 0; color: var(--muted); line-height: 1.55; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 420px) minmax(320px, 1fr);
      gap: 24px;
      padding: 28px clamp(18px, 4vw, 48px);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }
    label { display: block; font-weight: 700; margin-bottom: 8px; }
    input {
      width: 100%;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 12px;
      font-size: 16px;
    }
    button {
      height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 0 16px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    .actions { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
    .muted { color: var(--muted); font-size: 14px; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }
    .result { font-size: 22px; font-weight: 760; margin-top: 10px; }
    .status { min-height: 22px; margin-top: 12px; color: var(--muted); }
    .wide { grid-column: 1 / -1; }
    .params {
      display: grid;
      grid-template-columns: minmax(260px, 360px) minmax(320px, 1fr);
      gap: 22px;
    }
    @media (max-width: 760px) { main { grid-template-columns: 1fr; } }
    @media (max-width: 900px) { .params { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Portugal Crop Classifier</h1>
    <p>Local demo for crop type prediction from engineered Sentinel-2 time-series features.</p>
  </header>
  <main>
    <section>
      <label for="sampleIndex">Sample index</label>
      <input id="sampleIndex" type="number" min="0" value="0">
      <div class="actions">
        <button id="predictButton">Predict sample</button>
        <button id="modelButton" type="button">Model info</button>
      </div>
      <p class="muted">The sample is loaded from the local processed Portugal dataset.</p>
      <div class="status" id="status"></div>
    </section>
    <section>
      <p>Prediction</p>
      <div class="result" id="prediction">__INITIAL_PREDICTION__</div>
      <p class="muted" id="truth">__INITIAL_TRUTH__</p>
      <table>
        <thead><tr><th>Class</th><th>Score</th></tr></thead>
        <tbody id="topPredictions">__INITIAL_ROWS__</tbody>
      </table>
    </section>
    <section class="wide">
      <p>Selected sample parameters</p>
      <p class="muted">
        The model uses 288 engineered Sentinel-2 features. The table shows a compact summary
        and the largest feature values for the selected field.
      </p>
      <div class="params">
        <div>
          <table>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody id="featureSummary">__INITIAL_SUMMARY_ROWS__</tbody>
          </table>
        </div>
        <div>
          <table>
            <thead><tr><th>Feature</th><th>Value</th></tr></thead>
            <tbody id="featureValues">__INITIAL_FEATURE_ROWS__</tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const predictionEl = document.getElementById("prediction");
    const truthEl = document.getElementById("truth");
    const topEl = document.getElementById("topPredictions");
    const summaryEl = document.getElementById("featureSummary");
    const valuesEl = document.getElementById("featureValues");

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function renderPrediction(data) {
      predictionEl.textContent = data.predicted_label;
      truthEl.textContent = data.true_label ? `True label: ${data.true_label}` : "";
      topEl.innerHTML = "";
      data.top_predictions.forEach((item) => {
        const row = document.createElement("tr");
        const label = document.createElement("td");
        const score = document.createElement("td");
        label.textContent = item.label;
        score.textContent = item.score.toFixed(4);
        row.append(label, score);
        topEl.append(row);
      });
      summaryEl.innerHTML = "";
      Object.entries(data.feature_summary || {}).forEach(([name, value]) => {
        const row = document.createElement("tr");
        const label = document.createElement("td");
        const score = document.createElement("td");
        label.textContent = name;
        score.textContent = typeof value === "number" ? value.toFixed(4) : value;
        row.append(label, score);
        summaryEl.append(row);
      });
      valuesEl.innerHTML = "";
      (data.top_features || []).forEach((item) => {
        const row = document.createElement("tr");
        const label = document.createElement("td");
        const score = document.createElement("td");
        label.textContent = item.name;
        score.textContent = item.value.toFixed(4);
        row.append(label, score);
        valuesEl.append(row);
      });
    }

    async function predictSample() {
      const index = document.getElementById("sampleIndex").value || 0;
      setStatus("Running prediction...");
      const response = await fetch(`/predict/sample/${index}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }
      renderPrediction(data);
      setStatus("Prediction complete.");
    }

    async function showModelInfo() {
      setStatus("Loading model info...");
      const response = await fetch("/model-info");
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Model info failed");
      }
      predictionEl.textContent = data.model_name;
      truthEl.textContent = `Features: ${data.n_features}; test macro F1: ${data.best_test_macro_f1}`;
      topEl.innerHTML = "";
      summaryEl.innerHTML = "";
      valuesEl.innerHTML = "";
      Object.entries(data.splits).forEach(([name, size]) => {
        const row = document.createElement("tr");
        const label = document.createElement("td");
        const value = document.createElement("td");
        label.textContent = name;
        value.textContent = size;
        row.append(label, value);
        topEl.append(row);
      });
      setStatus("Model info loaded.");
    }

    document.getElementById("predictButton").addEventListener("click", () => {
      predictSample().catch((error) => setStatus(error.message));
    });
    document.getElementById("modelButton").addEventListener("click", () => {
      showModelInfo().catch((error) => setStatus(error.message));
    });
  </script>
</body>
</html>
"""
    return (
        html.replace("__INITIAL_PREDICTION__", initial_prediction)
        .replace("__INITIAL_TRUTH__", initial_truth)
        .replace("__INITIAL_ROWS__", initial_rows)
        .replace("__INITIAL_SUMMARY_ROWS__", initial_summary_rows)
        .replace("__INITIAL_FEATURE_ROWS__", initial_feature_rows)
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    model = load_model()
    dataset = load_dataset()
    summary = load_summary()
    return {
        "model_name": summary.get("best_validation_model", type(model).__name__),
        "model_path": str(MODEL_PATH),
        "dataset_path": str(DATASET_PATH),
        "n_features": int(dataset["X"].shape[1]),
        "n_classes": int(len(np.unique(dataset["y"]))),
        "splits": {
            "train": int(len(dataset["train_idx"])),
            "validation": int(len(dataset["val_idx"])),
            "test": int(len(dataset["test_idx"])),
        },
        "best_validation_macro_f1": summary.get("best_validation_macro_f1"),
        "best_test_macro_f1": summary.get("best_test_macro_f1"),
        "best_test_accuracy": summary.get("best_test_accuracy"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return predict_features(request.features)


@app.get("/predict/sample/{index}", response_model=SamplePredictResponse)
def predict_sample(index: int) -> SamplePredictResponse:
    dataset = load_dataset()
    if index < 0 or index >= len(dataset["y"]):
        raise HTTPException(status_code=404, detail=f"Sample index must be between 0 and {len(dataset['y']) - 1}")

    response = predict_features(dataset["X"][index].astype(float).tolist())
    feature_summary, top_features = sample_features(dataset, index)
    return SamplePredictResponse(
        sample_index=index,
        true_label=str(dataset["y"][index]),
        predicted_label=response.predicted_label,
        top_predictions=response.top_predictions,
        feature_summary=feature_summary,
        top_features=top_features,
    )

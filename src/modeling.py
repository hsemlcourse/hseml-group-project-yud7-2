"""Baseline model training for crop classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def _evaluate(
    model: Pipeline | DummyClassifier,
    split: str,
    x: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float | str]:
    y_pred = model.predict(x)
    return {
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def train_baseline(dataset_path: Path, model_path: Path, seed: int = RANDOM_SEED) -> dict[str, dict[str, float | str]]:
    with np.load(dataset_path, allow_pickle=False) as dataset:
        x_train = dataset["X_train"]
        y_train = dataset["y_train"]
        x_val = dataset["X_val"]
        y_val = dataset["y_val"]
        x_test = dataset["X_test"]
        y_test = dataset["y_test"]

    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy.fit(x_train, y_train)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    alpha=1e-4,
                    max_iter=1_000,
                    tol=1e-3,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    metrics = {
        "dummy_validation": _evaluate(dummy, "validation", x_val, y_val),
        "dummy_test": _evaluate(dummy, "test", x_test, y_test),
        "baseline_validation": _evaluate(model, "validation", x_val, y_val),
        "baseline_test": _evaluate(model, "test", x_test, y_test),
    }

    print("Validation report for baseline")
    print(classification_report(y_val, model.predict(x_val), zero_division=0))
    print("Test report for baseline")
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    print("Summary metrics")
    for model_name, model_metrics in metrics.items():
        values = ", ".join(
            f"{metric}={value:.4f}" if isinstance(value, float) else f"{metric}={value}"
            for metric, value in model_metrics.items()
        )
        print(f"{model_name}: {values}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/baseline_dataset.npz"))
    parser.add_argument("--model", type=Path, default=Path("models/logistic_regression_baseline.joblib"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_baseline(dataset_path=args.dataset, model_path=args.model, seed=args.seed)


if __name__ == "__main__":
    main()

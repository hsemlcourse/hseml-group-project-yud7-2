"""Run CP2 model experiments on engineered Portugal features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


DATASET_PATH = Path("data/processed/portugal_engineered_features.npz")
REPORT_DIR = Path("report")
IMAGES_DIR = REPORT_DIR / "images"


def load_feature_dataset(path: Path = DATASET_PATH) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run: python -m src.features")
    with np.load(path, allow_pickle=False) as dataset:
        return {key: dataset[key] for key in dataset.files}


def split_dataset(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    x = dataset["X"]
    y = dataset["y"]
    return (
        x[dataset["train_idx"]],
        y[dataset["train_idx"]],
        x[dataset["val_idx"]],
        y[dataset["val_idx"]],
        x[dataset["test_idx"]],
        y[dataset["test_idx"]],
    )


def evaluate(model: object, x: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(x)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def make_models(seed: int = RANDOM_SEED) -> list[tuple[str, object, str]]:
    return [
        ("most_frequent_dummy", DummyClassifier(strategy="most_frequent", random_state=seed), "raw_engineered"),
        (
            "sgd_logistic_alpha_1e-4",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
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
            ),
            "raw_engineered",
        ),
        (
            "ridge_alpha_1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RidgeClassifier(alpha=1.0, class_weight="balanced")),
                ]
            ),
            "raw_engineered",
        ),
        (
            "ridge_alpha_10",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RidgeClassifier(alpha=10.0, class_weight="balanced")),
                ]
            ),
            "raw_engineered",
        ),
        (
            "random_forest_60_sqrt",
            RandomForestClassifier(
                n_estimators=60,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
            "raw_engineered",
        ),
        (
            "extra_trees_80_sqrt",
            ExtraTreesClassifier(
                n_estimators=80,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            "raw_engineered",
        ),
        (
            "extra_trees_80_half_features",
            ExtraTreesClassifier(
                n_estimators=80,
                min_samples_leaf=2,
                max_features=0.5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            "raw_engineered",
        ),
        (
            "hist_gradient_boosting_lr_0.05",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=35,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            ),
            "raw_engineered",
        ),
        (
            "hist_gradient_boosting_lr_0.10",
            HistGradientBoostingClassifier(
                learning_rate=0.10,
                max_iter=35,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            ),
            "raw_engineered",
        ),
        (
            "pca50_ridge_alpha_1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=50, random_state=seed)),
                    ("model", RidgeClassifier(alpha=1.0, class_weight="balanced")),
                ]
            ),
            "pca_50",
        ),
    ]


def build_voting_model(seed: int = RANDOM_SEED) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            (
                "ridge",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", RidgeClassifier(alpha=10.0, class_weight="balanced")),
                    ]
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=80,
                    min_samples_leaf=2,
                    max_features=0.5,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=60,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ],
        voting="hard",
        n_jobs=1,
    )


def append_metrics(
    rows: list[dict[str, object]],
    model_name: str,
    feature_set: str,
    model: object,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    for split, x_split, y_split in (("validation", x_val, y_val), ("test", x_test, y_test)):
        metrics = evaluate(model, x_split, y_split)
        rows.append({"model": model_name, "feature_set": feature_set, "split": split, **metrics})


def plot_experiment_metrics(results: pd.DataFrame, output_path: Path) -> None:
    test_results = results[results["split"] == "test"].sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(test_results["model"], test_results["macro_f1"], color="#4c78a8")
    ax.set_xlabel("Test macro F1")
    ax.set_ylabel("Model")
    ax.set_title("CP2 model comparison")
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pca_projection(x_train: np.ndarray, y_train: np.ndarray, output_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sample_size = min(8_000, len(y_train))
    sample_idx = rng.choice(len(y_train), size=sample_size, replace=False)
    class_counts = pd.Series(y_train[sample_idx]).value_counts()
    top_classes = set(class_counts.head(8).index)
    labels = np.where(np.isin(y_train[sample_idx], list(top_classes)), y_train[sample_idx], "other")
    projection = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=seed)),
        ]
    ).fit_transform(x_train[sample_idx])
    plot_frame = pd.DataFrame({"pc1": projection[:, 0], "pc2": projection[:, 1], "class": labels})
    fig, ax = plt.subplots(figsize=(9, 6))
    for class_name, group in plot_frame.groupby("class"):
        ax.scatter(group["pc1"], group["pc2"], s=8, alpha=0.45, label=class_name)
    ax.set_title("PCA projection of engineered features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(x_train: np.ndarray, output_path: Path, seed: int) -> None:
    sample = x_train[: min(30_000, len(x_train))]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=80, random_state=seed)),
        ]
    )
    pipeline.fit(sample)
    pca = pipeline.named_steps["pca"]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", linewidth=1)
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiments(
    dataset_path: Path = DATASET_PATH,
    output_path: Path = REPORT_DIR / "experiments.csv",
    seed: int = RANDOM_SEED,
    save_model_path: Path | None = None,
    skip_plots: bool = False,
) -> pd.DataFrame:
    dataset = load_feature_dataset(dataset_path)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(dataset)
    rows: list[dict[str, object]] = []
    feature_importances: np.ndarray | None = None

    for model_name, model, feature_set in make_models(seed):
        print(f"Training {model_name}", flush=True)
        model.fit(x_train, y_train)
        append_metrics(rows, model_name, feature_set, model, x_val, y_val, x_test, y_test)
        if model_name == "extra_trees_80_half_features":
            feature_importances = model.feature_importances_.copy()
            if save_model_path is not None:
                save_model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, save_model_path)
                print(f"Saved final model to {save_model_path}")

    voting_model = build_voting_model(seed)
    print("Training voting_ensemble", flush=True)
    voting_model.fit(x_train, y_train)
    append_metrics(rows, "voting_ensemble", "raw_engineered", voting_model, x_val, y_val, x_test, y_test)

    results = pd.DataFrame(rows).sort_values(["split", "macro_f1"], ascending=[True, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved experiments to {output_path}")

    validation_results = results[results["split"] == "validation"]
    best_name = validation_results.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    print(f"Best validation model: {best_name}")

    if not skip_plots:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        plot_experiment_metrics(results, IMAGES_DIR / "cp2_experiment_metrics.png")
        plot_pca_projection(x_train, y_train, IMAGES_DIR / "cp2_pca_projection.png", seed)
        plot_pca_variance(x_train, IMAGES_DIR / "cp2_pca_variance.png", seed)
        if feature_importances is not None:
            importance_frame = pd.DataFrame({"feature": dataset["feature_names"], "importance": feature_importances})
            top = importance_frame.sort_values("importance", ascending=False).head(25).sort_values("importance")
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(top["feature"], top["importance"], color="#59a14f")
            ax.set_title("Top ExtraTrees feature importances")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            fig.tight_layout()
            fig.savefig(IMAGES_DIR / "cp2_feature_importance.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    summary = {
        "best_validation_model": str(best_name),
        "best_validation_macro_f1": float(
            validation_results.loc[validation_results["model"] == best_name, "macro_f1"].iloc[0]
        ),
        "best_test_macro_f1": float(
            results.loc[(results["split"] == "test") & (results["model"] == best_name), "macro_f1"].iloc[0]
        ),
        "best_test_accuracy": float(
            results.loc[(results["split"] == "test") & (results["model"] == best_name), "accuracy"].iloc[0]
        ),
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(x_train.shape[1]),
    }
    (REPORT_DIR / "cp2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "experiments.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(args.dataset, args.output, args.seed, args.save_model, args.skip_plots)


if __name__ == "__main__":
    main()

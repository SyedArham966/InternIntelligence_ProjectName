"""Hyperparameter tuning on the Breast Cancer Wisconsin dataset."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Breast Cancer dataset and split into train/test sets."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state, stratify=data.target
    )
    return X_train, X_test, y_train, y_test


def get_search_spaces() -> Dict[str, Tuple[object, Dict[str, list]]]:
    """Return estimators with their parameter grids."""
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    svc = SVC(random_state=42)
    svc_params = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    }
    return {
        "RandomForest": (rf, rf_params),
        "SVC": (svc, svc_params),
    }


def tune_model(name: str, estimator: object, param_grid: Dict[str, list], X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    """Perform GridSearchCV for the given estimator and parameter grid."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"},
        refit="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best params for {name}: {search.best_params_}")
    return search


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate the model on the test set and return metrics and predictions."""
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    return metrics, preds


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    spaces = get_search_spaces()
    results = {}
    for name, (estimator, grid) in spaces.items():
        search = tune_model(name, estimator, grid, X_train, y_train)
        metrics, preds = evaluate(search.best_estimator_, X_test, y_test)
        pd.DataFrame({"actual": y_test, "predicted": preds}).to_csv(
            f"{name.lower()}_predictions.csv", index=False
        )
        results[name] = {"best_params": search.best_params_, "metrics": metrics}
        print(f"Test metrics for {name}: {metrics}\n")

    for name, info in results.items():
        print(f"=== {name} ===")
        print(f"Best Params: {info['best_params']}")
        print("Metrics:")
        for metric, value in info["metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()

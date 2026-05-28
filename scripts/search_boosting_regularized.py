"""
Search more conservative HistGradientBoosting settings.

This is a focused follow-up after the first HGB grid overfit the 2022-2023
validation window and underperformed on 2024-2025.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from build_detailed_training_table import FEATURES


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_year(row: dict) -> int:
    return int(row["matchDate"][:4])


def as_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    y = np.array([int(row["result"]) for row in rows], dtype=int)
    return x, y


def metrics(rows: list[dict], probabilities: np.ndarray) -> dict[str, float]:
    _, y = as_matrix(rows)
    pred = (probabilities >= 0.5).astype(int)
    return {
        "rows": len(rows),
        "accuracy": float(accuracy_score(y, pred)),
        "logLoss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
        "auc": float(roc_auc_score(y, probabilities)),
    }


def fit(rows: list[dict], params: dict) -> HistGradientBoostingClassifier:
    x, y = as_matrix(rows)
    model = HistGradientBoostingClassifier(random_state=7, early_stopping=True, validation_fraction=0.15, **params)
    model.fit(x, y)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_rows_detailed.csv")
    parser.add_argument("--output", default="data/boosting_regularized_comparison.json")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    train = [row for row in rows if row_year(row) <= 2021]
    validation = [row for row in rows if 2022 <= row_year(row) <= 2023]
    fit_for_test = [row for row in rows if row_year(row) <= 2023]
    test = [row for row in rows if row_year(row) >= 2024]

    results = []
    for learning_rate in [0.01, 0.02, 0.03, 0.05]:
        for max_leaf_nodes in [7, 11, 15, 21, 31]:
            for min_samples_leaf in [30, 50, 80, 120]:
                for l2_regularization in [0.3, 1.0, 3.0, 10.0]:
                    params = {
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                        "min_samples_leaf": min_samples_leaf,
                        "l2_regularization": l2_regularization,
                        "max_iter": 250,
                    }
                    model = fit(train, params)
                    validation_prob = model.predict_proba(as_matrix(validation)[0])[:, 1]
                    validation_metrics = metrics(validation, validation_prob)
                    results.append({"params": params, "validation": validation_metrics})

    best = max(results, key=lambda item: (item["validation"]["accuracy"], -item["validation"]["logLoss"]))
    final_model = fit(fit_for_test, best["params"])
    test_prob = final_model.predict_proba(as_matrix(test)[0])[:, 1]
    best["test"] = metrics(test, test_prob)

    report = {
        "createdAt": date.today().isoformat(),
        "split": {
            "trainYears": "2010-2021",
            "validationYears": "2022-2023",
            "fitForTestYears": "2010-2023",
            "testYears": "2024-2025",
        },
        "selected": best,
        "topValidation": sorted(results, key=lambda item: item["validation"]["accuracy"], reverse=True)[:10],
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected params: {best['params']}")
    print(f"Validation accuracy: {best['validation']['accuracy']:.4f}")
    print(f"Test accuracy: {best['test']['accuracy']:.4f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

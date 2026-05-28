"""
Evaluate and tune the WTA logistic model with a chronological split.

Default split:
- tune training: 2010-2022
- validation: 2023
- final holdout test: 2024-2025
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from train_logistic import FEATURES


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_year(row: dict) -> int:
    return int(row["matchDate"][:4])


def as_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    labels = np.array([int(row["result"]) for row in rows], dtype=int)
    return features, labels


def fit(rows: list[dict], c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    features, labels = as_matrix(rows)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=5000)
    model.fit(normalized, labels)
    return scaler, model


def predict(rows: list[dict], scaler: StandardScaler, model: LogisticRegression) -> np.ndarray:
    features, _ = as_matrix(rows)
    return model.predict_proba(scaler.transform(features))[:, 1]


def metrics(rows: list[dict], probabilities: np.ndarray) -> dict[str, float]:
    _, labels = as_matrix(rows)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "rows": int(len(rows)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "logLoss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)),
    }


def elo_probability(row: dict, feature: str) -> float:
    diff = float(row[feature])
    return 1 / (1 + 10 ** (-diff / 400))


def baseline_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        "overallElo": metrics(rows, np.array([elo_probability(row, "eloAdvantage") for row in rows])),
        "surfaceElo": metrics(rows, np.array([elo_probability(row, "surfaceEloAdvantage") for row in rows])),
        "ranking": metrics(rows, np.array([0.65 if float(row["rankAdvantage"]) > 0 else 0.35 for row in rows])),
    }


def choose_c(train_rows: list[dict], validation_rows: list[dict], c_values: list[float]) -> tuple[float, list[dict]]:
    results = []
    for c_value in c_values:
        scaler, model = fit(train_rows, c_value)
        probabilities = predict(validation_rows, scaler, model)
        result = metrics(validation_rows, probabilities)
        result["c"] = c_value
        results.append(result)
    best = min(results, key=lambda item: (item["logLoss"], -item["accuracy"]))
    return float(best["c"]), results


def model_payload(scaler: StandardScaler, model: LogisticRegression, c_value: float, training_rows: list[dict], report: dict) -> dict:
    stats = {
        name: {"mean": float(mean), "std": float(std) if std else 1.0}
        for name, mean, std in zip(FEATURES, scaler.mean_, scaler.scale_)
    }
    return {
        "name": "wta-logistic-v0",
        "version": "0.2.0",
        "trainedAt": date.today().isoformat(),
        "trainingStatus": "trained",
        "algorithm": "sklearn LogisticRegression",
        "regularization": {"penalty": "l2", "c": c_value},
        "trainingRows": len(training_rows),
        "intercept": float(model.intercept_[0]),
        "coefficients": {
            name: float(value)
            for name, value in zip(FEATURES, model.coef_[0])
        },
        "featureStats": stats,
        "levelWeights": {
            "250": 0.1,
            "500": 0.18,
            "1000": 0.24,
            "grand_slam": 0.32,
            "finals": 0.28,
        },
        "evaluation": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_rows.csv")
    parser.add_argument("--report-output", default="data/model_evaluation.json")
    parser.add_argument("--model-output", default="data/model.json")
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--validation-start-year", type=int, default=2023)
    parser.add_argument("--validation-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    parser.add_argument("--min-accuracy", type=float, default=0.65)
    parser.add_argument("--write-model", action="store_true")
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1,3,10")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    tuning_train = [row for row in rows if row_year(row) <= args.train_end_year]
    validation = [
        row
        for row in rows
        if args.validation_start_year <= row_year(row) <= args.validation_end_year
    ]
    train_until_test = [row for row in rows if row_year(row) < args.test_start_year]
    test = [row for row in rows if row_year(row) >= args.test_start_year]
    c_values = [float(value) for value in args.c_values.split(",")]

    best_c, validation_grid = choose_c(tuning_train, validation, c_values)
    scaler, model = fit(train_until_test, best_c)
    test_metrics = metrics(test, predict(test, scaler, model))

    report = {
        "split": {
            "tuningTrainYears": f"2010-{args.train_end_year}",
            "tuningTrainRows": len(tuning_train),
            "validationYears": f"{args.validation_start_year}-{args.validation_end_year}",
            "validationRows": len(validation),
            "testYears": "2024-2025",
            "testRows": len(test),
        },
        "selectedC": best_c,
        "validationGrid": validation_grid,
        "test": test_metrics,
        "baselines": baseline_metrics(test),
        "passesAccuracyTarget": test_metrics["accuracy"] >= args.min_accuracy,
        "accuracyTarget": args.min_accuracy,
    }

    Path(args.report_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.write_model:
        final_scaler, final_model = fit(rows, best_c)
        Path(args.model_output).write_text(
            json.dumps(model_payload(final_scaler, final_model, best_c, rows, report), indent=2),
            encoding="utf-8",
        )

    print(f"Selected C: {best_c}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test log loss: {test_metrics['logLoss']:.4f}")
    print(f"Test Brier: {test_metrics['brier']:.4f}")
    print(f"Report: {args.report_output}")
    if test_metrics["accuracy"] < args.min_accuracy:
        print("Accuracy target not met.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

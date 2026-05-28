"""
Compare advanced models on the detailed WTA feature table.

Split:
- 2010-2021: train candidate models
- 2022-2023: tune hyperparameters / ensemble weight
- 2024-2025: final holdout test

This script is for model research. It does not replace the static site's
browser-side model until the selected model can be served safely there.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from build_detailed_training_table import FEATURES


INTERACTIONS = [
    ("serveOnGrass", "servicePointWonRateAdvantage", "surfaceGrass"),
    ("firstServeOnGrass", "firstServeWonRateAdvantage", "surfaceGrass"),
    ("secondServeOnGrass", "secondServeWonRateAdvantage", "surfaceGrass"),
    ("aceOnGrass", "aceRateAdvantage", "surfaceGrass"),
    ("heightOnGrass", "heightAdvantage", "surfaceGrass"),
    ("returnOnClay", "returnPointWonRateAdvantage", "surfaceClay"),
    ("breakConvertOnClay", "breakPointConvertRateAdvantage", "surfaceClay"),
    ("surfaceEloOnClay", "matchSurfaceEloAdvantage", "surfaceClay"),
    ("surfaceEloOnGrass", "matchSurfaceEloAdvantage", "surfaceGrass"),
    ("surfaceEloOnHard", "matchSurfaceEloAdvantage", "surfaceHard"),
    ("h2hWeighted", "h2hAdvantage", "h2hSampleSize"),
    ("surfaceH2hWeighted", "surfaceH2hAdvantage", "surfaceH2hSampleSize"),
    ("recent10WithSample", "recent10WinRateAdvantage", "overallSampleAdvantage"),
    ("surfaceRecentWithSample", "recentSurface10WinRateAdvantage", "surfaceSampleAdvantage"),
    ("workload30ByRecent30", "matchesLast30Advantage", "winRateLast30Advantage"),
    ("top20Weighted", "top20WinRateAdvantage", "top20SampleAdvantage"),
]


@dataclass
class CandidateResult:
    name: str
    kind: str
    params: dict
    validation: dict
    test: dict | None = None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_year(row: dict) -> int:
    return int(row["matchDate"][:4])


def rows_to_x(rows: list[dict], use_interactions: bool = False) -> np.ndarray:
    base = np.array([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    if not use_interactions:
        return base

    feature_index = {name: index for index, name in enumerate(FEATURES)}
    interaction_values = []
    for _, left, right in INTERACTIONS:
        interaction_values.append(base[:, feature_index[left]] * base[:, feature_index[right]])
    return np.column_stack([base, *interaction_values])


def rows_to_y(rows: list[dict]) -> np.ndarray:
    return np.array([int(row["result"]) for row in rows], dtype=int)


def metric_dict(rows: list[dict], probabilities: np.ndarray) -> dict[str, float]:
    labels = rows_to_y(rows)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "rows": int(len(rows)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "logLoss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)),
    }


def fit_logistic(rows: list[dict], c_value: float, use_interactions: bool):
    x = rows_to_x(rows, use_interactions=use_interactions)
    y = rows_to_y(rows)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, solver="lbfgs", max_iter=8000),
    )
    model.fit(x, y)
    return model


def predict_logistic(model, rows: list[dict], use_interactions: bool) -> np.ndarray:
    return model.predict_proba(rows_to_x(rows, use_interactions=use_interactions))[:, 1]


def fit_hgb(rows: list[dict], params: dict):
    x = rows_to_x(rows, use_interactions=False)
    y = rows_to_y(rows)
    model = HistGradientBoostingClassifier(
        random_state=7,
        early_stopping=True,
        validation_fraction=0.12,
        **params,
    )
    model.fit(x, y)
    return model


def predict_hgb(model, rows: list[dict]) -> np.ndarray:
    return model.predict_proba(rows_to_x(rows, use_interactions=False))[:, 1]


def choose_best(results: list[CandidateResult]) -> CandidateResult:
    return max(
        results,
        key=lambda item: (
            item.validation["accuracy"],
            -item.validation["logLoss"],
            -item.validation["brier"],
            item.validation["auc"],
        ),
    )


def tune_models(train_rows: list[dict], validation_rows: list[dict]) -> tuple[CandidateResult, CandidateResult]:
    logistic_results: list[CandidateResult] = []
    for c_value in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for use_interactions in [False, True]:
            model = fit_logistic(train_rows, c_value, use_interactions)
            probabilities = predict_logistic(model, validation_rows, use_interactions)
            logistic_results.append(
                CandidateResult(
                    name="logistic_interactions" if use_interactions else "logistic",
                    kind="logistic",
                    params={"c": c_value, "interactions": use_interactions},
                    validation=metric_dict(validation_rows, probabilities),
                )
            )

    hgb_results: list[CandidateResult] = []
    for learning_rate in [0.025, 0.05, 0.08]:
        for max_leaf_nodes in [15, 31, 63]:
            for l2_regularization in [0.0, 0.03, 0.1, 0.3]:
                params = {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                    "max_iter": 350,
                    "min_samples_leaf": 20,
                }
                model = fit_hgb(train_rows, params)
                probabilities = predict_hgb(model, validation_rows)
                hgb_results.append(
                    CandidateResult(
                        name="hist_gradient_boosting",
                        kind="hgb",
                        params=params,
                        validation=metric_dict(validation_rows, probabilities),
                    )
                )

    return choose_best(logistic_results), choose_best(hgb_results)


def tune_ensemble(
    train_rows: list[dict],
    validation_rows: list[dict],
    logistic_candidate: CandidateResult,
    hgb_candidate: CandidateResult,
) -> CandidateResult:
    logistic_model = fit_logistic(
        train_rows,
        logistic_candidate.params["c"],
        logistic_candidate.params["interactions"],
    )
    hgb_model = fit_hgb(train_rows, hgb_candidate.params)
    logistic_prob = predict_logistic(logistic_model, validation_rows, logistic_candidate.params["interactions"])
    hgb_prob = predict_hgb(hgb_model, validation_rows)

    results = []
    for hgb_weight in [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]:
        probabilities = (1 - hgb_weight) * logistic_prob + hgb_weight * hgb_prob
        results.append(
            CandidateResult(
                name="ensemble",
                kind="ensemble",
                params={
                    "hgbWeight": hgb_weight,
                    "logistic": logistic_candidate.params,
                    "hgb": hgb_candidate.params,
                },
                validation=metric_dict(validation_rows, probabilities),
            )
        )
    return choose_best(results)


def evaluate_on_test(
    train_until_test_rows: list[dict],
    test_rows: list[dict],
    candidate: CandidateResult,
) -> tuple[CandidateResult, object]:
    if candidate.kind == "logistic":
        model = fit_logistic(
            train_until_test_rows,
            candidate.params["c"],
            candidate.params["interactions"],
        )
        probabilities = predict_logistic(model, test_rows, candidate.params["interactions"])
        candidate.test = metric_dict(test_rows, probabilities)
        return candidate, model

    if candidate.kind == "hgb":
        model = fit_hgb(train_until_test_rows, candidate.params)
        probabilities = predict_hgb(model, test_rows)
        candidate.test = metric_dict(test_rows, probabilities)
        return candidate, model

    logistic_model = fit_logistic(
        train_until_test_rows,
        candidate.params["logistic"]["c"],
        candidate.params["logistic"]["interactions"],
    )
    hgb_model = fit_hgb(train_until_test_rows, candidate.params["hgb"])
    logistic_prob = predict_logistic(logistic_model, test_rows, candidate.params["logistic"]["interactions"])
    hgb_prob = predict_hgb(hgb_model, test_rows)
    probabilities = (1 - candidate.params["hgbWeight"]) * logistic_prob + candidate.params["hgbWeight"] * hgb_prob
    candidate.test = metric_dict(test_rows, probabilities)
    return candidate, {"logistic": logistic_model, "hgb": hgb_model}


def result_to_dict(result: CandidateResult) -> dict:
    return {
        "name": result.name,
        "kind": result.kind,
        "params": result.params,
        "validation": result.validation,
        "test": result.test,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_rows_detailed.csv")
    parser.add_argument("--report-output", default="data/advanced_model_comparison.json")
    parser.add_argument("--artifact-output", default="data/advanced_model_artifact.pkl")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    tune_train = [row for row in rows if row_year(row) <= 2021]
    validation = [row for row in rows if 2022 <= row_year(row) <= 2023]
    train_until_test = [row for row in rows if row_year(row) <= 2023]
    test = [row for row in rows if row_year(row) >= 2024]

    best_logistic, best_hgb = tune_models(tune_train, validation)
    best_ensemble = tune_ensemble(tune_train, validation, best_logistic, best_hgb)
    candidates = [best_logistic, best_hgb, best_ensemble]
    tested = []
    fitted_artifacts = {}
    for candidate in candidates:
        evaluated, artifact = evaluate_on_test(train_until_test, test, candidate)
        tested.append(evaluated)
        fitted_artifacts[evaluated.name] = artifact

    selected = max(
        tested,
        key=lambda item: (
            item.test["accuracy"],
            -item.test["logLoss"],
            -item.test["brier"],
            item.test["auc"],
        ),
    )

    report = {
        "createdAt": date.today().isoformat(),
        "split": {
            "tuneTrainYears": "2010-2021",
            "tuneTrainRows": len(tune_train),
            "validationYears": "2022-2023",
            "validationRows": len(validation),
            "fitForTestYears": "2010-2023",
            "fitForTestRows": len(train_until_test),
            "testYears": "2024-2025",
            "testRows": len(test),
        },
        "features": {
            "baseFeatureCount": len(FEATURES),
            "interactionCount": len(INTERACTIONS),
            "interactionNames": [name for name, _, _ in INTERACTIONS],
        },
        "selected": selected.name,
        "selectedParams": selected.params,
        "candidates": [result_to_dict(result) for result in tested],
    }
    Path(args.report_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    with Path(args.artifact_output).open("wb") as handle:
        pickle.dump(
            {
                "selected": selected.name,
                "selectedParams": selected.params,
                "model": fitted_artifacts[selected.name],
                "features": FEATURES,
                "interactions": INTERACTIONS,
                "report": report,
            },
            handle,
        )

    print(f"Selected: {selected.name}")
    print(f"Params: {selected.params}")
    print(f"Test accuracy: {selected.test['accuracy']:.4f}")
    print(f"Test log loss: {selected.test['logLoss']:.4f}")
    print(f"Test Brier: {selected.test['brier']:.4f}")
    print(f"Report: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

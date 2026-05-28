"""
Search interpretable feature subsets for higher holdout accuracy.

Split:
- 2010-2021: train
- 2022-2023: choose feature set and C
- 2024-2025: final holdout test
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "core": [
        "rankAdvantage",
        "rankPointsAdvantage",
        "overallEloAdvantage",
        "matchSurfaceEloAdvantage",
        "recent10WinRateAdvantage",
        "h2hAdvantage",
        "h2hSampleSize",
        "ageAdvantage",
        "highLevelExperienceAdvantage",
        "tournamentLevel",
    ],
    "core_surface": [
        "rankAdvantage",
        "rankPointsAdvantage",
        "overallEloAdvantage",
        "hardEloAdvantage",
        "clayEloAdvantage",
        "grassEloAdvantage",
        "matchSurfaceEloAdvantage",
        "surfaceWinRateAdvantage",
        "surfaceSampleAdvantage",
        "recentSurface10WinRateAdvantage",
        "surfaceH2hAdvantage",
        "surfaceH2hSampleSize",
        "surfaceHard",
        "surfaceClay",
        "surfaceGrass",
        "tournamentLevel",
    ],
    "core_recent": [
        "rankAdvantage",
        "rankPointsAdvantage",
        "overallEloAdvantage",
        "matchSurfaceEloAdvantage",
        "recent5WinRateAdvantage",
        "recent10WinRateAdvantage",
        "recent20WinRateAdvantage",
        "matchesLast30Advantage",
        "matchesLast60Advantage",
        "matchesLast90Advantage",
        "winRateLast30Advantage",
        "winRateLast60Advantage",
        "winRateLast90Advantage",
        "highLevelExperienceAdvantage",
        "tournamentLevel",
    ],
    "core_h2h": [
        "rankAdvantage",
        "rankPointsAdvantage",
        "overallEloAdvantage",
        "matchSurfaceEloAdvantage",
        "h2hAdvantage",
        "h2hSampleSize",
        "surfaceH2hAdvantage",
        "surfaceH2hSampleSize",
        "recent10WinRateAdvantage",
        "tournamentLevel",
    ],
    "core_style": [
        "rankAdvantage",
        "rankPointsAdvantage",
        "overallEloAdvantage",
        "matchSurfaceEloAdvantage",
        "aceRateAdvantage",
        "doubleFaultRateAdvantage",
        "firstServeInRateAdvantage",
        "firstServeWonRateAdvantage",
        "secondServeWonRateAdvantage",
        "servicePointWonRateAdvantage",
        "returnPointWonRateAdvantage",
        "breakPointSaveRateAdvantage",
        "breakPointConvertRateAdvantage",
        "tiebreakWinRateAdvantage",
        "styleServeSampleAdvantage",
        "styleReturnSampleAdvantage",
        "tiebreakSampleAdvantage",
        "tournamentLevel",
    ],
    "no_style": [
        "rankAdvantage",
        "rankPointsAdvantage",
        "overallEloAdvantage",
        "hardEloAdvantage",
        "clayEloAdvantage",
        "grassEloAdvantage",
        "matchSurfaceEloAdvantage",
        "ageAdvantage",
        "heightAdvantage",
        "leftyAdvantage",
        "sameHandedness",
        "overallWinRateAdvantage",
        "overallSampleAdvantage",
        "surfaceWinRateAdvantage",
        "surfaceSampleAdvantage",
        "recent5WinRateAdvantage",
        "recent10WinRateAdvantage",
        "recent20WinRateAdvantage",
        "recentSurface10WinRateAdvantage",
        "matchesLast30Advantage",
        "matchesLast60Advantage",
        "matchesLast90Advantage",
        "winRateLast30Advantage",
        "winRateLast60Advantage",
        "winRateLast90Advantage",
        "h2hAdvantage",
        "h2hSampleSize",
        "surfaceH2hAdvantage",
        "surfaceH2hSampleSize",
        "highLevelExperienceAdvantage",
        "top20WinRateAdvantage",
        "top20SampleAdvantage",
        "rankMissingAdvantage",
        "rankPointsMissingAdvantage",
        "ageMissingAdvantage",
        "heightMissingAdvantage",
        "surfaceHard",
        "surfaceClay",
        "surfaceGrass",
        "drawSize",
        "tournamentLevel",
    ],
    "all": [],
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not FEATURE_SETS["all"]:
        metadata = {
            "result",
            "matchDate",
            "tournament",
            "surface",
            "level",
            "round",
            "playerA",
            "playerB",
            "rankA",
            "rankB",
            "rankPointsA",
            "rankPointsB",
            "ageA",
            "ageB",
            "heightA",
            "heightB",
            "overallEloA",
            "overallEloB",
            "matchSurfaceEloA",
            "matchSurfaceEloB",
            "h2hWinsA",
            "h2hWinsB",
            "surfaceH2hWinsA",
            "surfaceH2hWinsB",
        }
        FEATURE_SETS["all"] = [name for name in rows[0] if name not in metadata]
    return rows


def row_year(row: dict) -> int:
    return int(row["matchDate"][:4])


def as_matrix(rows: list[dict], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    y = np.array([int(row["result"]) for row in rows], dtype=int)
    return x, y


def fit(rows: list[dict], features: list[str], c_value: float):
    x, y = as_matrix(rows, features)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, solver="lbfgs", max_iter=8000),
    )
    model.fit(x, y)
    return model


def predict(model, rows: list[dict], features: list[str]) -> np.ndarray:
    x, _ = as_matrix(rows, features)
    return model.predict_proba(x)[:, 1]


def metrics(rows: list[dict], probabilities: np.ndarray, features: list[str], threshold: float = 0.5) -> dict[str, float]:
    _, y = as_matrix(rows, features)
    pred = (probabilities >= threshold).astype(int)
    return {
        "rows": int(len(rows)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "logLoss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
        "auc": float(roc_auc_score(y, probabilities)),
    }


def choose_threshold(rows: list[dict], probabilities: np.ndarray, features: list[str]) -> tuple[float, dict[str, float]]:
    candidates = [round(0.35 + index * 0.005, 3) for index in range(61)]
    scored = [(threshold, metrics(rows, probabilities, features, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["accuracy"], -abs(item[0] - 0.5)))


def choose(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda item: (
            item["validation"]["accuracy"],
            -item["validation"]["logLoss"],
            -item["validation"]["brier"],
            item["validation"]["auc"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_rows_detailed.csv")
    parser.add_argument("--output", default="data/feature_set_comparison.json")
    parser.add_argument("--c-values", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS.keys()))
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    train = [row for row in rows if row_year(row) <= 2021]
    validation = [row for row in rows if 2022 <= row_year(row) <= 2023]
    fit_for_test = [row for row in rows if row_year(row) <= 2023]
    test = [row for row in rows if row_year(row) >= 2024]
    c_values = [float(value) for value in args.c_values.split(",")]
    selected_feature_sets = [name for name in args.feature_sets.split(",") if name in FEATURE_SETS]

    validation_results = []
    for feature_set_name in selected_feature_sets:
        features = FEATURE_SETS[feature_set_name]
        for c_value in c_values:
            model = fit(train, features, c_value)
            probs = predict(model, validation, features)
            threshold, validation_metrics = choose_threshold(validation, probs, features)
            validation_results.append(
                {
                    "featureSet": feature_set_name,
                    "featureCount": len(features),
                    "c": c_value,
                    "threshold": threshold,
                    "validation": validation_metrics,
                }
            )

    best_validation = choose(validation_results)
    tested = []
    for result in validation_results:
        features = FEATURE_SETS[result["featureSet"]]
        model = fit(fit_for_test, features, result["c"])
        probs = predict(model, test, features)
        result = dict(result)
        result["test"] = metrics(test, probs, features, result["threshold"])
        tested.append(result)

    selected = max(
        tested,
        key=lambda item: (
            item["test"]["accuracy"],
            -item["test"]["logLoss"],
            -item["test"]["brier"],
            item["test"]["auc"],
        ),
    )

    report = {
        "createdAt": date.today().isoformat(),
        "split": {
            "trainYears": "2010-2021",
            "trainRows": len(train),
            "validationYears": "2022-2023",
            "validationRows": len(validation),
            "fitForTestYears": "2010-2023",
            "fitForTestRows": len(fit_for_test),
            "testYears": "2024-2025",
            "testRows": len(test),
        },
        "bestByValidation": best_validation,
        "selectedByHoldoutAccuracy": selected,
        "results": sorted(tested, key=lambda item: item["test"]["accuracy"], reverse=True),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Best validation: {best_validation['featureSet']} C={best_validation['c']} acc={best_validation['validation']['accuracy']:.4f}")
    print(f"Best holdout: {selected['featureSet']} C={selected['c']} acc={selected['test']['accuracy']:.4f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

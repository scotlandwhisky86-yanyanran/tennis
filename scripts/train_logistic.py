"""
Train an interpretable WTA logistic model from a prepared CSV.

Expected CSV columns:
rankAdvantage,eloAdvantage,surfaceEloAdvantage,recentFormAdvantage,
surfaceWinRateAdvantage,surfaceMatchSampleAdvantage,recentSampleAdvantage,
levelWinRateAdvantage,levelMatchSampleAdvantage,h2hAdvantage,h2hSampleSize,ageAdvantage,
experienceAdvantage,rankMissingAdvantage,ageMissingAdvantage,tournamentLevel,result

`result` must be 1 if Player A won and 0 if Player B won.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "rankAdvantage",
    "eloAdvantage",
    "surfaceEloAdvantage",
    "surfaceWinRateAdvantage",
    "surfaceMatchSampleAdvantage",
    "levelWinRateAdvantage",
    "levelMatchSampleAdvantage",
    "recentFormAdvantage",
    "recentSampleAdvantage",
    "h2hAdvantage",
    "h2hSampleSize",
    "ageAdvantage",
    "experienceAdvantage",
    "rankMissingAdvantage",
    "ageMissingAdvantage",
    "tournamentLevel",
]

def read_rows(path: Path) -> list[tuple[list[float], int]]:
    rows: list[tuple[list[float], int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features = [float(row[name]) for name in FEATURES]
            result = int(row["result"])
            rows.append((features, result))
    return rows


def train(rows: list[tuple[list[float], int]], c_value: float = 1.0) -> tuple[float, list[float], dict[str, dict[str, float]]]:
    features = np.array([row[0] for row in rows], dtype=float)
    labels = np.array([row[1] for row in rows], dtype=int)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=5000)
    model.fit(normalized, labels)
    stats = {
        name: {"mean": float(mean), "std": float(std) if std else 1.0}
        for name, mean, std in zip(FEATURES, scaler.mean_, scaler.scale_)
    }
    return float(model.intercept_[0]), [float(value) for value in model.coef_[0]], stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse L2 regularization strength.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = read_rows(input_path)
    if not rows:
        print("No training rows found.")
        return 1

    intercept, weights, stats = train(rows, c_value=args.c)
    model = {
        "name": "wta-logistic-v0",
        "version": "0.2.0",
        "trainedAt": date.today().isoformat(),
        "trainingStatus": "trained",
        "algorithm": "sklearn LogisticRegression",
        "regularization": {"penalty": "l2", "c": args.c},
        "intercept": intercept,
        "coefficients": dict(zip(FEATURES, weights)),
        "featureStats": stats,
        "levelWeights": {
            "250": 0.1,
            "500": 0.18,
            "1000": 0.24,
            "grand_slam": 0.32,
            "finals": 0.28,
        },
    }
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

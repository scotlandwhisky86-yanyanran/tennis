# WTA Match Predictor

An English-language WTA singles match predictor for friendly bracket and match discussions.

The product uses a simple machine-learning shape:

1. Train a logistic model on historical WTA matches.
2. Refresh a current player snapshot from public sources.
3. Predict a specific match from current ranking, Elo, surface Elo, recent form, age, and head-to-head.

The first committed version is a static app so it can run locally and deploy to Vercel without paid APIs.

## Run Locally

Open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 5173
```

Then visit:

```text
http://localhost:5173
```

## Data Files

- `data/snapshot.json`: current player data used at prediction time.
- `data/model.json`: logistic model coefficients.

The included snapshot is a demo seed. Before sharing serious predictions, refresh it from public tennis sites:

```bash
python scripts/refresh_snapshot.py
```

Validate freshness and required fields with:

```bash
python scripts/validate_snapshot.py
```

The repository also includes `.github/workflows/refresh-snapshot.yml`, which refreshes the snapshot every Monday and Thursday after the project is on GitHub. The snapshot keeps the active WTA top 250 by current ranking. If the repo is connected to Vercel, that commit can redeploy the static site.

## Model Evaluation

The current model comparison is stored in `data/model_comparison.json`.

- Candidate A: train 2010-2022, tune on 2023, test on 2024-2025.
- Candidate B: train 2010-2020, tune on 2021-2023, test on 2024-2025.

Both reached 66.35% accuracy on the 2024-2025 holdout set. Candidate B is selected because its log loss and Brier score were slightly better.

## Training Plan

The intended training source is historical WTA match data, such as Jeff Sackmann's non-commercial tennis datasets. The training table should contain one row per pre-match player pair with:

- player ranking difference
- overall Elo difference
- surface Elo difference
- recent-form difference
- current pre-match H2H difference
- age/experience signals
- match result

Important rule: every feature must be computed from data available before that match date. No future leakage.

## Public Data Snapshot Plan

For V1, current data should be refreshed into `data/snapshot.json` from public pages or curated CSV exports. The site itself reads the local snapshot, which makes the prediction flow stable and avoids paid API keys.

The snapshot is not meant to be permanent. `meta.updatedAt` and `meta.maxAgeDays` define whether the prediction data is fresh enough. The app warns when the active snapshot is stale. The product limit is 15 days; during Grand Slams or WTA 1000 weeks, refresh more often.

Suggested public sources:

- Tennis Abstract WTA Elo reports
- WTA public ranking pages
- Curated historical match CSVs for recent form and H2H

Long-term refresh options:

- Manual: update `data/snapshot.json` before sharing a new event.
- Semi-automatic: run a local script that reads exported CSVs from public sources and rewrites `data/snapshot.json`.
- Automatic: add a Vercel Cron job or GitHub Action that refreshes the snapshot on a schedule and redeploys the site.

See `docs/data-refresh.md` for the snapshot freshness rule.

## Deployment

This app is static. Vercel can deploy it directly from the repository root.

## Attribution

If Jeff Sackmann data is used for training or snapshots, keep the required attribution and non-commercial usage note visible in documentation or the app footer.

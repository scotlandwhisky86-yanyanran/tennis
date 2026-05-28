# Current Snapshot Refresh

The product should never rely on a frozen player snapshot. The app reads a local `data/snapshot.json`, but that file represents the latest public-data snapshot, not a permanent database.

## Freshness Rule

`data/snapshot.json` includes:

```json
{
  "meta": {
    "updatedAt": "2026-05-26",
    "maxAgeDays": 15
  }
}
```

If the file is older than `maxAgeDays`, the app shows a stale-data warning. The validator also fails. The product limit is 15 days:

```bash
python scripts/validate_snapshot.py
```

## Refresh Strategy

V1 should use public sources and normalize them into `data/snapshot.json`.

Recommended fields:

- `rank`: current WTA singles ranking
- `elo`: current overall Elo
- `hardElo`, `clayElo`, `grassElo`: current surface Elo
- `recent`: wins/losses over a chosen recent window, such as last 10 matches or last 90 days
- `h2h`: current cumulative head-to-head against known opponents

The default snapshot keeps the active WTA top 250 by current ranking:

```bash
python scripts/refresh_snapshot.py --player-limit 250
```

Recommended public-source workflow:

1. Pull or export current Elo/surface Elo from Tennis Abstract reports.
2. Pull or export current WTA rankings from public ranking pages.
3. Use recent match CSVs to calculate last-10 or last-90-days form.
4. Use the same recent/historical match rows to calculate current H2H.
5. Write `data/snapshot.json`.
6. Run `python scripts/validate_snapshot.py`.

## Automation Path

The product uses `scripts/refresh_snapshot.py` to pull public tennis data directly:

```bash
python scripts/refresh_snapshot.py
python scripts/validate_snapshot.py
```

On Windows, the local one-command refresh is:

```powershell
.\scripts\refresh_local.ps1
```

The script currently reads:

- Tennis Abstract WTA Elo report for current ranking, age, Elo, and surface Elo
- Jeff Sackmann WTA match CSVs for recent form and head-to-head

For hosted refresh, `.github/workflows/refresh-snapshot.yml` runs every Monday and Thursday and commits a refreshed `data/snapshot.json`. When this repository is connected to Vercel, that commit can trigger redeployment automatically.

For a more reliable hosted version, add one of:

- a Vercel Cron endpoint that refreshes data and stores the JSON in Blob storage
- a local refresh command that you run before each tournament

The important product rule is simple: prediction uses a current snapshot, and snapshots older than 15 days fail validation.

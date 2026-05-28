# Detailed Feature Dataset

`scripts/build_detailed_training_table.py` builds a richer historical feature table for the next model iteration.

Default command:

```bash
python scripts/build_detailed_training_table.py
```

Output:

```text
data/training_rows_detailed.csv
```

## Scope

- Output rows: WTA 500 and above, 2010-2025.
- State updates: the same 2010-2025 WTA 500+ main-draw singles rows.
- Feature timing: every row is created before the match is used to update player state.

## Feature Groups

- Ranking: rank, rank points, missing flags.
- Elo: overall Elo, hard Elo, clay Elo, grass Elo, active-surface Elo.
- Surface records: career surface win rate and sample size.
- Recent form: last 5, last 10, last 20, surface recent form, and 30/60/90-day workload.
- H2H: overall H2H and active-surface H2H.
- Style: ace rate, double-fault rate, first-serve rate, first/second-serve points won, service points won, return points won, break-point save/convert rates, tiebreak win rate.
- Physical/context: age, height, handedness, draw size, surface flags, tournament level.

## Limits

Public historical CSVs do not include reliable injury, fatigue, coach, weather, ball type, or granular court-speed data. The current "court condition" fields are surface, level, draw size, and active-surface records.

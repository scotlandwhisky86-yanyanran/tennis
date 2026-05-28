python scripts\refresh_snapshot.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\validate_snapshot.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -c "import json; s=json.load(open('data/snapshot.json',encoding='utf-8')); print(f\"Snapshot {s['meta']['updatedAt']} generated {s['meta'].get('generatedAt')} with {len(s['players'])} players\")"

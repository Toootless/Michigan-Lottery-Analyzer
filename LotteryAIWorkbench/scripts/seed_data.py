"""Seed placeholder synthetic data artifacts.

Currently this is a stub because the app generates synthetic frequencies in-memory.
Future implementation may:
- Pull real historical draw data.
- Normalize schemas.
- Persist frequency aggregates to `data/` directory.
"""
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "src" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sample_payload = {
    "game": "Powerball",
    "generated": True,
    "note": "Synthetic scaffold seed; replace with real ingestion pipeline.",
}

out_file = DATA_DIR / "seed_sample.json"
out_file.write_text(json.dumps(sample_payload, indent=2))
print(f"Wrote {out_file}")

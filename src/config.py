from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DB_PATH = DATA_DIR / "nfl_pbp.db"

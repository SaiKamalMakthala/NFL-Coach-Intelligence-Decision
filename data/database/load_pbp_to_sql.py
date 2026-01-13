import pandas as pd
import sqlite3
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/pbp_field_position.csv")
DB_PATH = Path("data/database/nfl_pbp.db")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Connect to SQLite
# -----------------------------
conn = sqlite3.connect(DB_PATH)

# -----------------------------
# Write to SQL
# -----------------------------
df.to_sql(
    name="pbp_field_position",
    con=conn,
    if_exists="replace",
    index=False
)

conn.close()

print("NFL play-by-play data successfully stored in SQLite.")

import sqlite3
import numpy as np
import pandas as pd

from src.config import DB_PATH

def main():
    conn = sqlite3.connect(DB_PATH)

    # Load predictions (must exist from Day 16)
    df = pd.read_sql("SELECT * FROM fourth_down_predictions", conn)

    # Ensure numeric
    for c in ["ydstogo", "yardline_100", "go_probability"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ydstogo", "yardline_100"])
    if "go_probability" not in df.columns:
        # if you don't have go_probability saved, create a placeholder baseline
        df["go_probability"] = np.clip(0.55 - 0.03*df["ydstogo"] + 0.002*(50 - df["yardline_100"]), 0, 1)

    # FG range rule (rough): opponent 35-yard line or closer => yardline_100 <= 35
    in_fg_range = df["yardline_100"] <= 35

    # Punt territory: backed up deep
    backed_up = df["yardline_100"] >= 80

    # Aggressive go rules (simple but realistic)
    short = df["ydstogo"] <= 2
    medium = (df["ydstogo"] > 2) & (df["ydstogo"] <= 5)

    # Decision policy:
    # - If backed up: Punt
    # - If FG range: FG unless strong GO probability + short
    # - Else: Go if high go_prob and short/medium, else Punt/Lean Punt
    df["recommended_action"] = "PUNT"

    df.loc[backed_up, "recommended_action"] = "PUNT"

    df.loc[in_fg_range, "recommended_action"] = "FIELD_GOAL"
    df.loc[in_fg_range & short & (df["go_probability"] >= 0.60), "recommended_action"] = "GO"

    df.loc[~in_fg_range & ~backed_up & short & (df["go_probability"] >= 0.55), "recommended_action"] = "GO"
    df.loc[~in_fg_range & ~backed_up & medium & (df["go_probability"] >= 0.62), "recommended_action"] = "GO"

    # If not clear GO, but also not deep punt, call it lean punt (optional)
    df.loc[~in_fg_range & ~backed_up & (df["recommended_action"] == "PUNT"), "recommended_action"] = "LEAN_PUNT"

    keep_cols = [c for c in [
        "game_id", "play_id", "posteam", "defteam",
        "down", "ydstogo", "yardline_100",
        "go_probability", "recommended_action"
    ] if c in df.columns]

    out = df[keep_cols].copy()

    out.to_sql("fourth_down_recommendations", conn, if_exists="replace", index=False)
    conn.close()

    print("✅ Rebuilt fourth_down_recommendations")
    print(out["recommended_action"].value_counts())

if __name__ == "__main__":
    main()

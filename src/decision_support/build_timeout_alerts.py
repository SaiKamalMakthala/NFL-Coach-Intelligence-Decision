import sqlite3
import pandas as pd
from src.config import DB_PATH
from src.utils.log import log


def main():
    log("Day 23: Timeout alerts (rule-based baseline)")

    conn = sqlite3.connect(DB_PATH)
    # Use latest feature table; if you keep updating, change to feature_table_5/6 as needed.
    df = pd.read_sql("SELECT * FROM feature_table_5", conn)
    conn.close()

    # If score differential exists, use it; else create a placeholder
    if "score_differential" not in df.columns:
        df["score_differential"] = 0

    # Late game proxy: if you have clock_pressure_norm or clock_pressure, use it
    clock_col = "clock_pressure_norm" if "clock_pressure_norm" in df.columns else ("clock_pressure" if "clock_pressure" in df.columns else None)
    if clock_col is None:
        df["clock_proxy"] = 0.0
        clock_col = "clock_proxy"

    # Alert rule:
    # - high clock pressure
    # - close score
    # - high pressure context or late down
    df["close_game"] = df["score_differential"].abs() <= 7
    df["high_clock_pressure"] = df[clock_col] >= 0.7
    df["late_down"] = df["down"].isin([3, 4])

    df["timeout_alert"] = (
        df["close_game"] &
        df["high_clock_pressure"] &
        (df.get("pressure_bucket", "low").astype(str) == "high") &
        df["late_down"]
    ).astype(int)

    alerts = df[df["timeout_alert"] == 1].copy()

    # Keep a small alert table for UI later
    keep_cols = [c for c in ["game_id", "play_id", "posteam", "defteam", "down", "ydstogo", "yardline_100", "pressure_score"] if c in alerts.columns]
    alerts = alerts[keep_cols].head(500)

    conn = sqlite3.connect(DB_PATH)
    alerts.to_sql("timeout_alerts", conn, if_exists="replace", index=False)
    conn.close()

    log(f"Saved → timeout_alerts  (rows: {len(alerts)})")


if __name__ == "__main__":
    main()

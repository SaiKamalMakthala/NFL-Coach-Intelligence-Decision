import sqlite3
import numpy as np
import pandas as pd
from src.config import DB_PATH
from src.utils.log import log


def load_pbp_from_sqlite():
    """
    Use your existing engineered table as the source of EPA samples.
    This avoids relying on CSV paths.
    """
    conn = sqlite3.connect(DB_PATH)

    # Prefer the most recent table that contains EPA + play_type
    # Adjust if your latest is feature_table_6 later.
    table_candidates = ["feature_table_5", "feature_table_4", "feature_table_3"]

    for t in table_candidates:
        try:
            df = pd.read_sql(f"SELECT down, ydstogo, yardline_100, play_type, epa FROM {t}", conn)
            conn.close()
            return df, t
        except Exception:
            continue

    conn.close()
    raise ValueError(
        "Could not load pbp from SQLite. None of the expected feature tables exist "
        "(feature_table_5/4/3) with required columns."
    )


def sample_endgame_total_epa(pbp, n_plays=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    draws = rng.choice(pbp["epa"].values, size=n_plays, replace=True)
    return float(draws.sum())


def main():
    log("Day 24: End-game simulations (simplified Monte Carlo horizon)")

    pbp, source_table = load_pbp_from_sqlite()
    log(f"Loaded EPA samples from SQLite table: {source_table} (rows={len(pbp)})")

    # Clean + filter to realistic offensive plays
    pbp["play_type"] = pbp["play_type"].fillna("other").astype(str).str.lower()
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    pbp["epa"] = pd.to_numeric(pbp["epa"], errors="coerce")
    pbp = pbp.dropna(subset=["epa"])

    if len(pbp) < 500:
        raise ValueError(f"Not enough EPA rows after filtering. Remaining rows: {len(pbp)}")

    # Simulation settings
    n_sims = 10000
    n_plays = 5
    rng = np.random.default_rng(42)

    totals = np.array([sample_endgame_total_epa(pbp, n_plays=n_plays, rng=rng) for _ in range(n_sims)])

    out = pd.DataFrame([{
        "source_table": source_table,
        "n_sims": n_sims,
        "n_plays_horizon": n_plays,
        "expected_total_epa": float(np.mean(totals)),
        "p05_total_epa": float(np.quantile(totals, 0.05)),
        "p50_total_epa": float(np.quantile(totals, 0.50)),
        "p95_total_epa": float(np.quantile(totals, 0.95)),
        "prob_negative_total_epa": float(np.mean(totals < 0)),
        "std_total_epa": float(np.std(totals))
    }])

    print(out)

    conn = sqlite3.connect(DB_PATH)
    out.to_sql("endgame_simulation_summary", conn, if_exists="replace", index=False)
    conn.close()

    log("Saved → endgame_simulation_summary ")
    log("Day 24 complete.")


if __name__ == "__main__":
    main()

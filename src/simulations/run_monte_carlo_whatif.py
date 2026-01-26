import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DB_PATH
from src.utils.log import log


def find_pbp_source():
    candidates = [
        Path("data/processed/pbp_clean.csv"),
        Path("data/processed/pbp.csv"),
        Path("data/raw/nfl_pbp_2023_week1.csv"),
        Path("data/raw/nfl_pbp_2023.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find a pbp CSV in data/raw or data/processed.")


def load_pbp_for_sampling():
    pbp_path = find_pbp_source()
    log(f"Loading pbp for sampling from: {pbp_path}")

    # Keep minimal columns
    cols = ["down", "ydstogo", "yardline_100", "play_type", "epa"]
    pbp = pd.read_csv(pbp_path, usecols=lambda c: c in cols)

    pbp["play_type"] = pbp["play_type"].fillna("other").astype(str).str.lower()
    pbp = pbp[pbp["play_type"].isin(["pass", "run", "punt", "field_goal"])].copy()
    pbp = pbp.dropna(subset=["epa", "ydstogo", "yardline_100", "down"])
    return pbp


def bucket_context(df):
    # Context buckets: helps build conditional EPA distributions
    df["ydstogo_bucket"] = pd.cut(df["ydstogo"], bins=[-0.1, 2, 5, 10, 20, 100],
                                  labels=["short", "medium", "long", "very_long", "extreme"])
    df["field_bucket"] = pd.cut(df["yardline_100"], bins=[-0.1, 20, 50, 80, 100],
                                labels=["scoring_range", "midfield", "backed_up", "goal_line"])
    return df


def get_epa_distribution(pbp, play_type, ydstogo, yardline_100):
    # Choose bucket
    tmp = pbp.copy()
    tmp = bucket_context(tmp)

    y_bucket = pd.cut([ydstogo], bins=[-0.1, 2, 5, 10, 20, 100],
                      labels=["short", "medium", "long", "very_long", "extreme"])[0]
    f_bucket = pd.cut([yardline_100], bins=[-0.1, 20, 50, 80, 100],
                      labels=["scoring_range", "midfield", "backed_up", "goal_line"])[0]

    subset = tmp[
        (tmp["play_type"] == play_type) &
        (tmp["ydstogo_bucket"] == y_bucket) &
        (tmp["field_bucket"] == f_bucket)
    ]["epa"].values

    # Fallback: if bucket too small, relax the filter
    if len(subset) < 50:
        subset = tmp[tmp["play_type"] == play_type]["epa"].values

    return subset


def simulate_strategy(epa_samples, n_sims=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    draws = rng.choice(epa_samples, size=n_sims, replace=True)
    summary = {
        "expected_epa": float(np.mean(draws)),
        "p05_epa": float(np.quantile(draws, 0.05)),
        "p50_epa": float(np.quantile(draws, 0.50)),
        "p95_epa": float(np.quantile(draws, 0.95)),
        "prob_negative_epa": float(np.mean(draws < 0)),
        "std_epa": float(np.std(draws))
    }
    return summary


def save_results_to_sql(df, table_name="monte_carlo_whatif_results"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def main():
    log("Monte Carlo What-If Simulation")

   
    # 1) Load pbp for sampling EPA distributions
    
    pbp = load_pbp_for_sampling()
    log(f"PBP rows available for sampling: {len(pbp)}")

   
    # 2) Choose an example scenario (you can change these later)
    
    # Example: 4th & 3 at opponent 45 (yardline_100 = 55 means 55 yards from end zone)
    scenario = {
        "down": 4,
        "ydstogo": 3,
        "yardline_100": 55
    }
    log(f"Scenario: {scenario}")

    
    # 3) Build EPA distributions for each strategy
   
    rng = np.random.default_rng(42)

    # GO (we’ll approximate using pass+run as separate strategies)
    pass_epa = get_epa_distribution(pbp, "pass", scenario["ydstogo"], scenario["yardline_100"])
    run_epa = get_epa_distribution(pbp, "run", scenario["ydstogo"], scenario["yardline_100"])

    # NO-GO options
    punt_epa = get_epa_distribution(pbp, "punt", scenario["ydstogo"], scenario["yardline_100"])
    fg_epa = get_epa_distribution(pbp, "field_goal", scenario["ydstogo"], scenario["yardline_100"])

    
    # 4) Monte Carlo simulate each option
    
    results = []

    for name, samples in [
        ("GO_PASS", pass_epa),
        ("GO_RUN", run_epa),
        ("PUNT", punt_epa),
        ("FIELD_GOAL", fg_epa)
    ]:
        sim = simulate_strategy(samples, n_sims=5000, rng=rng)
        sim_row = {**scenario, "strategy": name, **sim, "n_samples_pool": int(len(samples))}
        results.append(sim_row)

    out = pd.DataFrame(results)
    log("Simulation complete. Results:")
    print(out)

    
    # 5) Save to SQLite
   
    save_results_to_sql(out)
    log("Saved → monte_carlo_whatif_results ")
   


if __name__ == "__main__":
    main()

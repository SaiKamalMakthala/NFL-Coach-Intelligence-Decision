import sqlite3
import pandas as pd
from src.config import DB_PATH
from src.utils.log import log


def load_table(name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df


def main():
    log("Building decision classifier (recommendation layer)")

    preds = load_table("fourth_down_predictions")
    sims = load_table("monte_carlo_whatif_results")

    # Use Monte Carlo expected EPA by strategy for the current scenario.
    # If you ran Day 17 with a single scenario, sims is small and easy.
    strat_value = sims.set_index("strategy")["expected_epa"].to_dict()

    # Map strategy expected EPA
    exp_go_pass = strat_value.get("GO_PASS", 0.0)
    exp_go_run = strat_value.get("GO_RUN", 0.0)
    exp_punt = strat_value.get("PUNT", 0.0)
    exp_fg = strat_value.get("FIELD_GOAL", 0.0)

    # Choose "GO value" as best of pass/run
    exp_go = max(exp_go_pass, exp_go_run)

    # Build a simple decision rule:
    # - If go_probability is high AND expected value beats punt/FG by margin -> recommend GO
    # - else recommend best of PUNT/FG by expected EPA
    margin = 0.05  # small threshold to avoid flip-flopping

    preds["exp_go_epa"] = exp_go
    preds["exp_punt_epa"] = exp_punt
    preds["exp_fg_epa"] = exp_fg

    # Pick best no-go option
    preds["best_nogo_action"] = preds.apply(
        lambda r: "FIELD_GOAL" if r["exp_fg_epa"] >= r["exp_punt_epa"] else "PUNT",
        axis=1
    )
    preds["best_nogo_epa"] = preds.apply(
        lambda r: r["exp_fg_epa"] if r["best_nogo_action"] == "FIELD_GOAL" else r["exp_punt_epa"],
        axis=1
    )

    preds["recommended_action"] = preds.apply(
        lambda r: "GO"
        if (r["go_probability"] >= 0.55 and (r["exp_go_epa"] - r["best_nogo_epa"]) >= margin)
        else r["best_nogo_action"],
        axis=1
    )

    # Save recommendations
    conn = sqlite3.connect(DB_PATH)
    preds.to_sql("fourth_down_recommendations", conn, if_exists="replace", index=False)
    conn.close()

    log("Saved → fourth_down_recommendations ")


if __name__ == "__main__":
    main()

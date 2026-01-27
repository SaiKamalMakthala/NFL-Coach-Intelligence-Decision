import sqlite3
import pandas as pd
from src.config import DB_PATH
from src.utils.log import log


def load(name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df


def main():
    log("Day 22: Generating coach insights (narrative)")

    tend = load("playcalling_tendencies")
    recs = load("fourth_down_recommendations")

    # Example insights:
    # 1) Teams that pass a lot in high pressure on 3rd down
    high_pressure_pass = tend[
        (tend["pressure_bucket"] == "high") &
        (tend["down"] == 3)
    ].sort_values("pass_rate", ascending=False).head(10)

    # 2) Aggregate recommendation distribution
    rec_summary = recs["recommended_action"].value_counts(dropna=False).reset_index()
    rec_summary.columns = ["recommended_action", "count"]

    # Build narrative strings
    insights = []
    insights.append("Top teams by 3rd-down high-pressure pass tendency:")
    for _, r in high_pressure_pass.iterrows():
        insights.append(
            f"- {r['posteam']}: pass_rate={r['pass_rate']:.2f}, success_rate={r['success_rate']:.2f}, avg_epa={r['avg_epa']:.2f}"
        )

    insights.append("\n4th-down recommendation mix across evaluated plays:")
    for _, r in rec_summary.iterrows():
        insights.append(f"- {r['recommended_action']}: {int(r['count'])}")

    out = pd.DataFrame({"insight": insights})

    conn = sqlite3.connect(DB_PATH)
    out.to_sql("coach_insights", conn, if_exists="replace", index=False)
    conn.close()

    log("Saved → coach_insights")
    print("\n".join(insights))


if __name__ == "__main__":
    main()

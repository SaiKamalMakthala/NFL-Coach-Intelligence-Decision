import sqlite3
import pandas as pd
from datetime import datetime
from src.config import DB_PATH
from src.utils.log import log


def load(table: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def main():
    log("Day 25: Generating narrative summaries")

    # Load what exists (skip gracefully if missing)
    tables = []
    conn = sqlite3.connect(DB_PATH)
    tnames = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    conn.close()

    summaries = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "fourth_down_recommendations" in tnames:
        recs = load("fourth_down_recommendations")

        # Overall recommendation mix
        mix = recs["recommended_action"].value_counts(dropna=False).to_dict()
        summaries.append(f"[{ts}] 4th-down Recommendation Mix: {mix}")

        # Top GO calls (highest go_probability)
        if "go_probability" in recs.columns:
            top_go = recs.sort_values("go_probability", ascending=False).head(5)
            for _, r in top_go.iterrows():
                summaries.append(
                    f"- High-confidence GO: game={r.get('game_id')} play={r.get('play_id')} "
                    f"team={r.get('posteam')} ydstogo={r.get('ydstogo', 'NA')} "
                    f"yardline_100={r.get('yardline_100', 'NA')} go_prob={r.get('go_probability', 0):.2f}"
                )

    if "coach_insights" in tnames:
        ci = load("coach_insights")
        summaries.append("Coach Insights Highlights:")
        for s in ci["insight"].dropna().head(15).tolist():
            summaries.append(f"- {s}")

    if "fourth_down_feature_importance" in tnames:
        imp = load("fourth_down_feature_importance").sort_values("importance_mean", ascending=False).head(5)
        summaries.append("Top drivers of 4th-down aggressiveness (permutation importance):")
        for _, r in imp.iterrows():
            summaries.append(f"- {r['feature']}: importance={r['importance_mean']:.4f} ± {r['importance_std']:.4f}")

    out = pd.DataFrame({"summary_line": summaries})

    conn = sqlite3.connect(DB_PATH)
    out.to_sql("narrative_summaries", conn, if_exists="replace", index=False)
    conn.close()

    log("Saved → narrative_summaries ")
    print("\n".join(summaries))


if __name__ == "__main__":
    main()

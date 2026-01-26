import sqlite3
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.config import DB_PATH
from src.utils.log import log


def load(name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df


def pick_col(df, candidates):
    """Return the first existing column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    log("Day 19: Validating 4th-down model (accuracy & bias slices)")

    df = load("fourth_down_predictions")
    log(f"Columns in fourth_down_predictions: {df.columns.tolist()}")

    # -----------------------------
    # 1) Basic metrics
    # -----------------------------
    y_true = df["actual_go"].astype(int)
    y_prob = df["go_probability"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    log(f"ROC AUC: {auc:.4f}")
    log(f"Confusion matrix:\n{cm}")

    # -----------------------------
    # 2) Slice metrics (robust)
    # -----------------------------
    ydstogo_col = pick_col(df, ["ydstogo", "yards_to_go", "to_go"])
    yardline_col = pick_col(df, ["yardline_100", "yardline", "yardline100"])

    if ydstogo_col is None or yardline_col is None:
        log("⚠️ Skipping slice report: missing ydstogo and/or yardline columns in fourth_down_predictions.")
        log(f"Found ydstogo_col={ydstogo_col}, yardline_col={yardline_col}")
        log("Fix option: include ydstogo and yardline_100 when saving Day 16 predictions.")
        return

    df[ydstogo_col] = pd.to_numeric(df[ydstogo_col], errors="coerce")
    df[yardline_col] = pd.to_numeric(df[yardline_col], errors="coerce")

    df["ydstogo_bucket"] = pd.cut(
        df[ydstogo_col],
        bins=[-0.1, 2, 5, 10, 20, 100],
        labels=["short", "medium", "long", "very_long", "extreme"]
    )

    df["field_bucket"] = pd.cut(
        df[yardline_col],
        bins=[-0.1, 20, 50, 80, 100],
        labels=["scoring_range", "midfield", "backed_up", "goal_line"]
    )

    slice_report = df.groupby(["ydstogo_bucket", "field_bucket"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "go_rate_true": float(g["actual_go"].mean()),
            "go_rate_pred": float((g["go_probability"] >= 0.5).mean()),
            "avg_go_prob": float(g["go_probability"].mean())
        })
    ).reset_index()

    conn = sqlite3.connect(DB_PATH)
    slice_report.to_sql("fourth_down_validation_slices", conn, if_exists="replace", index=False)
    conn.close()

    log("Saved → fourth_down_validation_slices")
    print(slice_report.head(30))


if __name__ == "__main__":
    main()

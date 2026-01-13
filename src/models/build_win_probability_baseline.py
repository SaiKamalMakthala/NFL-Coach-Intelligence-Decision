import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.config import DB_PATH


FEATURES = [
    "down",
    "ydstogo",
    "yardline_100",
    "is_red_zone",
    "is_goal_to_go",
    "is_backed_up",
    "field_position_score"
]


def load_feature_table():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feature_table", conn)
    conn.close()
    return df


def build_target(df):
    df["win_proxy"] = (df["epa"] > 0).astype(int)
    return df


def train_model(df):
    X = df[FEATURES]
    y = df["win_proxy"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    print(f"Baseline Win Probability AUC: {auc:.3f}")

    return model, preds


def save_outputs(df, preds):
    df["win_probability_baseline"] = preds

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("win_probability_baseline", conn, if_exists="replace", index=False)
    conn.close()


def main():
    print("Loading feature table...")
    df = load_feature_table()

    print("Building win proxy target...")
    df = build_target(df)

    print("Training baseline model...")
    model, preds = train_model(df)

    print("Saving baseline probabilities...")
    save_outputs(df, preds)

    print("✅ Day 8 complete — baseline WP model built.")


if __name__ == "__main__":
    main()

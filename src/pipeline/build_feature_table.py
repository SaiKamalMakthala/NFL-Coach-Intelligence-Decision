import pandas as pd
import sqlite3
from src.config import PROCESSED_DATA_DIR, DB_PATH


def load_features():
    path = PROCESSED_DATA_DIR / "pbp_field_position.csv"
    df = pd.read_csv(path)
    return df


def build_feature_table(df):
    # Explicit feature selection (no accidental columns)
    feature_cols = [
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
        "epa",
        "success",
        "distance_bucket",
        "down_distance",
        "is_third_down",
        "is_fourth_down",
        "is_long_yardage",
        "field_zone",
        "is_red_zone",
        "is_goal_to_go",
        "is_backed_up",
        "field_position_score"
    ]

    return df[feature_cols]


def write_to_sql(df):
    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        "feature_table",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()


def verify_write():
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )
    conn.close()
    print("Tables in DB:", tables["name"].tolist())


def main():
    print("Loading processed features...")
    df = load_features()

    print("Building feature table...")
    feature_table = build_feature_table(df)

    print(f"Rows: {len(feature_table)} | Columns: {len(feature_table.columns)}")

    print("Writing to SQLite...")
    write_to_sql(feature_table)

    verify_write()
    print("✅ Feature table built successfully.")


if __name__ == "__main__":
    main()

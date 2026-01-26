import pandas as pd
from src.utils.db import read_table, write_table
from src.utils.log import log
from src.config import LATEST_FEATURE_TABLE


def normalize_play_type(df):
    df["play_type_clean"] = df["play_type"].fillna("other").str.lower()
    df.loc[df["play_type_clean"].str.contains("pass"), "play_type_clean"] = "pass"
    df.loc[df["play_type_clean"].str.contains("run"), "play_type_clean"] = "run"
    df.loc[~df["play_type_clean"].isin(["pass", "run"]), "play_type_clean"] = "other"
    return df


def add_flags(df):
    df["is_pass"] = (df["play_type_clean"] == "pass").astype(int)
    df["is_run"] = (df["play_type_clean"] == "run").astype(int)
    return df


def build_tendencies(df):
    group_cols = ["posteam", "down", "distance_bucket", "field_zone", "pressure_bucket"]

    return df.groupby(group_cols).agg(
        plays=("play_id", "count"),
        avg_ydstogo=("ydstogo", "mean"),
        avg_epa=("epa", "mean"),
        success_rate=("success", "mean"),
        pass_rate=("is_pass", "mean"),
        run_rate=("is_run", "mean")
    ).reset_index()


def main():
    log("Starting Day 14 refactor demo: Day 12 using utils")

    df = read_table(LATEST_FEATURE_TABLE)
    log(f"Loaded {len(df)} rows from {LATEST_FEATURE_TABLE}")

    df = normalize_play_type(df)
    df = add_flags(df)

    tendencies = build_tendencies(df)
    log(f"Built tendencies rows: {len(tendencies)}")

    write_table(tendencies, "playcalling_tendencies")
    log("Saved playcalling_tendencies to SQLite ")


if __name__ == "__main__":
    main()

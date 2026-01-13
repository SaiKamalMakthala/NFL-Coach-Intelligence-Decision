import pandas as pd
from pathlib import Path

# Paths
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load all weekly CSVs
pbp_files = list(RAW_DIR.glob("nfl_pbp_2023_week*.csv"))

df_list = [pd.read_csv(file) for file in pbp_files]
pbp_df = pd.concat(df_list, ignore_index=True)

print("PBP shape:", pbp_df.shape)
pbp_df.head()

#Creating Game level table
game_cols = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score"
]

games_df = (
    pbp_df[game_cols]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Games table shape:", games_df.shape)
games_df.head()

#Handling missing values
games_df[["home_score", "away_score"]] = (
    games_df[["home_score", "away_score"]]
    .fillna(0)
)
#fixing data types
games_df["home_score"] = games_df["home_score"].astype(int)
games_df["away_score"] = games_df["away_score"].astype(int)
games_df["week"] = games_df["week"].astype(int)
games_df["season"] = games_df["season"].astype(int)  #Prevents silent bugs later in ML models

#Adding winner column  (Feature Engineering)
def get_winner(row):
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    elif row["away_score"] > row["home_score"]:
        return row["away_team"]
    else:
        return "TIE"

games_df["winner"] = games_df.apply(get_winner, axis=1)
#save table
output_path = PROCESSED_DIR / "games_cleaned.csv"
games_df.to_csv(output_path, index=False)

print(f"Saved cleaned game table to {output_path}")

games_df.info()
games_df.isna().sum()
games_df["week"].value_counts().sort_index()


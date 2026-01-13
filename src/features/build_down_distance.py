import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.load_data import load_pbp_data

# ... rest of your code

pbp_df = load_pbp_data()

cols = [
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "play_type",
    "epa",
    "success"
]

pbp_df = pbp_df[cols]

#Cleaning invalid plays
#Removes penalties, timeouts, kneels
#Focuses on decision plays
#Cleaner signal for ML
pbp_df = pbp_df.dropna(subset=["down", "ydstogo"])
pbp_df = pbp_df[pbp_df["play_type"].isin(["run", "pass"])]

#Create Distance Buckets (Key Feature Engineering)
#Instead of raw yards:
#Humans think in short / medium / long
#Models learn patterns faster
#Easier interpretability
def distance_bucket(yards):
    if yards <= 3:
        return "short"
    elif yards <= 7:
        return "medium"
    else:
        return "long"

pbp_df["distance_bucket"] = pbp_df["ydstogo"].apply(distance_bucket)

#Create down + distance situation label
pbp_df["down_distance"] = (
    pbp_df["down"].astype(int).astype(str)
    + " & "
    + pbp_df["distance_bucket"]
)
#High-leverage situations
pbp_df["is_third_down"] = pbp_df["down"] == 3
pbp_df["is_fourth_down"] = pbp_df["down"] == 4
pbp_df["is_long_yardage"] = pbp_df["ydstogo"] >= 7

from pathlib import Path

OUTPUT_PATH = Path("data/processed/pbp_down_distance.csv")
pbp_df.to_csv(OUTPUT_PATH, index=False)

import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/pbp_down_distance.csv")
df = pd.read_csv(INPUT_PATH)
#NFL convention:
#yardline_100 = 100 → offense at own goal line
#yardline_100 = 0 → touchdown
#Smaller number = closer to scoring
#Larger number = backed up

#Creating field zones
def field_zone(yardline):
    if yardline <= 20:
        return "red_zone"
    elif yardline <= 40:
        return "scoring_range"
    elif yardline <= 60:
        return "midfield"
    elif yardline <= 80:
        return "own_territory"
    else:
        return "backed_up"

df["field_zone"] = df["yardline_100"].apply(field_zone)

#Binary scoring context flags
df["is_red_zone"] = df["yardline_100"] <= 20
df["is_goal_to_go"] = (df["yardline_100"] <= 10) & (df["ydstogo"] <= df["yardline_100"])
df["is_backed_up"] = df["yardline_100"] >= 90

#Field Position Leverage Score
df["field_position_score"] = 1 - (df["yardline_100"] / 100)

OUTPUT_PATH = Path("data/processed/pbp_field_position.csv")
df.to_csv(OUTPUT_PATH, index=False)

import pandas as pd

df = pd.read_csv("data/processed/pbp_field_position.csv")

#Down pressure
df["down_load"] = df["down"] / 4

#Distance pressure
distance_map = {
    "short": 0.2,
    "medium": 0.5,
    "long": 0.8
}

df["distance_load"] = df["distance_bucket"].map(distance_map)

#Field position pressure
df["field_load"] = df["field_position_score"]

#Situational flags (binary cognitive stressors)
df["red_zone_load"] = df["is_red_zone"] * 0.15
df["goal_to_go_load"] = df["is_goal_to_go"] * 0.20
df["backed_up_load"] = df["is_backed_up"] * 0.20
df["long_yardage_load"] = df["is_long_yardage"] * 0.15

#Combine into Snap Load Proxy
df["snap_load_proxy"] = (
    0.30 * df["down_load"] +
    0.25 * df["distance_load"] +
    0.25 * df["field_load"] +
    df["red_zone_load"] +
    df["goal_to_go_load"] +
    df["backed_up_load"] +
    df["long_yardage_load"]
)

#Clamp values (safety)
df["snap_load_proxy"] = df["snap_load_proxy"].clip(0, 1)

df.to_csv("data/processed/pbp_snap_load.csv", index=False)

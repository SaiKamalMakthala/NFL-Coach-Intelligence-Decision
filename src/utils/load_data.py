import pandas as pd
from pathlib import Path
#Modular loading
#Reusable across future weeks
RAW_DIR = Path("data/raw")

def load_pbp_data():
    files = list(RAW_DIR.glob("nfl_pbp_2023_week*.csv"))
    df_list = [pd.read_csv(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

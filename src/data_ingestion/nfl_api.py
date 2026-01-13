import nflreadpy as nfl
import pandas as pd
from pathlib import Path
from .utils import save_csv, create_dir
import time

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def fetch_pbp(season: int, week: int) -> pd.DataFrame:
    """
    Fetch NFL play-by-play data for a season using nflreadpy,
    then filter by week.
    """
    try:
        print(f"Loading play-by-play data for season {season}...")
        pbp = nfl.load_pbp(season)  # loads season data
        df = pbp.to_pandas()       # convert to pandas
    except Exception as e:
        print(f"Failed to load nflreadpy pbp data: {e}")
        return pd.DataFrame()

    # Filter for the given week
    df_week = df[df['week'] == week]

    if df_week.empty:
        print(f"No data found for week {week} of season {season}")
        return df_week

    # Save CSV
    create_dir(RAW_DATA_DIR)
    file_path = RAW_DATA_DIR / f"nfl_pbp_{season}_week{week}.csv"
    save_csv(df_week, file_path)

    return df_week

def fetch_season_pbp(season: int, max_weeks: int = 17, sleep_sec: int = 1):
    """
    Fetch and save pbp data for each week in a season.
    """
    for week in range(1, max_weeks + 1):
        print(f"Fetching data for Season {season}, Week {week}...")
        df_week = fetch_pbp(season, week)
        time.sleep(sleep_sec)

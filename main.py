from src.data_ingestion.nfl_api import fetch_season_pbp

if __name__ == "__main__":
    SEASON = 2023
    MAX_WEEKS = 17           # regular NFL season
    SLEEP_SECONDS = 1

    fetch_season_pbp(SEASON, max_weeks=MAX_WEEKS, sleep_sec=SLEEP_SECONDS)

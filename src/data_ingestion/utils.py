import pandas as pd
from pathlib import Path

def save_csv(df: pd.DataFrame, file_path: str):
    """Save DataFrame to CSV"""
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV into DataFrame"""
    return pd.read_csv(file_path)

def create_dir(path: str):
    """Create directory if it does not exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

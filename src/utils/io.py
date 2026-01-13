import pandas as pd

def load_csv(path):
    """Load CSV safely into DataFrame."""
    return pd.read_csv(path)

def save_csv(df, path):
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)

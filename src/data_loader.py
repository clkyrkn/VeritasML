import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the CSV file and returns as a DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['text', 'label'])  # Clear missing lines
    df = df[df['label'].isin([0, 1])]         # Drop only valid tags
    df = df[['text', 'label']]                # Keep required columns
    return df
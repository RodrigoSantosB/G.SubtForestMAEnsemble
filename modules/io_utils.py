import os
import pandas as pd


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_csv(file_path: str) -> pd.DataFrame:
    """Wrapper to read CSV, matching pandas default behavior."""
    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """Wrapper to save CSV, matching pandas default behavior."""
    ensure_dir(os.path.dirname(file_path) or '.')
    df.to_csv(file_path, index=index)
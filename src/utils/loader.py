import pandas as pd
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def set_timeseries(df: pd.DataFrame, resample_freq: str) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
    df = df.set_index('timestamp').resample(resample_freq).size().reset_index(name='txn_count')
    df['txn_count'] = df['txn_count'].fillna(0)

    return df

def prepare_train_data(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    """Prepare train and test splits for transaction counts.

    Args:
        df: DataFrame containing at least a 'txn_count' column and optionally a 'timestamp' column.
        train_frac: Fraction of the data to use for training (0 < train_frac < 1).

    Returns:
        A tuple (train_series, test_series) containing the transaction counts as pandas Series.
    """

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    if 'txn_count' not in df.columns:
        raise ValueError("DataFrame must contain a 'txn_count' column")

    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be a float between 0 and 1 (e.g. 0.8 for 80% train)")

    n_train = int(len(df) * train_frac)
    train = df['txn_count'].iloc[:n_train]
    test = df['txn_count'].iloc[n_train:]

    return train, test
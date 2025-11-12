import pandas as pd
from typing import Tuple

def set_timeseries(df: pd.DataFrame, resample_freq: str) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
    df = df.set_index('timestamp').resample(resample_freq).size().reset_index(name='txn_count')
    df['txn_count'] = df['txn_count'].fillna(0)

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['timestamp'].dt.dayofweek > 4).astype(int)

    return df

def split_train_test_multivariate(df: pd.DataFrame, train_frac: float):
    n_train = int(len(df) * train_frac)
    FEATURE_COLS = ['hour', 'day_of_week', 'is_weekend']
    TARGET_COL = 'txn_count'
    
    X_train = df[FEATURE_COLS].iloc[:n_train]
    y_train = df[TARGET_COL].iloc[:n_train]
    X_test = df[FEATURE_COLS].iloc[n_train:]
    y_test = df[TARGET_COL].iloc[n_train:]
    
    return X_train, y_train, X_test, y_test

def prepare_train_data(df: pd.DataFrame, train_frac: float = 0.8, multivariate: bool = False):
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    if 'txn_count' not in df.columns:
        raise ValueError("DataFrame must contain a 'txn_count' column")

    n_train = int(len(df) * train_frac)

    if multivariate:
        # Return full DataFrames with all columns (features + target)
        train = df.iloc[:n_train]
        test = df.iloc[n_train:]
    else:
        # Return only txn_count column as Series
        train = df['txn_count'].iloc[:n_train]
        test = df['txn_count'].iloc[n_train:]

    return train, test

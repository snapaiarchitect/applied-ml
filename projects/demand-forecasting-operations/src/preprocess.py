"""
Data preprocessing module for bike sharing demand forecasting.
Works with REAL UCI Bike Sharing data (hour.csv / day.csv).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List


def load_hourly_data(data_dir: str = '../data') -> pd.DataFrame:
    """Load hourly bike sharing data from UCI dataset."""
    df = pd.read_csv(f'{data_dir}/hour.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    return df


def load_daily_data(data_dir: str = '../data') -> pd.DataFrame:
    """Load daily bike sharing data from UCI dataset."""
    df = pd.read_csv(f'{data_dir}/day.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df


def create_time_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """Create time-based features for forecasting."""
    df = df.copy()
    
    if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        dt = df[datetime_col]
    else:
        dt = pd.to_datetime(df[datetime_col])
    
    # Basic time features
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['month'] = dt.dt.month
    df['day_of_month'] = dt.dt.day
    df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['quarter'] = dt.dt.quarter
    df['year'] = dt.dt.year
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'cnt',
                       lags: List[int] = [1, 2, 3, 7, 14, 24]) -> pd.DataFrame:
    """Create lag features for time series."""
    df = df.copy().sort_values('datetime')
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = 'cnt',
                           windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """Create rolling window statistics."""
    df = df.copy().sort_values('datetime')
    
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_roll_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
    
    return df


def prepare_forecasting_features(df: pd.DataFrame, target_col: str = 'cnt') -> pd.DataFrame:
    """Full feature engineering pipeline for forecasting."""
    # Time features
    if 'datetime' in df.columns:
        df = create_time_features(df)
    
    # Lag features
    df = create_lag_features(df, target_col)
    
    # Rolling features
    df = create_rolling_features(df, target_col)
    
    # Drop rows with NaN from lags
    df = df.dropna()
    
    return df


def split_time_series(df: pd.DataFrame, test_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series into train/test sets by date."""
    df = df.sort_values('datetime')
    
    # Find split date
    max_date = df['datetime'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = df[df['datetime'] <= split_date].copy()
    test = df[df['datetime'] > split_date].copy()
    
    return train, test


def get_weather_description(weathersit: int) -> str:
    """Map weather situation code to description."""
    mapping = {
        1: 'Clear/Partly Cloudy',
        2: 'Mist/Cloudy',
        3: 'Light Snow/Rain',
        4: 'Heavy Rain/Snow'
    }
    return mapping.get(weathersit, 'Unknown')


def get_season_description(season: int) -> str:
    """Map season code to description."""
    mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    return mapping.get(season, 'Unknown')


if __name__ == '__main__':
    print("Testing bike sharing preprocessing...")
    
    try:
        hourly = load_hourly_data()
        print(f"Hourly data: {hourly.shape[0]:,} records")
        
        processed = prepare_forecasting_features(hourly)
        print(f"Processed: {processed.shape[0]:,} records, {processed.shape[1]} features")
        
        train, test = split_time_series(processed)
        print(f"Train: {train.shape[0]:,} | Test: {test.shape[0]:,}")
        print("\nPreprocessing module ready.")
    except FileNotFoundError:
        print("hour.csv not found — run fetch_bike_data.py first.")

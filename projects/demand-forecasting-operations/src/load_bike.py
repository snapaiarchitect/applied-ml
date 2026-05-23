"""
UCI Bike Sharing Data Loader
Loads real bike rental data from UCI Machine Learning Repository.

Source: UCI ML Repository — Bike Sharing Dataset
Citation: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors 
          and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15.
URL: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_hourly_data():
    """Load hourly bike sharing data (17,379 records)."""
    path = Path(__file__).parent.parent / 'data' / 'hour.csv'
    df = pd.read_csv(path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    return df


def load_daily_data():
    """Load daily aggregated bike sharing data (731 records)."""
    path = Path(__file__).parent.parent / 'data' / 'day.csv'
    df = pd.read_csv(path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df


def get_weather_description(weathersit):
    """Map weather situation code to description."""
    mapping = {
        1: 'Clear/Partly Cloudy',
        2: 'Mist/Cloudy',
        3: 'Light Snow/Rain',
        4: 'Heavy Rain/Snow'
    }
    return mapping.get(weathersit, 'Unknown')


def get_season_description(season):
    """Map season code to description."""
    mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    return mapping.get(season, 'Unknown')


if __name__ == '__main__':
    hourly = load_hourly_data()
    daily = load_daily_data()
    
    print(f"Hourly: {hourly.shape[0]:,} records")
    print(f"Daily: {daily.shape[0]:,} records")
    print(f"Date range: {hourly['dteday'].min()} to {hourly['dteday'].max()}")
    print(f"Total rentals: {hourly['cnt'].sum():,}")
    print("Data loaded successfully.")

"""
NASA C-MAPSS FD001 Data Loader
Loads real turbofan engine degradation data from NASA Prognostics Data Repository.

Source: NASA Ames Prognostics Center of Excellence
Citation: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set"
URL: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_train_data():
    """Load training data (run-to-failure for 100 engines)."""
    path = Path(__file__).parent.parent / 'data' / 'train_FD001.txt'
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    df = pd.read_csv(path, sep='\s+', header=None, names=columns)
    df['unit_number'] = df['unit_number'].astype(int)
    return df


def load_test_data():
    """Load test data (truncated before failure)."""
    path = Path(__file__).parent.parent / 'data' / 'test_FD001.txt'
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    df = pd.read_csv(path, sep='\s+', header=None, names=columns)
    df['unit_number'] = df['unit_number'].astype(int)
    return df


def load_rul():
    """Load true RUL values for test set."""
    path = Path(__file__).parent.parent / 'data' / 'RUL_FD001.txt'
    rul = pd.read_csv(path, header=None, names=['RUL'])
    rul['unit_number'] = rul.index + 1
    return rul


def add_rul_to_train(df):
    """Add Remaining Useful Life column to training data."""
    # For each engine, RUL = max_cycles - current_cycle
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop('max_cycle', axis=1)
    return df


def get_sensor_columns(df):
    """Return list of sensor measurement columns."""
    return [c for c in df.columns if c.startswith('sensor_')]


if __name__ == '__main__':
    # Verify data loads correctly
    train = load_train_data()
    test = load_test_data()
    rul = load_rul()
    
    print(f"Training: {train.shape[0]:,} rows, {train['unit_number'].nunique()} engines")
    print(f"Test: {test.shape[0]:,} rows, {test['unit_number'].nunique()} engines")
    print(f"RUL values: {len(rul)} engines")
    print(f"Sensors: {len(get_sensor_columns(train))}")
    print("Data loaded successfully.")

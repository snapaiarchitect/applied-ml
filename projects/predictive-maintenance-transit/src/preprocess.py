"""
Data preprocessing module for predictive maintenance.
Handles data cleaning, feature engineering, and dataset preparation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional


def load_transit_data(data_dir: str = '../data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all transit datasets."""
    assets = pd.read_csv(f'{data_dir}/transit_assets.csv')
    assets['installation_date'] = pd.to_datetime(assets['installation_date'])
    assets['last_maintenance'] = pd.to_datetime(assets['last_maintenance'])
    
    failures = pd.read_csv(f'{data_dir}/failure_history.csv')
    failures['failure_date'] = pd.to_datetime(failures['failure_date'])
    
    telemetry = pd.read_csv(f'{data_dir}/sensor_telemetry.csv')
    telemetry['date'] = pd.to_datetime(telemetry['date'])
    
    maintenance = pd.read_csv(f'{data_dir}/maintenance_logs.csv')
    maintenance['maintenance_date'] = pd.to_datetime(maintenance['maintenance_date'])
    
    return assets, failures, telemetry, maintenance


def engineer_features(assets: pd.DataFrame, 
                     failures: pd.DataFrame, 
                     maintenance: pd.DataFrame,
                     reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """Engineer features for survival analysis and anomaly detection."""
    if reference_date is None:
        reference_date = datetime(2024, 12, 31)
    
    # Merge asset and failure data
    asset_failures = assets.merge(
        failures.groupby('asset_id').agg(
            failure_count=('failure_date', 'count'),
            first_failure=('failure_date', 'min'),
            last_failure=('failure_date', 'max'),
            total_downtime=('downtime_hours', 'sum'),
            total_repair_cost=('repair_cost', 'sum'),
            avg_severity=('severity', lambda x: x.map({'Minor': 1, 'Moderate': 2, 'Major': 3, 'Severe': 4}).mean())
        ).reset_index(),
        on='asset_id', how='left'
    )
    
    # Fill missing values
    asset_failures['failure_count'] = asset_failures['failure_count'].fillna(0)
    asset_failures['total_downtime'] = asset_failures['total_downtime'].fillna(0)
    asset_failures['total_repair_cost'] = asset_failures['total_repair_cost'].fillna(0)
    
    # Days since last maintenance
    asset_failures['days_since_maintenance'] = (reference_date - asset_failures['last_maintenance']).dt.days
    
    # Maintenance frequency
    maintenance_freq = maintenance.groupby('asset_id').size().reset_index(name='maintenance_frequency')
    asset_failures = asset_failures.merge(maintenance_freq, on='asset_id', how='left')
    asset_failures['maintenance_frequency'] = asset_failures['maintenance_frequency'].fillna(0)
    
    # Encode criticality
    asset_failures['criticality_encoded'] = asset_failures['criticality'].map(
        {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    )
    
    # Survival target
    asset_failures['time_to_failure'] = np.where(
        pd.notna(asset_failures['first_failure']),
        (asset_failures['first_failure'] - asset_failures['installation_date']).dt.days,
        (reference_date - asset_failures['installation_date']).dt.days
    )
    asset_failures['failed'] = pd.notna(asset_failures['first_failure']).astype(int)
    
    return asset_failures


def prepare_sensor_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sensor telemetry into asset-level features."""
    sensor_features = telemetry.groupby(['asset_id', 'sensor_type']).agg(
        mean_reading=('reading', 'mean'),
        std_reading=('reading', 'std'),
        max_reading=('reading', 'max'),
        min_reading=('reading', 'min'),
        range_reading=('reading', lambda x: x.max() - x.min()),
        last_reading=('reading', 'last'),
        reading_trend=('reading', lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 10 else 0)
    ).reset_index()
    
    # Pivot to wide format
    sensor_wide = sensor_features.pivot(index='asset_id', columns='sensor_type')
    sensor_wide.columns = [f'{col[1]}_{col[0]}' for col in sensor_wide.columns]
    sensor_wide = sensor_wide.reset_index()
    
    return sensor_wide


def create_spc_features(telemetry: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Create Statistical Process Control features."""
    telemetry = telemetry.sort_values(['asset_id', 'sensor_type', 'date']).copy()
    
    # Rolling statistics
    telemetry['rolling_mean'] = telemetry.groupby(['asset_id', 'sensor_type'])['reading'].transform(
        lambda x: x.rolling(window=window, min_periods=10).mean()
    )
    telemetry['rolling_std'] = telemetry.groupby(['asset_id', 'sensor_type'])['reading'].transform(
        lambda x: x.rolling(window=window, min_periods=10).std()
    )
    
    # Control limits
    telemetry['upper_control'] = telemetry['rolling_mean'] + 3 * telemetry['rolling_std']
    telemetry['lower_control'] = telemetry['rolling_mean'] - 3 * telemetry['rolling_std']
    
    # Anomaly flag
    telemetry['spc_anomaly'] = (
        (telemetry['reading'] > telemetry['upper_control']) | 
        (telemetry['reading'] < telemetry['lower_control'])
    )
    
    return telemetry


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed dataframe to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")


if __name__ == '__main__':
    # Test the preprocessing pipeline
    assets, failures, telemetry, maintenance = load_transit_data()
    features = engineer_features(assets, failures, maintenance)
    save_processed_data(features, '../data/processed_features.csv')
    print(f"Processed {len(features)} assets")

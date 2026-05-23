"""
Anomaly detection module for predictive maintenance.
Implements Isolation Forest and Statistical Process Control methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class AnomalyDetector:
    """Anomaly detection for transit asset sensor data."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit Isolation Forest on sensor features."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self.fitted = True
        self.feature_names = feature_names
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 = anomaly, 1 = normal)."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def detect(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Run full anomaly detection pipeline on dataframe."""
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                          if c not in ['asset_id', 'anomaly', 'anomaly_score']]
        
        X = df[feature_cols].fillna(df[feature_cols].median()).values
        
        if not self.fitted:
            self.fit(X, feature_names=feature_cols)
        
        df = df.copy()
        df['anomaly'] = self.predict(X) == -1
        df['anomaly_score'] = self.score(X)
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of detected anomalies."""
        anomalies = df[df['anomaly']]
        normal = df[~df['anomaly']]
        
        return {
            'total_assets': len(df),
            'anomalous_assets': len(anomalies),
            'anomaly_rate': df['anomaly'].mean(),
            'mean_anomaly_score': df['anomaly_score'].mean(),
            'top_anomalies': anomalies.nsmallest(10, 'anomaly_score')['asset_id'].tolist()
        }


class StatisticalProcessControl:
    """Statistical Process Control for sensor telemetry."""
    
    def __init__(self, window: int = 30, sigma: int = 3):
        self.window = window
        self.sigma = sigma
        
    def fit_transform(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Apply SPC to telemetry data."""
        telemetry = telemetry.sort_values(['asset_id', 'sensor_type', 'date']).copy()
        
        # Rolling statistics
        telemetry['rolling_mean'] = telemetry.groupby(['asset_id', 'sensor_type'])['reading'].transform(
            lambda x: x.rolling(window=self.window, min_periods=10).mean()
        )
        telemetry['rolling_std'] = telemetry.groupby(['asset_id', 'sensor_type'])['reading'].transform(
            lambda x: x.rolling(window=self.window, min_periods=10).std()
        )
        
        # Control limits
        telemetry['upper_control'] = telemetry['rolling_mean'] + self.sigma * telemetry['rolling_std']
        telemetry['lower_control'] = telemetry['rolling_mean'] - self.sigma * telemetry['rolling_std']
        
        # Anomaly flag
        telemetry['spc_anomaly'] = (
            (telemetry['reading'] > telemetry['upper_control']) | 
            (telemetry['reading'] < telemetry['lower_control'])
        )
        
        return telemetry
    
    def get_anomaly_rate(self, telemetry: pd.DataFrame) -> float:
        """Get overall anomaly rate."""
        return telemetry['spc_anomaly'].mean()


def combine_risk_scores(survival_risk: pd.DataFrame, 
                       anomaly_scores: pd.DataFrame,
                       survival_weight: float = 0.6) -> pd.DataFrame:
    """Combine survival and anomaly risk scores."""
    # Normalize
    scaler = MinMaxScaler()
    
    survival_risk['risk_norm'] = scaler.fit_transform(survival_risk[['risk_score']])
    anomaly_scores['anomaly_norm'] = scaler.fit_transform(-anomaly_scores[['anomaly_score']])
    
    # Combine
    combined = survival_risk[['asset_id', 'risk_norm']].merge(
        anomaly_scores[['asset_id', 'anomaly_norm']], on='asset_id', how='outer'
    )
    combined = combined.fillna(0)
    
    combined['combined_risk'] = (
        survival_weight * combined['risk_norm'] + 
        (1 - survival_weight) * combined['anomaly_norm']
    )
    
    return combined


if __name__ == '__main__':
    # Test anomaly detection
    print("Testing anomaly detection pipeline...")
    
    # Generate test data
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    # Inject some anomalies
    X[:50] += np.random.randn(50, 5) * 5
    
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(X)
    
    labels = detector.predict(X)
    scores = detector.score(X)
    
    print(f"Anomalies detected: {(labels == -1).sum()}")
    print(f"Mean anomaly score: {scores.mean():.3f}")
    print("Anomaly detection module ready.")

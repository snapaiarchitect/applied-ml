"""
Evaluation metrics module for demand forecasting operations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Optional


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute standard forecasting metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE with handling for zeros
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }


def calculate_mape_by_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                               horizons: List[int] = [1, 7, 14, 30]) -> Dict:
    """Calculate MAPE at different forecast horizons."""
    results = {}
    
    for h in horizons:
        if len(y_true) >= h:
            mape = np.mean(np.abs((y_true[:h] - y_pred[:h]) / np.maximum(y_true[:h], 1))) * 100
            results[f'MAPE_{h}d'] = mape
    
    return results


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Analyze residuals for model diagnostics."""
    residuals = y_true - y_pred
    
    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'max_residual': np.max(np.abs(residuals)),
        'residual_skewness': pd.Series(residuals).skew(),
        'residual_kurtosis': pd.Series(residuals).kurtosis(),
        'pct_within_1std': np.mean(np.abs(residuals) < np.std(residuals)),
        'pct_within_2std': np.mean(np.abs(residuals) < 2 * np.std(residuals))
    }


def compare_forecasts(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare multiple forecasting models."""
    results = []
    
    for model_name, y_pred in predictions.items():
        metrics = forecast_metrics(y_true, y_pred)
        metrics['model'] = model_name
        results.append(metrics)
    
    return pd.DataFrame(results)


def print_metrics(metrics: Dict, title: str = "Forecast Performance"):
    """Pretty print forecasting metrics."""
    print(f"\n=== {title} ===")
    print(f"  MAE:  {metrics['MAE']:.0f}")
    print(f"  RMSE: {metrics['RMSE']:.0f}")
    print(f"  MAPE: {metrics['MAPE']:.1f}%")
    print(f"  R²:   {metrics['R2']:.3f}")
    
    if 'Directional_Accuracy' in metrics:
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.1%}")


if __name__ == '__main__':
    # Test metrics
    y_true = np.array([1000, 1200, 1100, 1300, 1400, 1500, 1600])
    y_pred = np.array([1050, 1150, 1120, 1280, 1420, 1480, 1550])
    
    metrics = forecast_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Forecast")
    
    residuals = residual_analysis(y_true, y_pred)
    print(f"\nResiduals: {residuals}")
    print("\nEvaluation module ready.")

"""
Evaluation metrics module for predictive maintenance models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List
from scipy import stats


def survival_metrics(y_true: np.ndarray, y_pred: np.ndarray, event_observed: np.ndarray) -> Dict:
    """Evaluate survival model predictions."""
    # C-index approximation
    concordant = 0
    permissible = 0
    
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if event_observed[i] == 1 or event_observed[j] == 1:
                permissible += 1
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                   (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
    
    c_index = concordant / permissible if permissible > 0 else 0.5
    
    return {
        'c_index': c_index,
        'concordant_pairs': concordant,
        'permissible_pairs': permissible
    }


def anomaly_validation(normal_group: pd.DataFrame, 
                      anomalous_group: pd.DataFrame,
                      metric_col: str = 'failure_count') -> Dict:
    """Validate anomalies by comparing with failure outcomes."""
    normal_mean = normal_group[metric_col].mean()
    anomalous_mean = anomalous_group[metric_col].mean()
    
    t_stat, p_value = stats.ttest_ind(
        normal_group[metric_col], 
        anomalous_group[metric_col]
    )
    
    return {
        'normal_mean': normal_mean,
        'anomalous_mean': anomalous_mean,
        'ratio': anomalous_mean / normal_mean if normal_mean > 0 else float('inf'),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Standard forecasting metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Classification metrics for document categorization."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def print_metrics(metrics: Dict, title: str = "Model Performance"):
    """Pretty print metrics."""
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    
    metrics = forecast_metrics(y_true, y_pred)
    print_metrics(metrics, "Forecast Test")
    
    print("\nEvaluation module ready.")

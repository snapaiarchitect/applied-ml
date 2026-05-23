"""
Evaluation metrics module for NLP text classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, List, Optional


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          target_names: Optional[List[str]] = None) -> Dict:
    """Compute comprehensive classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    per_class = {}
    if target_names:
        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(y_true, y_pred)
        for i, name in enumerate(target_names):
            per_class[name] = {
                'precision': precision_per[i],
                'recall': recall_per[i],
                'f1': f1_per[i],
                'support': int(support_per[i])
            }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': per_class,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=target_names)
    }


def compare_models(y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                  target_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare multiple model predictions."""
    results = []
    
    for model_name, y_pred in predictions.items():
        metrics = classification_metrics(y_true, y_pred, target_names)
        results.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1_score']
        })
    
    return pd.DataFrame(results)


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                  texts: List[str], label_map: Optional[Dict] = None) -> pd.DataFrame:
    """Analyze misclassified examples."""
    errors = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            error = {
                'index': i,
                'text': texts[i][:100],
                'true_label': true,
                'predicted_label': pred
            }
            
            if label_map:
                error['true_category'] = label_map.get(true, str(true))
                error['predicted_category'] = label_map.get(pred, str(pred))
            
            errors.append(error)
    
    return pd.DataFrame(errors)


def print_metrics(metrics: Dict, title: str = "Model Performance"):
    """Pretty print classification metrics."""
    print(f"\n=== {title} ===")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    if metrics.get('per_class'):
        print(f"\n  Per-class F1:")
        for name, scores in metrics['per_class'].items():
            print(f"    {name}: {scores['f1']:.4f}")


if __name__ == '__main__':
    # Test metrics
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3])
    
    metrics = classification_metrics(y_true, y_pred, ['Budget', 'Policy', 'Service', 'FOIA'])
    print_metrics(metrics, "Test Classification")
    print("\nEvaluation module ready.")

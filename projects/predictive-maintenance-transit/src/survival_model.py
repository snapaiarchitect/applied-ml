"""
Survival analysis models for predictive maintenance.
Implements Kaplan-Meier, Cox PH, and Weibull survival models.
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullFitter
from lifelines.statistics import logrank_test
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class SurvivalModel:
    """Wrapper for survival analysis models."""
    
    def __init__(self, model_type: str = 'cox'):
        self.model_type = model_type
        self.model = None
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, duration_col: str = 'time_to_failure',
            event_col: str = 'failed', feature_cols: Optional[List[str]] = None):
        """Fit survival model."""
        if feature_cols is None:
            feature_cols = ['age_years', 'condition_score', 'daily_usage_hours',
                          'days_since_maintenance', 'maintenance_frequency',
                          'criticality_encoded']
        
        # Convert duration to years for stability
        df_fit = df[feature_cols + [duration_col, event_col]].copy()
        df_fit[duration_col] = df_fit[duration_col] / 365.25
        df_fit = df_fit.dropna()
        
        if self.model_type == 'cox':
            self.model = CoxPHFitter(penalizer=0.01)
            self.model.fit(df_fit, duration_col=duration_col, event_col=event_col)
        elif self.model_type == 'weibull':
            self.model = WeibullFitter()
            self.model.fit(df_fit[duration_col], df_fit[event_col])
        elif self.model_type == 'kaplan_meier':
            self.model = KaplanMeierFitter()
            self.model.fit(df_fit[duration_col], event_observed=df_fit[event_col])
        
        self.fitted = True
        self.feature_cols = feature_cols
        return self
    
    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Predict risk scores."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type == 'cox':
            return self.model.predict_partial_hazard(df[self.feature_cols]).values
        else:
            # For non-Cox models, use a simple heuristic
            return np.ones(len(df))
    
    def predict_survival(self, df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Predict survival probabilities at given times."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        if self.model_type == 'cox':
            return self.model.predict_survival_function(df[self.feature_cols], times=times).T.values
        elif self.model_type == 'weibull':
            return np.array([self.model.predict(t) for t in times])
        else:
            return self.model.predict(times)
    
    def get_summary(self) -> Dict:
        """Get model summary statistics."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        summary = {
            'model_type': self.model_type,
            'concordance_index': getattr(self.model, 'concordance_index_', None),
            'AIC': getattr(self.model, 'AIC_', None),
        }
        
        if self.model_type == 'cox':
            summary['hazard_ratios'] = np.exp(self.model.summary['coef']).to_dict()
            summary['p_values'] = self.model.summary['p'].to_dict()
        elif self.model_type == 'weibull':
            summary['lambda_'] = self.model.lambda_
            summary['rho_'] = self.model.rho_
        
        return summary
    
    def plot_survival_curve(self, df: pd.DataFrame = None, 
                           label: str = 'All Assets',
                           color: str = 'steelblue'):
        """Plot survival curve."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        if df is None:
            self.model.plot_survival_function(color=color, linewidth=2, label=label)
        else:
            self.model.predict_survival_function(df[self.feature_cols]).plot(
                color=color, linewidth=2, label=label
            )
    
    def save(self, path: str):
        """Save model (placeholder for pickle/joblib)."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved model to: {path}")
    
    def load(self, path: str):
        """Load model."""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.fitted = True
        print(f"Loaded model from: {path}")


def run_logrank_test(df: pd.DataFrame, group_col: str, 
                     duration_col: str = 'time_to_failure',
                     event_col: str = 'failed') -> Dict:
    """Run log-rank test between two groups."""
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Log-rank test requires exactly 2 groups")
    
    mask_a = df[group_col] == groups[0]
    mask_b = df[group_col] == groups[1]
    
    T_a = df.loc[mask_a, duration_col] / 365.25
    T_b = df.loc[mask_b, duration_col] / 365.25
    E_a = df.loc[mask_a, event_col]
    E_b = df.loc[mask_b, event_col]
    
    result = logrank_test(T_a, T_b, event_observed_A=E_a, event_observed_B=E_b)
    
    return {
        'test_statistic': result.test_statistic,
        'p_value': result.p_value,
        'significant': result.p_value < 0.05
    }


if __name__ == '__main__':
    # Test survival model
    df = pd.read_csv('../data/processed_features.csv')
    
    print("Fitting Cox PH model...")
    model = SurvivalModel(model_type='cox')
    model.fit(df)
    
    summary = model.get_summary()
    print(f"C-index: {summary['concordance_index']:.3f}")
    print(f"AIC: {summary['AIC']:.2f}")
    
    # Predict risk scores
    risk_scores = model.predict_risk(df)
    print(f"Risk scores: mean={risk_scores.mean():.3f}, std={risk_scores.std():.3f}")

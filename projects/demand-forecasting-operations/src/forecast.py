"""
Forecasting models module for demand forecasting operations.
Implements ARIMA, Prophet, XGBoost, and ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DemandForecaster:
    """Time series forecasting wrapper."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.fitted = False
        self.feature_cols = None
        
    def fit(self, df: pd.DataFrame, target_col: str = 'demand',
            feature_cols: Optional[List[str]] = None):
        """Train forecasting model."""
        if feature_cols is None:
            # Auto-select features
            exclude = ['datetime', 'week_start', target_col]
            feature_cols = [c for c in df.columns if c not in exclude]
        
        self.feature_cols = feature_cols
        X = df[feature_cols]
        y = df[target_col]
        
        if self.model_type == 'xgboost':
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model.fit(X, y)
            self.fitted = True
            
        elif self.model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.fitted = True
            
        elif self.model_type == 'arima':
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y, order=(7, 1, 1))
            self.model = self.model.fit()
            self.fitted = True
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type in ['xgboost', 'linear']:
            X = df[self.feature_cols]
            return self.model.predict(X)
        elif self.model_type == 'arima':
            steps = len(df)
            return self.model.forecast(steps=steps)
        
    def forecast(self, df: pd.DataFrame, steps: int = 30) -> pd.DataFrame:
        """Forecast future demand."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        # For tree-based models, we need to create future features
        if self.model_type in ['xgboost', 'linear']:
            # Use last known values and project forward
            last_row = df.iloc[-1:].copy()
            future_rows = []
            
            for i in range(steps):
                future_row = last_row.copy()
                # Update time-based features
                if 'day_of_week' in future_row.columns:
                    future_row['day_of_week'] = (future_row['day_of_week'] + i) % 7
                future_rows.append(future_row)
            
            future_df = pd.concat(future_rows, ignore_index=True)
            predictions = self.predict(future_df)
            
        else:
            predictions = self.model.forecast(steps=steps)
            future_df = pd.DataFrame(index=pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            ))
        
        future_df['forecast'] = predictions
        return future_df
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'demand') -> Dict:
        """Evaluate model performance."""
        y_true = df[target_col].values
        y_pred = self.predict(df)
        
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
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance (for tree-based models)."""
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None
    
    def save(self, path: str):
        """Save model to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_cols': self.feature_cols
            }, f)
        print(f"Saved model to: {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_cols = data['feature_cols']
        self.fitted = True
        print(f"Loaded model from: {path}")


class EnsembleForecaster:
    """Ensemble of multiple forecasting models."""
    
    def __init__(self, models: Dict[str, DemandForecaster], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0 for name in models}
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            pred = model.predict(df)
            weight = self.weights.get(name, 1.0)
            predictions.append(pred * weight / total_weight)
        
        return np.sum(predictions, axis=0)
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'demand') -> Dict:
        """Evaluate ensemble performance."""
        y_true = df[target_col].values
        y_pred = self.predict(df)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'models': list(self.models.keys()),
            'weights': self.weights
        }


def run_prophet_forecast(df: pd.DataFrame, periods: int = 30,
                        holidays: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run Prophet forecasting."""
    try:
        from prophet import Prophet
        
        # Prepare data
        prophet_df = df.reset_index()[['datetime', 'demand']].rename(
            columns={'datetime': 'ds', 'demand': 'y'}
        )
        
        # Fit
        m = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True)
        m.fit(prophet_df)
        
        # Predict
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
    except ImportError:
        print("Prophet not installed. Install with: pip install prophet")
        return None


if __name__ == '__main__':
    # Test forecaster
    print("Testing demand forecasting module...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample = pd.DataFrame({
        'datetime': dates,
        'demand': np.random.randint(1000, 5000, 100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    
    forecaster = DemandForecaster('linear')
    forecaster.fit(sample, target_col='demand', feature_cols=['feature1', 'feature2'])
    
    preds = forecaster.predict(sample)
    metrics = forecaster.evaluate(sample)
    
    print(f"Predictions: {preds[:5]}")
    print(f"Metrics: {metrics}")
    print("\nForecasting module ready.")

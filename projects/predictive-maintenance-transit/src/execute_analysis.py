import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print('CELL 1: Libraries loaded')

# CELL 3: Load data
cols = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
train_df = pd.read_csv('data/raw/train_FD001.txt', sep='\s+', header=None, names=cols)
test_df = pd.read_csv('data/raw/test_FD001.txt', sep='\s+', header=None, names=cols)
true_rul = pd.read_csv('data/raw/RUL_FD001.txt', header=None, names=['RUL']).values.flatten()
print(f'Train: {train_df.shape}, Test: {test_df.shape}, RUL: {len(true_rul)}')

# CELL 6: Compute RUL
max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']
train_df = train_df.merge(max_cycles, on='unit_number')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
train_df = train_df.drop('max_cycle', axis=1)

engine_lifetimes = train_df.groupby('unit_number')['time_in_cycles'].max()
print(f'RUL range: {train_df["RUL"].min()} to {train_df["RUL"].max()}')

# RUL distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train_df['RUL'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Remaining Useful Life (cycles)')
axes[0].set_ylabel('Count')
axes[0].set_title('Training RUL Distribution')
axes[1].hist(engine_lifetimes, bins=20, color='coral', edgecolor='white')
axes[1].set_xlabel('Engine Lifetime (cycles)')
axes[1].set_ylabel('Count')
axes[1].set_title('Engine Failure Time Distribution')
plt.tight_layout()
plt.savefig('figures/rul_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/rul_distributions.png')

# CELL 8: Sensor degradation curves
sample_engines = [1, 25, 50, 75]
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
fig, axes = plt.subplots(3, 7, figsize=(20, 10))
axes = axes.flatten()
for idx, sensor in enumerate(sensor_cols):
    ax = axes[idx]
    for engine in sample_engines:
        engine_data = train_df[train_df['unit_number'] == engine]
        ax.plot(engine_data['time_in_cycles'], engine_data[sensor], alpha=0.7, label=f'Engine {engine}')
    ax.set_title(sensor)
    ax.set_xlabel('Cycle')
    ax.tick_params(labelsize=8)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')
plt.suptitle('Sensor Degradation Curves (4 Sample Engines)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/sensor_degradation_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/sensor_degradation_curves.png')

# CELL 9: Identify constant sensors
sensor_variance = train_df[sensor_cols].var().sort_values()
constant_sensors = sensor_variance[sensor_variance < 0.01].index.tolist()
useful_sensors = [s for s in sensor_cols if s not in constant_sensors]
print(f'Constant sensors dropped: {constant_sensors}')
print(f'Useful sensors: {len(useful_sensors)}/{len(sensor_cols)}')

# CELL 11: Feature engineering
def create_features(df, sensors, window=10):
    feature_df = df.copy()
    for sensor in sensors:
        feature_df[f'{sensor}_rolling_mean'] = (
            feature_df.groupby('unit_number')[sensor]
            .rolling(window=window, min_periods=1)
            .mean().reset_index(level=0, drop=True)
        )
        feature_df[f'{sensor}_rolling_std'] = (
            feature_df.groupby('unit_number')[sensor]
            .rolling(window=window, min_periods=1)
            .std().reset_index(level=0, drop=True)
        ).fillna(0)
        feature_df[f'{sensor}_diff'] = (
            feature_df.groupby('unit_number')[sensor]
            .diff().fillna(0)
        )
    feature_df['cycle_norm'] = feature_df.groupby('unit_number')['time_in_cycles'].transform(lambda x: x / x.max())
    feature_df['op_setting_sum'] = feature_df['op_setting_1'] + feature_df['op_setting_2'] + feature_df['op_setting_3']
    return feature_df

print('Engineering features...')
train_features = create_features(train_df, useful_sensors, window=10)
test_features = create_features(test_df, useful_sensors, window=10)
print(f'Train features: {len(train_features.columns)}, Test features: {len(test_features.columns)}')

# CELL 12: Prepare feature matrix
engineered_cols = [c for c in train_features.columns if '_rolling_mean' in c or '_rolling_std' in c or '_diff' in c or c in ['cycle_norm', 'op_setting_sum']]
feature_cols = useful_sensors + engineered_cols
X_train = train_features[feature_cols].fillna(0)
y_train = train_features['RUL']
test_last = test_features.groupby('unit_number').last().reset_index()
X_test = test_last[feature_cols].fillna(0)
print(f'Training matrix: {X_train.shape}, Test matrix: {X_test.shape}')

# CELL 14: Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Features standardized')

# CELL 15: Train models
print('Training Random Forest...')
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print('Training XGBoost...')
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, objective='reg:squarederror')
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
print('Models trained')

# CELL 17: Evaluate
rf_mae = mean_absolute_error(true_rul, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(true_rul, rf_pred))
rf_r2 = r2_score(true_rul, rf_pred)

xgb_mae = mean_absolute_error(true_rul, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(true_rul, xgb_pred))
xgb_r2 = r2_score(true_rul, xgb_pred)

print(f'RF  MAE={rf_mae:.2f} RMSE={rf_rmse:.2f} R2={rf_r2:.3f}')
print(f'XGB MAE={xgb_mae:.2f} RMSE={xgb_rmse:.2f} R2={xgb_r2:.3f}')

# CELL 18: Predicted vs Actual scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, (name, pred) in enumerate([('Random Forest', rf_pred), ('XGBoost', xgb_pred)]):
    ax = axes[idx]
    ax.scatter(true_rul, pred, alpha=0.6, c='steelblue', edgecolors='white', s=60)
    max_val = max(true_rul.max(), pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    mae = rf_mae if name == 'Random Forest' else xgb_mae
    rmse = rf_rmse if name == 'Random Forest' else xgb_rmse
    ax.set_title(f'{name}\nMAE={mae:.1f}, RMSE={rmse:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs Actual RUL — NASA CMAPSS FD001', fontsize=13)
plt.tight_layout()
plt.savefig('figures/predicted_vs_actual_rul.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/predicted_vs_actual_rul.png')

# CELL 19: Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (name, pred) in enumerate([('Random Forest', rf_pred), ('XGBoost', xgb_pred)]):
    residuals = true_rul - pred
    ax = axes[idx]
    ax.hist(residuals, bins=25, color='coral', edgecolor='white')
    ax.axvline(x=0, color='black', linestyle='--', lw=2)
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name} Residuals\nMean={residuals.mean():.1f}, Std={residuals.std():.1f}')
plt.suptitle('RUL Prediction Residuals', fontsize=13)
plt.tight_layout()
plt.savefig('figures/rul_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/rul_residuals.png')

# CELL 21: Feature importance
importance = pd.DataFrame({'feature': feature_cols, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=True)
top_n = 20
top_features = importance.tail(top_n)
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(range(len(top_features)), top_features['importance'], color='teal')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=9)
ax.set_xlabel('Feature Importance (XGBoost)')
ax.set_title(f'Top {top_n} Features for RUL Prediction — NASA CMAPSS FD001')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/feature_importance.png')

# CELL 23: Degradation trajectories
X_test_all = test_features[feature_cols].fillna(0)
X_test_all_scaled = scaler.transform(X_test_all)
test_features['RUL_pred'] = xgb_model.predict(X_test_all_scaled)

sample_test_engines = [1, 25, 50, 75]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, engine in enumerate(sample_test_engines):
    ax = axes[idx]
    engine_data = test_features[test_features['unit_number'] == engine]
    ax.plot(engine_data['time_in_cycles'], engine_data['RUL_pred'], 'b-', lw=2, label='Predicted RUL')
    true_final = true_rul[engine - 1]
    ax.axhline(y=true_final, color='r', linestyle='--', lw=2, label=f'True final RUL = {true_final}')
    last_cycle = engine_data['time_in_cycles'].max()
    last_pred = engine_data['RUL_pred'].iloc[-1]
    ax.scatter([last_cycle], [last_pred], color='blue', s=100, zorder=5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title(f'Test Engine {engine}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('RUL Degradation Trajectories — XGBoost Predictions', fontsize=13)
plt.tight_layout()
plt.savefig('figures/rul_degradation_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/rul_degradation_trajectories.png')

# CELL 25: Correlation analysis
sensor_rul_corr = train_df[useful_sensors + ['RUL']].corr()['RUL'].drop('RUL').sort_values(key=abs, ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['darkgreen' if c > 0 else 'darkred' for c in sensor_rul_corr.values]
ax.barh(range(len(sensor_rul_corr)), sensor_rul_corr.values, color=colors)
ax.set_yticks(range(len(sensor_rul_corr)))
ax.set_yticklabels(sensor_rul_corr.index, fontsize=10)
ax.set_xlabel('Correlation with RUL')
ax.set_title('Sensor Correlation with Remaining Useful Life')
ax.axvline(x=0, color='black', lw=1)
plt.tight_layout()
plt.savefig('figures/sensor_rul_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: figures/sensor_rul_correlation.png')

print()
print('=== FINAL RESULTS ===')
print(f'Dataset: {train_df["unit_number"].nunique()} train engines, {test_df["unit_number"].nunique()} test engines')
print(f'Sensors used: {len(useful_sensors)}/21')
print(f'Features: {len(feature_cols)}')
print(f'RF  MAE={rf_mae:.2f} RMSE={rf_rmse:.2f} R2={rf_r2:.3f}')
print(f'XGB MAE={xgb_mae:.2f} RMSE={xgb_rmse:.2f} R2={xgb_r2:.3f}')
print('All visualizations saved to figures/')

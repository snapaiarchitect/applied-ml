import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Demand Forecasting — Operations", layout="wide")

st.title("📊 Demand Forecasting Dashboard")
st.caption("Bike-sharing demand prediction — Powered by UCI ML Repository / Capital Bikeshare real data")

@st.cache_data
def load_data():
    day_df = pd.read_csv("data/day.csv")
    hour_df = pd.read_csv("data/hour.csv")
    forecast_df = pd.read_csv("figures/forecast_predictions.csv")
    importance_df = pd.read_csv("figures/feature_importance.csv")
    return day_df, hour_df, forecast_df, importance_df

try:
    day_df, hour_df, forecast_df, importance_df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Data load error: {e}")
    data_loaded = False

if data_loaded:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Records", f"{len(day_df):,}")
    col2.metric("Hourly Records", f"{len(hour_df):,}")
    
    # Forecast accuracy KPIs
    forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
    mae = (forecast_df['actual'] - forecast_df['ensemble']).abs().mean()
    rmse = ((forecast_df['actual'] - forecast_df['ensemble']) ** 2).mean() ** 0.5
    mape = ((forecast_df['actual'] - forecast_df['ensemble']).abs() / forecast_df['actual'].clip(lower=1)).mean() * 100
    
    col3.metric("MAE", f"{mae:.1f}")
    col4.metric("RMSE", f"{rmse:.1f}")
    
    st.caption(f"MAPE: {mape:.1f}%  ·  Forecast horizon: {len(forecast_df):,} hours")

    # Daily patterns
    st.subheader("Daily Rental Patterns")
    daily_rentals = day_df.groupby('weekday')['cnt'].mean()
    weekday_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    daily_rentals.index = [weekday_names[i] for i in daily_rentals.index]
    st.bar_chart(daily_rentals)

    # Hourly patterns
    st.subheader("Hourly Rental Patterns")
    hourly_rentals = hour_df.groupby('hr')['cnt'].mean()
    st.bar_chart(hourly_rentals)

    # Feature importance
    st.subheader("Top Feature Importance")
    st.bar_chart(importance_df.set_index('feature')['importance'].head(10))

    # Forecast vs Actual
    st.subheader("Forecast vs Actual (Last 168 Hours)")
    recent = forecast_df.tail(168).set_index('datetime')[['actual', 'ensemble']]
    st.line_chart(recent)

    # Data tables
    st.subheader("Forecast Samples")
    st.dataframe(forecast_df.tail(20), use_container_width=True)

else:
    st.info("Run the forecasting pipeline to generate data files, then refresh this dashboard.")
    st.code("""
python src/fetch_bike_data.py
python src/preprocess.py
python src/forecast.py
python src/evaluate.py
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Sierra Napier · AI Architect & Data Science Leader")

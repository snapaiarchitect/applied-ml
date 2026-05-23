import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Predictive Maintenance — Transit Systems", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")
st.caption("WMATA transit asset risk scoring — Powered by real MTA/NYC Open Data")

@st.cache_data
def load_data():
    df = pd.read_csv("data/combined_risk_scores.csv")
    with open("figures/sensor_degradation_interactive.json") as f:
        charts = json.load(f)
    with open("results.json") as f:
        results = json.load(f)
    return df, charts, results

try:
    df, charts, results = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Data load error: {e}")
    data_loaded = False

if data_loaded:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Assets", f"{len(df):,}")
    col2.metric("Critical Risk", f"{(df['risk_tier']=='Critical').sum():,}")
    col3.metric("Anomalies Detected", f"{df['anomaly'].sum():,}")
    col4.metric("Mean Risk Score", f"{df['combined_risk'].mean():.3f}")

    # Risk distribution
    st.subheader("Risk Tier Distribution")
    tier_counts = df['risk_tier'].value_counts()
    st.bar_chart(tier_counts)

    # Anomaly vs Risk
    st.subheader("Anomaly Detection by Risk Tier")
    anomaly_by_tier = df.groupby('risk_tier')['anomaly'].mean().sort_values(ascending=False)
    st.bar_chart(anomaly_by_tier)

    # Data table
    st.subheader("Asset Risk Table")
    st.dataframe(df.sort_values('combined_risk', ascending=False).head(50), use_container_width=True)

    # Model results
    st.subheader("Model Performance")
    st.json(results)

else:
    st.info("Run the notebooks to generate data files, then refresh this dashboard.")
    st.code("""
python notebooks/01_data_collection.ipynb
python notebooks/02_feature_engineering.ipynb
python notebooks/03_model_training.ipynb
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Sierra Napier · AI Architect & Data Science Leader")

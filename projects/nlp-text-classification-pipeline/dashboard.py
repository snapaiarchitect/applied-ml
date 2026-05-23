import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="NLP Text Classification Pipeline", layout="wide")

st.title("📝 NLP Text Classification Dashboard")
st.caption("20 Newsgroups classification — Powered by scikit-learn real benchmark data")

@st.cache_data
def load_data():
    stats_df = pd.read_csv("data/raw/20newsgroups_stats.csv")
    with open("results.json") as f:
        results = json.load(f)
    return stats_df, results

try:
    stats_df, results = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Data load error: {e}")
    data_loaded = False

if data_loaded:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Samples", f"{results['dataset']['train_samples']:,}")
    col2.metric("Test Samples", f"{results['dataset']['test_samples']:,}")
    col3.metric("Categories", f"{results['dataset']['n_categories']}")
    best_model = max(results['models'], key=lambda k: results['models'][k]['accuracy'])
    col4.metric("Best Accuracy", f"{results['models'][best_model]['accuracy']:.1%}")

    # Class distribution
    st.subheader("Class Distribution")
    st.bar_chart(stats_df.set_index('subset')['n_samples'])

    # Model comparison
    st.subheader("Model Performance Comparison")
    model_df = pd.DataFrame({
        k: {
            'Accuracy': v['accuracy'],
            'Precision': v['precision'],
            'Recall': v['recall'],
            'F1': v['f1']
        }
        for k, v in results['models'].items()
    }).T
    st.dataframe(model_df.style.format({
        'Accuracy': '{:.1%}',
        'Precision': '{:.1%}',
        'Recall': '{:.1%}',
        'F1': '{:.1%}'
    }), use_container_width=True)

    # Model comparison chart
    st.subheader("Accuracy by Model")
    st.bar_chart(model_df['Accuracy'])

    # Confusion matrix image
    st.subheader("Confusion Matrix")
    cm_path = Path("figures/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Run the training pipeline to generate the confusion matrix figure.")

    # Feature vectorizer settings
    st.subheader("Feature Engineering Configuration")
    st.json(results['feature_engineering'])

    # Dataset metadata
    st.subheader("Dataset Metadata")
    st.json(results['dataset'])

else:
    st.info("Run the NLP pipeline to generate data and results, then refresh this dashboard.")
    st.code("""
python src/fetch_20newsgroups.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Sierra Napier · AI Architect & Data Science Leader")

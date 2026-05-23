# applied-ml — Containerized Data Science Portfolio
FROM python:3.10-slim

WORKDIR /app

# Install system deps for data science
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose all dashboard ports
EXPOSE 8501
EXPOSE 8502
EXPOSE 8503

# Default: show available dashboards
CMD ["python", "-c", "\nprint('  streamlit run predictive-maintenance -> http://localhost:8501')\nprint('  streamlit run nlp-classification -> http://localhost:8502')\nprint('  streamlit run demand-forecasting -> http://localhost:8503')\n"]

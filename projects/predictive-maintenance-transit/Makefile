.PHONY: install test run clean lint format fetch

install:
	pip install -r requirements.txt

fetch:
	python src/fetch_cmapss_data.py

test:
	pytest tests/ -v

run:
	python src/execute_analysis.py
	jupyter notebook notebooks/predictive_maintenance_analysis.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/processed/*

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

.PHONY: install test run clean lint format fetch

install:
	pip install -r requirements.txt

fetch:
	python src/fetch_20newsgroups.py

test:
	pytest tests/ -v

run:
	python src/run_analysis.py
	jupyter notebook notebooks/nlp_classification_analysis.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| 20 Newsgroups Dataset | Public Dataset | Classic text classification dataset — 20,000 posts across 20 topics | http://qwone.com/~jason/20Newsgroups/ |

## Data Provenance

- Fetched via sklearn.datasets.fetch_20newsgroups (built-in sklearn dataset loader)
- 20,000 newsgroup posts across 20 categories
- Standard benchmark for text classification and NLP pipelines

## Data Files

| File | Description | Size (approx) |
|------|-------------|---------------|
| 20newsgroups (fetched in-memory) | Newsgroup posts for text classification | ~20,000 documents |

## Refresh Strategy

- Re-fetch via sklearn.datasets.fetch_20newsgroups when needed
- Static dataset — no external updates expected

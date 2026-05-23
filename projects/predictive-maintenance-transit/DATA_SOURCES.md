# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| NASA C-MAPSS (Turbofan Engine Degradation Simulation) | Public Dataset | Simulated sensor data for predicting remaining useful life of aircraft engines | https://ti.arc.nasa.gov/c/6/ |

## Data Provenance

- Downloaded from NASA Ames Prognostics Center of Excellence
- Includes training, test, and RUL (Remaining Useful Life) files
- Standard benchmark for predictive maintenance and RUL estimation

## Data Files

| File | Description | Size (approx) |
|------|-------------|---------------|
| train_FD001.txt | Training data — sensor readings for multiple engines | N/A |
| test_FD001.txt | Test data — sensor readings ending before failure | N/A |
| RUL_FD001.txt | Ground truth remaining useful life for test set | N/A |

## Refresh Strategy

- Re-download from NASA C-MAPSS repository if updated
- Static dataset — no external updates expected

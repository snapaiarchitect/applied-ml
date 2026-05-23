# Bike Sharing Dataset — Data Dictionary

**Source**: UCI Machine Learning Repository  
**Dataset**: Capital Bikeshare (Washington, D.C.)  
**Period**: 2011-01-01 to 2012-12-31  
**Citation**: Fanaee-T, H., & Gama, J. (2013). Progress in Artificial Intelligence.

---

## Files

| File | Records | Granularity | Description |
|------|---------|-------------|-------------|
| `day.csv` | 731 | Daily | One row per day with aggregated counts |
| `hour.csv` | 17,379 | Hourly | One row per hour with detailed weather |

---

## Columns (both files unless noted)

| Column | Type | Description |
|--------|------|-------------|
| `instant` | int | Row index (1-based) |
| `dteday` | date | Date in YYYY-MM-DD format |
| `season` | int | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| `yr` | int | Year: 0=2011, 1=2012 |
| `mnth` | int | Month: 1–12 |
| `hr` | int | **Hour only** — Hour of day: 0–23 |
| `holiday` | int | 1=holiday, 0=not holiday |
| `weekday` | int | Day of week: 0=Sunday, …, 6=Saturday |
| `workingday` | int | 1=working day (neither weekend nor holiday), 0=otherwise |
| `weathersit` | int | Weather situation: 1=Clear/Partly cloudy, 2=Mist/Cloudy, 3=Light snow/rain, 4=Heavy rain/snow |
| `temp` | float | Normalized temperature in Celsius (max=41, divided by 41) |
| `atemp` | float | Normalized feeling temperature (max=50, divided by 50) |
| `hum` | float | Normalized humidity (divided by 100) |
| `windspeed` | float | Normalized wind speed (max=67, divided by 67) |
| `casual` | int | Count of casual (non-registered) users |
| `registered` | int | Count of registered users |
| `cnt` | int | **Total rental count** = casual + registered |

---

## Key Statistics (hour.csv)

- **Total rentals (2 years)**: ~3.29 million
- **Peak hour**: 977 rentals (hour 17, late summer weekday)
- **Minimum hour**: 1 rental (early morning, bad weather)
- **Mean hourly rentals**: ~189
- **Weather impact**: Clear days ≈ 2.8× higher than heavy rain/snow

---

## Use in This Project

All forecasting models, seasonal decompositions, and visualizations use **only** these real datasets. No synthetic or generated data is used anywhere in the analysis pipeline.

---

*Generated automatically by `src/fetch_bike_data.py`*

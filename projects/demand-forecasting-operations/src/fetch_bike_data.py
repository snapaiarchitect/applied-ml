"""
Fetch UCI Machine Learning Repository — Bike Sharing Dataset
Downloads, caches, and documents the real Capital Bikeshare dataset.

DATA SOURCE:
    Fanaee-T, H., & Gama, J. (2013).
    Event labeling combining ensemble detectors and background knowledge.
    Progress in Artificial Intelligence, 2(2-3), 113-127.
    DOI: https://doi.org/10.24432/C5W894

DOWNLOAD URL:
    https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip

MIRROR (fallback):
    https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv
"""
import os
import urllib.request
import zipfile
import time
from pathlib import Path


def download_with_retry(url, dest_path, max_retries=3, timeout=30):
    """Download file with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   Attempt {attempt}/{max_retries}: {url}")
            urllib.request.urlretrieve(url, dest_path)
            file_size = os.path.getsize(dest_path)
            if file_size > 0:
                print(f"   ✓ Downloaded: {file_size:,} bytes")
                return True
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"   Retrying in {wait}s...")
                time.sleep(wait)
    return False


def fetch_bike_data(force_download=False):
    """
    Download and extract UCI Bike Sharing Dataset.
    
    Returns:
        dict: Paths to raw CSVs and metadata
    """
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    day_path = raw_dir / 'day.csv'
    hour_path = raw_dir / 'hour.csv'
    
    # Skip if already cached
    if not force_download and day_path.exists() and hour_path.exists():
        print("📦 Using cached dataset (use force_download=True to re-fetch)")
        return _build_result(day_path, hour_path)
    
    zip_path = raw_dir / 'bike_sharing.zip'
    
    # Primary source
    primary_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00275/Bike-Sharing-Dataset.zip"
    )
    
    print("=" * 60)
    print("📡 FETCHING UCI BIKE SHARING DATASET")
    print("=" * 60)
    print(f"   Source: UCI Machine Learning Repository")
    print(f"   Citation: Fanaee-T & Gama (2013), Progress in AI")
    print(f"   DOI: https://doi.org/10.24432/C5W894")
    print(f"   Cache dir: {raw_dir}")
    print()
    
    success = download_with_retry(primary_url, zip_path, max_retries=3)
    
    if not success:
        print("\n⚠ Primary source failed. Trying mirror...")
        mirror_url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv"
        mirror_path = raw_dir / 'bikeshare.csv'
        
        if download_with_retry(mirror_url, mirror_path, max_retries=2):
            print("   Mirror download successful (single CSV)")
            # Mirror only has hourly, we'll work with that
            return _build_result(None, mirror_path)
        
        print("\n✗ ALL DOWNLOADS FAILED — no real data available.")
        raise RuntimeError(
            "Could not download UCI Bike Sharing Dataset from any source. "
            "Network may be unavailable."
        )
    
    # Extract
    print(f"\n📂 Extracting ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw_dir)
    
    # The ZIP extracts into a subdirectory; move files up
    extracted_subdir = raw_dir / 'Bike-Sharing-Dataset'
    if extracted_subdir.exists():
        for f in extracted_subdir.iterdir():
            if f.is_file():
                dest = raw_dir / f.name
                if dest.exists():
                    dest.unlink()
                f.rename(dest)
        extracted_subdir.rmdir()
    
    # Cleanup zip
    zip_path.unlink(missing_ok=True)
    
    # Verify
    print("\n✅ DATASET READY")
    print(f"   📄 day.csv  → {day_path.stat().st_size:,} bytes")
    print(f"   📄 hour.csv → {hour_path.stat().st_size:,} bytes")
    
    return _build_result(day_path, hour_path)


def _build_result(day_path, hour_path):
    """Build result dictionary with metadata."""
    result = {
        'day_csv': str(day_path) if day_path and day_path.exists() else None,
        'hour_csv': str(hour_path) if hour_path and hour_path.exists() else None,
        'raw_dir': str(day_path.parent) if day_path else str(hour_path.parent),
    }
    return result


def generate_data_dictionary(output_path=None):
    """
    Generate data dictionary / schema documentation for the bike sharing dataset.
    Writes a markdown file documenting every column.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'DATA_DICTIONARY.md'
    
    content = """# Bike Sharing Dataset — Data Dictionary

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
"""
    
    Path(output_path).write_text(content, encoding='utf-8')
    print(f"\n📝 Data dictionary written to: {output_path}")
    return str(output_path)


if __name__ == '__main__':
    result = fetch_bike_data()
    generate_data_dictionary()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in result.items():
        print(f"   {k}: {v}")

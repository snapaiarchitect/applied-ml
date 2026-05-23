import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Verify all src modules are importable."""
    import fetch_bike_data
    import load_bike
    import preprocess
    import forecast
    import evaluate
    assert True


def test_data_dir_exists():
    """Verify data directory structure exists."""
    data_dir = Path(__file__).parent.parent / "data"
    assert data_dir.exists()
    assert (data_dir / "raw").exists()


def test_raw_data_files_exist():
    """Verify UCI Bike Sharing raw data files are present."""
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    required_files = ["hour.csv", "day.csv", "Readme.txt"]
    for f in required_files:
        assert (raw_dir / f).exists(), f"Missing: {f}"


def test_figures_dir_exists():
    """Verify figures directory exists."""
    figures_dir = Path(__file__).parent.parent / "figures"
    assert figures_dir.exists()


def test_results_json_exists():
    """Verify results.json is present and valid."""
    import json
    results_file = Path(__file__).parent.parent / "results.json"
    assert results_file.exists()
    with open(results_file) as f:
        data = json.load(f)
    assert "models" in data
    assert "xgboost" in data["models"]
    assert data["models"]["xgboost"]["mape"] < 0.25


def test_data_dictionary_exists():
    """Verify DATA_DICTIONARY.md is present."""
    dd_file = Path(__file__).parent.parent / "data" / "DATA_DICTIONARY.md"
    assert dd_file.exists()
    content = dd_file.read_text()
    assert "UCI" in content
    assert "Bike Sharing" in content

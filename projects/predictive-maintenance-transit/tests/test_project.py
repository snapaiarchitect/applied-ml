import sys
from pathlib import Path
import pytest

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Verify all src modules are importable."""
    import fetch_cmapss_data
    import load_nasa
    import preprocess
    import evaluate
    import survival_model
    import anomaly_detector
    assert True


def test_data_dir_exists():
    """Verify data directory structure exists."""
    data_dir = Path(__file__).parent.parent / "data"
    assert data_dir.exists()
    assert (data_dir / "raw").exists()


def test_raw_data_files_exist():
    """Verify NASA C-MAPSS raw data files are present."""
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    required_files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    for f in required_files:
        assert (raw_dir / f).exists(), f"Missing: {f}"


def test_figures_dir_exists():
    """Verify figures directory exists."""
    figures_dir = Path(__file__).parent.parent / "figures"
    assert figures_dir.exists()


def test_results_json_exists():
    """Verify results.json is present and valid."""
    results_file = Path(__file__).parent.parent / "results.json"
    assert results_file.exists()
    import json
    with open(results_file) as f:
        data = json.load(f)
    assert "models" in data
    assert "dataset" in data

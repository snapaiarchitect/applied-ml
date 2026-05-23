import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Verify all src modules are importable."""
    import fetch_20newsgroups
    import preprocess
    import train
    import evaluate
    import run_analysis
    import serve
    import verify_newsgroups
    assert True


def test_data_dir_exists():
    """Verify data directory structure exists."""
    data_dir = Path(__file__).parent.parent / "data"
    assert data_dir.exists()
    assert (data_dir / "raw").exists()


def test_raw_data_files_exist():
    """Verify 20 Newsgroups raw data files are present."""
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    required_files = ["20newsgroups_train.json", "20newsgroups_test.json", "20newsgroups_stats.csv"]
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
    assert "naive_bayes" in data["models"]
    assert data["models"]["naive_bayes"]["accuracy"] > 0.6

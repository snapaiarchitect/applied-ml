"""
CMAPSS Data Fetcher
Downloads and extracts NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

Data Source:
- NASA Prognostics Center of Excellence (PCoE)
- https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Mirror used (NASA direct download currently redirects to HTML):
- GitHub: https://github.com/ankur-mali/Predictive-Maintenance-Using-AI-
"""

import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# CMAPSS FD001 files - the classic turbofan engine degradation dataset
CMAPSS_FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]

# Raw data URLs from a reliable GitHub mirror
# (NASA direct download: https://ti.arc.nasa.gov/m/project/prognostic-repository/download/ 
#  currently serves HTML instead of the zip file)
GITHUB_MIRROR = "https://raw.githubusercontent.com/ankur-mali/Predictive-Maintenance-Using-AI-/main/"

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest: Path, retries: int = 3, backoff: float = 2.0) -> bool:
    """
    Download a file with exponential backoff retry logic.
    
    Args:
        url: Remote file URL
        dest: Local destination path
        retries: Number of retry attempts
        backoff: Initial backoff seconds (doubles each retry)
    
    Returns:
        True if download succeeded, False otherwise
    """
    wait = backoff
    for attempt in range(1, retries + 1):
        try:
            print(f"  [Attempt {attempt}/{retries}] Downloading {url} ...")
            urllib.request.urlretrieve(url, str(dest))
            if dest.stat().st_size > 0:
                print(f"  ✓ Saved to {dest} ({dest.stat().st_size:,} bytes)")
                return True
            else:
                raise RuntimeError("Downloaded file is empty")
        except Exception as e:
            print(f"  ✗ Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"  → Retrying in {wait:.1f}s ...")
                time.sleep(wait)
                wait *= 2
    return False

def fetch_cmapss_data(raw_dir: Path = None, processed_dir: Path = None) -> dict:
    """
    Download CMAPSS dataset files to local directories.
    
    Returns:
        dict with 'success', 'files', 'raw_dir', 'error' keys
    """
    project_root = Path(__file__).resolve().parent.parent
    
    if raw_dir is None:
        raw_dir = project_root / "data" / "raw"
    if processed_dir is None:
        processed_dir = project_root / "data" / "processed"
    
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    
    downloaded = []
    failed = []
    
    print(f"Downloading CMAPSS dataset to {raw_dir}")
    print(f"Source: NASA Prognostics Center of Excellence")
    print(f"Mirror: GitHub raw (ankur-mali/Predictive-Maintenance-Using-AI-)")
    print("-" * 50)
    
    for fname in CMAPSS_FILES:
        dest = raw_dir / fname
        if dest.exists():
            print(f"  ⏭ {fname} already exists ({dest.stat().st_size:,} bytes) — skipping")
            downloaded.append(dest)
            continue
        
        url = GITHUB_MIRROR + fname
        if download_file(url, dest, retries=3, backoff=2.0):
            downloaded.append(dest)
        else:
            failed.append(fname)
    
    print("-" * 50)
    
    if failed:
        return {
            "success": False,
            "files": downloaded,
            "raw_dir": raw_dir,
            "error": f"Failed to download: {', '.join(failed)}"
        }
    
    print(f"✓ All {len(downloaded)} CMAPSS files ready in {raw_dir}")
    return {
        "success": True,
        "files": downloaded,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "error": None
    }

def main():
    result = fetch_cmapss_data()
    if not result["success"]:
        print(f"\n✗ FAILED: {result['error']}")
        sys.exit(1)
    print("\n✓ CMAPSS data fetch complete. Ready for analysis.")

if __name__ == "__main__":
    main()

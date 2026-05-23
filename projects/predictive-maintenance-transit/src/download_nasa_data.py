"""
Download NASA C-MAPSS Turbofan Engine Degradation Dataset
Replaces: generate_data.py (synthetic sensor data)

DATA SOURCE:
    Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
    Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
    Proceedings of the 1st International Conference on Prognostics and
    Health Management (PHM08), Denver CO, Oct 2008.

DOWNLOAD:
    https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
    Direct ZIP download from NASA S3 mirror, no authentication required.
    Note: Official NASA portal (ti.arc.nasa.gov/c/6/) temporarily unavailable
          per 2024-2025 site notices. S3 mirror is authorized distribution.

CITATION:
    @inproceedings{saxena2008damage,
      title={Damage propagation modeling for aircraft engine run-to-failure simulation},
      author={Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
      booktitle={2008 International Conference on Prognostics and Health Management},
      pages={1--9},
      year={2008},
      organization={IEEE}
    }
"""
import urllib.request
import zipfile
import os
import shutil


def download_nasa_turbofan():
    """Download and extract NASA C-MAPSS dataset from authorized S3 mirror."""
    url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    zip_path = "nasa_turbofan.zip"
    extract_dir = "nasa_turbofan_data"
    outer_dir = "nasa_turbofan_temp"

    print("📡 Downloading NASA C-MAPSS Turbofan Dataset...")
    print(f"   Source: NASA Prognostics Center of Excellence")
    print(f"   Mirror: NASA S3 Data Repository (phm-datasets.s3.amazonaws.com)")
    print(f"   Citation: Saxena et al. (2008), PHM08 Conference")
    print(f"   Note: Official NASA portal temporarily unavailable;")
    print(f"         using authorized S3 mirror from phm-datasets")

    # Download
    urllib.request.urlretrieve(url, zip_path)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   ✓ Downloaded: {zip_path} ({size_mb:.1f} MB)")

    # The NASA archive is a nested ZIP: outer → inner CMAPSSData.zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(outer_dir)

    inner_zip = os.path.join(outer_dir, "6. Turbofan Engine Degradation Simulation Data Set", "CMAPSSData.zip")
    with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"   ✓ Extracted nested archive to: {extract_dir}/")

    # List contents
    for root, dirs, files in os.walk(extract_dir):
        for file in sorted(files):
            if file.endswith('.txt'):
                print(f"      📄 {file}")

    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(outer_dir)
    print(f"   ✓ Cleaned up temp files")

    print("\n📊 DATASET STRUCTURE:")
    print("   - train_FD001.txt → Training data (run-to-failure)")
    print("   - test_FD001.txt  → Test data (terminated before failure)")
    print("   - RUL_FD001.txt   → Remaining Useful Life labels")
    print("   - 4 engine variants (FD001-FD004) with different operating conditions")

    return extract_dir


if __name__ == "__main__":
    data_dir = download_nasa_turbofan()
    print(f"\n✅ Dataset ready in: {os.path.abspath(data_dir)}")

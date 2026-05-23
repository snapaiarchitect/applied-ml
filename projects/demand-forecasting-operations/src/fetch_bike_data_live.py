"""
Live Capital Bikeshare GBFS Ingestion Pipeline
===============================================
Replaces static UCI archive with real-time station availability feed.

DATA SOURCE:
    Capital Bikeshare (Lyft Bikes)
    GBFS v1.1 Station Status Feed
    https://gbfs.lyft.com/gbfs/1.1/dca-cabi/en/station_status.json
    Updates: Every ~60 seconds during operating hours

WHAT IT DOES:
    1. Pulls live station status (bikes available, docks open, station health)
    2. Appends snapshot to time-series database (CSV/Parquet)
    3. Computes system-wide aggregates (total bikes, % utilization, dead stations)
    4. Returns structured DataFrame for forecasting pipeline

FORECASTING TARGET:
    Instead of static historical trip counts (2011–2012), we now forecast:
    - System-wide bike availability (next 1–6 hours)
    - Station-level utilization rate (predict empty/full stations)
    - Demand surge detection (anomalous checkout patterns)

This is a genuine production operations pipeline hitting a live public API.
"""

import json
import urllib.request
import time
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────────

GBFS_STATUS_URL = "https://gbfs.lyft.com/gbfs/1.1/dca-cabi/en/station_status.json"
GBFS_INFO_URL   = "https://gbfs.lyft.com/gbfs/1.1/dca-cabi/en/station_information.json"
PROJECT_ROOT    = Path(__file__).parent.parent
TIMESERIES_DIR  = PROJECT_ROOT / "data" / "live_timeseries"
SNAPSHOT_DIR    = PROJECT_ROOT / "data" / "live_snapshots"
CACHE_FILE      = TIMESERIES_DIR / "station_timeseries.parquet"
LATEST_CSV      = PROJECT_ROOT / "data" / "processed" / "latest_system_status.csv"

TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_json(url: str, timeout: int = 30, retries: int = 3) -> dict:
    """Fetch JSON with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {e}")
            wait = 2 ** attempt
            print(f"   Retry {attempt}/{retries} in {wait}s...")
            time.sleep(wait)


def fetch_live_station_status() -> pd.DataFrame:
    """
    Pull live station status from Capital Bikeshare GBFS API.
    Returns DataFrame with one row per station + system aggregates.
    """
    print("📡 FETCHING LIVE CAPITAL BIKESHARE STATUS")
    print(f"   Endpoint: {GBFS_STATUS_URL}")
    print(f"   Time:     {datetime.now(timezone.utc).isoformat()}")
    print()

    # ── 1. Fetch feeds ──────────────────────────────────────────────────────
    status_data = _fetch_json(GBFS_STATUS_URL)
    info_data   = _fetch_json(GBFS_INFO_URL)

    last_updated = status_data.get("last_updated", int(time.time()))
    ttl          = status_data.get("ttl", 60)

    stations_status = {s["station_id"]: s for s in status_data["data"]["stations"]}
    stations_info   = {s["station_id"]: s for s in info_data["data"]["stations"]}

    # ── 2. Merge status + info ──────────────────────────────────────────────
    records = []
    for sid, status in stations_status.items():
        info = stations_info.get(sid, {})
        capacity = info.get("capacity", 0)
        total_docks = status.get("num_docks_available", 0) + status.get("num_bikes_available", 0)
        capacity = capacity if capacity > 0 else total_docks
        utilization = (status.get("num_bikes_available", 0) / capacity) if capacity > 0 else 0

        records.append({
            "station_id":           sid,
            "legacy_id":            status.get("legacy_id", ""),
            "name":                 info.get("name", "Unknown"),
            "lat":                  info.get("lat", 0),
            "lon":                  info.get("lon", 0),
            "capacity":             capacity,
            "num_bikes_available":  status.get("num_bikes_available", 0),
            "num_ebikes_available": status.get("num_ebikes_available", 0),
            "num_docks_available":  status.get("num_docks_available", 0),
            "num_bikes_disabled":   status.get("num_bikes_disabled", 0),
            "num_docks_disabled":   status.get("num_docks_disabled", 0),
            "is_renting":           bool(status.get("is_renting", 0)),
            "is_returning":         bool(status.get("is_returning", 0)),
            "is_installed":         bool(status.get("is_installed", 0)),
            "last_reported":        status.get("last_reported", 0),
            "utilization_rate":     round(utilization, 4),
            "timestamp_utc":        datetime.now(timezone.utc),
            "gbfs_last_updated":    datetime.fromtimestamp(last_updated, tz=timezone.utc),
        })

    df = pd.DataFrame(records)

    # ── 3. System-wide aggregates ───────────────────────────────────────────
    active_stations = df[df["is_renting"] == True]
    system_stats = {
        "total_stations":      len(df),
        "active_stations":     len(active_stations),
        "dead_stations":       len(df[df["is_renting"] == False]),
        "total_bikes":         int(df["num_bikes_available"].sum()),
        "total_ebikes":        int(df["num_ebikes_available"].sum()),
        "total_docks_open":    int(df["num_docks_available"].sum()),
        "system_utilization":  round(df["utilization_rate"].mean(), 4),
        "empty_stations":      int((df["num_bikes_available"] == 0).sum()),
        "full_stations":       int((df["num_docks_available"] == 0).sum()),
        "timestamp_utc":       datetime.now(timezone.utc),
        "gbfs_ttl_seconds":    ttl,
    }

    print(f"✅ LIVE SNAPSHOT CAPTURED")
    print(f"   Stations:      {system_stats['total_stations']} total | {system_stats['active_stations']} active | {system_stats['dead_stations']} dead")
    print(f"   Bikes:         {system_stats['total_bikes']} regular + {system_stats['total_ebikes']} ebikes")
    print(f"   Docks open:    {system_stats['total_docks_open']}")
    print(f"   Utilization:   {system_stats['system_utilization']:.1%}")
    print(f"   Empty:         {system_stats['empty_stations']} | Full: {system_stats['full_stations']}")
    print(f"   API updated:   {system_stats.get('gbfs_last_updated', 'N/A')}")

    return df, system_stats


def append_to_timeseries(df: pd.DataFrame) -> Path:
    """Append current snapshot to persistent time-series store."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Parquet append (efficient for large history) ────────────────────────
    if CACHE_FILE.exists():
        existing = pd.read_parquet(CACHE_FILE)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_parquet(CACHE_FILE, index=False)

    # ── Also save raw JSON snapshot for audit/debug ──────────────────────────
    snapshot_path = SNAPSHOT_DIR / f"snapshot_{timestamp}.json"
    snapshot_path.write_text(df.to_json(orient="records", indent=2))

    # ── Export latest processed CSV for downstream models ────────────────────
    LATEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LATEST_CSV, index=False)

    print(f"\n💾 PERSISTED")
    print(f"   Time-series: {CACHE_FILE} ({len(combined):,} total rows)")
    print(f"   Snapshot:    {snapshot_path}")
    print(f"   Latest CSV:  {LATEST_CSV}")

    return CACHE_FILE


def fetch_bike_data_live(force_refresh: bool = True) -> dict:
    """
    Main entry point — replaces the legacy UCI static fetcher.
    Returns dict compatible with downstream preprocess.py
    """
    df, system_stats = fetch_live_station_status()
    cache_path = append_to_timeseries(df)

    return {
        "source":        "Capital Bikeshare GBFS v1.1 (Live)",
        "endpoint":      GBFS_STATUS_URL,
        "data_type":     "station-level availability",
        "forecast_target": "system-wide utilization + station-level availability",
        "stations":      len(df),
        "system_stats":  system_stats,
        "dataframe":     df,
        "timeseries_path": str(cache_path),
        "latest_csv":    str(LATEST_CSV),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    result = fetch_bike_data_live()
    print("\n" + "=" * 60)
    print("LIVE INGESTION COMPLETE")
    print("=" * 60)
    for k, v in result.items():
        if k not in ("dataframe", "system_stats"):
            print(f"   {k}: {v}")
    print(f"\n   Next step: python src/preprocess.py")
    print("   Then:      python src/forecast.py")

# NASA C-MAPSS FD001 Data Dictionary

**Dataset**: NASA Prognostics Center of Excellence — Turbofan Engine Degradation Simulation  
**Source**: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)  
**Last Updated**: 2008 (static dataset)  
**Update Frequency**: N/A — benchmark dataset  

---

## Files

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `train_FD001.txt` | Space-delimited | 20,631 | 100 engines run-to-failure |
| `test_FD001.txt` | Space-delimited | 13,096 | 100 engines truncated before failure |
| `RUL_FD001.txt` | Single column | 100 | True Remaining Useful Life per test engine |

---

## Columns

| Position | Column | Type | Description |
|----------|--------|------|-------------|
| 1 | `unit_id` | int | Engine identifier (1–100) |
| 2 | `time_cycles` | int | Operational cycle number |
| 3–5 | `setting_1`–`setting_3` | float | Operating settings (altitude, Mach, throttle) |
| 6–26 | `sensor_1`–`sensor_21` | float | Sensor measurements (temperature, pressure, speed, etc.) |

**Sensor Names** (informative):
- `sensor_1`–`sensor_3`: Fan inlet temperature, LPC outlet temperature, HPC outlet temperature
- `sensor_4`–`sensor_6`: LPT outlet temperature, Fan inlet pressure, Bypass duct pressure
- `sensor_7`–`sensor_9`: HPC outlet pressure, Physical fan speed, Physical core speed
- `sensor_10`–`sensor_12`: Engine pressure ratio, HPC outlet static pressure, Ratio of fuel flow to Ps30
- `sensor_13`–`sensor_15`: Corrected fan speed, Corrected core speed, Bypass ratio
- `sensor_16`–`sensor_18`: Burner fuel-air ratio, Bleed enthalpy, Required fan speed
- `sensor_19`–`sensor_21`: Required fan conversion speed, High-pressure turbines cool air flow, Low-pressure turbines cool air flow

---

## Data Quality Notes

- All engines operate at **sea level** (FD001 condition)
- No missing values
- `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` are **constant** (no variance) — drop during preprocessing
- Degradation is gradual; failure defined by threshold crossing

---

## API vs. Fallback

| Scenario | Behavior |
|----------|----------|
| NASA website available | `src/fetch_cmapss_data.py` downloads directly from NASA repository |
| NASA website down | Uses cached local copy in `data/raw/` (checked into repo) |
| Fresh download needed | Manual download from [NASA Prognostics](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |

---

## Citation

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

"""Is the downloaded history complete?  Run:  python check_history_completeness.py
Compares historical_reported_observations.csv with train_coverage_summary.csv (written by the collector)."""
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent
f = H / "historical_reported_observations.csv"
if not f.exists(): raise SystemExit(f"Put historical_reported_observations.csv in {H} first.")
cov = pd.read_csv(H / "train_coverage_summary.csv"); exp = int(cov["rows_with_any_actual_time"].sum())
o = pd.read_csv(f, usecols=["train_no"], dtype=str)
have = set(o["train_no"].astype(int)); want = set(cov["train_no"].astype(int)); miss = sorted(want - have)
print(f"rows in file: {len(o):,}   expected: {exp:,}   ({100 * len(o) / exp:.0f}%)")
print(f"trains in file: {len(have)} of {len(want)}")
if len(o) in (1048575, 1048576) or miss:
    print("PROBLEM: the file is cut off (1,048,575 = Excel's row limit). Missing trains:", miss[:30], "..." if len(miss) > 30 else "")
    print("Fix: run collect_railway_history.py again and never open the CSV in Excel.")
else:
    print("OK: the file is complete.")

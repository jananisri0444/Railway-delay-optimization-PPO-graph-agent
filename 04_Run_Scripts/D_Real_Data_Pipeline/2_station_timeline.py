"""Station-by-station REAL delay timeline for one day / date range / month.
   python 2_station_timeline.py --date 2025-03-05      (or --from/--to, --month 2025-03, or nothing = busiest day)
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
from tools.real_timeline import main
main()

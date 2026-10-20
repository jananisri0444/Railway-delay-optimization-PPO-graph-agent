"""Table alone (no dashboard file). Usage: python 3b_table_only.py --date 2025-03-05   (same options as 3_dashboard_and_table.py)"""
import sys, runpy, pathlib
sys.argv += ["--only", "table"]
runpy.run_path(str(pathlib.Path(__file__).with_name("3_dashboard_and_table.py")), run_name="__main__")

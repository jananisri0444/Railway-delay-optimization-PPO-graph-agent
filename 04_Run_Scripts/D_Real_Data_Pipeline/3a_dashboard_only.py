"""Dashboard alone (no table files). Usage: python 3a_dashboard_only.py --date 2025-03-05   (same options as 3_dashboard_and_table.py)"""
import sys, runpy, pathlib
sys.argv += ["--only", "dashboard"]
runpy.run_path(str(pathlib.Path(__file__).with_name("3_dashboard_and_table.py")), run_name="__main__")

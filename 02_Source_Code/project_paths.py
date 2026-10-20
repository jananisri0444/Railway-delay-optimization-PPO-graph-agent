"""Single place that knows where everything lives. Every run script imports this first."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # hides the harmless "OMP Error #15" message on Windows / Anaconda
from pathlib import Path

CODE = Path(__file__).resolve().parent            # 02_Source_Code
ROOT = CODE.parent                                # project root
SIM_DATA = ROOT / '01_Data' / '1_Simulator_Timetable'
HISTORY = ROOT / '01_Data' / '2_Real_Railway_History'
REAL_PROC = ROOT / '01_Data' / '3_Real_Delay_Data_Processed'
MODELS = ROOT / '03_Trained_Models'
RESULTS = ROOT / '05_Results'
OUT_SYN = RESULTS / 'A_Synthetic_Testing'
OUT_REAL = RESULTS / 'B_Real_Data_Replay'
OUT_COMPARE = OUT_REAL / 'compare_policies'
OUT_REAL_EVAL = OUT_REAL / 'evaluate_on_real_scenarios'
OUT_DASH = OUT_REAL / 'dashboard_and_table'

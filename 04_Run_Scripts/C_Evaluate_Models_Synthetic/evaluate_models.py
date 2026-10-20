"""Compare all policies on the SAME unseen synthetic stress scenarios.
   python evaluate_models.py --seeds 21 --steps 200      -> 05_Results/A_Synthetic_Testing/evaluation.csv
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
from tools.policy_eval import main
main()

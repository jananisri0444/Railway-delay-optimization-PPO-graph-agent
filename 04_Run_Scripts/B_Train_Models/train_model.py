"""Train a PPO model (graph or flat). Same options as decision_layer/train_ppo.py.

Examples (run from anywhere):
  python train_model.py --obs-mode graph --teacher conflict_aware --bc-episodes 60 --episodes 300 --decision-steps 200 --stress --lr 1e-4 --ent 0.003 --seed-offset 100
  python train_model.py --obs-mode flat --bc-episodes 60 --episodes 300 --decision-steps 200 --stress --lr 2e-4 --ent 0.005 --seed-offset 100 --save ../../03_Trained_Models/policy_flat.pt
By default the model is saved to 03_Trained_Models/policy_final.pt (COPY the folder first if you want to keep the old model).
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
from decision_layer.train_ppo import main
main()

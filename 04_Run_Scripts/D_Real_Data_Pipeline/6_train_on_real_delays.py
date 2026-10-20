"""OPTIONAL STEP 3: fine-tune / train a model on the REAL-delay TRAIN split (older dates).
Same arguments as decision_layer/train_ppo.py. Examples (run from the project folder):
  python train_on_real_delays.py --obs-mode graph --init output/checkpoints/policy_final.pt --episodes 200 --decision-steps 200 --lr 5e-5 --ent 0.003 --save output/checkpoints/policy_real_graph.pt
  python train_on_real_delays.py --obs-mode flat  --init output/checkpoints/policy_flat.pt  --episodes 200 --decision-steps 200 --lr 1e-4 --ent 0.003 --save output/checkpoints/policy_real_flat.pt
Then test on the held-out TEST days with evaluate_real_delays.py --real-graph-ckpt ... / --real-flat-ckpt ...
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import os, sys
import decision_layer.train_ppo as T
from tools.real_delay_env import RealDelayEnv
T.RailwayPPOEnv = lambda start, end, stress_mode=False, weekday=None, obs_mode='graph', **k: RealDelayEnv(start, end, stress_mode=stress_mode, obs_mode=obs_mode, split='train')
if __name__ == '__main__': T.main()

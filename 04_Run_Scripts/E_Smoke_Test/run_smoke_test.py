"""Quick check that the simulator + graph + environment work (runs in a few seconds)."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
from decision_layer.environment import RailwayPPOEnv
import numpy as np

env = RailwayPPOEnv(480, 600, stress_mode=True, seed=42, max_ticks=200)
obs, _ = env.reset(seed=42)
decisions = 0
for _ in range(100):
    obs, done = env.advance_to_decision(20)
    if done:
        break
    decisions += len(env.decision_candidates())
    obs, reward, term, trunc, info = env.step(np.zeros(5, dtype=int))
    if term or trunc:
        break
print(f"state_dim={env.state_dim}")
print(f"decision_candidates_seen={decisions}")
print(f"simulation_time={env._rs['simulation_time_min']}")
print(f"controllable_delay={env.total_delay():.1f}")

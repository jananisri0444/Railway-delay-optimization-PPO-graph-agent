"""STEP 2: test your models on REAL historical delay days (held-out test split).

Usage:
    python evaluate_real_delays.py                       # graph PPO, flat PPO and the baselines on the test days
    python evaluate_real_delays.py --max-days 40 --steps 200
    python evaluate_real_delays.py --stress              # also add synthetic disruptions on top of real lateness
    python evaluate_real_delays.py --real-graph-ckpt output/checkpoints/policy_real_graph.pt   # a model fine-tuned on real days
Reward = -(importance-weighted delay added by dispatching); closer to 0 is better. Entry lateness is NOT charged.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import argparse, csv, os, sys
import numpy as np
from tools.policy_eval import ppo_policy, run_episode
from tools.real_delay_env import RealDelayEnv
from decision_layer.baselines import always_proceed, always_hold, priority_rule, conflict_aware_rule

ap = argparse.ArgumentParser()
ap.add_argument('--scenarios', default=str(P.REAL_PROC / 'real_day_scenarios.json')); ap.add_argument('--split', default='test')
ap.add_argument('--max-days', type=int, default=30); ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--start', type=float, default=480); ap.add_argument('--end', type=float, default=720)
ap.add_argument('--stress', action='store_true')
ap.add_argument('--ckpt', default=str(P.MODELS / 'policy_final.pt')); ap.add_argument('--flat-ckpt', default=str(P.MODELS / 'policy_flat.pt'))
ap.add_argument('--real-graph-ckpt', default=''); ap.add_argument('--real-flat-ckpt', default='')
a = ap.parse_args()
mk = lambda mode: RealDelayEnv(a.start, a.end, scenario_file=a.scenarios, split=a.split, stress_mode=a.stress, obs_mode=mode)
flat_env = mk('flat'); graph_env = None
n = min(a.max_days, len(flat_env.scenarios)); seeds = list(range(n))
pols = {'always_hold': (always_hold, flat_env), 'always_proceed': (always_proceed, flat_env),
        'priority_rule': (priority_rule, flat_env), 'conflict_aware_rule': (conflict_aware_rule, flat_env)}
if os.path.exists(a.flat_ckpt): pols['flat_ppo'] = (ppo_policy(a.flat_ckpt), flat_env)
if a.real_flat_ckpt: pols['flat_ppo_real'] = (ppo_policy(a.real_flat_ckpt), flat_env)
for name, path in (('graph_ppo', a.ckpt), ('graph_ppo_real', a.real_graph_ckpt)):
    if path and os.path.exists(path):
        graph_env = graph_env or mk('graph'); pols[name] = (ppo_policy(path), graph_env)
print(f'{n} real days from split "{a.split}", window {a.start/60:.0f}:00-{a.end/60:.0f}:00' + (' (+ synthetic disruptions)' if a.stress else ''))
res, dec = {}, {}
for name, (pol, env) in pols.items():
    r, d = [], 0
    for s in seeds:
        x, ac = run_episode(env, pol, s, a.steps); r.append(x); d += sum(ac.values())
    res[name] = np.array(r); dec[name] = d / n; print('finished', name, flush=True)
print(f"\n{'policy':<22}{'mean':>9}{'std':>8}{'sem':>7}   decisions/day")
for k, v in res.items(): print(f'{k:<22}{v.mean():9.1f}{v.std(ddof=1):8.1f}{v.std(ddof=1)/np.sqrt(len(v)):7.1f}   {dec[k]:.1f}')
for m in [k for k in res if 'ppo' in k]:
    for b in ('always_proceed', 'priority_rule'):
        d = res[m] - res[b]; se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = 'better' if d.mean() - 2 * se > 0 else ('worse' if d.mean() + 2 * se < 0 else 'no clear difference')
        print(f'{m} - {b:<15} = {d.mean():+.1f} +/- {2*se:.1f} (95%)  -> {verdict}')
os.makedirs(P.OUT_REAL_EVAL, exist_ok=True)
with open(str(P.OUT_REAL_EVAL / 'real_results.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['day_index', 'date', 'weekday'] + list(res))
    for i, s in enumerate(seeds): w.writerow([i, flat_env.scenarios[i]['date'], flat_env.scenarios[i]['weekday']] + [round(float(res[k][i]), 2) for k in res])
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = sorted(res, key=lambda k: res[k].mean()); fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ks, [res[k].mean() for k in ks], xerr=[2 * res[k].std(ddof=1) / np.sqrt(n) for k in ks], capsize=4,
            color=['#14B8A6' if 'graph' in k else '#F59E0B' if 'flat' in k else '#64748B' for k in ks])
    ax.set_xlabel('Average reward on real historical days (closer to 0 = better). Bars = 95% margin'); ax.set_title('Policies on real delay days')
    plt.tight_layout(); plt.savefig(str(P.OUT_REAL_EVAL / 'real_comparison.png'), dpi=150)
except Exception as e: print('chart skipped:', e)
print('\nSaved results in', P.OUT_REAL_EVAL)

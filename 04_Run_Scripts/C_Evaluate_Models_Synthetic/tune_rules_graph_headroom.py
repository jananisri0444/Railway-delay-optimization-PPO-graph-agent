"""IS THERE ANYTHING FOR A GRAPH MODEL TO WIN? Searches the best hand-made rule that sees ONLY the priority gap (what a flat model can
see) and the best rule that also sees the whole conflict neighbourhood (what the graph provides), then tests both on unseen scenarios.
If the second is not clearly better, no graph model can be clearly better in this simulator.
    python tune_rules_graph_headroom.py --train-seeds 10 --test-seeds 24      (takes ~20-30 min on a laptop CPU)
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P
import argparse, json, itertools, time
import numpy as np
from decision_layer.environment import RailwayPPOEnv, MAX_DECISIONS
from decision_layer.baselines import priority_rule
from tools.policy_eval import run_episode

ap = argparse.ArgumentParser(); ap.add_argument('--train-seeds', type=int, default=10); ap.add_argument('--test-seeds', type=int, default=24); ap.add_argument('--steps', type=int, default=200)
a = ap.parse_args(); env = RailwayPPOEnv(480, 720, stress_mode=True, obs_mode='flat'); w = lambda p: (7 - p) / 2.0

def gap_rule(h, l):                      # FLAT-visible information only: priority gap to the main blocked train
    def rule(e):
        m = e.action_mask(); act = priority_rule(e)
        for i, c in enumerate(e._candidates[:MAX_DECISIONS]):
            if c['kind'] != 'section': continue
            g = e._train_priority(c['train_number']) - e._train_priority(c['opponent']); act[i] = 1
            if g >= h: act[i] = 0
            if g >= l and m[i, 2]: act[i] = 2
            if not m[i, act[i]]: act[i] = 0
        return act
    return rule

def neighbourhood_rule(al, be):          # GRAPH-visible information: total blocking cost of EVERY blocked train vs cost of holding
    def rule(e):
        m = e.action_mask(); act = priority_rule(e)
        for i, c in enumerate(e._candidates[:MAX_DECISIONS]):
            if c['kind'] != 'section': continue
            pl = e._train_priority(c['train_number']); B = sum(0.25 * (pl - e._train_priority(x)) * w(e._train_priority(x)) for x in c['blocked'] if pl > e._train_priority(x)); H = w(pl)
            act[i] = 1
            if B > al * H: act[i] = 0
            if B > be * H and m[i, 2]: act[i] = 2
            if not m[i, act[i]]: act[i] = 0
        return act
    return rule

def score(rule, seeds): return np.array([run_episode(env, rule, s, a.steps)[0] for s in seeds])
tr = list(range(100, 100 + a.train_seeds)); te = list(range(2000, 2000 + a.test_seeds)); res = {'gap': {}, 'neigh': {}}; t0 = time.time()
for h, l in itertools.product((2, 3, 4, 5), (4, 6, 99)):
    res['gap'][f'{h},{l}'] = float(score(gap_rule(h, l), tr).mean()); print(f'gap rule hold>={h} loop>={l}: {res["gap"][f"{h},{l}"]:.1f}   [{time.time() - t0:.0f}s]', flush=True)
for al, be in itertools.product((0.7, 1.0, 1.5, 2.5), (2, 4, 99)):
    res['neigh'][f'{al},{be}'] = float(score(neighbourhood_rule(al, be), tr).mean()); print(f'neighbourhood rule alpha={al} beta={be}: {res["neigh"][f"{al},{be}"]:.1f}   [{time.time() - t0:.0f}s]', flush=True)
bg = max(res['gap'], key=res['gap'].get); bn = max(res['neigh'], key=res['neigh'].get); h, l = map(int, bg.split(',')); al, be = map(float, bn.split(','))
print(f'\nbest flat-visible rule: gap {bg} (train {res["gap"][bg]:.1f});  best graph-visible rule: {bn} (train {res["neigh"][bn]:.1f})')
S = {'priority_rule': score(priority_rule, te), 'best_gap_rule': score(gap_rule(h, l), te), 'best_neighbourhood_rule': score(neighbourhood_rule(al, be), te)}
print(f'\nHELD-OUT TEST on {a.test_seeds} unseen scenarios (higher is better):')
for k, v in S.items(): print(f'  {k:26s} {v.mean():8.1f}  +/- {2 * v.std(ddof=1) / np.sqrt(len(v)):.1f} (95%)')
d = S['best_neighbourhood_rule'] - S['best_gap_rule']; se = d.std(ddof=1) / np.sqrt(len(d))
print(f'  neighbourhood minus gap rule = {d.mean():+.1f} +/- {2 * se:.1f} (95%)  ->', 'graph-visible information HELPS' if d.mean() - 2 * se > 0 else ('graph-visible information HURTS' if d.mean() + 2 * se < 0 else 'no clear difference'))
os.makedirs(P.OUT_SYN, exist_ok=True); json.dump({'sweep': res, 'best_gap': bg, 'best_neigh': bn, 'test': {k: v.tolist() for k, v in S.items()}}, open(P.OUT_SYN / 'rule_sweep.json', 'w'))

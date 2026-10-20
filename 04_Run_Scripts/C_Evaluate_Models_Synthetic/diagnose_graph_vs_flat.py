"""WHY do Graph PPO and Flat PPO give (almost) the same result?  Answers it with numbers.

For many decision moments the SAME situation is shown to both models and we count how often they choose the same action,
how often each agrees with the two rule teachers, and how different their probabilities are.
    python diagnose_graph_vs_flat.py --seeds 12          -> prints a report, saves 05_Results/A_Synthetic_Testing/diagnosis_graph_vs_flat.csv
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P
import argparse, collections
import numpy as np, torch
from decision_layer.environment import RailwayPPOEnv, MAX_DECISIONS
from decision_layer.ppo_agent import GraphPPOActorCritic
from decision_layer.baselines import priority_rule, conflict_aware_rule
from representation_layer.flat_state import build_flat_state

ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=12); ap.add_argument('--first-seed', type=int, default=2000)
ap.add_argument('--steps', type=int, default=200); ap.add_argument('--graph-ckpt', default=str(P.MODELS / 'policy_final.pt')); ap.add_argument('--flat-ckpt', default=str(P.MODELS / 'policy_flat.pt'))
a = ap.parse_args()

def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False); use_graph = ck.get('use_graph', any(k.startswith('graph_encoder.') for k in ck['actor_critic'])); m = GraphPPOActorCritic(ck['state_dim'], use_graph=use_graph); m.load_state_dict(ck['actor_critic'], strict=False); m.eval(); return m
G, F = load(a.graph_ckpt), load(a.flat_ckpt)
env = RailwayPPOEnv(480, 720, stress_mode=True, obs_mode='graph')
def probs(model, obs, mask, ctx=None):
    with torch.no_grad(): dists, _ = model.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0), graph_contexts=[ctx] if model.use_graph else None)
    return np.stack([d.probs.squeeze(0).numpy() for d in dists])
rows = []; agree = collections.Counter(); tot = collections.Counter(); pdiff = []; ag_g = collections.Counter(); ag_f = collections.Counter(); ag_gf_t = collections.Counter(); share = collections.Counter()
for sd in range(a.first_seed, a.first_seed + a.seeds):
    env.reset(seed=sd); n = 0
    while n < a.steps:
        _, done = env.advance_to_decision()
        if done: break
        if not env._candidates: continue
        n += 1; mask = env.action_mask(); gobs = env._obs
        fobs = np.concatenate([build_flat_state(env._rs), env._slot_features()]).astype(np.float32)
        ctx = env.graph_context()
        pg, pf = probs(G, gobs, mask, ctx), probs(F, fobs, mask); ag, af = pg.argmax(1), pf.argmax(1)
        tp, tc = priority_rule(env), conflict_aware_rule(env)
        for i, c in enumerate(env._candidates[:MAX_DECISIONS]):
            k = c['kind']; tot[k] += 1; agree[k] += int(ag[i] == af[i]); pdiff.append(float(np.abs(pg[i] - pf[i]).max()))
            ag_g[k, 'priority'] += int(ag[i] == tp[i]); ag_g[k, 'conflict_aware'] += int(ag[i] == tc[i]); ag_f[k, 'priority'] += int(af[i] == tp[i]); ag_f[k, 'conflict_aware'] += int(af[i] == tc[i])
            share[k, 'graph', int(ag[i])] += 1; share[k, 'flat', int(af[i])] += 1
            rows.append((sd, k, int(ag[i]), int(af[i]), round(float(pg[i].max()), 3), round(float(pf[i].max()), 3), c.get('opponent') is not None and env._train_priority(c['train_number']) - env._train_priority(c['opponent'])))
        env.step(ag if True else af)          # drive with the Graph PPO's own choices
N = sum(tot.values()); print(f'\n{N} decisions looked at ({a.seeds} stress scenarios). Both models saw the identical situation each time.\n')
print('Same action chosen by Graph and Flat PPO:')
for k in tot: print(f'  {k:8s}: {100 * agree[k] / tot[k]:5.1f}%  of {tot[k]} decisions')
print(f'  all     : {100 * sum(agree.values()) / N:5.1f}%      mean largest probability difference: {np.mean(pdiff):.3f}')
print('\nAgreement with the rule teachers (% of decisions):\n  kind      model   priority_rule  conflict_aware_rule')
for k in tot:
    for nm, d in (('graph', ag_g), ('flat', ag_f)): print(f'  {k:8s}  {nm:5s}   {100 * d[k, "priority"] / tot[k]:6.1f}%       {100 * d[k, "conflict_aware"] / tot[k]:6.1f}%')
print('\nAction mix (0 = HOLD, 1 = PROCEED / GRANT PLATFORM, 2 = LOOP):')
for k in tot: print(f'  {k:8s} graph', {x: share[k, "graph", x] for x in (0, 1, 2)}, ' flat', {x: share[k, 'flat', x] for x in (0, 1, 2)})
os.makedirs(P.OUT_SYN, exist_ok=True)
import csv
with open(P.OUT_SYN / 'diagnosis_graph_vs_flat.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['seed', 'kind', 'graph_action', 'flat_action', 'graph_max_prob', 'flat_max_prob', 'priority_gap']); w.writerows(rows)
print('\nSaved', P.OUT_SYN / 'diagnosis_graph_vs_flat.csv')

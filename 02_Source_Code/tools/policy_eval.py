"""Compare PPO against baselines on the SAME unseen seeds.

Usage (from the project folder):
    python evaluate.py                                   # 30 seeds, stress mode
    python evaluate.py --seeds 50 --ckpt output/checkpoints/policy_best.pt
    python evaluate.py --realistic                       # realistic mode instead of stress
Reward is a cost: closer to zero is better.
"""
import argparse, csv, os, sys
import numpy as np, torch
import project_paths as P
from decision_layer.environment import RailwayPPOEnv, MAX_DECISIONS
from decision_layer.ppo_agent import GraphPPOActorCritic
from decision_layer.baselines import always_proceed, always_hold, priority_rule, conflict_aware_rule

def ppo_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    use_graph = ck.get('use_graph', any(k.startswith('graph_encoder.') for k in ck['actor_critic']))
    ag = GraphPPOActorCritic(ck['state_dim'], use_graph=use_graph); ag.load_state_dict(ck['actor_critic'], strict=False); ag.eval()
    def f(env):
        m = torch.as_tensor(env.action_mask()).unsqueeze(0); sv = torch.as_tensor(env.slot_valid()).unsqueeze(0)
        st = torch.as_tensor(env._obs, dtype=torch.float32).unsqueeze(0)
        ctx = env.graph_context() if ag.use_graph else None
        with torch.no_grad(): a, _, _ = ag.act(st, m, deterministic=True, slot_valid=sv, graph_contexts=[ctx] if ag.use_graph else None)
        return a[0].numpy()
    return f

def run_episode(env, policy, seed, steps):
    """steps = max number of DECISIONS. advance_to_decision() gives up after 100 quiet minutes, so keep calling it until done."""
    env.reset(seed=seed); total = 0.0; acts = {0: 0, 1: 0, 2: 0}; n = 0
    while n < steps:
        _, done = env.advance_to_decision()
        if done: break
        if not env._candidates: continue
        a = policy(env); n += 1
        for i in range(min(len(env._candidates), MAX_DECISIONS)): acts[int(a[i])] += 1
        _, r, term, trunc, _ = env.step(a); total += r
        if term or trunc: break
    return total, acts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30); ap.add_argument('--first-seed', type=int, default=2000)
    ap.add_argument('--steps', type=int, default=150); ap.add_argument('--start', type=float, default=480)
    ap.add_argument('--end', type=float, default=720); ap.add_argument('--realistic', action='store_true')
    ap.add_argument('--weekday', type=int, default=-1, help='0=Mon..6=Sun, -1 = all days')
    ap.add_argument('--ckpt', default=str(P.MODELS / 'policy_final.pt'))
    ap.add_argument('--flat-ckpt', default=str(P.MODELS / 'policy_flat.pt'))
    a = ap.parse_args()
    env = RailwayPPOEnv(a.start, a.end, stress_mode=not a.realistic, weekday=None if a.weekday < 0 else a.weekday)
    pols = {'always_hold': always_hold, 'always_proceed': always_proceed, 'priority_rule': priority_rule, 'conflict_aware_rule': conflict_aware_rule}
    if os.path.exists(a.ckpt): pols['trained_ppo'] = ppo_policy(a.ckpt)
    else: print(f'[warn] {a.ckpt} not found, skipping PPO')
    envs = {n: env for n in pols}
    if os.path.exists(a.flat_ckpt):    # plain RL, NO graph: separate env with flat observations
        envs['flat_ppo'] = RailwayPPOEnv(a.start, a.end, stress_mode=not a.realistic, weekday=None if a.weekday < 0 else a.weekday, obs_mode='flat')
        pols['flat_ppo'] = ppo_policy(a.flat_ckpt)
    ctrl = str(P.MODELS / 'policy_flat_same_teacher.pt')   # control: flat PPO distilled from the same teacher as graph PPO
    if os.path.exists(ctrl):
        envs['flat_same_teacher'] = envs.get('flat_ppo') or RailwayPPOEnv(a.start, a.end, stress_mode=not a.realistic, weekday=None if a.weekday < 0 else a.weekday, obs_mode='flat')
        pols['flat_same_teacher'] = ppo_policy(ctrl)
    seeds = list(range(a.first_seed, a.first_seed + a.seeds)); res = {}; used = {}
    for name, pol in pols.items():
        rs, tot = [], {0: 0, 1: 0, 2: 0}
        for s in seeds:
            r, ac = run_episode(envs[name], pol, s, a.steps); rs.append(r)
            for k in tot: tot[k] += ac[k]
        res[name] = np.array(rs); used[name] = tot; print(f'finished {name}', flush=True)
    print(f"\n{a.seeds} seeds, {'realistic' if a.realistic else 'stress'} mode (reward = -cost, higher is better)\n")
    print(f"{'policy':<16}{'mean':>9}{'std':>8}{'sem':>7}   hold/proceed/loop")
    for n, r in res.items():
        print(f"{n:<16}{r.mean():9.1f}{r.std(ddof=1):8.1f}{r.std(ddof=1)/np.sqrt(len(r)):7.1f}   {used[n][0]}/{used[n][1]}/{used[n][2]}")
    for fl in [k for k in ('flat_ppo', 'flat_same_teacher') if k in res and 'trained_ppo' in res]:
        d = res['trained_ppo'] - res[fl]; se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"graph PPO - {fl} = {d.mean():+.1f} +/- {2*se:.1f} (95%)  -> {'graph better' if d.mean()-2*se>0 else ('flat better' if d.mean()+2*se<0 else 'no clear difference')}")
    for m in [k for k in ('trained_ppo', 'flat_ppo', 'flat_same_teacher') if k in res]:
        for b in ('always_proceed', 'priority_rule', 'conflict_aware_rule'):
            d = res[m] - res[b]; se = d.std(ddof=1) / np.sqrt(len(d))
            verdict = 'PPO better' if d.mean() - 2 * se > 0 else ('PPO worse' if d.mean() + 2 * se < 0 else 'no clear difference')
            print(f"{m} - {b:<15} = {d.mean():+.1f} +/- {2*se:.1f} (95%)  wins {int((d>0).sum())}/{len(d)}  -> {verdict}")
    os.makedirs(P.OUT_SYN, exist_ok=True)
    with open(str(P.OUT_SYN / 'evaluation.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['seed'] + list(res))
        for i, s in enumerate(seeds): w.writerow([s] + [round(float(res[n][i]), 2) for n in res])
    print('\nper-seed results saved to', P.OUT_SYN / 'evaluation.csv')
if __name__ == '__main__': main()

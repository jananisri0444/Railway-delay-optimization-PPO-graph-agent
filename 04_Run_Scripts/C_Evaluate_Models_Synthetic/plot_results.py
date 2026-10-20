"""Make result charts. Run AFTER evaluate.py (and training).
    python plot_results.py      -> 05_Results/A_Synthetic_Testing/plots/*.png
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
PL = P.OUT_SYN / 'plots'; os.makedirs(PL, exist_ok=True)
NAMES = {'always_hold': 'Always hold', 'always_proceed': 'Always proceed', 'priority_rule': 'Priority rule',
         'conflict_aware_rule': 'Conflict-aware rule', 'trained_ppo': 'Graph PPO', 'flat_ppo': 'Flat PPO (no graph)',
         'flat_same_teacher': 'Flat PPO (same teacher)'}
COL = {'trained_ppo': '#14B8A6', 'flat_ppo': '#F59E0B', 'flat_same_teacher': '#FBBF24'}

if os.path.exists(str(P.OUT_SYN / 'evaluation.csv')):
    d = pd.read_csv(str(P.OUT_SYN / 'evaluation.csv')); EXCLUDE = {'flat_same_teacher'}   # control model, left out of the charts
    cols = [c for c in d.columns if c != 'seed' and c not in EXCLUDE]
    m = d[cols].mean(); se = d[cols].std(ddof=1) / np.sqrt(len(d)); order = m.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([NAMES.get(c, c) for c in order], m[order], xerr=2 * se[order], color=[COL.get(c, '#64748B') for c in order], capsize=4)
    ax.set_xlabel('Average reward over seeds (closer to 0 = better).  Bars = 95% margin'); ax.set_title('Policy comparison')
    plt.tight_layout(); plt.savefig(str(PL / '1_comparison.png'), dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(9, 4.5)); x = d.seed
    for c in ('priority_rule', 'trained_ppo', 'flat_ppo'):
        if c in d: ax.plot(range(len(d)), d[c], marker='o', ms=3, label=NAMES[c], color=COL.get(c, '#64748B'))
    ax.set_xlabel('test scenario (seed)'); ax.set_ylabel('reward'); ax.legend(); ax.set_title('Per-scenario reward')
    plt.tight_layout(); plt.savefig(str(PL / '2_per_seed.png'), dpi=150); plt.close()
for mode, c in (('graph', '#14B8A6'), ('flat', '#F59E0B')):
    f = f'{P.OUT_SYN}/training_logs/training_log_{mode}.csv'
    if os.path.exists(f):
        t = pd.read_csv(f); fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t.episode, t.reward, alpha=.3, color=c); ax.plot(t.episode, t.reward.rolling(20, min_periods=1).mean(), color=c, lw=2)
        ax.set_xlabel('episode'); ax.set_ylabel('episode reward'); ax.set_title(f'Training curve ({mode} PPO), thick = 20-episode average')
        plt.tight_layout(); plt.savefig(f'{PL}/3_training_{mode}.png', dpi=150); plt.close()
print('saved charts in', PL)

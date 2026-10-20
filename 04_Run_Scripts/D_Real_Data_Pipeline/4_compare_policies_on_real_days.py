"""STEP B: replay REAL days in the simulator with YOUR models and compare, station by station, with what really happened.

For each real day: same weekday timetable, only the trains that really ran, their real entry delays (delay at the first corridor
station) -> simulator -> your policy takes the decisions -> simulated delay at every station is compared with the REAL delay there.

    python 4_compare_policies_on_real_days.py --date 2025-03-05
    python 4_compare_policies_on_real_days.py --from 2025-03-03 --to 2025-03-09
    python 4_compare_policies_on_real_days.py --month 2025-03 --max-days 10
Options: --start 0 --end 1440 (window in minutes, default whole day), --stress, --real-graph-ckpt / --real-flat-ckpt for fine-tuned models.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import argparse, os, sys
import numpy as np, pandas as pd
from tools.policy_eval import ppo_policy
from tools.real_delay_env import RealDayEnv
from tools.real_timeline import select, corridor_order, day_events
from decision_layer.environment import MAX_DECISIONS
from decision_layer.baselines import always_proceed, priority_rule, conflict_aware_rule

ap = argparse.ArgumentParser(); ap.add_argument('--clean', default=str(P.REAL_PROC / 'real_delays_clean.csv'))
ap.add_argument('--date'); ap.add_argument('--from'); ap.add_argument('--to'); ap.add_argument('--month'); ap.add_argument('--max-days', type=int, default=7)
ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float, default=1440); ap.add_argument('--max-steps', type=int, default=4000)
ap.add_argument('--stress', action='store_true'); ap.add_argument('--ckpt', default=str(P.MODELS / 'policy_final.pt')); ap.add_argument('--flat-ckpt', default=str(P.MODELS / 'policy_flat.pt'))
ap.add_argument('--real-graph-ckpt', default=''); ap.add_argument('--real-flat-ckpt', default='')
a = ap.parse_args()
df = pd.read_csv(a.clean, parse_dates=['cal_date']); sel, label = select(df, a)
days = sorted(sel['cal_date'].unique())[:a.max_days]
if not days: raise SystemExit('No days found for that selection.')
rank = {c: i for i, c in enumerate(corridor_order())}
flat = RealDayEnv(a.start, a.end, stress_mode=a.stress, obs_mode='flat'); graph = None
pols = {'always_proceed': (always_proceed, flat), 'priority_rule': (priority_rule, flat), 'conflict_aware_rule': (conflict_aware_rule, flat)}
if os.path.exists(a.flat_ckpt): pols['flat_ppo'] = (ppo_policy(a.flat_ckpt), flat)
if a.real_flat_ckpt: pols['flat_ppo_real'] = (ppo_policy(a.real_flat_ckpt), flat)
for n, p in (('graph_ppo', a.ckpt), ('graph_ppo_real', a.real_graph_ckpt)):
    if p and os.path.exists(p): graph = graph or RealDayEnv(a.start, a.end, stress_mode=a.stress, obs_mode='graph'); pols[n] = (ppo_policy(p), graph)

def replay(env, pol, wd, trains, delays):
    env.set_day(wd, trains, delays); rec = {}; orig = env._sim.step
    def step():
        rs = orig()
        for tn, st in env._sim.states.items():
            if st.status.name == 'HALTED' and st.station_code and (int(tn), st.station_code) not in rec: rec[(int(tn), st.station_code)] = float(st.delay_min)
        return rs
    env._sim.step = step; holds = loops = decs = 0
    try:
        env.reset(seed=0)
        n_steps = 0
        while n_steps < a.max_steps:
            _, done = env.advance_to_decision()
            if done: break
            if not env._candidates: continue      # 100 quiet minutes passed, keep going until the day ends
            n_steps += 1; act = pol(env); n = min(len(env._candidates), MAX_DECISIONS); decs += n; holds += int((act[:n] == 0).sum()); loops += int((act[:n] == 2).sum())
            _, _, te, tr, _ = env.step(act)
            if te or tr: break
        added = env.total_delay()
    finally: del env._sim.step
    return rec, holds, loops, decs, added

rows, prof, per_day = [], [], []
for d in days:
    g = df[df['cal_date'] == d].copy(); g['rank'] = g['station_code'].map(rank); g = g.sort_values(['train_no', 'rank'])
    entry = df[df['cal_date'] == d].sort_values(['train_no', 'seq']).groupby('train_no').first()['late_min'].clip(upper=240).round(1); wd = int(pd.Timestamp(d).weekday())
    real = g.set_index(['train_no', 'station_code'])['late_min']; ev = day_events(g)
    for name, (pol, env) in pols.items():
        rec, holds, loops, decs, added = replay(env, pol, wd, entry.index, entry.to_dict())
        sim = pd.Series(rec); sim.index = pd.MultiIndex.from_tuples(sim.index, names=['train_no', 'station_code'])
        j = pd.concat([real.rename('real'), sim.rename('sim')], axis=1, join='inner'); j = j[~j.index.duplicated()]
        if j.empty: continue
        seqmap = g.drop_duplicates(['train_no', 'station_code']).set_index(['train_no', 'station_code'])['seq']; j = j.reset_index(); j['seq'] = [seqmap.get((t_, s_)) for t_, s_ in zip(j['train_no'], j['station_code'])]; first = j.sort_values('seq').groupby('train_no').first(); last = j.sort_values('seq').groupby('train_no').last()
        per_day.append({'date': f'{d:%Y-%m-%d}', 'policy': name, 'matched_records': len(j), 'real_trains': g['train_no'].nunique(), 'sim_trains': j['train_no'].nunique(),
                        'MAE_vs_real': (j['sim'] - j['real']).abs().mean(), 'bias_sim_minus_real': (j['sim'] - j['real']).mean(),
                        'real_growth': (last['real'] - first['real']).mean(), 'sim_growth': (last['sim'] - first['sim']).mean(), 'added_delay': added,
                        'decisions': decs, 'holds': holds, 'loop_overtakes': loops, 'real_holds': int(ev.holds.sum()), 'real_order_swaps': int(ev.order_swaps.sum())})
        prof.append(j.assign(policy=name, date=f'{d:%Y-%m-%d}'))
    print('finished', f'{d:%Y-%m-%d}', flush=True)
out = str(P.OUT_COMPARE); os.makedirs(out, exist_ok=True)
pdv = pd.DataFrame(per_day); pdv.to_csv(f'{out}/per_day_policy.csv', index=False)
P = pd.concat(prof); P.groupby(['policy', 'station_code'])[['real', 'sim']].mean().round(2).to_csv(f'{out}/station_profile.csv')
S = pdv.groupby('policy').agg(days=('date', 'nunique'), matched=('matched_records', 'sum'), MAE_vs_real=('MAE_vs_real', 'mean'), bias=('bias_sim_minus_real', 'mean'),
        real_growth=('real_growth', 'mean'), sim_growth=('sim_growth', 'mean'), added_delay=('added_delay', 'mean'), decisions=('decisions', 'mean'), holds=('holds', 'mean'),
        loops=('loop_overtakes', 'mean'), real_holds=('real_holds', 'mean'), real_swaps=('real_order_swaps', 'mean')).round(2)
S.to_csv(f'{out}/summary.csv'); pd.set_option('display.width', 250); pd.set_option('display.max_columns', None); print(f'\n=== {len(days)} real day(s) [{label}] ===\n', S.T)
print('\nHow to read: real_growth = how much real delay rose from the first to the last corridor station (everything that really happened).')
print('sim_growth = same for the simulator under each policy: ONLY dispatching-related delay. A gap to the real number is delay from causes the simulator does not model.')
print('Compare policies with each other (sim_growth, added_delay). Compare with reality only as context, not as proof of beating the real dispatchers.')
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    st = [c for c in corridor_order() if c in set(P['station_code'])]; fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(st, [P[P.station_code == c]['real'].mean() for c in st], color='black', lw=2.5, label='REAL')
    for n in pols:
        q = P[P.policy == n].groupby('station_code')['sim'].mean(); ax.plot(st, [q.get(c, np.nan) for c in st], lw=1.5, label=n)
    ax.set_ylabel('mean arrival delay (min)'); ax.set_title(f'Delay along the corridor: real vs simulated ({label})'); ax.legend(fontsize=7); plt.xticks(rotation=90, fontsize=6); plt.tight_layout(); plt.savefig(f'{out}/profile_real_vs_sim.png', dpi=150)
except Exception:
    import traceback; print('\n!!! CHART FAILED, full error below. Run: python plot_real_compare.py'); traceback.print_exc()
print('Saved to', out)

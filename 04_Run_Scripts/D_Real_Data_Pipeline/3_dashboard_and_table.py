"""Interactive dashboard: REAL arrivals vs Graph PPO vs Flat PPO, on real days.

    python 3_dashboard_and_table.py --date 2025-03-05
    python 3_dashboard_and_table.py --from 2025-03-03 --to 2025-03-09 --max-days 7
    python 3_dashboard_and_table.py --date 2025-03-05 --only dashboard      # dashboard alone
    python 3_dashboard_and_table.py --date 2025-03-05 --only table          # table alone
    python 3_dashboard_and_table.py --date 2025-03-05 --mode observed
ONE run writes the dashboard AND the table into 05_Results/B_Real_Data_Replay/dashboard_and_table/<dates>_<mode>/ :
    real_dashboard.html      interactive dashboard (double-click to open, works offline)
    comparison_table.html    REAL vs Graph PPO vs Flat PPO, train by train, station by station (+ per-train and overall summaries)
    comparison_table.csv, train_summary.csv, overall_summary.csv

--mode full      (default) simulate EVERY train the timetable runs that weekday. Trains that were observed get their real entry delay,
                 the others draw a delay at random from the real delays of that day. The day is busy enough that the models face decisions.
--mode observed  simulate ONLY the trains that really ran. Real days are often so quiet that the models face almost NO decisions;
                 then Graph PPO and Flat PPO give identical results BY CONSTRUCTION (nothing to decide). The dashboard says so.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import argparse, json, os, sys, collections
import numpy as np, pandas as pd
from tools.policy_eval import ppo_policy
from tools.real_delay_env import RealDayEnv
from tools.real_timeline import select, corridor_order, mins
from decision_layer.environment import MAX_DECISIONS

ap = argparse.ArgumentParser(); ap.add_argument('--clean', default=str(P.REAL_PROC / 'real_delays_clean.csv'))
ap.add_argument('--date'); ap.add_argument('--from'); ap.add_argument('--to'); ap.add_argument('--month'); ap.add_argument('--max-days', type=int, default=3)
ap.add_argument('--only', default='both', choices=['both', 'dashboard', 'table'], help='both (default) | dashboard | table')
ap.add_argument('--name', default='', help='output sub-folder name (default: <dates>_<mode>)')
ap.add_argument('--mode', default='full', choices=['full', 'observed']); ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float, default=1440)
ap.add_argument('--max-steps', type=int, default=4000); ap.add_argument('--stress', action='store_true')
ap.add_argument('--graph-ckpt', default=str(P.MODELS / 'policy_final.pt')); ap.add_argument('--flat-ckpt', default=str(P.MODELS / 'policy_flat.pt'))
a = ap.parse_args()
df = pd.read_csv(a.clean, parse_dates=['cal_date']); sel, label = select(df, a)
days = sorted(sel['cal_date'].unique())[:a.max_days]
if not days: raise SystemExit('No days found for that selection.')
order = corridor_order(); rank = {c: i for i, c in enumerate(order)}
models = {'graph': (ppo_policy(a.graph_ckpt), RealDayEnv(a.start, a.end, stress_mode=a.stress, obs_mode='graph')),
          'flat': (ppo_policy(a.flat_ckpt), RealDayEnv(a.start, a.end, stress_mode=a.stress, obs_mode='flat'))}

def replay(env, pol, wd, trains, delays, fill):
    env.set_day(wd, trains, delays, fill=fill); rec = {}; orig = env._sim.step
    def step():
        rs = orig(); t = getattr(rs, 'simulation_time_min', None)
        for tn, st in env._sim.states.items():
            if st.status.name == 'HALTED' and st.station_code and (int(tn), st.station_code) not in rec: rec[(int(tn), st.station_code)] = (t, float(st.delay_min))
        return rs
    env._sim.step = step; acts = collections.Counter(); n = 0
    try:
        env.reset(seed=0)
        while n < a.max_steps:
            _, done = env.advance_to_decision()
            if done: break
            if not env._candidates: continue
            act = pol(env); n += 1
            for i, c in enumerate(env._candidates[:MAX_DECISIONS]): acts[('HOLD', 'PROCEED / GRANT PLATFORM', 'LOOP')[int(act[i])]] += 1
            _, _, te, tr, _ = env.step(act)
            if te or tr: break
        added = env.total_delay(); simulated = len(env._sim.states)
    finally: del env._sim.step
    return rec, {'decisions': n, 'actions': dict(acts), 'added_delay': round(float(added), 1), 'trains_simulated': simulated}

out = {'meta': {'label': label, 'mode': a.mode, 'start': a.start, 'end': a.end, 'real_trains_in_data': int(df['train_no'].nunique())}, 'stations': order, 'days': {}}
for d in days:
    day = df[df['cal_date'] == d]; wd = int(pd.Timestamp(d).weekday())
    entry = day.sort_values(['train_no', 'seq']).groupby('train_no').first()['late_min'].clip(upper=240).round(1)
    trains = entry.index if a.mode == 'observed' else None; fill = [x for x in entry if x > 0] or [10.0]
    real = day.drop_duplicates(['train_no', 'station_code']).set_index(['train_no', 'station_code'])
    t = real[['train_name', 'sch_arrival', 'delay_arr']].copy(); t['schm'] = mins(t['sch_arrival']).values; t['realm'] = t['schm'] + t['delay_arr']; info = {}
    for name, (pol, env) in models.items():
        rec, info[name] = replay(env, pol, wd, trains, entry.to_dict(), fill)
        t[name + 'm'] = [rec.get(k, (np.nan, np.nan))[0] for k in t.index]; t[name + 'd'] = [rec.get(k, (np.nan, np.nan))[1] for k in t.index]
    t = t.dropna(subset=['graphm', 'flatm']).reset_index().sort_values(['train_no', 'realm'])
    rows = [{'t': int(r.train_no), 'n': str(r.train_name), 's': r.station_code, 'rk': rank.get(r.station_code, -1), 'sch': r.sch_arrival, 'schm': float(r.schm),
             'rm': round(float(r.realm), 1), 'rd': round(float(r.delay_arr), 1), 'gm': round(float(r.graphm), 1), 'gd': round(float(r.graphd), 1),
             'fm': round(float(r.flatm), 1), 'fd': round(float(r.flatd), 1)} for r in t.itertuples()]
    out['days'][f'{d:%Y-%m-%d}'] = {'weekday': wd, 'rows': rows, 'real_trains': int(day['train_no'].nunique()), 'info': info}
    print('finished', f'{d:%Y-%m-%d}', '| compared records:', len(rows), '| decisions graph/flat:', info['graph']['decisions'], info['flat']['decisions'], flush=True)
    if len(rows) and info['graph']['decisions'] + info['flat']['decisions'] == 0: print('  NOTE: no decisions on this day -> Graph PPO and Flat PPO are identical by construction. Try --mode full.')
# ---------- output folder ----------
od = str(P.OUT_DASH / (a.name or f'{label}_{a.mode}')); os.makedirs(od, exist_ok=True)

def hhmm(m):
    d, r = divmod(int(round(m)), 1440); return f'{r // 60:02d}:{r % 60:02d}' + (f' (+{d}d)' if d > 0 else f' ({d}d)' if d < 0 else '')

# ---------- the TABLE files (same data as the dashboard, no second replay needed) ----------
recs = [dict(r, date=dk) for dk, v in out['days'].items() for r in v['rows']]
if recs and a.only in ('both', 'table'):
    T = pd.DataFrame(recs).sort_values(['date', 't', 'rm'])
    tab = pd.DataFrame({'date': T['date'], 'train_no': T['t'], 'train_name': T['n'], 'station': T['s'], 'scheduled_arrival': T['sch'],
        'REAL_arrival': T['rm'].map(hhmm), 'REAL_delay_min': T['rd'], 'GraphPPO_arrival': T['gm'].map(hhmm), 'GraphPPO_delay_min': T['gd'],
        'FlatPPO_arrival': T['fm'].map(hhmm), 'FlatPPO_delay_min': T['fd'], 'Graph_minus_REAL_min': (T['gm'] - T['rm']).round(1), 'Flat_minus_REAL_min': (T['fm'] - T['rm']).round(1)})
    summ = tab.groupby(['date', 'train_no']).agg(train_name=('train_name', 'first'), stations_compared=('station', 'count'), first_station=('station', 'first'), last_station=('station', 'last'),
        REAL_avg_delay=('REAL_delay_min', 'mean'), GraphPPO_avg_delay=('GraphPPO_delay_min', 'mean'), FlatPPO_avg_delay=('FlatPPO_delay_min', 'mean'),
        REAL_last_delay=('REAL_delay_min', 'last'), GraphPPO_last_delay=('GraphPPO_delay_min', 'last'), FlatPPO_last_delay=('FlatPPO_delay_min', 'last')).round(1).reset_index()
    dg, df_, rd_ = tab['GraphPPO_delay_min'], tab['FlatPPO_delay_min'], tab['REAL_delay_min'].clip(lower=0)
    dsum = lambda k: sum(v['info'][k]['decisions'] for v in out['days'].values())
    ov = pd.DataFrame({'': ['Records compared', 'Average delay (min)', 'Avg |arrival difference to real| (min)', 'Avg arrival difference to real (min, + = later than real)', 'Decisions faced', 'Records where Graph and Flat differ'],
        'REAL': [len(tab), round(rd_.mean(), 2), '-', '-', '-', '-'], 'Graph PPO': ['', round(dg.mean(), 2), round(tab['Graph_minus_REAL_min'].abs().mean(), 2), round(tab['Graph_minus_REAL_min'].mean(), 2), dsum('graph'), int(((dg - df_).abs() > .05).sum())],
        'Flat PPO': ['', round(df_.mean(), 2), round(tab['Flat_minus_REAL_min'].abs().mean(), 2), round(tab['Flat_minus_REAL_min'].mean(), 2), dsum('flat'), '']})
    tab.to_csv(f'{od}/comparison_table.csv', index=False); summ.to_csv(f'{od}/train_summary.csv', index=False); ov.to_csv(f'{od}/overall_summary.csv', index=False)
    css = 'body{font-family:Segoe UI,Arial;margin:20px}table{border-collapse:collapse;font-size:12px}th{background:#0B1F3A;color:#fff;padding:5px 8px}td{padding:4px 8px;border-bottom:1px solid #dde}tr:nth-child(even){background:#f6f8fb}.r{background:#fde8e8}'
    def tohtml(d_, title):
        cols = list(d_.columns); h = f'<h3>{title}</h3><table><tr>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>'
        for _, r in d_.iterrows(): h += '<tr>' + ''.join(f'<td class="{"r" if (c.endswith("delay_min") and isinstance(r[c], (int, float)) and r[c] > 10) else ""}">{r[c]}</td>' for c in cols) + '</tr>'
        return h + '</table>'
    note = f'<p style="color:#6b7688">Real data in this project covers {out["meta"]["real_trains_in_data"]} of 127 trains. Simulation mode: {a.mode}. Only train-station records present in both the real data and the simulator are shown.</p>'
    open(f'{od}/comparison_table.html', 'w', encoding='utf-8').write(f'<html><head><meta charset="utf-8"><style>{css}</style></head><body><h2>Real vs Graph PPO vs Flat PPO ({label})</h2>{note}'
        + tohtml(ov, 'Overall') + tohtml(summ, 'Per train') + tohtml(tab, 'Station by station (red = delay over 10 min)') + '</body></html>')
    print('Table     :', f'{od}/comparison_table.html', '(+ comparison_table.csv, train_summary.csv, overall_summary.csv)')

# ---------- the DASHBOARD ----------
if a.only in ('both', 'dashboard'):
    html = open(P.CODE / 'tools' / 'real_dashboard_template.html', encoding='utf-8').read().replace('__DATA__', json.dumps(out))
    open(f'{od}/real_dashboard.html', 'w', encoding='utf-8').write(html); print('Dashboard :', f'{od}/real_dashboard.html', '  (double-click to open in your browser)')

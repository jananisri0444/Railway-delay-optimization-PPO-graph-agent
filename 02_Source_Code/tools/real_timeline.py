"""STEP A: station-by-station REAL delay timeline for a day / date range / month (from real_delays_clean.csv).

Run build_real_delays.py first, then e.g.:
    python real_timeline.py --date 2025-03-05
    python real_timeline.py --from 2025-03-03 --to 2025-03-09          # a week
    python real_timeline.py --month 2025-03
    python real_timeline.py                                           # the busiest day available
Output folder 01_Data/3_Real_Delay_Data_Processed/timeline_<label>/ :
    station_timeline.csv  every train x station: scheduled, actual, arrival delay, departure delay, extra dwell (a hold proxy)
    delay_matrix.csv      arrival lateness, rows = date+train, columns = stations in corridor order
    station_summary.csv   mean / median / p90 lateness and share >5 min at each station
    real_events.csv       per date+station: trains, holds (extra dwell >= 3 min) and order swaps (overtakes proxy)
    delay_heatmap.png / delay_profile.png
"""
import argparse, os
import numpy as np, pandas as pd
import project_paths as P

def corridor_order(path=None):
    path = path or P.SIM_DATA / 'chennai_bangalore_stations.csv'
    s = pd.read_csv(path).sort_values('distanceKm'); return list(s['stationCode'])

def mins(t):
    h = pd.to_datetime(t, format='%H:%M', errors='coerce'); return h.dt.hour * 60 + h.dt.minute

def day_events(g, hold_min=3.0, window=60.0):
    """g = real rows of ONE calendar day. Returns per-station counts of holds and order swaps."""
    g = g.copy(); g['sch'] = mins(g['sch_arrival']); g['act'] = g['sch'] + g['delay_arr']; out = []
    for st, h in g.dropna(subset=['sch', 'act']).groupby('station_code'):
        h = h.sort_values('sch'); s, a = h['sch'].values, h['act'].values; swaps = 0
        for i in range(len(h)):
            for j in range(i + 1, len(h)):
                if s[j] - s[i] > window: break
                swaps += int(a[i] > a[j])
        holds = int(((h['delay_dep'] - h['delay_arr']) >= hold_min).sum())
        out.append({'station_code': st, 'trains': len(h), 'holds': holds, 'order_swaps': swaps})
    return pd.DataFrame(out)

def select(df, a):
    d = df['cal_date']
    if a.date: return df[d == pd.Timestamp(a.date)], a.date
    if a.month: return df[d.dt.strftime('%Y-%m') == a.month], a.month
    if a.__dict__['from'] or a.to:
        lo, hi = pd.Timestamp(a.__dict__['from'] or d.min()), pd.Timestamp(a.to or d.max()); return df[(d >= lo) & (d <= hi)], f"{lo:%Y-%m-%d}_to_{hi:%Y-%m-%d}"
    best = df.groupby('cal_date')['train_no'].nunique().idxmax(); return df[d == best], f'{best:%Y-%m-%d}'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--clean', default=str(P.REAL_PROC / 'real_delays_clean.csv'))
    ap.add_argument('--date'); ap.add_argument('--from'); ap.add_argument('--to'); ap.add_argument('--month'); ap.add_argument('--out', default=str(P.REAL_PROC))
    a = ap.parse_args()
    df = pd.read_csv(args := a.clean, parse_dates=['cal_date'])
    sel, label = select(df, a)
    if sel.empty: raise SystemExit('No rows for that selection. Check the dates available in real_delays_clean.csv')
    order = corridor_order(); rank = {c: i for i, c in enumerate(order)}
    sel = sel.assign(station_order=sel['station_code'].map(rank), extra_dwell_min=sel['delay_dep'] - sel['delay_arr']).sort_values(['cal_date', 'train_no', 'station_order'])
    out = f'{a.out}/timeline_{label}'; os.makedirs(out, exist_ok=True)
    cols = ['cal_date', 'weekday', 'train_no', 'train_name', 'station_order', 'station_code', 'sch_arrival', 'act_arrival', 'sch_departure', 'act_departure', 'delay_arr', 'delay_dep', 'extra_dwell_min']
    sel[[c for c in cols if c in sel]].to_csv(f'{out}/station_timeline.csv', index=False)
    mat = sel.pivot_table(index=['cal_date', 'train_no'], columns='station_code', values='late_min', aggfunc='first').reindex(columns=[c for c in order if c in set(sel['station_code'])])
    mat.to_csv(f'{out}/delay_matrix.csv')
    sm = sel.groupby('station_code')['delay_min'].agg(['count', 'mean', 'median', lambda x: x.quantile(.9)]).rename(columns={'<lambda_0>': 'p90'})
    sm['pct_late_gt5'] = sel.groupby('station_code')['delay_min'].apply(lambda x: (x > 5).mean() * 100); sm = sm.reindex([c for c in order if c in sm.index]).round(2); sm.to_csv(f'{out}/station_summary.csv')
    ev = pd.concat([day_events(g).assign(cal_date=d) for d, g in sel.groupby('cal_date')]); ev.to_csv(f'{out}/real_events.csv', index=False)
    print(f'Selection {label}: {sel["cal_date"].nunique()} day(s), {sel["train_no"].nunique()} trains, {len(sel)} station records')
    print(f'Real dispatching proxies: {int(ev.holds.sum())} extra-dwell holds (>=3 min) and {int(ev.order_swaps.sum())} overtake-type order swaps')
    print('Saved to', out)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5)); ax.plot(sm.index, sm['mean'], marker='o', color='#14B8A6'); ax.set_ylabel('mean arrival delay (min)')
        ax.set_title(f'Real delay along the corridor ({label})'); plt.xticks(rotation=90, fontsize=6); plt.tight_layout(); plt.savefig(f'{out}/delay_profile.png', dpi=150); plt.close()
        m = mat.head(40); fig, ax = plt.subplots(figsize=(10, 6)); im = ax.imshow(m.fillna(np.nan).values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=60)
        ax.set_yticks(range(len(m))); ax.set_yticklabels([f'{t}' for _, t in m.index], fontsize=6); ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns, rotation=90, fontsize=6)
        plt.colorbar(im, label='late minutes'); ax.set_title(f'Train x station delay ({label})'); plt.tight_layout(); plt.savefig(f'{out}/delay_heatmap.png', dpi=150)
    except Exception as e: print('charts skipped:', e)


if __name__ == '__main__':
    main()

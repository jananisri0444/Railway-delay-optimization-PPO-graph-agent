"""Draw the real-vs-simulated charts from the CSVs that compare_with_real_dispatch.py already wrote (no need to re-run the replay).
    python plot_real_compare.py
Needs output/real_compare/station_profile.csv and summary.csv. Saves PNGs in the same folder and prints the full error if anything fails."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import os, sys, traceback
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = str(P.OUT_COMPARE)
try:
    prof = pd.read_csv(f'{D}/station_profile.csv'); summ = pd.read_csv(f'{D}/summary.csv')
    print('station_profile rows:', len(prof), '| policies:', list(prof['policy'].unique()), '| stations:', prof['station_code'].nunique())
    try:
        order = list(pd.read_csv(P.SIM_DATA / 'chennai_bangalore_stations.csv').sort_values('distanceKm')['stationCode'])
    except Exception:
        order = list(prof['station_code'].unique())
    st = [c for c in order if c in set(prof['station_code'])]; x = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(11, 5))
    real = prof.groupby('station_code')['real'].mean().reindex(st)
    ax.plot(x, real.values, color='black', lw=2.8, marker='o', ms=3, label='REAL')
    for name, g in prof.groupby('policy'):
        sim = g.set_index('station_code')['sim'].reindex(st); ax.plot(x, sim.values, lw=1.6, marker='.', label=name)
    ax.set_xticks(x); ax.set_xticklabels(st, rotation=90, fontsize=6); ax.set_ylabel('mean arrival delay (min)')
    ax.set_title('Delay along the corridor: real vs simulated'); ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f'{D}/profile_real_vs_sim.png', dpi=150); plt.close(); print('saved', f'{D}/profile_real_vs_sim.png')
    fig, ax = plt.subplots(figsize=(8, 4.5)); s = summ.set_index('policy')
    names = ['REAL'] + list(s.index); vals = [float(s['real_growth'].mean())] + [float(v) for v in s['sim_growth']]
    ax.bar(names, vals, color=['black'] + ['#14B8A6' if 'graph' in n else '#F59E0B' if 'flat' in n else '#64748B' for n in s.index])
    ax.set_ylabel('delay growth, first to last station (min)'); ax.set_title('Delay growth along the corridor'); plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(f'{D}/growth_real_vs_sim.png', dpi=150); plt.close(); print('saved', f'{D}/growth_real_vs_sim.png')
except Exception:
    traceback.print_exc(); sys.exit(1)

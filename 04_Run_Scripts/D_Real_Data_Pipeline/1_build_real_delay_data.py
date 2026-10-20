"""STEP 1: turn the scraped historical files into clean real-delay data + real-day scenarios.

Usage (from the project folder):
    python 1_build_real_delay_data.py --obs <path to historical_reported_observations.csv>   (default: 01_Data/2_Real_Railway_History/)
Optional:
    --all historical_all_rows.csv      use this instead (it also contains unreported stops; they are ignored)
    --min-trains 5                     keep a calendar day only if >= this many trains have an entry delay
    --test-frac 0.3                    newest 30% of days become the TEST split

Outputs go to 01_Data/3_Real_Delay_Data_Processed/ :
    real_delays_clean.csv         one row per train x date x corridor station, with computed delays
    delay_summary_by_train.csv    mean / median / p90 delay per train
    delay_summary_by_station.csv  same per station
    real_day_scenarios.json       one scenario per calendar day: {train: entry delay in minutes}, train/test split
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import argparse, json, os
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore', category=DeprecationWarning)

ap = argparse.ArgumentParser()
ap.add_argument('--obs', default=str(P.HISTORY / 'historical_reported_observations.csv'))
ap.add_argument('--all', default=None)
ap.add_argument('--timetable', default=str(P.SIM_DATA / 'dataset.csv'), help='project timetable (defines corridor stations and the 127 trains)')
ap.add_argument('--min-trains', type=int, default=5)
ap.add_argument('--test-frac', type=float, default=0.3)
ap.add_argument('--max-delay', type=float, default=720, help='drop |delay| above this many minutes as data errors')
ap.add_argument('--out', default=str(P.REAL_PROC))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

src = a.all or a.obs
print('Reading', src)
df = pd.read_csv(src, dtype={'train_no': str, 'station_code': str}, low_memory=False)
print('  rows read:', len(df))
_cov = P.HISTORY / 'train_coverage_summary.csv'
if _cov.exists():                                   # data-completeness check (the Excel 1,048,575-row problem)
    _c = pd.read_csv(_cov); _exp = int(_c['total_station_rows' if a.all else 'rows_with_any_actual_time'].sum())
    _have = df['train_no'].nunique()
    if len(df) in (1048575, 1048576) or len(df) < 0.98 * _exp:
        print(f'\n  !!! WARNING: the collection summary says about {_exp:,} rows for {len(_c)} trains, but this file has {len(df):,} rows covering {_have} trains.')
        print('  !!! 1,048,575 rows is the Excel row limit: the file was probably opened and re-saved in Excel, which silently cut it off.')
        print('  !!! Re-run 01_Data/2_Real_Railway_History/collect_railway_history.py and DO NOT open the CSV in Excel.\n')
tt = pd.read_csv(a.timetable)
corridor = set(tt['stationCode'].astype(str)); trains = set(tt['trainNumber'].astype(int))
df['train_no'] = pd.to_numeric(df['train_no'], errors='coerce')
df = df[df['train_no'].isin(trains)].copy(); df['train_no'] = df['train_no'].astype(int)
if 'reported' in df: df = df[pd.to_numeric(df['reported'], errors='coerce').eq(1)]
print('  rows for your trains (reported):', len(df))
df = df[df['station_code'].isin(corridor)].copy()
print('  rows at corridor stations:', len(df), '| distinct stations matched:', df['station_code'].nunique(), 'of', len(corridor))

def dt(date_col, time_col):
    s = df[date_col].astype('string').str.strip() + ' ' + df[time_col].astype('string').str.strip()
    return pd.to_datetime(s, format='%d-%b-%y %H:%M', errors='coerce')
sch_a, act_a = dt('sch_arrival_date', 'sch_arrival'), dt('act_arrival_date', 'act_arrival')
sch_d, act_d = dt('sch_departure_date', 'sch_departure'), dt('act_departure_date', 'act_departure')
df['delay_arr'] = (act_a - sch_a).dt.total_seconds() / 60
df['delay_dep'] = (act_d - sch_d).dt.total_seconds() / 60
for col, src_col in (('delay_arr', 'delay_arrival_min'), ('delay_dep', 'delay_departure_min')):
    df[col] = df[col].fillna(pd.to_numeric(df[src_col], errors='coerce'))   # fall back to the dataset's own delay column
bad = (df['delay_arr'].abs() > a.max_delay) | (df['delay_dep'].abs() > a.max_delay)
print('  dropped as implausible (> %d min):' % a.max_delay, int(bad.sum()))
df.loc[df['delay_arr'].abs() > a.max_delay, 'delay_arr'] = np.nan
df.loc[df['delay_dep'].abs() > a.max_delay, 'delay_dep'] = np.nan
df['delay_min'] = df['delay_arr'].fillna(df['delay_dep'])
df['cal_date'] = sch_a.dt.normalize().fillna(sch_d.dt.normalize())           # real calendar day the train was at this station
df['weekday'] = df['cal_date'].dt.weekday                                    # 0=Mon .. 6=Sun, same as the simulator
df = df.dropna(subset=['delay_min', 'cal_date'])
df['late_min'] = df['delay_min'].clip(lower=0)                               # the simulator only models lateness
keep = ['cal_date', 'weekday', 'train_no', 'train_name', 'train_type', 'seq', 'station_code', 'sch_arrival', 'act_arrival',
        'sch_departure', 'act_departure', 'delay_arr', 'delay_dep', 'delay_min', 'late_min', 'date']
df[[c for c in keep if c in df]].sort_values(['cal_date', 'train_no', 'seq']).to_csv(f'{a.out}/real_delays_clean.csv', index=False)
print('  clean rows with a usable delay:', len(df))

def summ(g):
    x = g['delay_min']
    return pd.Series({'n': len(x), 'mean': x.mean(), 'median': x.median(), 'p90': x.quantile(.9), 'pct_late_gt5': (x > 5).mean() * 100})
df.groupby('train_no').apply(summ).round(2).to_csv(f'{a.out}/delay_summary_by_train.csv')
df.groupby('station_code').apply(summ).round(2).to_csv(f'{a.out}/delay_summary_by_station.csv')

# ---- real-day scenarios: delay of each train when it ENTERS the corridor (first corridor stop it reports) ----
df = df.sort_values(['train_no', 'date', 'seq'])
entry = df.groupby(['train_no', 'date'], as_index=False).first()
entry = entry[entry['cal_date'].notna()]
entry['entry_delay'] = entry['late_min'].clip(upper=240).round(1)
days = []
for d, g in entry.groupby('cal_date'):
    if len(g) >= a.min_trains:
        days.append({'date': d.strftime('%Y-%m-%d'), 'weekday': int(d.weekday()), 'n_trains': int(len(g)),
                     'delays': {str(int(t)): float(x) for t, x in zip(g['train_no'], g['entry_delay'])}})
days.sort(key=lambda s: s['date']); cut = int(len(days) * (1 - a.test_frac))
for i, s in enumerate(days): s['split'] = 'train' if i < cut else 'test'
json.dump({'meta': {'source': src, 'min_trains': a.min_trains, 'n_days': len(days)}, 'scenarios': days}, open(f'{a.out}/real_day_scenarios.json', 'w'))

allv = entry['entry_delay']
print('\n=== REAL ENTRY-DELAY STATISTICS (%d train runs) ===' % len(entry))
print('  share of runs entering > 5 min late : %.1f%%  (synthetic stress mode assumes 40%%)' % ((allv > 5).mean() * 100))
print('  mean delay of the late ones         : %.1f min  (synthetic assumes 25)' % allv[allv > 5].mean())
print('  median / 90th percentile (all runs) : %.1f / %.1f min' % (allv.median(), allv.quantile(.9)))
print('  scenarios (calendar days) kept      : %d  -> train %d / test %d' % (len(days), cut, len(days) - cut))
print('\nSaved to', a.out)

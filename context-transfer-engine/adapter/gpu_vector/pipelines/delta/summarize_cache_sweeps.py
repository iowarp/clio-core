#!/usr/bin/env python3
"""Collect the six per-workload residency sweeps into ONE faults-vs-time table.

The question these sweeps exist to answer is whether page faults cost time.
Each workload's own post block prints its curve in its own units; this pulls
them together so the relationship can be read across workloads at once.

Usage:  python3 summarize_cache_sweeps.py [results_root]
"""
import os
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/gv_pipeline_results')

# (results dir, per-node deck in the units that dir's cache column uses,
#  the column holding the cache setting)
SWEEPS = [
    ('cache_kmeans',    'kmeans',    32000, 'wl.cache_mb'),
    ('cache_weights',   'weights',   32000, 'wl.cache_mb'),
    ('cache_grayscott', 'grayscott', 32000, 'wl.cache_mb'),
    ('cache_gmx',       'gmx',        1024, 'wl.cap'),
    ('cache_lbann',     'lbann',      3072, 'wl.cap'),
    ('cache_lammps_md', 'lammps_md',     0, 'wl.vram_mb'),
]

# The six benches do not agree on which timing they print. Coalescing in a
# fixed order and RECORDING THE SOURCE is what keeps a workload from going
# silently missing: lbann fills only ms_per_step, gmx only spread/gather_ms,
# and a table pivoted on bench_ms alone renders those rows blank while still
# looking like a complete sweep.
METRICS = ('bench_ms', 'ms_per_step', 'gather_ms', 'spread_ms')


def load(d, name, deck, cache_col):
    path = os.path.join(ROOT, d, 'results.csv')
    if not os.path.exists(path):
        return None, '%s: NO results.csv (job did not run or was cleared)' % name
    df = pd.read_csv(path)
    if not len(df):
        return None, '%s: results.csv is empty' % name

    df['workload'] = name
    df['metric_ms'] = pd.NA
    df['metric_src'] = ''
    for col in METRICS:
        if col in df.columns:
            take = df['metric_ms'].isna() & df[col].notna()
            df.loc[take, 'metric_ms'] = df.loc[take, col]
            df.loc[take, 'metric_src'] = col

    if cache_col in df.columns and deck:
        df['residency_pct'] = (100.0 * df[cache_col] / deck).round(1)
    else:
        df['residency_pct'] = pd.NA

    for c in ('faults', 'evicts', 'completed', 'gates_pass', 'runtime',
              'vram_peak_mb', 'blocks_resolved'):
        if c not in df.columns:
            df[c] = pd.NA
    return df, None


frames, problems = [], []
for d, name, deck, col in SWEEPS:
    df, err = load(d, name, deck, col)
    if err:
        problems.append(err)
    else:
        frames.append(df)

if not frames:
    print('no results at all under %s' % ROOT)
    for p in problems:
        print('  ' + p)
    sys.exit(1)

all_df = pd.concat(frames, ignore_index=True, sort=False)
pd.set_option('display.width', 220)

cols = ['workload', 'residency_pct', 'blocks_resolved', 'completed',
        'gates_pass', 'faults', 'evicts', 'metric_ms', 'metric_src',
        'runtime', 'vram_peak_mb']
print('\n=== ALL CELLS ===')
print(all_df[cols].to_string(index=False))

print('\n=== FAULTS vs TIME, per workload ===')
print('x_resident = this cell / that workload\'s 100%-residency cell.')
print('A workload whose x_resident stays ~1.0 while faults span orders of')
print('magnitude is NOT paying for its page faults -- that is a result, not')
print('a broken cell.\n')

for name, g in all_df.groupby('workload', sort=False):
    g = g[(g['completed'] == 1) & g['metric_ms'].notna()].copy()
    if not len(g):
        print('%-11s no completed cell produced a metric' % name)
        continue
    full = g[g['residency_pct'] >= 100.0]
    base = float(full['metric_ms'].min()) if len(full) else None
    g = g.sort_values('residency_pct')
    print('%s  (%s)' % (name, g['metric_src'].iloc[0]))
    for _, r in g.iterrows():
        f = r['faults']
        fs = '%9d' % int(f) if pd.notna(f) else '     n/a'
        x = ('%5.2fx' % (r['metric_ms'] / base)) if base else '    -'
        print('   %6s%%  faults=%s  ms=%10.1f  %s'
              % (r['residency_pct'], fs, float(r['metric_ms']), x))
    if base and pd.notna(g['faults']).any():
        sub = g[g['faults'].notna() & (g['faults'] > 0)]
        if len(sub) > 1:
            fr = float(sub['faults'].max()) / max(1.0, float(sub['faults'].min()))
            tr = float(sub['metric_ms'].max()) / float(sub['metric_ms'].min())
            print('   -> faults span %.1fx, time spans %.2fx' % (fr, tr))
            if fr > 5 and tr < 1.5:
                print('      PAGING IS NOT ON THE CRITICAL PATH HERE.')
    print()

# gmx and lbann do not have their paging line parsed into results.csv, so
# their faults column is EMPTY -- not zero. Saying so matters: a blank read
# as "no faults" would invert the conclusion for two of six workloads.
blank = [n for n, g in all_df.groupby('workload')
         if g['faults'].isna().all()]
if blank:
    print('FAULTS NOT PARSED (blank, NOT zero) for: %s' % ', '.join(blank))
    print('  Read ~/.ppi-jarvis/shared/gv_cache_<wl>_a100*/wl/gvw_dist/'
          '*.node0.log for "paging: faults=... evicts=...".')

bad = all_df[all_df['completed'] != 1]
if len(bad):
    print('\nCELLS THAT DID NOT COMPLETE (%d):' % len(bad))
    print(bad[['workload', 'residency_pct', 'runtime']].to_string(index=False))
for p in problems:
    print('MISSING: ' + p)

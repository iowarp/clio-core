#!/usr/bin/env python3
"""Collect sweep.py result JSONs (CTE-API producer-consumer and wfrun runs) into one CSV.

Usage: collect.py [--root ~/jarvis-runs/dtschedule-results] [--out results.csv] [EXP ...]
One row per <exp>/<run>.json: makespan, validity, producer/consumer phase times and
dtschedule stats. A run is valid when status is ok and the consumer saw no bad files.
"""
import argparse
import csv
import glob
import json
import os


def row_of(path):
    """Flatten one result JSON into a dict."""
    with open(path) as fh:
        d = json.load(fh)
    m = d.get('makespan') or {}
    p, c, s = m.get('producer') or {}, m.get('consumer') or {}, d.get('stats') or {}
    bad = c.get('bad', 0) or 0
    return {
        'exp': d.get('exp'), 'run': d.get('run'), 'started': d.get('started'),
        'makespan_s': (m.get('total_ms') or 0) / 1000.0,
        'valid': int(m.get('status') == 'ok' and bad == 0 and d.get('rc') == 0),
        'bad': bad,
        'prod_compute_s': p.get('compute_s'), 'prod_write_s': p.get('write_s'),
        'prod_flush_s': p.get('flush_s'), 'prod_wall_s': p.get('wall_s'),
        'cons_wait_s': c.get('file_wait_s', c.get('wait_s')),
        'cons_read_s': c.get('read_s'), 'cons_cksum_s': c.get('cksum_s'),
        'cons_wall_s': c.get('wall_s'),
        'bytes_in': s.get('dtschedule.bytes_in'), 'bytes_out': s.get('dtschedule.bytes_out'),
        'compressed': s.get('dtschedule.compressed'), 'puts': s.get('dtschedule.puts'),
        'mean_ratio': s.get('dtschedule.mean_ratio'),
        'mean_select_ms': s.get('dtschedule.mean_select_ms'),
        'overrides': json.dumps(d.get('overrides', {}), sort_keys=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/jarvis-runs/dtschedule-results'))
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('exps', nargs='*')
    a = ap.parse_args()
    exps = a.exps or sorted(os.path.basename(x) for x in glob.glob(os.path.join(a.root, '*')))
    rows = []
    for e in exps:
        for f in sorted(glob.glob(os.path.join(a.root, e, '*.json'))):
            try:
                rows.append(row_of(f))
            except (OSError, ValueError, KeyError) as err:
                print(f'skip {f}: {err}')
    if not rows:
        print('no results'); return
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} rows -> {a.out}')


if __name__ == '__main__':
    main()

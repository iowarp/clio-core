#!/usr/bin/env python3
"""Reduce a run_float32_sweep.sh results dir to plain metric CSVs.

    collect_float32.py RESULTS_DIR

Reads RESULTS_DIR/<workload>/results.csv (corpus_sweep rows) and writes
    RESULTS_DIR/per_chunk.csv  one row per (workload, chunk, setting)
    RESULTS_DIR/summary.csv    one row per (workload, setting), over its chunks
Columns: codec, its settings without the shuffle, shuffle (none/byte/bit),
compression ratio, compress / decompress GPU time in ms (CUDA events around
the codec call only; the shuffle is not included) and throughput in GB/s.
summary.csv: ratio = total input bytes / total compressed bytes; times are
the mean over chunks; ok_chunks / chunks counts bit-exact round trips, and
only ok chunks enter the metrics.
"""

import csv
import os
import sys
from collections import OrderedDict


def split_shuffle(settings):
    """Separate the shuffle mode from a corpus_sweep settings string.

    @param settings "k=v k=v" as written by corpus_sweep
    @return (settings without shuffle, shuffle mode)
    """
    rest, shuffle = [], 'none'
    for tok in settings.split():
        if tok.startswith('shuffle='):
            shuffle = tok.split('=', 1)[1]
        else:
            rest.append(tok)
    return ' '.join(rest), shuffle


def gbps(nbytes, ms):
    """@return nbytes / ms as GB/s, '' when ms is zero."""
    return f"{nbytes / (ms * 1e6):.3f}" if ms > 0 else ''


def read_rows(results_dir):
    """Yield one normalized dict per corpus_sweep row of every workload.

    @param results_dir directory holding <workload>/results.csv
    """
    for wl in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, wl, 'results.csv')
        if not os.path.isfile(path):
            continue
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                settings, shuffle = split_shuffle(r['settings'])
                n, c = int(r['bytes']), int(r['comp_bytes'] or 0)
                yield {
                    'workload': wl, 'chunk': r['file'], 'bytes': n,
                    'codec': r['algorithm'], 'settings': settings,
                    'shuffle': shuffle, 'comp_bytes': c,
                    'comp_ms': float(r['comp_ms'] or 0),
                    'decomp_ms': float(r['decomp_ms'] or 0),
                    'ok': r['ok'] == '1',
                }


def write_per_chunk(rows, path):
    """Write one CSV row per (workload, chunk, setting)."""
    cols = ['workload', 'chunk', 'codec', 'settings', 'shuffle', 'bytes',
            'comp_bytes', 'ratio', 'comp_ms', 'decomp_ms', 'comp_gbps',
            'decomp_gbps', 'ok']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            ratio = f"{r['bytes'] / r['comp_bytes']:.4f}" if r['comp_bytes'] else ''
            w.writerow([r['workload'], r['chunk'], r['codec'], r['settings'],
                        r['shuffle'], r['bytes'], r['comp_bytes'], ratio,
                        f"{r['comp_ms']:.4f}", f"{r['decomp_ms']:.4f}",
                        gbps(r['bytes'], r['comp_ms']),
                        gbps(r['bytes'], r['decomp_ms']), int(r['ok'])])


def write_summary(rows, path):
    """Write one CSV row per (workload, setting), aggregated over chunks."""
    groups = OrderedDict()
    for r in rows:
        key = (r['workload'], r['codec'], r['settings'], r['shuffle'])
        g = groups.setdefault(key, {'chunks': 0, 'ok': 0, 'in': 0, 'out': 0,
                                    'cms': 0.0, 'dms': 0.0})
        g['chunks'] += 1
        if not r['ok']:
            continue
        g['ok'] += 1
        g['in'] += r['bytes']
        g['out'] += r['comp_bytes']
        g['cms'] += r['comp_ms']
        g['dms'] += r['decomp_ms']
    cols = ['workload', 'codec', 'settings', 'shuffle', 'chunks', 'ok_chunks',
            'input_bytes', 'comp_bytes', 'ratio', 'comp_ms_mean',
            'decomp_ms_mean', 'comp_gbps', 'decomp_gbps']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (wl, codec, settings, shuffle), g in groups.items():
            k = g['ok']
            w.writerow([wl, codec, settings, shuffle, g['chunks'], k,
                        g['in'], g['out'],
                        f"{g['in'] / g['out']:.4f}" if g['out'] else '',
                        f"{g['cms'] / k:.4f}" if k else '',
                        f"{g['dms'] / k:.4f}" if k else '',
                        gbps(g['in'], g['cms']), gbps(g['in'], g['dms'])])


def main():
    """Collect the results dir named on the command line."""
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        return 1
    d = sys.argv[1]
    rows = list(read_rows(d))
    write_per_chunk(rows, os.path.join(d, 'per_chunk.csv'))
    write_summary(rows, os.path.join(d, 'summary.csv'))
    print(f"{len(rows)} rows -> {d}/per_chunk.csv, {d}/summary.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())

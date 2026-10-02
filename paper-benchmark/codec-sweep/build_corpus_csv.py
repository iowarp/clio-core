#!/usr/bin/env python3
"""Turn corpus_sweep raw rows into a CSV laid out like benchmark_results_600k.csv.

    build_corpus_csv.py --corpus DIR --raw RAW.csv [RAW2.csv ...] --out OUT.csv
                        [--files FILES.txt] [--settings SETTINGS.txt]
                        [--ref benchmark_results_600k.csv]

Columns: the 600k CSV's, in its order, with `settings` after `algorithm`, no
`lib_throughput_mbps` (always 0 there), and four added at the end:
`comp_wall_ms`, `decomp_wall_ms`, `reps` and `note`.

How each value is defined:
  original_size / compressed_size   input bytes / the whole compressed stream
  compression_ratio                 original_size / compressed_size
  compression_time_ms               median CUDA-event time of the compress call
  decompression_time_ms             same, decompress call
  *_throughput_mbps                 original_size / 2**20 / seconds (MiB/s, as
                                    in the 600k CSV), from the event times
  comp_wall_ms / decomp_wall_ms     median host time incl. launch and syncs
  exact_match                       every rep round-tripped bit for bit
  success                           every rep compressed and decompressed
                                    without an error
  psnr_db / max_error / rmse        inf / 0 / 0 on an exact match, else empty
  entropy                           bytewise Shannon entropy, bits
  mad, second_derivative            mean |x - mean| and mean |x[i+2] - 2x[i+1]
                                    + x[i]|, divided by (max - min), as the
                                    600k CSV stores them
  shuffle / quantization / error_bound   0 / none / 0: no preprocessing,
                                    lossless only

Checks (exit 1 on any failure): every (file, setting) appears exactly once,
every listed file x every listed setting is present, original_size equals the
file's size on disk, and with --ref the statistics match the reference CSV.
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

NAME_RE = re.compile(r'^(?P<dtype>float32)_(?P<palette>[a-z_]+)_w(?P<w>\d+)'
                     r'_p(?P<p>\d+)_(?P<fill>[a-z]+)_(?P<size>\d+[km]b)\.bin$')
MIB = float(1 << 20)
STAT_TOL = 1e-6  # the 600k CSV rounds statistics to 6 decimals


def file_stats(path):
    """Statistics of one float32 file, defined as in the 600k CSV.

    @param path  raw float32 file
    @return (entropy, mad, second_derivative)
    """
    a = np.fromfile(path, dtype=np.float32)
    b = np.frombuffer(a.tobytes(), dtype=np.uint8)
    h = np.bincount(b, minlength=256)
    p = h[h > 0] / b.size
    f = a.astype(np.float64)
    rng = f.max() - f.min()
    rng = rng if rng > 0 else 1.0
    mad = np.mean(np.abs(f - f.mean())) / rng
    d2 = np.mean(np.abs(f[2:] - 2.0 * f[1:-1] + f[:-2])) / rng if f.size > 2 else 0.0
    return float(-np.sum(p * np.log2(p))), float(mad), float(d2)


def stats_table(corpus, names):
    """@return DataFrame file -> entropy, mad, second_derivative, disk_bytes."""
    rows = []
    for n in names:
        path = os.path.join(corpus, n)
        e, m, d = file_stats(path)
        rows.append((n, e, m, d, os.path.getsize(path)))
    return pd.DataFrame(rows, columns=['file', 'entropy', 'mad',
                                       'second_derivative', 'disk_bytes'])


def check_stats_against(ref_path, st):
    """Compare our statistics with the reference CSV's for shared files.

    @return list of problems, empty when every shared file matches
    """
    ref = (pd.read_csv(ref_path, usecols=['file', 'entropy', 'mad',
                                          'second_derivative'])
           .drop_duplicates('file').set_index('file'))
    j = st.set_index('file').join(ref, rsuffix='_ref', how='inner')
    if j.empty:
        return ['no file in common with the reference CSV']
    worst = max(float((j[c] - j[c + '_ref']).abs().max())
                for c in ['entropy', 'mad', 'second_derivative'])
    print(f"stats vs reference: {len(j)} files, worst |diff| {worst:.2e}")
    return [] if worst <= STAT_TOL else [f"stats differ from reference by {worst:.2e}"]


def canon(spec):
    """@return a codec spec with its settings in key order, as corpus_sweep
    writes them ("lz4 chunk=65536 bitshuffle=msb" -> "lz4 bitshuffle=msb
    chunk=65536")."""
    tok = spec.split()
    return ' '.join(tok[:1] + sorted(tok[1:], key=lambda t: t.split('=')[0]))


def check_coverage(raw, files, settings):
    """@return problems with duplicate or missing (file, setting) pairs."""
    probs = []
    spec = (raw['algorithm'] + ' ' + raw['settings']).map(canon)
    dup = int((raw['file'] + ' | ' + spec).duplicated().sum())
    if dup:
        probs.append(f"{dup} duplicated (file, setting) rows")
    if files is not None and settings is not None:
        want = len(files) * len(set(map(canon, settings)))
        have = raw[raw['file'].isin(files) &
                   spec.isin(set(map(canon, settings)))]
        if len(have) != want:
            probs.append(f"{len(have)} of {want} (file, setting) rows present")
    return probs


def fmt_num(x):
    """@return x the way the 600k CSV writes grid values (0.325, 1, 16)."""
    return f"{x:g}"


def build(raw, st):
    """Lay the joined rows out like the 600k CSV.

    @param raw  corpus_sweep rows
    @param st   per-file statistics
    @return the output DataFrame
    """
    d = raw.merge(st, on='file', how='left')
    meta = d['file'].str.extract(NAME_RE)
    ok = d['ok'] == 1
    fail_codec = d['note'].fillna('').str.match(
        r'^(compress failed|decompress failed|exception|setup|unknown|size not)')
    ratio = np.where(d['comp_bytes'] > 0, d['bytes'] / d['comp_bytes'].where(
        d['comp_bytes'] > 0), np.nan)
    sec_c = d['comp_ms'] / 1e3
    sec_d = d['decomp_ms'] / 1e3
    out = pd.DataFrame({
        'file': d['file'],
        'dtype': meta['dtype'],
        'palette': meta['palette'],
        'perturbation': (meta['p'].astype(int) / 1000).map(fmt_num),
        'bin_width': (meta['w'].astype(int) / 100).map(fmt_num),
        'fill_mode': meta['fill'],
        'algorithm': d['algorithm'],
        'settings': d['settings'].fillna(''),
        'shuffle': 0,
        'quantization': 'none',
        'error_bound': 0,
        'original_size': d['bytes'],
        'compressed_size': d['comp_bytes'],
        'compression_ratio': np.round(ratio, 6),
        'compression_time_ms': d['comp_ms'],
        'decompression_time_ms': d['decomp_ms'],
        'compression_throughput_mbps':
            np.round(np.where(sec_c > 0, d['bytes'] / MIB / sec_c.where(sec_c > 0), np.nan), 4),
        'decompression_throughput_mbps':
            np.round(np.where(sec_d > 0, d['bytes'] / MIB / sec_d.where(sec_d > 0), np.nan), 4),
        'psnr_db': np.where(ok, math.inf, np.nan),
        'max_error': np.where(ok, 0.0, np.nan),
        'rmse': np.where(ok, 0.0, np.nan),
        'entropy': np.round(d['entropy'], 6),
        'mad': np.round(d['mad'], 6),
        'second_derivative': np.round(d['second_derivative'], 6),
        'exact_match': ok,
        'success': ~fail_codec & (ok | d['note'].eq('output differs')),
        'comp_wall_ms': d['comp_wall_ms'],
        'decomp_wall_ms': d['decomp_wall_ms'],
        'reps': d['reps'],
        'note': d['note'].fillna(''),
    })
    return out, d


def main():
    """Read raw rows, check them, and write the output CSV."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--raw', required=True, nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--files', help='file list every setting must cover')
    ap.add_argument('--settings', help='settings list every file must cover')
    ap.add_argument('--ref', help='600k CSV to check statistics against')
    a = ap.parse_args()

    raw = pd.concat([pd.read_csv(p, keep_default_na=False,
                                 na_values={'comp_ms': [''], 'decomp_ms': ['']})
                     for p in a.raw], ignore_index=True)
    # A job killed at its time limit can leave a half-written last line.
    num = ['bytes', 'comp_bytes', 'comp_ms', 'comp_wall_ms', 'decomp_ms',
           'decomp_wall_ms', 'reps', 'ok']
    for c in num:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    cut = raw[num].isna().any(axis=1)
    if cut.any():
        print(f"dropped {int(cut.sum())} incomplete rows")
        raw = raw[~cut].reset_index(drop=True)
    raw['settings'] = raw['settings'].astype(str)
    raw['note'] = raw['note'].astype(str)
    files = settings = None
    if a.files:
        files = [l.strip() for l in open(a.files) if l.strip()]
    if a.settings:
        settings = [l.strip() for l in open(a.settings)
                    if l.strip() and not l.lstrip().startswith('#')]
    probs = check_coverage(raw, files, settings)

    st = stats_table(a.corpus, sorted(raw['file'].unique()))
    if a.ref:
        probs += check_stats_against(a.ref, st)
    out, d = build(raw, st)
    size_bad = int((d['bytes'] != d['disk_bytes']).sum())
    if size_bad:
        probs.append(f"{size_bad} rows whose original_size is not the file size")
    if out['palette'].isna().any():
        probs.append("file names that do not parse")

    out.to_csv(a.out, index=False)
    n_ok = int(out['exact_match'].sum())
    print(f"wrote {a.out}: {len(out)} rows, {out['file'].nunique()} files, "
          f"{(out['algorithm'] + ' ' + out['settings']).nunique()} settings, "
          f"{n_ok} exact, {len(out) - n_ok} not")
    for p in probs:
        print('CHECK FAILED:', p)
    return 1 if probs else 0


if __name__ == '__main__':
    sys.exit(main())

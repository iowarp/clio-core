#!/usr/bin/env python3
"""Settings grid and file lists for corpus_sweep.

    corpus_grid.py settings > settings.txt
    corpus_grid.py files --dir CORPUS [--sample-seed S] > files.txt

`settings` writes one codec spec per line (see gpu_codecs.cuh):
    nvcomp chunk sizes   64 KiB, 256 KiB, 1 MiB for all eight nvcomp codecs
    lz4        type char/short/int x bitshuffle none/msb/lsb
    snappy     chunk only
    zstd       chunk only
    gdeflate   level 0, 1, 2, 4, 5 (nvcomp: 3 is the same as 2)
    deflate    level 0, 1, 2, 4, 5
    ans        type char/float16
    cascaded   type char/short/int/longlong x rle 0-2 x delta 0-2 x bp 0/1,
               without rle=0 delta=0 bp=0 (stores the input unchanged)
    bitcomp    algo 0 (default)/1 (sparse) x type char/short/int/longlong
    ndzip, gpulz, spspeed, spratio   no settings

`files` lists every .bin file in the corpus, or with --sample-seed one file
per (palette, fill mode, size) with a seeded random bin width and
perturbation: 7 x 5 x 5 = 175 files.
"""

import argparse
import itertools
import os
import random
import re
import sys

CHUNKS = [64 << 10, 256 << 10, 1 << 20]
NAME_RE = re.compile(r'^float32_(?P<palette>[a-z_]+)_w(?P<w>\d+)_p(?P<p>\d+)'
                     r'_(?P<fill>[a-z]+)_(?P<size>\d+[km]b)\.bin$')


def settings(shuffles=('none',), ndzip_shapes=()):
    """Every codec spec line of the grid, in run order.

    @param shuffles      shuffle modes (none, byte, bit); every line of the
                         grid is repeated once per mode, "none" unsuffixed
    @param ndzip_shapes  extra ndzip shapes ("3600", "128x128") besides 1-D
    @return the spec lines
    """
    out = []
    for ch in CHUNKS:
        c = f"chunk={ch}"
        for t, b in itertools.product(['char', 'short', 'int'],
                                      ['none', 'msb', 'lsb']):
            out.append(f"lz4 {c} type={t} bitshuffle={b}")
        out.append(f"snappy {c}")
        out.append(f"zstd {c}")
        for base in ['gdeflate', 'deflate']:
            out += [f"{base} {c} level={lv}" for lv in [0, 1, 2, 4, 5]]
        out += [f"ans {c} type={t}" for t in ['char', 'float16']]
        for t, r, d, bp in itertools.product(
                ['char', 'short', 'int', 'longlong'], [0, 1, 2], [0, 1, 2],
                [0, 1]):
            if r == d == bp == 0:
                continue
            out.append(f"cascaded {c} type={t} rle={r} delta={d} bp={bp}")
        for a, t in itertools.product([0, 1],
                                      ['char', 'short', 'int', 'longlong']):
            out.append(f"bitcomp {c} algo={a} type={t}")
    out += ['ndzip'] + [f"ndzip shape={s}" for s in ndzip_shapes]
    out += ['gpulz', 'spspeed', 'spratio']
    return [line if sh == 'none' else f"{line} shuffle={sh}"
            for sh in shuffles for line in out]


def files(corpus_dir, sample_seed=None):
    """List corpus files, all or one per (palette, fill, size) stratum.

    @param corpus_dir   directory holding the float32_*.bin files
    @param sample_seed  None for every file, else the stratified sample's seed
    @return file names, sorted
    """
    names = sorted(n for n in os.listdir(corpus_dir) if NAME_RE.match(n))
    if sample_seed is None:
        return names
    strata = {}
    for n in names:
        m = NAME_RE.match(n)
        strata.setdefault((m['palette'], m['fill'], m['size']), []).append(n)
    rng = random.Random(sample_seed)
    return sorted(rng.choice(strata[k]) for k in sorted(strata))


def main():
    """Write the requested list to stdout."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    st = sub.add_parser('settings')
    st.add_argument('--shuffle', default='none',
                    help="comma list of none, byte, bit (default none)")
    st.add_argument('--ndzip-shape', action='append', default=[],
                    help='extra ndzip shape, e.g. 128x128 (repeatable)')
    f = sub.add_parser('files')
    f.add_argument('--dir', required=True)
    f.add_argument('--sample-seed', type=int, default=None)
    a = ap.parse_args()
    if a.cmd == 'settings':
        lines = settings(a.shuffle.split(','), a.ndzip_shape)
    else:
        lines = files(a.dir, a.sample_seed)
    sys.stdout.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

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
    ans        type char/float16/float8 (nvcomp 5 FP8 E4M3)
    cascaded   type char/short/int/longlong x rle 0-2 x delta 0-2 x bp 0/1,
               without rle=0 delta=0 bp=0 (stores the input unchanged)
    bitcomp    algo 0 (default)/1 (sparse) x type char/short/int/longlong
    ndzip, gpulz, spspeed, spratio   no settings
    store      stored uncompressed, once (no shuffle variants)

`settings --profile float32` writes a grid trimmed to float32 data, at the
same three nvcomp chunk sizes:
    lz4        type=char bitshuffle=none (plain LZ4; type only drives the
               bitshuffle), type=int bitshuffle=msb/lsb (4-byte words)
    snappy, zstd       chunk only
    gdeflate, deflate  level 0 (fastest) and 5 (best ratio) only
    ans        type=char (float16 would split every float in half)
    cascaded   type=int (one float word) and char (one byte plane, the
               layout a byte shuffle produces) x rle 0-2 x delta 0-2 x bp
               0/1, without rle=0 delta=0 bp=0
    bitcomp    algo 0/1 x type=int/char, same reasoning
    ndzip, gpulz, spspeed, spratio   as in the full grid
    store      once, without a shuffle: the chunk stored uncompressed
               (ratio 1, a device-to-device copy each way), the baseline
               every codec has to beat

`settings --profile cpu [--dtype float|double] [--threads 1,128]` writes the
CPU grid for cpu_corpus_sweep (cpu_codecs.h), every line at each thread
count:
    zstd 1/3/19, lz4 / lz4-hc 9, zlib 1/6, bzip2 9, xz 0/6, brotli 1/6,
               snappy, lzo        each with --shuffle's modes (byte =
                                  an HDF5-style shuffle at the element size)
    blosc2     blosclz/lz4/zstd x shuffle/bitshuffle/bytedelta, clevel 5
    fpzip, zfp (reversible), ndzip (CPU)   1-D and each --ndzip-shape
    store      stored uncompressed (memcpy), no shuffle

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


def with_shuffles(lines, shuffles, dtype):
    """Repeat GPU spec lines once per shuffle mode.

    @param lines    spec lines without a shuffle
    @param shuffles shuffle modes (none, byte, bit); "none" stays unsuffixed
    @param dtype    float or double: the byte shuffle's element size (4 / 8)
    @return the suffixed lines, mode by mode
    """
    tail = {'none': '', 'byte': ' shuffle=byte', 'bit': ' shuffle=bit'}
    if dtype == 'double':
        tail['byte'] += ' elem=8'
    return [line + tail[sh] for sh in shuffles for line in lines]


def settings(shuffles=('none',), ndzip_shapes=(), dtype='float'):
    """Every codec spec line of the grid, in run order.

    @param shuffles      shuffle modes (none, byte, bit); every line of the
                         grid is repeated once per mode, "none" unsuffixed
    @param ndzip_shapes  extra ndzip shapes ("3600", "128x128") besides 1-D
    @param dtype         float or double, the byte shuffle's element type
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
        out += [f"ans {c} type={t}" for t in ['char', 'float16', 'float8']]
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
    return with_shuffles(out, shuffles, dtype) + ['store']


def settings_float32(shuffles=('none',), ndzip_shapes=(), dtype='float'):
    """The grid trimmed to float32 data (see the module docstring).

    @param shuffles      shuffle modes (none, byte, bit); every line of the
                         grid is repeated once per mode, "none" unsuffixed
    @param ndzip_shapes  extra ndzip shapes ("3600", "128x128") besides 1-D
    @param dtype         float or double, the byte shuffle's element type
    @return the spec lines
    """
    out = []
    word_types = ['int', 'char']
    for ch in CHUNKS:
        c = f"chunk={ch}"
        out.append(f"lz4 {c} type=char bitshuffle=none")
        out += [f"lz4 {c} type=int bitshuffle={b}" for b in ['msb', 'lsb']]
        out.append(f"snappy {c}")
        out.append(f"zstd {c}")
        for base in ['gdeflate', 'deflate']:
            out += [f"{base} {c} level={lv}" for lv in [0, 5]]
        out.append(f"ans {c} type=char")
        for t, r, d, bp in itertools.product(word_types, [0, 1, 2],
                                             [0, 1, 2], [0, 1]):
            if r == d == bp == 0:
                continue
            out.append(f"cascaded {c} type={t} rle={r} delta={d} bp={bp}")
        for a, t in itertools.product([0, 1], word_types):
            out.append(f"bitcomp {c} algo={a} type={t}")
    out += ['ndzip'] + [f"ndzip shape={s}" for s in ndzip_shapes]
    out += ['gpulz', 'spspeed', 'spratio']
    return with_shuffles(out, shuffles, dtype) + ['store']


def settings_cpu(shuffles=('none',), shapes=(), dtype='float',
                 threads=(1, 128)):
    """The CPU grid for cpu_corpus_sweep (see cpu_codecs.h).

    @param shuffles shuffle modes for the general-purpose codecs (none, byte,
                    bit); Blosc2 filters itself and the float codecs need the
                    values intact, so neither is shuffled
    @param shapes   extra fpzip / zfp / ndzip shapes ("3600", "128x128")
                    besides 1-D
    @param dtype    element type of the data, float or double
    @param threads  thread counts; every line is repeated once per count,
                    1 unsuffixed
    @return the spec lines
    """
    general = ([f"zstd level={lv}" for lv in [1, 3, 19]]
               + [f"lz4 level={lv}" for lv in [0, 9]]
               + [f"zlib level={lv}" for lv in [1, 6]]
               + ["bzip2 level=9"]
               + [f"xz preset={p}" for p in [0, 6]]
               + [f"brotli quality={q}" for q in [1, 6]]
               + ["snappy", "lzo"])
    # The byte shuffle is HDF5-style, at the element size; the bit shuffle
    # is the GPU sweep's 32-word one.
    tail = {'none': '', 'byte': f" shuffle=byte type={dtype}",
            'bit': " shuffle=bit"}
    out = [line + tail[sh] for sh in shuffles for line in general]
    out += [f"blosc2 codec={c} filter={f} clevel=5 type={dtype}"
            for c in ['blosclz', 'lz4', 'zstd']
            for f in ['shuffle', 'bitshuffle', 'bytedelta']]
    for base in ['fpzip', 'zfp', 'ndzip']:
        out.append(f"{base} type={dtype}")
        out += [f"{base} type={dtype} shape={s}" for s in shapes]
    out.append("store")
    return [line if t == 1 else f"{line} threads={t}"
            for t in threads for line in out]


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
    st.add_argument('--profile', choices=['full', 'float32', 'cpu'],
                    default='full',
                    help='full GPU grid, the GPU grid trimmed to float32 '
                         'data, or the CPU grid (cpu_corpus_sweep)')
    st.add_argument('--dtype', choices=['float', 'double'], default='float',
                    help='element type: the byte shuffle element size, and the '
                         'cpu float codecs\' type')
    st.add_argument('--threads', default='1,128',
                    help='comma list of thread counts, for the cpu profile')
    f = sub.add_parser('files')
    f.add_argument('--dir', required=True)
    f.add_argument('--sample-seed', type=int, default=None)
    a = ap.parse_args()
    if a.cmd == 'settings' and a.profile == 'cpu':
        lines = settings_cpu(a.shuffle.split(','), a.ndzip_shape, a.dtype,
                             [int(t) for t in a.threads.split(',')])
    elif a.cmd == 'settings':
        make = settings_float32 if a.profile == 'float32' else settings
        lines = make(a.shuffle.split(','), a.ndzip_shape, a.dtype)
    else:
        lines = files(a.dir, a.sample_seed)
    sys.stdout.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

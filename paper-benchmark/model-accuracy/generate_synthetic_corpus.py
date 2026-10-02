#!/usr/bin/env python3
"""Generate the 9800 synthetic float32 files behind benchmark_results_600k.csv.

Port of NeuroPress syntheticGeneration/generator.py (deleted in NeuroPress
29d049f), reduced to the one batch that produced the corpus:

    generator.py batch --mode training -s 16KB,64KB,256KB,1MB,4MB

Grid (7 x 7 x 8 x 5 x 5 = 9800 files, ~10.2 GiB):
    palette       uniform normal gamma exponential bimodal grayscott high_entropy
    bin_width     0.1 0.12 0.15 0.25 0.5 1.0 16
    perturbation  0 0.1 0.2 0.325 0.5 0.75 0.95 1.0
    fill_mode     const linear quad sin rand
    size          16KB 64KB 256KB 1MB 4MB

Each file is raw little-endian float32, no header, named
    float32_<palette>_w<bin_width*100>_p<perturbation*1000>_<fill>_<size>.bin
e.g. float32_bimodal_w100_p0_const_16kb.bin. Seeds follow the original
enumeration order, so the bytes match the files the CSV was measured on.

Usage:
    generate_synthetic_corpus.py --output-dir /path/to/synthetic [--workers N]
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

N_BINS = 32

PALETTES = ['uniform', 'normal', 'gamma', 'exponential',
            'bimodal', 'grayscott', 'high_entropy']
BIN_WIDTHS = [0.1, 0.12, 0.15, 0.25, 0.5, 1.0, 16.0]
PERTURBATIONS = [0.0, 0.1, 0.2, 0.325, 0.5, 0.75, 0.95, 1.0]
FILL_MODES = ['const', 'linear', 'quad', 'sin', 'rand']
SIZES = [('16KB', 16 * 1024), ('64KB', 64 * 1024), ('256KB', 256 * 1024),
         ('1MB', 1024 * 1024), ('4MB', 4 * 1024 * 1024)]


# ============================================================
# Bin weights per palette
# ============================================================

def _uniform_weights(n):
    """Equal share per bin.

    @param n  number of bins
    @return   weights summing to 1
    """
    return np.ones(n) / n


def _normal_weights(n):
    """Gaussian over x in [-3, 3], peaked on the middle bins.

    @param n  number of bins
    @return   weights summing to 1
    """
    x = np.linspace(-3, 3, n)
    w = np.exp(-0.5 * x**2)
    return w / w.sum()


def _gamma_weights(n):
    """Shape-2 gamma over x in [0.1, 6], skewed toward low bins.

    @param n  number of bins
    @return   weights summing to 1
    """
    x = np.linspace(0.1, 6.0, n)
    w = x * np.exp(-x)
    return w / w.sum()


def _exponential_weights(n):
    """99.99% in bin 0, the rest decaying by half per bin.

    @param n  number of bins
    @return   weights summing to 1
    """
    w = np.zeros(n)
    w[0] = 0.9999
    decay = np.power(0.5, np.arange(1, n).astype(float))
    w[1:] = decay / decay.sum() * 0.0001
    return w


def _bimodal_weights(n):
    """40% on each end pair of bins, 20% spread across the middle.

    @param n  number of bins
    @return   weights summing to 1
    """
    w = np.full(n, 0.2 / max(n - 4, 1))
    w[0], w[1] = 0.24, 0.16
    w[-2], w[-1] = 0.16, 0.24
    return w / w.sum()


def _grayscott_weights(n):
    """Gray-Scott-like: 70% background (low bins), 20% spots (high bins),
    10% edges spread across the middle.

    @param n  number of bins
    @return   weights summing to 1
    """
    w = np.zeros(n)
    bg = max(2, n // 4)
    sp = max(2, n // 4)
    for i in range(bg):
        t = i / bg
        w[i] = 0.7 * np.exp(-2 * t * t)
    for i in range(sp):
        t = i / sp
        w[n - 1 - i] = 0.2 * np.exp(-2 * t * t)
    w[bg:n - sp] = 0.1 / (n - sp - bg)
    return w / w.sum()


WEIGHT_FUNCS = {
    'uniform': _uniform_weights,
    'normal': _normal_weights,
    'gamma': _gamma_weights,
    'exponential': _exponential_weights,
    'bimodal': _bimodal_weights,
    'grayscott': _grayscott_weights,
    'high_entropy': _uniform_weights,  # entropy comes from the bin layout
}


# ============================================================
# Bin layout
# ============================================================

def standard_bins(n, bin_width):
    """Lay n bins of width bin_width over [0, 1].

    Bins that fit are spaced with equal gaps; bins that do not fit start
    at even steps and overlap, clipped at 1.

    @param n          number of bins
    @param bin_width  absolute bin width in value space
    @return           (bin_lo, bin_hi) arrays
    """
    if bin_width * n <= 1.0:
        gap = (1.0 - bin_width * n) / (n - 1)
        bin_lo = np.arange(n) * (bin_width + gap)
    else:
        bin_lo = np.arange(n) * (1.0 / n)
    return bin_lo, np.minimum(bin_lo + bin_width, 1.0)


def high_entropy_bins(n):
    """Narrow bins centred at 1e-6 .. 1, log-spaced, so values span many
    float exponents and the bytes are diverse. Ignores bin_width.

    @param n  number of bins
    @return   (bin_lo, bin_hi) arrays
    """
    bin_lo = np.empty(n)
    bin_hi = np.empty(n)
    for i in range(n):
        center = 10.0 ** (-6.0 + (i / (n - 1)) * 6.0)
        width = center * 0.1
        bin_lo[i] = max(0.0, center - width / 2)
        bin_hi[i] = min(1.0, center + width / 2)
    return bin_lo, bin_hi


# ============================================================
# Generator
# ============================================================

def plan_bursts(targets, perturbation, seed):
    """Choose the sequence of (bin, run length) bursts.

    Each step picks a random unfinished bin and emits
    max(1, remaining * (1 - perturbation)) values from it, so
    perturbation 0 gives one run per bin and 1 gives single values.

    @param targets       per-bin element counts, summing to the chunk size
    @param perturbation  0.0 = long runs, 1.0 = fully interleaved
    @param seed          seed for the burst RNG
    @return              (burst_bins, burst_sizes) as numpy arrays
    """
    rng = random.Random(seed)
    tgt = targets.tolist()
    num_elements = sum(tgt)
    counts = [0] * len(tgt)
    active = [i for i in range(len(tgt)) if tgt[i] > 0]
    burst_bins, burst_sizes = [], []
    pos = 0
    while pos < num_elements and active:
        pick = rng.randrange(len(active))
        bi = active[pick]
        rem = tgt[bi] - counts[bi]
        burst = max(1, int(rem * (1.0 - perturbation)))
        burst = min(burst, rem, num_elements - pos)
        burst_bins.append(bi)
        burst_sizes.append(burst)
        pos += burst
        counts[bi] += burst
        if counts[bi] >= tgt[bi]:
            active[pick] = active[-1]
            active.pop()
    return (np.array(burst_bins, dtype=np.int32),
            np.array(burst_sizes, dtype=np.int64))


def fill_values(lo, hi, bp_sizes, fill_mode, seed):
    """Fill each burst with values from its bin.

    @param lo         per-element bin lower bound
    @param hi         per-element bin upper bound
    @param bp_sizes   burst lengths, in order
    @param fill_mode  const | linear | quad | sin | rand
    @param seed       seed for the value RNG (rand only)
    @return           float64 values, one per element
    """
    n = lo.size
    mid = (lo + hi) * 0.5
    data = np.empty(n, dtype=np.float64)
    if fill_mode == 'const':
        data[:] = lo
        return data
    if fill_mode == 'rand':
        same = lo == hi
        data[same] = lo[same]
        diff = ~same
        n_diff = diff.sum()
        if n_diff > 0:
            u = np.random.RandomState(seed).random_sample(n_diff)
            data[diff] = lo[diff] + u * (hi[diff] - lo[diff])
        return data

    # linear, quad and sin need each element's position inside its burst.
    starts = np.zeros(len(bp_sizes), dtype=np.int64)
    starts[1:] = np.cumsum(bp_sizes)[:-1]
    within = np.arange(n, dtype=np.float64) - np.repeat(starts, bp_sizes)
    sizes_f = np.repeat(bp_sizes.astype(np.float64), bp_sizes)
    single = sizes_f == 1.0
    span = hi - lo
    if fill_mode == 'linear':
        t = within / np.maximum(sizes_f - 1.0, 1.0)
        data[:] = np.where(single, mid, lo + t * span)
    elif fill_mode == 'quad':
        phase = within / np.maximum(sizes_f - 1.0, 1.0) - 0.5
        vals = mid + (4.0 * (span / 4.0) / 0.25) * phase * phase - span / 4.0
        data[:] = np.where(single, mid,
                           np.clip(vals, np.minimum(lo, hi), np.maximum(lo, hi)))
    elif fill_mode == 'sin':
        t = within / np.maximum(sizes_f, 1.0)
        data[:] = np.where(single, mid, mid + span * 0.5 * np.sin(2.0 * np.pi * t))
    else:
        raise ValueError(f"unknown fill mode: {fill_mode}")
    return data


def generate(num_elements, palette, bin_width, perturbation, fill_mode, seed):
    """Generate one synthetic float32 chunk.

    @param num_elements  number of float32 values
    @param palette       bin weight distribution (see PALETTES)
    @param bin_width     absolute bin width in [0, 1] value space
    @param perturbation  0.0 = long runs, 1.0 = fully interleaved
    @param fill_mode     value pattern inside a burst (see FILL_MODES)
    @param seed          RNG seed
    @return              float32 array of num_elements values
    """
    weights = WEIGHT_FUNCS[palette](N_BINS)
    if palette == 'high_entropy':
        bin_lo, bin_hi = high_entropy_bins(N_BINS)
    else:
        bin_lo, bin_hi = standard_bins(N_BINS, bin_width)

    targets = (weights * num_elements).astype(int)
    targets[-1] = num_elements - targets[:-1].sum()

    bp_bins, bp_sizes = plan_bursts(targets, perturbation, seed)
    bin_idx = np.repeat(bp_bins, bp_sizes)
    data = fill_values(bin_lo[bin_idx], bin_hi[bin_idx], bp_sizes, fill_mode, seed)
    return data.astype(np.float32)


# ============================================================
# Batch
# ============================================================

def file_name(palette, bin_width, perturbation, fill_mode, size_label):
    """Corpus file name for one configuration.

    @return  e.g. float32_bimodal_w100_p0_const_16kb.bin
    """
    return (f"float32_{palette}_w{int(round(bin_width * 100))}"
            f"_p{int(round(perturbation * 1000))}_{fill_mode}"
            f"_{size_label.lower()}.bin")


def work_items():
    """Every (file name, element count, palette, width, perturbation, fill,
    seed) in the corpus. Seed = 1 + position in the original enumeration
    order (palette, width, perturbation, fill, size).
    """
    items = []
    for pal in PALETTES:
        for bw in BIN_WIDTHS:
            for pert in PERTURBATIONS:
                for fm in FILL_MODES:
                    for label, nbytes in SIZES:
                        seed = len(items) + 1
                        items.append((file_name(pal, bw, pert, fm, label),
                                      nbytes // 4, pal, bw, pert, fm, seed))
    return items


def write_one(output_dir, name, num_elements, palette, bin_width,
              perturbation, fill_mode, seed):
    """Generate one file and write it as raw float32.

    @return  the file name written
    """
    data = generate(num_elements, palette, bin_width, perturbation, fill_mode, seed)
    data.tofile(os.path.join(output_dir, name))
    return name


def main():
    """Parse arguments and write all 9800 files in parallel."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--output-dir', '-o', required=True,
                    help='directory to write the .bin files into')
    ap.add_argument('--workers', '-t', type=int, default=os.cpu_count() or 4,
                    help='worker processes (default: CPU count)')
    args = ap.parse_args()

    items = work_items()
    total_bytes = sum(n * 4 for _, n, *_ in items)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing {len(items)} files ({total_bytes / 2**30:.1f} GiB) to "
          f"{args.output_dir} with {args.workers} workers")

    start = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(write_one, args.output_dir, *it) for it in items]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 200 == 0 or done == len(items):
                elapsed_ms = (time.monotonic() - start) * 1000.0
                print(f"  {done}/{len(items)}  {elapsed_ms:.0f} ms", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

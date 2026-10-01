#!/usr/bin/env python3
"""
QTable training-data collector.

Walks a workflow output directory, reads files in chunks of multiple
sizes, runs every available compressor at three presets (low / med /
high == FAST / BALANCED / BEST in clio_ctp/compress), and writes a
CSV with everything the qtable predictor and lossy-quality analyses
need.

CSV columns
-----------
workflow_id            Logical workflow name (from --workflow-id)
file                   Path relative to --input-dir
compression_method     Library: zstd, lz4, bzip2, zlib, lzma, brotli,
                       snappy, blosc2, zfp, sz3, fpzip
compression_level      One of {low, med, high} (maps to FAST/BALANCED/BEST)
chunk_size             Bytes (65536, 131072, 262144, 524288, 1048576)
error_bound            Absolute-error tolerance for lossy; "" for lossless
io_time_ns             Wall-time to read the chunk from disk
cpu_time_ns            Process CPU time of the compression call
decompress_time_ns     Process CPU time of the decompression call
compression_ratio      uncompressed_size / compressed_size

  Chunk features (computed once per chunk, joined to every compressor row,
  formulas match data_stats.h::CalculateAllStatistics, all bytewise):
shannon_entropy        Bytewise Shannon entropy, bits in [0, 8]
mad                    Mean absolute deviation from byte mean
second_derivative      mean(|x[i+2] - 2 x[i+1] + x[i]|) over bytes

  Lossy quality (computed on float32-reinterpretation, "" for lossless):
psnr                   10 log10(range^2 / MSE)
ssim                   Global SSIM, range derived from original chunk
pointwise_max_error    max(|orig - reconstructed|)

Parallelization
---------------
Work unit = (file, offset, chunk_size). Each unit reads its chunk once
and runs every available compressor on it. Units are scheduled across
a ThreadPoolExecutor (compress C-extensions release the GIL).

Across the cluster, run one sbatch job per workflow.

Usage
-----
  qtable_data_collect.py \\
    --workflow-id montage \\
    --input-dir /home/llogan/jarvis-runs/montage \\
    --output-csv montage.csv \\
    --workers 24 \\
    --max-chunks-per-file 200
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------- Optional compressor imports ----------------------------------

_AVAIL: dict[str, object] = {}

import bz2
import lzma
import zlib
_AVAIL["bz2"] = bz2
_AVAIL["zlib"] = zlib
_AVAIL["lzma"] = lzma

try:
    import zstandard as _zstd
    _AVAIL["zstd"] = _zstd
except ImportError:
    pass
try:
    import lz4.block as _lz4block
    _AVAIL["lz4"] = _lz4block
except ImportError:
    pass
try:
    import brotli as _brotli
    _AVAIL["brotli"] = _brotli
except ImportError:
    pass
try:
    import snappy as _snappy
    _AVAIL["snappy"] = _snappy
except ImportError:
    pass
try:
    import blosc2 as _blosc2
    _AVAIL["blosc2"] = _blosc2
except ImportError:
    pass
try:
    import zfpy as _zfpy
    _AVAIL["zfp"] = _zfpy
except ImportError:
    pass
try:
    import pylibpressio as _plp
    _AVAIL["libpressio"] = _plp
except ImportError:
    pass

CHUNK_SIZES = [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]


# ---------- Compressor registry -----------------------------------------

# Each entry: (method, level, compress_fn, decompress_fn, error_bound, is_lossy)
# *_fn takes/returns bytes (compress_fn input is always raw bytes; the
# fn handles any needed reinterpretation internally for lossy).
COMPRESSORS: list = []

def _register():
    if "zstd" in _AVAIL:
        # Use module-level zstandard.compress/decompress — each call creates
        # its own context, so this is thread-safe. ZstdCompressor instances
        # are NOT thread-safe and deadlock under concurrent compress() calls.
        _zmod = _AVAIL["zstd"]
        for lvl, c in (("low", 1), ("med", 3), ("high", 22)):
            COMPRESSORS.append(("zstd", lvl,
                                (lambda c=c: lambda b: _zmod.compress(b, level=c))(),
                                _zmod.decompress, None, False))

    if "lz4" in _AVAIL:
        block = _AVAIL["lz4"]
        COMPRESSORS.append(("lz4", "low",
                            lambda b: block.compress(b, mode="fast"),
                            block.decompress, None, False))
        COMPRESSORS.append(("lz4", "med",
                            lambda b: block.compress(b, mode="high_compression", compression=9),
                            block.decompress, None, False))
        COMPRESSORS.append(("lz4", "high",
                            lambda b: block.compress(b, mode="high_compression", compression=12),
                            block.decompress, None, False))

    if "bz2" in _AVAIL:
        for lvl, c in (("low", 1), ("med", 5), ("high", 9)):
            COMPRESSORS.append(("bzip2", lvl,
                                (lambda c=c: lambda b: bz2.compress(b, c))(),
                                bz2.decompress, None, False))

    if "zlib" in _AVAIL:
        for lvl, c in (("low", 1), ("med", 6), ("high", 9)):
            COMPRESSORS.append(("zlib", lvl,
                                (lambda c=c: lambda b: zlib.compress(b, c))(),
                                zlib.decompress, None, False))

    if "lzma" in _AVAIL:
        for lvl, c in (("low", 0), ("med", 6), ("high", 9)):
            COMPRESSORS.append(("lzma", lvl,
                                (lambda c=c: lambda b: lzma.compress(b, preset=c))(),
                                lzma.decompress, None, False))

    if "brotli" in _AVAIL:
        bro = _AVAIL["brotli"]
        for lvl, c in (("low", 1), ("med", 6), ("high", 11)):
            COMPRESSORS.append(("brotli", lvl,
                                (lambda c=c: lambda b: bro.compress(b, quality=c))(),
                                bro.decompress, None, False))

    if "snappy" in _AVAIL:
        COMPRESSORS.append(("snappy", "med",
                            _AVAIL["snappy"].compress,
                            _AVAIL["snappy"].decompress, None, False))

    if "blosc2" in _AVAIL:
        b2 = _AVAIL["blosc2"]
        for lvl, clevel, codec in (("low", 1, b2.Codec.LZ4),
                                    ("med", 5, b2.Codec.ZSTD),
                                    ("high", 9, b2.Codec.ZSTD)):
            COMPRESSORS.append(("blosc2", lvl,
                                (lambda cl=clevel, co=codec:
                                    lambda b: b2.compress(b, clevel=cl, codec=co))(),
                                b2.decompress, None, False))

    if "zfp" in _AVAIL:
        zfpy = _AVAIL["zfp"]
        def make_zfp(tol):
            def _c(b):
                arr = np.frombuffer(b, dtype=np.float32)
                return bytes(zfpy.compress_numpy(arr, tolerance=tol))
            def _d(b):
                arr = zfpy.decompress_numpy(np.frombuffer(b, dtype=np.uint8))
                return arr.tobytes()
            return _c, _d
        for lvl, tol in (("low", 1e-2), ("med", 1e-4), ("high", 1e-6)):
            c, d = make_zfp(tol)
            COMPRESSORS.append(("zfp", lvl, c, d, tol, True))

    if "libpressio" in _AVAIL:
        plp = _AVAIL["libpressio"]
        for name in ("sz3", "fpzip"):
            for lvl, tol in (("low", 1e-2), ("med", 1e-4), ("high", 1e-6)):
                def make_plp(n=name, t=tol):
                    def _c(b):
                        arr = np.frombuffer(b, dtype=np.float32).copy()
                        comp = plp.PressioCompressor.from_config({
                            "compressor_id": n,
                            "compressor_config": {"pressio:abs": t},
                        })
                        out = comp.encode(arr)
                        return bytes(out)
                    def _d(b):
                        comp = plp.PressioCompressor.from_config({
                            "compressor_id": n,
                            "compressor_config": {"pressio:abs": t},
                        })
                        # decode requires a template output array.
                        template = np.zeros(1, dtype=np.float32)
                        out = comp.decode(np.frombuffer(b, dtype=np.uint8), template)
                        return out.tobytes()
                    return _c, _d
                c, d = make_plp()
                COMPRESSORS.append((name, lvl, c, d, tol, True))

_register()


# ---------- Per-chunk feature extraction --------------------------------

def _chunk_features(chunk: bytes) -> tuple[float, float, float]:
    """Bytewise (shannon_entropy, MAD, second_derivative) per data_stats.h."""
    arr = np.frombuffer(chunk, dtype=np.uint8)
    # Shannon entropy from byte histogram
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    n = float(arr.size)
    p = hist[hist > 0] / n
    entropy = float(-(p * np.log2(p)).sum())
    # MAD = mean(|x - mean(x)|) over bytes
    f = arr.astype(np.float64)
    mean = f.mean()
    mad = float(np.abs(f - mean).mean())
    # Second derivative: mean(|x[i+2] - 2 x[i+1] + x[i]|)
    if f.size >= 3:
        sd = float(np.abs(f[2:] - 2.0 * f[1:-1] + f[:-2]).mean())
    else:
        sd = 0.0
    return entropy, mad, sd


def _lossy_quality(orig_bytes: bytes, decomp_bytes: bytes
                    ) -> tuple[float, float, float]:
    """PSNR, global SSIM, pointwise-max-error on float32 reinterpretation."""
    # Truncate to the shorter of the two if a lossy codec changed sample count.
    n = min(len(orig_bytes), len(decomp_bytes)) // 4
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    orig = np.frombuffer(orig_bytes,    dtype=np.float32, count=n).astype(np.float64)
    rec  = np.frombuffer(decomp_bytes,  dtype=np.float32, count=n).astype(np.float64)

    diff = orig - rec
    mse = float((diff * diff).mean())
    rng = float(orig.max() - orig.min())
    if mse <= 0.0 or rng <= 0.0 or not math.isfinite(rng):
        psnr = float("inf") if mse == 0.0 else float("nan")
    else:
        psnr = 10.0 * math.log10((rng * rng) / mse)

    mu_x, mu_y = orig.mean(), rec.mean()
    sx, sy = orig.std(), rec.std()
    sxy = float(((orig - mu_x) * (rec - mu_y)).mean())
    L = rng if rng > 0 else 1.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx ** 2 + sy ** 2 + c2)
    ssim = float(num / den) if den > 0 else float("nan")

    point_max = float(np.abs(diff).max())
    return psnr, ssim, point_max


# ---------- Worker -------------------------------------------------------

def _bench_features_only(path: str, offset: int, chunk_size: int) -> dict | None:
    """Read chunk, compute features only. No compression."""
    try:
        t0 = time.perf_counter_ns()
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(chunk_size)
        io_time = time.perf_counter_ns() - t0
    except OSError:
        return None
    if len(chunk) != chunk_size:
        return None
    cpu0 = time.thread_time_ns()
    entropy, mad, sd2 = _chunk_features(chunk)
    feat_cpu = time.thread_time_ns() - cpu0
    return {
        "compression_method": "",
        "compression_level": "",
        "error_bound": "",
        "io_time_ns": io_time,
        "cpu_time_ns": feat_cpu,        # cost of computing features
        "decompress_time_ns": 0,
        "compression_ratio": "",
        "shannon_entropy": f"{entropy:.6f}",
        "mad": f"{mad:.6f}",
        "second_derivative": f"{sd2:.6f}",
        "psnr": "",
        "ssim": "",
        "pointwise_max_error": "",
    }


def _bench_one(path: str, offset: int, chunk_size: int,
                comp_spec: tuple) -> dict | None:
    """Read chunk + run ONE compressor. Page cache makes repeated reads
    of the same (path, offset, chunk_size) cheap, so this expanded
    work-unit granularity costs little I/O but maximizes pool parallelism.
    """
    try:
        t0 = time.perf_counter_ns()
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(chunk_size)
        io_time = time.perf_counter_ns() - t0
    except OSError:
        return None
    if len(chunk) != chunk_size:
        return None

    entropy, mad, sd2 = _chunk_features(chunk)

    method, level, cfn, dfn, err, is_lossy = comp_spec
    try:
        cpu0 = time.thread_time_ns()
        comp = cfn(chunk)
        comp_t = time.thread_time_ns() - cpu0
        dcpu0 = time.thread_time_ns()
        decomp = dfn(comp)
        decomp_t = time.thread_time_ns() - dcpu0
    except Exception:
        print(f"[warn] {method}/{level} failed on {path}@{offset}+{chunk_size}: "
              f"{traceback.format_exc().splitlines()[-1]}",
              file=sys.stderr, flush=True)
        return None

    ratio = len(chunk) / max(1, len(comp))
    if is_lossy:
        psnr, ssim, pmax = _lossy_quality(chunk, decomp)
        psnr_s = "" if not math.isfinite(psnr) else f"{psnr:.4f}"
        ssim_s = "" if not math.isfinite(ssim) else f"{ssim:.6f}"
        pmax_s = "" if not math.isfinite(pmax) else f"{pmax:.6e}"
    else:
        psnr_s = ssim_s = pmax_s = ""

    return {
        "compression_method":  method,
        "compression_level":   level,
        "error_bound":         "" if err is None else f"{err:.6e}",
        "io_time_ns":          io_time,
        "cpu_time_ns":         comp_t,
        "decompress_time_ns":  decomp_t,
        "compression_ratio":   f"{ratio:.6f}",
        "shannon_entropy":     f"{entropy:.6f}",
        "mad":                 f"{mad:.6f}",
        "second_derivative":   f"{sd2:.6f}",
        "psnr":                psnr_s,
        "ssim":                ssim_s,
        "pointwise_max_error": pmax_s,
    }


def _enumerate_units(input_dir: Path, chunk_sizes: list[int],
                      max_chunks_per_file: int, rng: random.Random
                     ) -> list[tuple[str, int, int]]:
    units = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        if sz < min(chunk_sizes):
            continue
        for chunk in chunk_sizes:
            n = sz // chunk
            if n <= 0:
                continue
            offsets = list(range(0, n * chunk, chunk))
            if len(offsets) > max_chunks_per_file:
                offsets = rng.sample(offsets, max_chunks_per_file)
            for off in offsets:
                units.append((str(path), off, chunk))
    return units


# ---------- Main ---------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workflow-id", required=True)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-csv", required=True, type=Path,
                   help="Each rank writes to {output_csv}.rank{R}; the "
                   "launcher concatenates after all ranks finish.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--max-chunks-per-file", type=int, default=200,
                   help="Per-(file, chunk_size) sample cap. 0 = unlimited.")
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=CHUNK_SIZES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int,
                   default=int(os.environ.get("SLURM_PROCID", "0")),
                   help="Multi-node partition rank (default from SLURM_PROCID)")
    p.add_argument("--world-size", type=int,
                   default=int(os.environ.get("SLURM_NPROCS", "1")),
                   help="Multi-node partition size (default from SLURM_NPROCS)")
    p.add_argument("--features-only", action="store_true",
                   help="Skip compression. Read every chunk, compute "
                   "entropy/MAD/2nd-derivative, emit one row per chunk. "
                   "Use this to verify NFS read throughput before "
                   "running the compression sweep.")
    args = p.parse_args()

    if not args.input_dir.is_dir():
        print(f"FATAL: {args.input_dir} is not a directory", file=sys.stderr)
        return 2
    if not COMPRESSORS:
        print("FATAL: no compressors available", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    max_per = args.max_chunks_per_file if args.max_chunks_per_file > 0 else 10**9
    chunk_units = _enumerate_units(args.input_dir, args.chunk_sizes, max_per, rng)
    libs = sorted({c[0] for c in COMPRESSORS})

    if args.features_only:
        # One task per chunk; no compression. Used to validate NFS read
        # throughput at scale before running the full sweep.
        all_units = [(p, o, c, None) for (p, o, c) in chunk_units]
    else:
        # Expand to per-compressor tasks so the thread pool sees fine-grain
        # work and slow compressors don't block faster ones inside the same
        # chunk.
        all_units = [(path, off, chunk, spec)
                      for (path, off, chunk) in chunk_units
                      for spec in COMPRESSORS]
    units = all_units[args.rank::args.world_size]

    # Each rank writes its own CSV; the launcher concatenates after.
    if args.world_size > 1:
        out_path = args.output_csv.with_name(
            f"{args.output_csv.stem}.rank{args.rank:03d}{args.output_csv.suffix}")
    else:
        out_path = args.output_csv

    if args.rank == 0:
        if args.features_only:
            print(f"[{args.workflow_id}] features-only: {len(chunk_units)} "
                  f"chunks = {len(all_units)} total tasks", flush=True)
        else:
            print(f"[{args.workflow_id}] {len(chunk_units)} chunks × "
                  f"{len(COMPRESSORS)} variants = {len(all_units)} total tasks",
                  flush=True)
            print(f"[{args.workflow_id}] compressors registered: {libs}",
                  flush=True)
        print(f"[{args.workflow_id}] world_size={args.world_size}  "
              f"workers/rank={args.workers}  "
              f"total cores={args.world_size * args.workers}", flush=True)
    print(f"[{args.workflow_id}] rank {args.rank}/{args.world_size}: "
          f"{len(units)} tasks -> {out_path.name}",
          flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workflow_id", "file",
        "compression_method", "compression_level",
        "chunk_size", "error_bound",
        "io_time_ns", "cpu_time_ns", "decompress_time_ns",
        "compression_ratio",
        "shannon_entropy", "mad", "second_derivative",
        "psnr", "ssim", "pointwise_max_error",
    ]

    t_start_wall = time.monotonic()
    t_start_cpu = time.process_time()
    completed = 0
    total_cpu_ns = 0  # sum of compress + decompress CPU time across all rows
    with open(out_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            if args.features_only:
                futures = {
                    ex.submit(_bench_features_only, path, off, chunk):
                        (path, off, chunk, "features", "")
                    for path, off, chunk, _ in units
                }
            else:
                futures = {
                    ex.submit(_bench_one, path, off, chunk, spec):
                        (path, off, chunk, spec[0], spec[1])
                    for path, off, chunk, spec in units
                }
            for fut in as_completed(futures):
                path, off, chunk, m, l = futures[fut]
                try:
                    r = fut.result()
                except Exception:
                    print(f"[error] worker crashed on {path}@{off}+{chunk} {m}/{l}: "
                          f"{traceback.format_exc().splitlines()[-1]}",
                          file=sys.stderr, flush=True)
                    continue
                if r is None:
                    continue
                r["workflow_id"] = args.workflow_id
                r["file"] = os.path.relpath(path, args.input_dir)
                r["chunk_size"] = chunk
                writer.writerow(r)
                total_cpu_ns += int(r["cpu_time_ns"]) + int(r["decompress_time_ns"])
                completed += 1
                if completed % 1000 == 0:
                    csvf.flush()
                    wall = time.monotonic() - t_start_wall
                    cpu_sum_s = total_cpu_ns / 1e9
                    print(f"[{args.workflow_id} r{args.rank}] "
                          f"{completed}/{len(units)} rows  "
                          f"wall={wall:.1f}s  cpu={cpu_sum_s:.1f}s  "
                          f"({completed/wall:.1f} rows/s)",
                          flush=True)

    wall = time.monotonic() - t_start_wall
    cpu_sum_s = total_cpu_ns / 1e9
    par = cpu_sum_s / wall if wall > 0 else 0.0
    print(f"[{args.workflow_id} r{args.rank}] done: {completed} rows  "
          f"wall={wall:.1f}s  cpu_sum={cpu_sum_s:.1f}s  "
          f"parallelism={par:.2f}×  -> {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

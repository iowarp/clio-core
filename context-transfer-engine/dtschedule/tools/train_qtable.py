#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# All rights reserved.
#
# This file is part of IOWarp Core.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Train the dtschedule Q-table compression-characteristic model (``ccm: qtable``).

Pipeline
--------
1. **Corpus**: ``dt_datagen`` output for every data class at several
   noise levels, plus any real workflow output directories passed with
   ``--real-dir`` (e.g. the montage / seismology dry runs).
2. **Collect**: every chunk (64 KiB .. 1 MiB) of every corpus file is run
   through every available (library, preset) pair.  Presets mirror the C++
   ``LosslessMode`` / ``CompressionMode`` levels in
   ``clio_ctp/compress/lossless_modes.h`` and ``libpressio_modes.h``, not
   the levels of the generic ``qtable_data_collect.py`` sweep.  The chunk
   features (bytewise Shannon entropy, MAD, mean |second derivative|) are
   the collector's, which match ``data_stats.h::CalculateAllStatistics``.
3. **Bin**: quantile edges exactly like ``QTablePredictor::BuildBinningEdges``
   (``values[floor(i/n * N)]``, duplicates removed) and lookups exactly like
   ``DiscretizeFeatures`` (``lower_bound`` on float32 edges, clamped to
   ``n_bins - 1``).  15 bins for size and entropy, 10 for MAD and the
   second derivative.
4. **Table**: per-state running means of compression ratio, PSNR,
   compress time (ms) and decompress time (ms); the state tuple is
   ``[library_id, config_id, datatype_id, size_bin, entropy_bin, mad_bin,
   derivative_bin]`` with ``library_id`` = ``CompressionFactory`` base id,
   ``config_id`` = ``CompressPreset`` (kFast=1, kBalanced=2, kBest=3) and
   ``datatype_id`` = 0 char / 1 float / 2 int.
5. **Evaluate**: held-out R^2 (split by file), unknown-state rate, and an
   inference sanity check through a Python re-implementation of
   ``QTablePredictor::Load``.

Outputs ``<model-dir>/qtable.json`` and ``<model-dir>/binning_params.json``
in the layout ``QTablePredictor::Load`` parses, plus ``train_report.json``.

Usage
-----
  train_qtable.py --scratch $HOME/dtschedule-scratch/qtable \\
      --real-dir montage=$HOME/dtschedule-scratch/montage_dry \\
      --real-dir seis=$HOME/dtschedule-scratch/seis_dry \\
      --model-dir context-transfer-engine/dtschedule/models/qtable_v1
"""
from __future__ import annotations

import argparse
import bz2
import csv
import datetime as _dt
import hashlib
import json
import lzma
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CTE = _HERE.parent.parent
sys.path.insert(0, str(_CTE / "compressor" / "generator"))
sys.path.insert(0, str(_HERE / "wfcommons"))
import qtable_data_collect as qdc  # noqa: E402
import dt_datagen as dg  # noqa: E402

# ---------- Encoding tables (single source of truth: compress_factory.h) --

LIBRARY_BASE_ID = {
    "brotli": 6, "bzip2": 1, "blosc2": 8, "fpzip": 12, "lz4": 3, "lzma": 5,
    "snappy": 7, "sz3": 11, "zfp": 10, "zlib": 4, "zstd": 2,
}
"""``CompressorInfo::base_id`` per library (the ML id scheme)."""

SINGLE_MODE = {"snappy", "blosc2"}
"""Libraries whose ``GetLibraryId`` always encodes preset 2."""

PRESET_ID = {"fast": 1, "balanced": 2, "best": 3}
"""``CompressPreset`` numbering (compression_header.h)."""

DTYPE_ID = {"char": 0, "float": 1, "int": 2}
"""Datatype id as decoded by ``QTablePredictor::DiscretizeFeatures``."""

CLASS_DTYPE = {"float_field": "float", "text_seq": "char",
               "text_table": "char", "mixed_binary": "char"}
"""dt_datagen data class -> datatype name."""

STATE_FIELDS = ["library_id", "config_id", "datatype_id", "size_bin",
                "entropy_bin", "mad_bin", "derivative_bin"]
FEATURE_SLOTS = {"size": 3, "entropy": 4, "mad": 5, "derivative": 6}
"""Index of each continuous feature inside ``bin_edges`` (C++ reads 3/4/5)."""

N_BINS = {"size": 15, "entropy": 15, "mad": 10, "derivative": 10}
NOISE_LEVELS = (0.0, 0.05, 0.2, 0.5)
CHUNK_SIZES = [64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20]
TARGETS = ("compression_ratio", "compression_time_ms",
           "decompression_time_ms")


# ---------- Compressor specs matching the C++ presets --------------------

def build_compressor_specs() -> list:
    """Build (method, preset, compress_fn, decompress_fn, err, lossy) specs.

    Levels mirror ``lossless_modes.h`` (zlib 1/6/9, bzip2 1/6/9,
    zstd 1/3/19, lz4 default/HC6/HC12, lzma 0/6/9, brotli 1/6/11), the
    blosc2 default context, and ``libpressio_modes.h`` for zfp
    (rate 8 / rate 16 / accuracy 1e-3).  Libraries whose Python binding is
    missing are skipped and reported by the caller.

    :return: List of spec tuples accepted by ``qtable_data_collect._bench_one``.
    """
    specs = []
    for p, lvl in (("fast", 1), ("balanced", 6), ("best", 9)):
        specs.append(("zlib", p, (lambda c=lvl: lambda b: zlib.compress(b, c))(),
                      zlib.decompress, None, False))
        specs.append(("bzip2", p, (lambda c=lvl: lambda b: bz2.compress(b, c))(),
                      bz2.decompress, None, False))
    for p, lvl in (("fast", 0), ("balanced", 6), ("best", 9)):
        specs.append(("lzma", p,
                      (lambda c=lvl: lambda b: lzma.compress(b, preset=c))(),
                      lzma.decompress, None, False))
    if "zstd" in qdc._AVAIL:
        zmod = qdc._AVAIL["zstd"]
        for p, lvl in (("fast", 1), ("balanced", 3), ("best", 19)):
            specs.append(("zstd", p,
                          (lambda c=lvl: lambda b: zmod.compress(b, level=c))(),
                          zmod.decompress, None, False))
    if "lz4" in qdc._AVAIL:
        blk = qdc._AVAIL["lz4"]
        specs.append(("lz4", "fast", lambda b: blk.compress(b, mode="default"),
                      blk.decompress, None, False))
        for p, lvl in (("balanced", 6), ("best", 12)):
            specs.append(("lz4", p,
                          (lambda c=lvl: lambda b: blk.compress(
                              b, mode="high_compression", compression=c))(),
                          blk.decompress, None, False))
    if "brotli" in qdc._AVAIL:
        bro = qdc._AVAIL["brotli"]
        for p, lvl in (("fast", 1), ("balanced", 6), ("best", 11)):
            specs.append(("brotli", p,
                          (lambda c=lvl: lambda b: bro.compress(b, quality=c))(),
                          bro.decompress, None, False))
    if "snappy" in qdc._AVAIL:
        sn = qdc._AVAIL["snappy"]
        specs.append(("snappy", "balanced", sn.compress, sn.decompress,
                      None, False))
    if "blosc2" in qdc._AVAIL:
        b2 = qdc._AVAIL["blosc2"]
        specs.append(("blosc2", "balanced",
                      lambda b: b2.compress(b, clevel=5, codec=b2.Codec.BLOSCLZ,
                                            typesize=8),
                      b2.decompress, None, False))
    if "zfp" in qdc._AVAIL:
        specs.extend(_zfp_specs(qdc._AVAIL["zfp"]))
    return specs


def _zfp_specs(zfpy) -> list:
    """Build zfp specs: FAST rate 8 b/value, BALANCED rate 16, BEST acc 1e-3.

    :param zfpy: Imported ``zfpy`` module.
    :return: Three spec tuples (float32 reinterpretation of the chunk).
    """
    def make(kw):
        def _c(b):
            arr = np.frombuffer(b, dtype=np.float32)
            return bytes(zfpy.compress_numpy(arr, **kw))

        def _d(b):
            return zfpy.decompress_numpy(np.frombuffer(b, dtype=np.uint8)).tobytes()
        return _c, _d
    out = []
    for p, kw, err in (("fast", {"rate": 8.0}, 8.0),
                       ("balanced", {"rate": 16.0}, 16.0),
                       ("best", {"tolerance": 1e-3}, 1e-3)):
        c, d = make(kw)
        out.append(("zfp", p, c, d, err, True))
    return out


# ---------- Corpus -------------------------------------------------------

def build_synthetic_corpus(out_dir: Path, size_mb: int, seeds: int) -> Path:
    """Write dt_datagen files for every class x noise level x seed.

    :param out_dir: Directory for the generated files (created).
    :param size_mb: Size of each generated file in MiB.
    :param seeds: Number of seeds per (class, noise) pair.
    :return: ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nbytes = size_mb << 20
    for cls in dg.CLASSES:
        ext = dg.CLASS_EXT[cls] if hasattr(dg, "CLASS_EXT") else _class_ext(cls)
        for noise in NOISE_LEVELS:
            for s in range(seeds):
                name = f"synth_{cls}_n{noise:g}_s{s}{ext}"
                path = out_dir / name
                if path.exists() and path.stat().st_size == nbytes:
                    continue
                seed = (dg.seed_for(name) + 7919 * s) & 0xFFFFFFFF
                with open(path, "wb") as fp:
                    dg.write_file(fp, name, nbytes, seed=seed, data_class=cls,
                                  noise=noise)
    return out_dir


def _class_ext(cls: str) -> str:
    """Return a file extension that ``dt_datagen.classify`` maps to ``cls``.

    :param cls: Data class name.
    :return: Extension including the leading dot.
    """
    for ext, c in dg.EXT_TABLE.items():
        if c == cls:
            return ext
    return ".bin"


def file_dtype(path: str) -> int:
    """Datatype id of a corpus file from its dt_datagen class.

    :param path: File path (extension selects the class).
    :return: 0 for char, 1 for float, 2 for int.
    """
    return DTYPE_ID[CLASS_DTYPE[dg.classify(os.path.basename(path))]]


def enumerate_units(dirs: list, chunk_sizes: list, max_chunks: int,
                    max_files: int, rng: random.Random) -> list:
    """List (workflow, path, offset, chunk_size) work units over the corpus.

    Only files whose extension is in ``dt_datagen.EXT_TABLE`` are used, so
    logs, scripts and JSON manifests of a workflow run are skipped.

    :param dirs: List of (workflow_id, directory) pairs.
    :param chunk_sizes: Chunk sizes in bytes.
    :param max_chunks: Per-(file, chunk size) sample cap.
    :param max_files: Per-directory file cap (random sample).
    :param rng: Seeded random generator.
    :return: List of work units.
    """
    units = []
    for wf, d in dirs:
        files = sorted(p for p in Path(d).rglob("*")
                       if p.is_file() and p.suffix.lower() in dg.EXT_TABLE
                       and p.stat().st_size >= min(chunk_sizes))
        if len(files) > max_files:
            files = rng.sample(files, max_files)
        for path in files:
            sz = path.stat().st_size
            for chunk in chunk_sizes:
                offsets = list(range(0, (sz // chunk) * chunk, chunk))
                if len(offsets) > max_chunks:
                    offsets = rng.sample(offsets, max_chunks)
                units.extend((wf, str(path), off, chunk) for off in offsets)
    return units


# ---------- Collection ---------------------------------------------------

CSV_FIELDS = ["workflow_id", "file", "dtype_id", "compression_method",
              "compression_level", "chunk_size", "error_bound",
              "cpu_time_ns", "decompress_time_ns", "compression_ratio",
              "shannon_entropy", "mad", "second_derivative", "psnr"]


def collect(units: list, specs: list, workers: int, csv_path: Path) -> int:
    """Run every spec on every unit and write the training CSV.

    Lossy specs are only run on float files.  Uses
    ``qtable_data_collect._bench_one`` for the per-row measurement (process
    CPU time of the compress and decompress calls).

    :param units: Work units from :func:`enumerate_units`.
    :param specs: Compressor specs from :func:`build_compressor_specs`.
    :param workers: Thread pool size.
    :param csv_path: Output CSV path.
    :return: Number of rows written.
    """
    tasks = [(wf, path, off, chunk, spec) for wf, path, off, chunk in units
             for spec in specs if not spec[5] or file_dtype(path) == 1]
    print(f"[collect] {len(units)} chunks x {len(specs)} specs -> "
          f"{len(tasks)} tasks on {workers} workers", flush=True)
    t0 = time.monotonic()
    rows = 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(qdc._bench_one, path, off, chunk, spec):
                    (wf, path, chunk) for wf, path, off, chunk, spec in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                if r is None:
                    continue
                wf, path, chunk = futs[fut]
                r.update(workflow_id=wf, file=path, chunk_size=chunk,
                         dtype_id=file_dtype(path))
                w.writerow(r)
                rows += 1
                if rows % 5000 == 0:
                    print(f"[collect] {rows}/{len(tasks)} rows "
                          f"({time.monotonic() - t0:.0f} s)", flush=True)
    print(f"[collect] done: {rows} rows in {time.monotonic() - t0:.0f} s",
          flush=True)
    return rows


def load_rows(csv_path: Path) -> list:
    """Load the training CSV into a list of dict rows with numeric fields.

    :param csv_path: CSV written by :func:`collect`.
    :return: Rows with ``library_id``, ``config_id`` and ms timings added.
    """
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            lib = r["compression_method"]
            preset = 2 if lib in SINGLE_MODE else PRESET_ID[r["compression_level"]]
            psnr = float(r["psnr"]) if r["psnr"] else 0.0
            rows.append({
                "file": r["file"], "workflow_id": r["workflow_id"],
                "library": lib, "library_id": LIBRARY_BASE_ID[lib],
                "config_id": preset, "datatype_id": int(r["dtype_id"]),
                "size": float(r["chunk_size"]),
                "entropy": float(r["shannon_entropy"]),
                "mad": float(r["mad"]),
                "derivative": float(r["second_derivative"]),
                "compression_ratio": float(r["compression_ratio"]),
                "compression_time_ms": int(r["cpu_time_ns"]) / 1e6,
                "decompression_time_ms": int(r["decompress_time_ns"]) / 1e6,
                "psnr_db": psnr,
            })
    return rows


# ---------- Binning (mirrors QTablePredictor) ----------------------------

def quantile_edges(values: np.ndarray, n_bins: int) -> list:
    """Percentile bin edges exactly as ``QTablePredictor::BuildBinningEdges``.

    :param values: Feature values (one per unique chunk).
    :param n_bins: Number of bins; produces at most ``n_bins - 1`` edges.
    :return: Sorted, de-duplicated float32 edges as Python floats.
    """
    v = np.sort(values.astype(np.float32))
    edges = []
    for i in range(1, n_bins):
        pos = min(int(i / n_bins * v.size), v.size - 1)
        edges.append(float(v[pos]))
    out = []
    for e in edges:  # std::unique on adjacent duplicates
        if not out or out[-1] != e:
            out.append(e)
    return out


def discretize(value: float, edges: list, n_bins: int) -> int:
    """Bin index exactly as ``QTablePredictor::DiscretizeFeatures``.

    ``std::lower_bound`` == number of edges strictly less than the value.

    :param value: Feature value.
    :param edges: Edges from :func:`quantile_edges`.
    :param n_bins: Clamp bound (the C++ uses the global ``n_bins``).
    :return: Bin index in ``[0, n_bins - 1]``.
    """
    if not edges:
        return 0
    idx = int(np.searchsorted(np.asarray(edges, dtype=np.float32),
                              np.float32(value), side="left"))
    return max(0, min(idx, n_bins - 1))


def fit_edges(rows: list) -> dict:
    """Fit edges per continuous feature on the unique chunks of ``rows``.

    :param rows: Training rows.
    :return: ``{"size": [...], "entropy": [...], "mad": [...], "derivative": [...]}``.
    """
    seen = {}
    for r in rows:
        seen[(r["file"], r["size"], r["entropy"], r["mad"])] = r
    chunks = list(seen.values())
    return {f: quantile_edges(np.array([c[f] for c in chunks]), N_BINS[f])
            for f in N_BINS}


def state_of(r: dict, edges: dict) -> tuple:
    """State tuple of a row under the given edges (C++ clamp uses n_bins=15).

    :param r: Row from :func:`load_rows`.
    :param edges: Edges from :func:`fit_edges`.
    :return: 7-tuple of ints in ``STATE_FIELDS`` order.
    """
    clamp = N_BINS["size"]
    return (r["library_id"], r["config_id"], r["datatype_id"],
            discretize(r["size"], edges["size"], clamp),
            discretize(r["entropy"], edges["entropy"], clamp),
            discretize(r["mad"], edges["mad"], clamp),
            discretize(r["derivative"], edges["derivative"], clamp))


def build_table(rows: list, edges: dict) -> tuple:
    """Per-state means of ratio, PSNR, compress and decompress time.

    :param rows: Training rows.
    :param edges: Edges from :func:`fit_edges`.
    :return: ``(table, global_average)``; table maps state -> value dict.
    """
    acc = {}
    keys = ("compression_ratio", "psnr_db", "compression_time_ms",
            "decompression_time_ms")
    for r in rows:
        s = state_of(r, edges)
        a = acc.setdefault(s, {k: 0.0 for k in keys} | {"sample_count": 0})
        for k in keys:
            a[k] += r[k]
        a["sample_count"] += 1
    table = {s: {k: a[k] / a["sample_count"] for k in keys}
             | {"sample_count": a["sample_count"]} for s, a in acc.items()}
    n = len(rows)
    glob = {k: sum(r[k] for r in rows) / n for k in keys} | {"sample_count": n}
    return table, glob


# ---------- Evaluation ---------------------------------------------------

def split_by_file(rows: list, test_frac: float, seed: int) -> tuple:
    """Deterministic train/test split with whole files on one side.

    :param rows: All rows.
    :param test_frac: Fraction of files held out.
    :param seed: Hash salt.
    :return: ``(train_rows, test_rows)``.
    """
    def held(path):
        h = hashlib.sha1(f"{seed}:{path}".encode()).digest()
        return int.from_bytes(h[:4], "big") / 2 ** 32 < test_frac
    train = [r for r in rows if not held(r["file"])]
    test = [r for r in rows if held(r["file"])]
    return train, test


def evaluate(table: dict, glob: dict, edges: dict, test: list) -> dict:
    """Held-out R^2 and MAPE per target plus the unknown-state rate.

    :param table: State table from :func:`build_table` (train split).
    :param glob: Global average fallback.
    :param edges: Edges fitted on the train split.
    :param test: Held-out rows.
    :return: Report dict.
    """
    unknown = 0
    pred = {t: [] for t in TARGETS}
    true = {t: [] for t in TARGETS}
    for r in test:
        v = table.get(state_of(r, edges))
        if v is None:
            unknown += 1
            v = glob
        for t in TARGETS:
            pred[t].append(v[t])
            true[t].append(r[t])
    rep = {"n_test": len(test), "unknown_rate": unknown / max(1, len(test))}
    for t in TARGETS:
        y, p = np.array(true[t]), np.array(pred[t])
        ss_res = float(((y - p) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ly, lp = np.log(np.maximum(y, 1e-6)), np.log(np.maximum(p, 1e-6))
        rep[t] = {
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "r2_log": 1.0 - float(((ly - lp) ** 2).sum())
            / max(float(((ly - ly.mean()) ** 2).sum()), 1e-12),
            "mape": float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-9))),
        }
    return rep


# ---------- Model files --------------------------------------------------

def _round(x: float) -> float:
    """Round to 5 significant digits to keep the JSON small.

    :param x: Value.
    :return: Rounded float.
    """
    return float(f"{x:.5g}")


def write_model(model_dir: Path, edges: dict, table: dict, glob: dict,
                meta: dict) -> None:
    """Write ``binning_params.json`` and ``qtable.json`` for the C++ loader.

    ``bin_edges`` has 11 slots (one per ``CompressionFeatures::ToVector``
    index); the C++ discretizer reads slots 3 (size), 4 (entropy) and
    5 (MAD).  Slot 6 carries the derivative edges for the dtschedule
    runtime and this script.

    :param model_dir: Output directory (created).
    :param edges: Edges from :func:`fit_edges`.
    :param table: State table from :func:`build_table`.
    :param glob: Global average value dict.
    :param meta: Provenance block stored in both files.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = [[] for _ in range(11)]
    for f, slot in FEATURE_SLOTS.items():
        # Full precision: edges are exact float32 values and the C++ side
        # compares with lower_bound, so rounding would merge bins.
        bin_edges[slot] = [float(np.float32(e)) for e in edges[f]]
    binning = {"n_bins": N_BINS["size"], "use_nearest_neighbor": False,
               "nn_k": 5, "bin_edges": bin_edges,
               "bins_per_feature": N_BINS, "feature_slots": FEATURE_SLOTS,
               "meta": meta}
    (model_dir / "binning_params.json").write_text(json.dumps(binning, indent=1))
    states = [{"state": list(s), "compression_ratio": _round(v["compression_ratio"]),
               "psnr_db": _round(v["psnr_db"]),
               "compression_time_ms": _round(v["compression_time_ms"]),
               "decompression_time_ms": _round(v["decompression_time_ms"]),
               "sample_count": v["sample_count"]}
              for s, v in sorted(table.items())]
    ga = {k: (_round(v) if isinstance(v, float) else v) for k, v in glob.items()}
    doc = {"num_states": len(states), "state_fields": STATE_FIELDS,
           "encoding": {"library_id": LIBRARY_BASE_ID, "config_id": PRESET_ID,
                        "datatype_id": DTYPE_ID, "single_mode": sorted(SINGLE_MODE)},
           "meta": meta, "global_average": ga, "states": states}
    with open(model_dir / "qtable.json", "w") as fh:
        fh.write(json.dumps(doc, separators=(",", ":")))


class QTableModel:
    """Python mirror of ``QTablePredictor::Load`` + state lookup.

    Parses the two JSON files the way the C++ loader does: ``bin_edges``
    as float32 vectors, each ``state`` copied into an 11-int array that is
    zero-filled past the stored length.
    """

    def __init__(self, model_dir: Path):
        """Load a model directory.

        :param model_dir: Directory with ``qtable.json`` and ``binning_params.json``.
        """
        b = json.loads((model_dir / "binning_params.json").read_text())
        self.n_bins = int(b["n_bins"])
        self.edges = [np.asarray(e, dtype=np.float32) for e in b["bin_edges"]]
        q = json.loads((model_dir / "qtable.json").read_text())
        self.table = {}
        for e in q["states"]:
            bins = [0] * 11
            src = [int(x) for x in e["state"]]
            bins[:min(len(src), 11)] = src[:11]
            self.table[tuple(bins)] = e
        self.glob = q["global_average"]

    def state(self, lib: str, preset: str, dtype: str, size: float,
              entropy: float, mad: float, deriv: float) -> tuple:
        """Build the 11-int state for a (codec, chunk) query.

        :return: 11-tuple state key.
        """
        cfg = 2 if lib in SINGLE_MODE else PRESET_ID[preset]
        bins = [LIBRARY_BASE_ID[lib], cfg, DTYPE_ID[dtype]] + [0] * 8
        for f, v in (("size", size), ("entropy", entropy), ("mad", mad),
                     ("derivative", deriv)):
            bins[FEATURE_SLOTS[f]] = discretize(v, list(self.edges[FEATURE_SLOTS[f]]),
                                                self.n_bins)
        return tuple(bins)

    def predict(self, *args) -> tuple:
        """Look up a state; fall back to the global average.

        :return: ``(value_dict, known)``.
        """
        v = self.table.get(self.state(*args))
        return (v, True) if v is not None else (self.glob, False)


def sanity_check(model_dir: Path, specs: list) -> dict:
    """Inference sanity: decode every state, rank codecs on two fresh chunks.

    :param model_dir: Trained model directory.
    :param specs: Compressor specs (to know which libraries exist).
    :return: Dict with the decoded-state count and the ranking samples.
    """
    m = QTableModel(model_dir)
    out = {"states_loaded": len(m.table)}
    libs = sorted({(s[0], s[1]) for s in specs if not s[5]})
    for cls, noise in (("float_field", 0.05), ("text_table", 0.0),
                       ("mixed_binary", 0.5)):
        name = f"probe_{cls}{_class_ext(cls)}"
        chunk = dg.generate(name, 1 << 20, data_class=cls, noise=noise,
                            offset=3 << 20)
        ent, mad, der = qdc._chunk_features(chunk)
        dtype = CLASS_DTYPE[cls]
        ranked = []
        for lib, preset in libs:
            v, known = m.predict(lib, preset, dtype, float(len(chunk)), ent,
                                 mad, der)
            ranked.append((f"{lib}:{preset}", _round(v["compression_ratio"]),
                           _round(v["compression_time_ms"]), known))
        ranked.sort(key=lambda t: -t[1])
        out[cls] = {"entropy": _round(ent), "mad": _round(mad),
                    "derivative": _round(der),
                    "top3_by_ratio": ranked[:3],
                    "fastest3": sorted(ranked, key=lambda t: t[2])[:3],
                    "unknown": sum(1 for t in ranked if not t[3])}
    t0 = time.perf_counter()
    q = ("zstd", "balanced", "float", 1 << 20, 3.0, 40.0, 50.0)
    for _ in range(20000):
        m.predict(*q)
    out["python_lookup_us"] = _round((time.perf_counter() - t0) / 20000 * 1e6)
    return out


# ---------- Main ---------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", type=Path, required=True,
                   help="scratch dir for the synthetic corpus and CSV")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--real-dir", action="append", default=[],
                   metavar="ID=DIR", help="real workflow output directory")
    p.add_argument("--synth-mb", type=int, default=12,
                   help="size of each synthetic file (MiB)")
    p.add_argument("--synth-seeds", type=int, default=1)
    p.add_argument("--max-chunks-per-file", type=int, default=8)
    p.add_argument("--max-files-per-dir", type=int, default=40)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--reuse-csv", action="store_true",
                   help="skip collection if the CSV already exists")
    p.add_argument("--derivative-bins", type=int, default=N_BINS["derivative"],
                   help="bins for the 2nd-derivative feature; 0 drops it "
                        "from the state (bin always 0)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run corpus build, collection, binning, evaluation and model export.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Exit code.
    """
    a = parse_args(argv)
    N_BINS["derivative"] = max(0, a.derivative_bins)
    rng = random.Random(a.seed)
    specs = build_compressor_specs()
    libs = sorted({s[0] for s in specs})
    missing = sorted(set(LIBRARY_BASE_ID) - set(libs))
    print(f"[specs] {len(specs)} (library, preset) pairs: {libs}; "
          f"skipped (no Python binding): {missing}", flush=True)
    csv_path = a.scratch / "train_rows.csv"
    if not (a.reuse_csv and csv_path.exists()):
        dirs = [("synthetic", build_synthetic_corpus(a.scratch / "corpus",
                                                     a.synth_mb, a.synth_seeds))]
        dirs += [tuple(x.split("=", 1)) for x in a.real_dir]
        units = enumerate_units(dirs, CHUNK_SIZES, a.max_chunks_per_file,
                                a.max_files_per_dir, rng)
        collect(units, specs, a.workers, csv_path)
    rows = load_rows(csv_path)
    train, test = split_by_file(rows, a.test_frac, a.seed)
    edges_tr = fit_edges(train)
    table_tr, glob_tr = build_table(train, edges_tr)
    report = evaluate(table_tr, glob_tr, edges_tr, test)
    report["n_train"] = len(train)
    report["train_states"] = len(table_tr)
    # Ablation: same split without the derivative bin (6-tuple state).
    edges_nd = dict(edges_tr, derivative=[])
    table_nd, glob_nd = build_table(train, edges_nd)
    report["ablation_no_derivative"] = evaluate(table_nd, glob_nd, edges_nd, test)
    edges = fit_edges(rows)
    table, glob = build_table(rows, edges)
    meta = {"trained": _dt.datetime.now().isoformat(timespec="seconds"),
            "host": os.uname().nodename, "n_rows": len(rows),
            "n_chunks": len({(r["file"], r["size"], r["entropy"]) for r in rows}),
            "n_files": len({r["file"] for r in rows}),
            "libraries": libs, "skipped_libraries": missing,
            "chunk_sizes": CHUNK_SIZES, "noise_levels": list(NOISE_LEVELS),
            "command": " ".join(sys.argv), "heldout": report}
    write_model(a.model_dir, edges, table, glob, meta)
    report["sanity"] = sanity_check(a.model_dir, specs)
    report["num_states"] = len(table)
    report["model_bytes"] = sum(f.stat().st_size for f in a.model_dir.glob("*.json"))
    (a.model_dir / "train_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

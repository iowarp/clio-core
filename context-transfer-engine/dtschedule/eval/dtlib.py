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
Shared loaders, statistics and figure style for the dtschedule evaluation.

Inputs (all measured, never hand-entered):

* **Decision trace** -- ``<trace_path>.<container>.csv`` written by the
  ``clio::cte::dtschedule`` chimod (DESIGN.md section 9).  One row per
  PutBlob decision; GetBlob appends *decompress* rows that carry only
  ``obs_dtime_ms``.  :func:`load_traces` returns both kinds with a
  ``kind`` column (``decision`` / ``decompress``).
* **Candidate file** -- ``<trace_path>.cand.<node>.csv``
  (``trace_candidates: true``): one row per candidate (lib, preset) per
  decision with its predictions, ranking cost and the ``reason`` it was
  kept or filtered.  Scenario and tier are decision-level (in the trace).
  :func:`load_candidates`.
* **Sweep results** -- ``${HOME}/jarvis-runs/dtschedule-results/<exp>/<run>.json``
  written by ``pipelines/ares/dtschedule/sweep.py``: the run's knob
  overrides, every package's ``_get_stat`` output and the wfcommons
  ``makespan.json``.  :func:`load_results` flattens one row per run.

Figure style follows the ``dataviz`` skill: fixed-order categorical palette
(never cycled), one-hue sequential ramp, blue/gray/red diverging ramp,
hairline recessive axes, 8 pt fonts at single-column width (3.3 in).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Trace layout (DESIGN.md section 9)
# ----------------------------------------------------------------------

#: DESIGN section 9 columns (26).  The chimod appends ``obs_dtime_ms`` and
#: ``select_ms`` (28 columns): decision rows fill ``select_ms`` (SelectCodec
#: wall time) and leave ``obs_dtime_ms`` empty; decompress rows fill
#: ``obs_dtime_ms`` and leave the decision fields empty.  Puts stored raw are
#: written too (``chosen_lib == "raw"``, ``obs_ratio == 1``); ratios are
#: original / compressed (>= 1).
TRACE_COLUMNS = [
    "ts_ms", "node", "tag", "blob", "size", "dtype", "entropy", "mad", "d2",
    "producer_cpu", "consumer_node", "consumer_cpu", "owner_node",
    "n_candidates", "chosen_lib", "chosen_preset", "chosen_scenario",
    "chosen_tier", "pred_ctime_ms", "pred_dtime_ms", "pred_ratio",
    "obs_ctime_ms", "obs_ratio", "store_ms", "forced", "knobs_hash",
]
TRACE_COLUMNS_FULL = TRACE_COLUMNS + ["obs_dtime_ms", "select_ms"]

#: Numeric trace columns (everything else stays a string).
TRACE_NUMERIC = [
    "ts_ms", "size", "dtype", "entropy", "mad", "d2", "producer_cpu",
    "consumer_cpu", "n_candidates", "chosen_preset", "chosen_scenario",
    "pred_ctime_ms", "pred_dtime_ms", "pred_ratio", "obs_ctime_ms",
    "obs_ratio", "store_ms", "obs_dtime_ms", "select_ms",
]

#: Candidate file layout (``<trace_path>.cand.<node>.csv``): one row per
#: candidate per decision.  ``cost_ms`` is the ranker's cost at the
#: decision's placement (scenario / tier are decision-level, see the trace)
#: and is empty unless ``reason == "ok"``.
CANDIDATE_COLUMNS = [
    "ts_ms", "node", "tag", "blob", "lib", "preset",
    "pred_ctime_ms", "pred_dtime_ms", "pred_ratio", "cost_ms", "reason",
]

CANDIDATE_NUMERIC = ["ts_ms", "preset", "pred_ctime_ms", "pred_dtime_ms",
                     "pred_ratio", "cost_ms"]

#: ``reason`` values the chimod writes; only ``ok`` rows were ranked.
CANDIDATE_REASONS = ["ok", "qos_lossy_not_allowed", "qos_error_bound",
                     "qos_preference", "unavailable", "fixed", "skip_ratio"]

#: ``chosen_lib`` values that mean "stored raw".
NOT_COMPRESSED = {"", "none", "raw", "skip", "nan"}

#: Jarvis package id of the dtschedule package (prefix of its stats keys).
DT_PKG = "dtschedule"


def _as_float(value) -> float:
    """Parse a CSV cell as float, NaN when empty or malformed.

    :param value: Cell text (or None).
    :return: Float value or ``numpy.nan``.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _read_trace_csv(path: str) -> pd.DataFrame:
    """Read one decision-trace CSV, tolerating older decompress-row layouts.

    Primary layout: :data:`TRACE_COLUMNS_FULL` (28 columns, named
    ``obs_dtime_ms`` and ``select_ms``).  Older writers put the decompression
    time in a trailing 27th field, or (the phase-2 runtime) in the
    ``knobs_hash`` slot of a row whose decision fields are empty; both are
    still accepted, in that order of precedence.

    :param path: CSV path.
    :return: DataFrame with the header's columns plus ``obs_dtime_ms``.
    """
    with open(path, newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if not header or header[0] != "ts_ms":
            return pd.DataFrame(columns=TRACE_COLUMNS_FULL)
        rows = []
        for row in reader:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            record = dict(zip(header, row[: len(header)]))
            dtime = _as_float(record.get("obs_dtime_ms"))
            if np.isnan(dtime) and len(row) > len(header):
                dtime = _as_float(row[len(header)])
            if (np.isnan(dtime) and record.get("chosen_preset", "") == ""
                    and record.get("n_candidates", "") == ""):
                dtime = _as_float(record.get("knobs_hash"))
                if not np.isnan(dtime):
                    record["knobs_hash"] = ""
            record["obs_dtime_ms"] = dtime
            rows.append(record)
    return pd.DataFrame(rows)


def load_traces(pattern: str | Iterable[str]) -> pd.DataFrame:
    """Load decision traces matching a glob into one DataFrame.

    Decision rows have ``kind == "decision"``; decompress rows have
    ``kind == "decompress"``, ``obs_dtime_ms`` set and empty decision
    fields.  Adds ``src_file`` and ``t_s`` (seconds since the earliest row).

    :param pattern: Glob (or list of globs / paths) for ``*.csv`` traces;
        candidate files (``*.cand.*.csv`` / ``*.cand.csv``) are skipped.
    :return: DataFrame with :data:`TRACE_COLUMNS_FULL`, ``kind``,
        ``src_file``, ``t_s``.  Empty frame when nothing matches.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    paths = sorted(p for pat in patterns for p in glob.glob(os.path.expanduser(pat))
                   if not is_candidate_file(p))
    frames = []
    for path in paths:
        frame = _read_trace_csv(path)
        frame["src_file"] = path
        frames.append(frame)
    if not frames:
        df = pd.DataFrame(columns=TRACE_COLUMNS_FULL + ["src_file"])
    else:
        df = pd.concat(frames, ignore_index=True)
    return _finish_trace(df)


def _finish_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce trace column types and derive ``kind`` / ``t_s``.

    :param df: Raw trace rows (strings).
    :return: Typed DataFrame.
    """
    for col in TRACE_COLUMNS_FULL:
        if col not in df.columns:
            df[col] = ""
    for col in TRACE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    is_decomp = df["obs_dtime_ms"].notna() & df["n_candidates"].isna()
    df["kind"] = np.where(is_decomp, "decompress", "decision")
    df["chosen_lib"] = df["chosen_lib"].fillna("").astype(str).str.strip().str.lower()
    df["chosen_tier"] = df["chosen_tier"].fillna("").astype(str).str.strip().str.lower()
    t0 = df["ts_ms"].min() if len(df) else 0.0
    df["t_s"] = (df["ts_ms"] - t0) / 1000.0
    return df


def is_candidate_file(path: str) -> bool:
    """Whether a path is a candidate file (``*.cand.<node>.csv`` / ``*.cand.csv``).

    :param path: File path.
    :return: True for candidate files.
    """
    name = os.path.basename(path)
    return ".cand." in name or name.endswith(".cand.csv")


def load_candidates(pattern: str | Iterable[str]) -> pd.DataFrame:
    """Load candidate files (``<trace_path>.cand.<node>.csv``) into one frame.

    A ``reason`` column missing from an older writer is taken as ``ok``;
    ``cost_ms`` missing on an ``ok`` row falls back to
    ``pred_ctime_ms + pred_dtime_ms``.  Rows with any other reason keep
    ``cost_ms`` NaN; use :func:`ok_candidates` to select the ranked ones.

    :param pattern: Glob (or list) for candidate CSVs; a glob that also
        matches decision traces is filtered to candidate files.
    :return: DataFrame with :data:`CANDIDATE_COLUMNS` plus ``src_file``.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    paths = sorted(p for pat in patterns for p in glob.glob(os.path.expanduser(pat))
                   if is_candidate_file(p))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["src_file"] = path
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in CANDIDATE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in CANDIDATE_NUMERIC else ""
    for col in CANDIDATE_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["lib"] = df["lib"].fillna("").astype(str).str.strip().str.lower()
    df["reason"] = df["reason"].fillna("").astype(str).str.strip().str.lower()
    df.loc[df["reason"] == "", "reason"] = "ok"
    missing_cost = (df["reason"] == "ok") & df["cost_ms"].isna()
    df.loc[missing_cost, "cost_ms"] = (
        df.loc[missing_cost, "pred_ctime_ms"].fillna(0)
        + df.loc[missing_cost, "pred_dtime_ms"].fillna(0))
    return df


def ok_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    """Keep only the candidates the ranker actually scored (``reason == ok``).

    :param cand: Output of :func:`load_candidates`.
    :return: Filtered copy.
    """
    return cand[cand["reason"] == "ok"].copy()


# ----------------------------------------------------------------------
# Sweep results
# ----------------------------------------------------------------------

_REPEAT_RE = re.compile(r"[._-]r(\d+)$")


def _flatten_result(path: str, exp_dir: str) -> Dict[str, object]:
    """Flatten one ``sweep.py`` run JSON into a row dict.

    :param path: Path of ``<run>.json``.
    :param exp_dir: Experiment root (sub-directory name becomes ``cell``).
    :return: Row dict: run metadata, knobs (override keys with the package
        prefix stripped), every stat under its full key, ``makespan_ms``,
        ``level<i>_ms``, ``n_nodes`` and ``cell``.
    """
    with open(path) as fp:
        doc = json.load(fp)
    row: Dict[str, object] = {
        "run": doc.get("run", Path(path).stem),
        "exp": doc.get("exp", Path(exp_dir).name),
        "status": doc.get("status"),
        "rc": doc.get("rc"),
        "wall_s": doc.get("wall_s"),
        "json_path": path,
    }
    rel = os.path.relpath(os.path.dirname(path), exp_dir)
    row["cell"] = "" if rel in (".", "") else rel.replace(os.sep, "/")
    match = _REPEAT_RE.search(str(row["run"]))
    row["repeat"] = int(match.group(1)) if match else 0
    for key, value in (doc.get("overrides") or {}).items():
        knob = key.split(".", 1)[1] if "." in key else key
        row[knob] = value if not isinstance(value, (list, dict)) else json.dumps(value)
    for key, value in (doc.get("stats") or {}).items():
        row[key] = value
        if key.startswith(DT_PKG + "."):
            knob = key[len(DT_PKG) + 1:]
            if "." not in knob and knob not in row:
                row[knob] = value
    mk = doc.get("makespan") or {}
    row["makespan_ms"] = mk.get("total_ms", row.get("wfcommons.makespan_ms"))
    row["makespan_status"] = mk.get("status")
    row["n_nodes"] = len(mk.get("nodes") or []) or None
    for level in mk.get("levels") or []:
        row[f"level{level['level']}_ms"] = level.get("wall_ms")
    return row


def load_results(exp_dir: str, trace_subdir: str = "traces") -> pd.DataFrame:
    """Load every run JSON under an experiment results directory.

    Walks ``exp_dir`` recursively, so an experiment collected as one
    ``sweep.py --exp`` per grid cell (E5) can be loaded from the parent
    directory with the sub-directory name in the ``cell`` column.  If
    ``<exp_dir>/<trace_subdir>/<run>/`` exists its ``*.csv`` glob is put in
    ``trace_glob`` (copy the per-run traces there after each run; the
    runtime resets them at every configure).

    :param exp_dir: ``${HOME}/jarvis-runs/dtschedule-results/<exp>``.
    :param trace_subdir: Name of the per-run trace directory.
    :return: One row per run; numeric columns coerced where possible.
    """
    exp_dir = os.path.expanduser(exp_dir)
    rows = []
    for root, dirs, files in os.walk(exp_dir):
        dirs[:] = [d for d in dirs if d not in (trace_subdir, "yaml")]
        for name in sorted(files):
            if name.endswith(".json"):
                rows.append(_flatten_result(os.path.join(root, name), exp_dir))
    if not rows:
        return pd.DataFrame(columns=["run", "cell", "repeat", "makespan_ms"])
    df = pd.DataFrame(rows)
    keep_text = {"run", "exp", "cell", "status", "json_path", "makespan_status"}
    for col in df.columns:
        if df[col].dtype == object and col not in keep_text:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    df["cell"] = df["cell"].fillna("").astype(str)
    df["run"] = df["run"].astype(str)
    df["trace_glob"] = [
        os.path.join(exp_dir, cell, trace_subdir, run, "*.csv")
        if os.path.isdir(os.path.join(exp_dir, cell, trace_subdir, run))
        else "" for cell, run in zip(df["cell"], df["run"])]
    return df


def stat_prefix(row: pd.Series, prefix: str) -> Dict[str, float]:
    """Collect ``<prefix><name>`` stat columns of a result row into a dict.

    Used for the ``dtschedule.lib.<codec>``, ``dtschedule.scenario.<k>`` and
    ``dtschedule.tier.<name>`` histograms ``_get_stat`` emits.

    :param row: One row of :func:`load_results`.
    :param prefix: Key prefix, e.g. ``"dtschedule.lib."``.
    :return: ``{name: count}`` with NaN / zero entries dropped.
    """
    out = {}
    for key, value in row.items():
        if isinstance(key, str) and key.startswith(prefix):
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(count) and count > 0:
                out[key[len(prefix):]] = count
    return out


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], stat: Callable = np.mean,
                 n_boot: int = 2000, ci: float = 0.95,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval over repeats.

    :param values: Observations (e.g. makespans of the repeats of one knob
        setting).  NaNs are dropped.
    :param stat: Statistic applied to each resample (default mean).
    :param n_boot: Number of resamples.
    :param ci: Interval mass (0.95 -> 2.5th / 97.5th percentile).
    :param seed: RNG seed for reproducible intervals.
    :return: ``(estimate, lo, hi)``; ``lo == hi == estimate`` for a single
        observation and all NaN for none.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    est = float(stat(arr))
    if arr.size == 1:
        return (est, est, est)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    boots = np.apply_along_axis(stat, 1, samples)
    alpha = (1.0 - ci) / 2.0
    return (est, float(np.quantile(boots, alpha)),
            float(np.quantile(boots, 1.0 - alpha)))


def aggregate(df: pd.DataFrame, by: Sequence[str], value: str = "makespan_ms",
              **kwargs) -> pd.DataFrame:
    """Group repeats and attach bootstrap intervals.

    :param df: Result rows.
    :param by: Grouping columns (the knobs that identify one setting).
    :param value: Column to summarise.
    :param kwargs: Forwarded to :func:`bootstrap_ci`.
    :return: One row per group with ``<value>``, ``<value>_lo``,
        ``<value>_hi`` and ``n`` columns.
    """
    records = []
    for keys, grp in df.groupby(list(by), dropna=False, sort=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        est, lo, hi = bootstrap_ci(grp[value].tolist(), **kwargs)
        rec = dict(zip(by, keys))
        rec.update({value: est, f"{value}_lo": lo, f"{value}_hi": hi,
                    "n": int(grp[value].notna().sum())})
        records.append(rec)
    return pd.DataFrame(records)


def normalize_to_baseline(df: pd.DataFrame, value: str, baseline_mask,
                          by: Optional[Sequence[str]] = None,
                          out: Optional[str] = None) -> pd.DataFrame:
    """Divide ``value`` by the baseline's mean, per group.

    :param df: Rows with ``value`` (and the interval columns if present).
    :param value: Column to normalise (e.g. ``makespan_ms``).
    :param baseline_mask: Boolean mask selecting baseline rows.
    :param by: Columns defining independent groups (e.g. ``["cell"]``);
        None normalises the whole frame by one baseline.
    :param out: Output column name (default ``<value>_norm``).
    :return: Copy of ``df`` with ``out`` (and ``out_lo`` / ``out_hi`` when the
        interval columns exist).  Groups without a baseline get NaN.
    """
    out = out or f"{value}_norm"
    df = df.copy()
    if by:
        base = (df[baseline_mask].groupby(list(by))[value].mean()
                .rename("_base").reset_index())
        df = df.merge(base, on=list(by), how="left")
    else:
        df["_base"] = df.loc[baseline_mask, value].mean()
    for suffix in ("", "_lo", "_hi"):
        col = f"{value}{suffix}"
        if col in df.columns:
            df[f"{out}{suffix}"] = df[col] / df["_base"]
    return df.drop(columns="_base")


def size_bin_label(size_bytes: float) -> str:
    """Name the nearest power-of-two size bin (``64 KiB``, ``1 MiB`` ...).

    :param size_bytes: Blob size in bytes.
    :return: Human label; ``"?"`` for NaN / non-positive sizes.
    """
    if not np.isfinite(size_bytes) or size_bytes <= 0:
        return "?"
    exp = int(round(np.log2(size_bytes)))
    if exp >= 30:
        return f"{2 ** (exp - 30)} GiB"
    if exp >= 20:
        return f"{2 ** (exp - 20)} MiB"
    if exp >= 10:
        return f"{2 ** (exp - 10)} KiB"
    return f"{2 ** exp} B"


def timestep_index(df: pd.DataFrame) -> pd.Series:
    """Derive a workflow timestep per trace row.

    Uses the last integer in ``blob`` (Gray-Scott writes ``.../step_<n>`` or
    ADIOS2 ``<var>_<n>``), else the last integer in ``tag``, else the rank of
    the row's first-seen (tag, blob) in time order.

    :param df: Trace rows.
    :return: Integer Series aligned with ``df``.
    """
    pat = re.compile(r"(\d+)(?!.*\d)")

    def last_int(text):
        """Last integer in ``text`` or None.

        :param text: Blob or tag name.
        :return: Integer or None.
        """
        match = pat.search(str(text))
        return int(match.group(1)) if match else None

    steps = df["blob"].map(last_int)
    fallback = df["tag"].map(last_int)
    steps = steps.where(steps.notna(), fallback)
    if steps.isna().any():
        order = (df.sort_values("ts_ms").drop_duplicates(["tag", "blob"])
                 .reset_index()[["tag", "blob"]].reset_index()
                 .rename(columns={"index": "_rank"}))
        ranked = df[["tag", "blob"]].merge(order, on=["tag", "blob"], how="left")
        steps = steps.where(steps.notna(), ranked["_rank"].values)
    return steps.fillna(0).astype(int)


# ----------------------------------------------------------------------
# Figure style (dataviz skill reference palette, light mode)
# ----------------------------------------------------------------------

#: Fixed-order categorical slots: blue, orange, aqua, yellow, magenta,
#: green, violet, red.  Assign by entity, never by rank; never cycle past 8.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#b5b3ab"

#: One-hue sequential ramp (blue 100..700).
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]
#: Ordinal ramp for tiers (starts at step 250 so the lightest clears 2:1).
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
#: Diverging: blue (below) - neutral gray - red (above).
DIVERGING = ["#1c5cab", "#2a78d6", "#86b6ef", "#f0efec", "#f2a19f",
             "#e34948", "#a92a29"]
HATCH = "////"

#: Entity -> slot maps so a codec / scenario keeps its colour in every figure.
CODEC_COLORS = {
    "zstd": CAT[0], "lz4": CAT[1], "sz3": CAT[2], "zfp": CAT[3],
    "zlib": CAT[4], "bzip2": CAT[5], "fpzip": CAT[6], "brotli": CAT[7],
    "none": OTHER, "raw": OTHER, "": OTHER,
}
SCENARIO_COLORS = {"1": CAT[0], "2": CAT[1], "3": CAT[2], "auto": CAT[6]}
SCENARIO_LABELS = {"1": "S1 compress@P, store@P",
                   "2": "S2 compress@P, store@C",
                   "3": "S3 raw to C, compress@C", "auto": "auto"}
PREDICTOR_COLORS = {"qtable": CAT[0], "ema": CAT[1], "xgboost": CAT[2],
                    "oracle": CAT[5], "fixed": CAT[6]}
TIER_ORDER = ["ram", "nvme", "ssd", "nfs"]
TIER_COLORS = dict(zip(TIER_ORDER, ORDINAL))

FIG_W = 3.3


def codec_color(lib: str) -> str:
    """Colour for a codec name; unknown codecs fold into the neutral slot.

    :param lib: Codec library name as written in the trace.
    :return: Hex colour.
    """
    return CODEC_COLORS.get(str(lib).strip().lower(), OTHER)


def setup_style() -> None:
    """Apply the single-column paper style (8 pt, hairline chrome).

    Call once before building any figure.  Uses only matplotlib rcParams so
    every script shares the look without a style file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "lines.markeredgewidth": 0.8, "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4, "legend.columnspacing": 0.8,
        "legend.borderaxespad": 0.2,
        "hatch.linewidth": 0.4, "hatch.color": SURFACE,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def figure(rows: int = 1, cols: int = 1, height: float = 2.2,
           width: float = FIG_W, **kwargs):
    """Create a single-column figure with the shared style.

    :param rows: Subplot rows.
    :param cols: Subplot columns.
    :param height: Figure height in inches.
    :param width: Figure width in inches (3.3 in = one column).
    :param kwargs: Forwarded to ``plt.subplots``.
    :return: ``(fig, axes)`` as from ``plt.subplots``.
    """
    import matplotlib.pyplot as plt
    setup_style()
    return plt.subplots(rows, cols, figsize=(width, height), **kwargs)


def bar_kwargs(color: str, hatch: Optional[str] = None) -> Dict[str, object]:
    """Mark spec for a bar: series fill with a surface-colour gap edge.

    :param color: Fill colour.
    :param hatch: Optional hatch pattern (the texture channel).
    :return: Keyword arguments for ``ax.bar`` / ``ax.barh``.
    """
    return {"color": color, "edgecolor": SURFACE, "linewidth": 0.8,
            "hatch": hatch}


LEGEND_ROW_PT = 11.0


def demo_title(ax, text: str, demo: bool, width: int = 58) -> None:
    """Set an axes title, prefixing ``DEMO DATA`` for synthesised input.

    The title sits above any legend placed by :func:`legend_outside` (which
    records its row count on the axes), and is wrapped so it never overruns
    the single-column width.

    :param ax: Axes.
    :param text: Title text.
    :param demo: True when the figure was built from ``--demo`` data.
    :param width: Wrap width in characters.
    """
    import textwrap
    full = ("DEMO DATA: " if demo else "") + text
    lines = textwrap.wrap(full, width)
    rows = getattr(ax, "_dt_legend_rows", 0)
    fig = ax.get_figure()
    axes_pt = ax.get_position().height * fig.get_figheight() * 72.0
    y = 1.0 + (rows * LEGEND_ROW_PT + 4.0) / max(axes_pt, 1.0)
    ax.set_title("\n".join(lines), loc="left", fontsize=7.5,
                 color=CAT[7] if demo else INK, y=y, pad=0)


def savefig(fig, name: str, out_dir: Optional[str] = None) -> List[str]:
    """Write a figure as SVG and PDF into the output directory.

    :param fig: Matplotlib figure.
    :param name: Base file name without extension (e.g. ``e5_heatmap``).
    :param out_dir: Destination; default ``eval/out/`` next to this module.
    :return: Paths written.
    """
    out = Path(os.path.expanduser(out_dir)) if out_dir else Path(__file__).parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "pdf"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", pad_inches=0.02)
        paths.append(str(path))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def demo_out_dir() -> str:
    """Default output directory for ``--demo`` runs (never ``eval/out``).

    :return: ``${HOME}/dtschedule-scratch/eval-demo``.
    """
    return os.path.join(os.environ.get("HOME", "~"), "dtschedule-scratch",
                        "eval-demo")


def add_common_args(parser, results_help: str = "sweep results directory") -> None:
    """Add the ``--results``, ``--out`` and ``--demo`` flags every script has.

    :param parser: ``argparse.ArgumentParser``.
    :param results_help: Help text for ``--results``.
    """
    parser.add_argument("--results", default=None, help=results_help)
    parser.add_argument("--out", default=None,
                        help="figure directory (default eval/out; with "
                             "--demo ${HOME}/dtschedule-scratch/eval-demo)")
    parser.add_argument("--demo", action="store_true",
                        help="synthesise a small input only to exercise the "
                             "script; the figure is titled DEMO DATA")


def resolve_out(args) -> str:
    """Pick the output directory from the parsed arguments.

    :param args: Parsed arguments with ``out`` and ``demo``.
    :return: Directory path (demo runs never default to ``eval/out``).
    """
    if args.out:
        return args.out
    return demo_out_dir() if args.demo else str(Path(__file__).parent / "out")


def legend_outside(ax, ncol: int = 2, **kwargs) -> None:
    """Place a compact legend above the axes, outside the plot area.

    :param ax: Axes.
    :param ncol: Legend columns.
    :param kwargs: Forwarded to ``ax.legend``.
    """
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                       borderaxespad=0.0, **kwargs)
    n_entries = len(legend.get_texts())
    ax._dt_legend_rows = int(np.ceil(n_entries / float(max(ncol, 1))))
    return legend


def print_table(df: pd.DataFrame, title: str) -> None:
    """Print a DataFrame as the figure's companion table.

    :param df: Table to print.
    :param title: Heading line.
    """
    print(f"\n== {title}")
    with pd.option_context("display.width", 160, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))


# ----------------------------------------------------------------------
# Demo-data helpers shared by the scripts
# ----------------------------------------------------------------------

DEMO_CODECS = ["zstd", "lz4", "sz3", "zfp", "zlib"]


def demo_trace(n: int, seed: int = 0, duration_s: float = 600.0,
               codecs: Sequence[str] = DEMO_CODECS,
               scenarios: Sequence[str] = ("1", "2", "3"),
               tiers: Sequence[str] = ("ram", "nvme")) -> pd.DataFrame:
    """Synthesise a decision trace with the real column layout.

    Only for ``--demo``: exercises the loaders and plots, nothing else.

    :param n: Number of decision rows (one decompress row per decision).
    :param seed: RNG seed.
    :param duration_s: Span of the timestamps.
    :param codecs: Codec names to draw from.
    :param scenarios: Scenario ids to draw from.
    :param tiers: Tier names to draw from.
    :return: DataFrame as :func:`load_traces` returns.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, duration_s, n))
    size = rng.choice([65536, 1 << 20, 16 << 20], n)
    entropy = rng.uniform(1.0, 7.5, n)
    lib = rng.choice(list(codecs), n)
    ratio = np.clip(8.0 - entropy + rng.normal(0, 0.5, n), 1.05, 12)
    ctime = size / 1.0e6 * (0.6 + entropy / 8.0) * rng.lognormal(0, 0.15, n)
    rows = pd.DataFrame({
        "ts_ms": 1.7e12 + t * 1000.0, "node": rng.integers(0, 2, n),
        "tag": "tag" + (np.arange(n) % 8).astype(str),
        "blob": ["step_%d" % (i // 4) for i in range(n)],
        "size": size, "dtype": 1, "entropy": entropy,
        "mad": rng.uniform(0, 1, n), "d2": rng.uniform(0, 1, n),
        "producer_cpu": rng.uniform(5, 95, n), "consumer_node": 1,
        "consumer_cpu": rng.uniform(5, 95, n), "owner_node": 0,
        "n_candidates": 12, "chosen_lib": lib,
        "chosen_preset": rng.integers(0, 3, n),
        "chosen_scenario": rng.choice(list(scenarios), n),
        "chosen_tier": rng.choice(list(tiers), n),
        "pred_ctime_ms": ctime * rng.lognormal(0, 0.3, n),
        "pred_dtime_ms": ctime * 0.4 * rng.lognormal(0, 0.3, n),
        "pred_ratio": ratio * rng.lognormal(0, 0.25, n),
        "obs_ctime_ms": ctime, "obs_ratio": ratio,
        "store_ms": size / ratio / 2.0e6 * rng.lognormal(0, 0.2, n),
        "forced": 0, "knobs_hash": "demo", "obs_dtime_ms": np.nan,
        "select_ms": 0.05 * rng.lognormal(0, 0.3, n),
    })
    dec = rows.copy()
    dec["ts_ms"] = dec["ts_ms"] + 2000.0
    dec["obs_dtime_ms"] = ctime * 0.4 * rng.lognormal(0, 0.2, n)
    for col in ("dtype", "entropy", "mad", "d2", "producer_cpu", "consumer_cpu",
                "n_candidates", "chosen_preset", "chosen_scenario",
                "pred_ctime_ms", "pred_dtime_ms", "pred_ratio", "obs_ctime_ms",
                "obs_ratio", "store_ms", "select_ms"):
        dec[col] = np.nan
    dec["chosen_tier"] = ""
    df = pd.concat([rows, dec], ignore_index=True).sort_values("ts_ms")
    df["src_file"] = "demo"
    return _finish_trace(df.reset_index(drop=True))


def demo_results(settings: List[Dict[str, object]], base_ms: float = 300000.0,
                 repeats: int = 3, seed: int = 0,
                 effect: Optional[Callable[[Dict[str, object]], float]] = None,
                 n_nodes: int = 2) -> pd.DataFrame:
    """Synthesise sweep result rows (one per setting x repeat).

    :param settings: Knob dicts (each becomes columns of the rows).
    :param base_ms: Makespan of the reference setting.
    :param repeats: Repeats per setting.
    :param seed: RNG seed.
    :param effect: ``f(setting) -> multiplier`` on the base makespan.
    :param n_nodes: Node count reported in ``n_nodes``.
    :return: DataFrame shaped like :func:`load_results`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, setting in enumerate(settings):
        mult = effect(setting) if effect else 1.0
        for rep in range(repeats):
            row = dict(setting)
            row.update({
                "run": f"run{i}_r{rep}", "repeat": rep, "status": "success",
                "rc": 0, "cell": setting.get("cell", ""),
                "makespan_ms": base_ms * mult * rng.lognormal(0, 0.03),
                "n_nodes": n_nodes, "trace_glob": "",
                f"{DT_PKG}.puts": 400, f"{DT_PKG}.gets": 400,
                f"{DT_PKG}.bytes_in": 400 * (1 << 20),
                f"{DT_PKG}.mean_ratio": 2.5 * rng.lognormal(0, 0.05),
                f"{DT_PKG}.mean_obs_ctime_ms": 3.0 * rng.lognormal(0, 0.1),
                f"{DT_PKG}.mean_obs_dtime_ms": 1.2 * rng.lognormal(0, 0.1),
            })
            rows.append(row)
    return pd.DataFrame(rows)

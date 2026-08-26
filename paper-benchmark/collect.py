#!/usr/bin/env python3
"""Aggregate a paper-benchmark sweep into summary.csv and summary.md.

Shared by every workload under paper-benchmark/. Field names are derived from
the blob names rather than hardcoded, so a workload only has to name its blobs
"<field>/..." (LAMMPS: position/step_50/chunk_3) or use the AMReX dump naming
("plt00007/fab0000_comp00_density/chunk_1"), and the per-field columns follow.

Reads each run directory's meta.json (what was configured) and blobs.csv (what
happened to every chunk) and reports the numbers a paper needs: compression
ratio, bytes stored, how many chunks the codec actually shrank, the codec mix,
per-field ratios, and time.

It also carries the axes run_benchmark.sh sweeps -- mode (lossless/lossy),
cost model (balance/ratio) and the cost model's assumed bandwidth -- and
reports BEST and WORST per-chunk behaviour alongside the totals, because an
average over a run hides exactly the variation a benchmark is looking for.
Runs from several workloads may sit in one directory; the markdown groups by
workload rather than describing them all as the first one.

THROUGHPUT is bytes divided by the CODEC time (a CUDA-event bracket around the
codec call), not by wall clock -- so it measures the compressor, not the
harness around it. Decompression throughput covers only the chunks that HAVE a
measured decompression time: chunks stored raw have nothing to invert, and
nothing measures one at all unless CLIO_NEUROPRESS_EXPLORE_MEASURE_DT is set.
The dt_chunks column says how many chunks are behind the number.

Usage: ./collect.py results/ [-o OUTDIR]
"""
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict


def _f(row, key, default=0.0):
    """Float from a CSV row, tolerating an absent or empty column.

    blobs.csv gained decompress_ms after the first sweeps were run, and the
    WarpX converter emits it only when an exploration log was available, so a
    missing column has to read as "not measured" rather than crash.
    """
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _stats(values):
    """min/median/max, or zeros when nothing was measured."""
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), statistics.median(values), max(values)

def field_of(blob):
    """Physical field a blob belongs to, from its name.

    Two shapes in use. LAMMPS-style "<field>/step_N/chunk_M" puts the field
    first. AMReX-style dumps name the middle segment
    "fab%04d_comp%02d_<field>", so the field is what follows the component
    index -- taking the first segment there would group everything under a
    timestep instead, which is not a physical quantity.
    """
    parts = blob.split("/")
    if len(parts) >= 2:
        mid = parts[1]
        c = mid.find("_comp")
        if c != -1:
            u = mid.find("_", c + 5)
            if u != -1:
                return mid[u + 1:]
    return parts[0]


def load_run(d):
    """One run directory -> a dict of aggregates, or None if unusable."""
    meta_p, blobs_p = os.path.join(d, "meta.json"), os.path.join(d, "blobs.csv")
    if not (os.path.isfile(meta_p) and os.path.isfile(blobs_p)):
        return None
    with open(meta_p) as f:
        meta = json.load(f)
    rows = list(csv.DictReader(open(blobs_p)))
    if not rows:
        return None

    total_in = sum(int(r["bytes"]) for r in rows)
    # `stored` already accounts for a chunk the codec could not shrink: the
    # compressor stores the caller's bytes and reports lib=0, so stored ==
    # bytes there. Summing it is the true tier footprint.
    total_stored = sum(int(r["stored"]) for r in rows)
    compressed = [r for r in rows if r["lib"] != "0"]

    per_field_io = defaultdict(lambda: [0, 0])
    for r in rows:
        io = per_field_io[field_of(r["blob"])]
        io[0] += int(r["bytes"])
        io[1] += int(r["stored"])
    per_field = {f: (round(i / s, 4) if s else 0.0)
                 for f, (i, s) in per_field_io.items()}

    mix = Counter(r["codec"] if r["lib"] != "0" else "raw" for r in rows)
    ms = [_f(r, "compress_ms") for r in rows]

    # Per-chunk COMPRESSION throughput, MB/s, over the chunks the codec
    # actually timed. Two things leave a chunk untimed and both must be
    # excluded rather than counted as instantaneous: WarpX without an
    # exploration log has no timing at all, and a CPU codec in the action
    # space (brotli, zlib, ...) reports 0 because the measurement is a
    # CUDA-event bracket. ct_chunks below says how many are behind the rate.
    ct_pairs = [(int(r["bytes"]), _f(r, "compress_ms")) for r in rows
                if _f(r, "compress_ms") > 0.0]
    ct_rates = [b / (t * 1000.0) for b, t in ct_pairs]   # B/ms -> MB/s
    ct_sum = sum(t for _, t in ct_pairs)
    ct_bytes = sum(b for b, _ in ct_pairs)

    # DECOMPRESSION: measured only when CLIO_NEUROPRESS_EXPLORE_MEASURE_DT
    # asked for it, and never for a chunk stored raw -- there is nothing to
    # invert, and the driver correctly reports -1 rather than 0. Both are
    # excluded rather than counted as instantaneous.
    dt_pairs = [(int(r["bytes"]), _f(r, "decompress_ms", -1.0)) for r in rows
                if _f(r, "decompress_ms", -1.0) > 0.0]
    dt_rates = [b / (t * 1000.0) for b, t in dt_pairs]
    dt_sum = sum(t for _, t in dt_pairs)
    dt_bytes = sum(b for b, _ in dt_pairs)

    # Best and worst COMPRESSION RATIO seen on any single chunk. The run-level
    # ratio below is the aggregate; these two are what it averages over.
    chunk_ratios = [int(r["bytes"]) / int(r["stored"])
                    for r in rows if int(r["stored"]) > 0]
    ct_lo, ct_mid, ct_hi = _stats(ct_rates)
    dt_lo, dt_mid, dt_hi = _stats(dt_rates)
    cr_lo, cr_mid, cr_hi = _stats(chunk_ratios)

    # A workload may record its own verification outcome in meta.json -- the
    # in-situ one does, because its application is a stock binary whose stdout
    # says nothing about Clio. Fall back to scanning the driver's output.
    verified = meta.get("verify_result")
    if verified in (None, "n/a"):
        verified = None
    out_p = os.path.join(d, "stdout.log")
    if os.path.isfile(out_p):
        txt = open(out_p, errors="replace").read()
        # The BOUND verdict wins where it exists. On a lossy run under
        # --check-bound it is the only line that says anything about the DATA:
        # the VERIFIED:/FAILED: line beside it reports whether decompression
        # succeeded, not whether the values came back inside the error bound.
        if "BOUND OK:" in txt:
            verified = "bound-ok"
        elif "BOUND FAILED:" in txt:
            verified = "BOUND-FAIL"
        elif "VERIFIED:" in txt:
            verified = "pass"
        elif "FAILED:" in txt:
            verified = "FAIL"

    # Physics that differs from the deck's defaults. Part of the identity of a
    # run: two runs of the same config at different density or temperature are
    # different experiments and must not be averaged together.
    phys = meta.get("physics", {}) or {}
    phys_set = {k: v for k, v in phys.items() if v not in ("deck", "", None)}
    phys_key = " ".join(f"{k}={v}" for k, v in sorted(phys_set.items()))
    precision = phys.get("precision")

    return {
        "tag": meta.get("tag", os.path.basename(d)),
        "config": meta.get("config", ""),
        "workload": meta.get("workload", "lammps-ljmelt"),
        # Axes run_benchmark.sh sweeps. Defaults describe a run recorded
        # before they existed: lossless, whatever weights the config implied,
        # and NeuroPress's shipped 5e6 B/ms bandwidth.
        "mode": meta.get("mode", "lossless"),
        "cost_model": meta.get("cost_model", ""),
        "bw_bytes_per_ms": meta.get("bw_bytes_per_ms", 5e6),
        "bw_GBps": round(float(meta.get("bw_bytes_per_ms", 5e6)) / 1e6, 3),
        "error_bound": meta.get("error_bound", 0),
        "explore_k": meta.get("explore_k", 0),
        "explore_thresh": meta.get("explore_thresh", 0),
        "steps": meta.get("steps", 0),
        "files": meta.get("files", 0),
        "physics": phys_key or "deck defaults",
        "physics_precision": precision,
        "device": meta.get("device", ""),
        "atoms": meta.get("atoms", 0),
        "frames": meta.get("frames", 0),
        "chunks": len(rows),
        "payload_bytes": total_in,
        "stored_bytes": total_stored,
        "ratio": round(total_in / total_stored, 4) if total_stored else 0.0,
        "compressed_chunks": len(compressed),
        "raw_chunks": len(rows) - len(compressed),
        "compress_ms_sum": round(sum(ms), 1),
        "compress_ms_median": round(statistics.median(ms), 3),
        # Aggregate throughput = timed bytes / summed codec time. The per-chunk
        # best and worst beside it are the extremes that average hides.
        "compress_MBps": round(ct_bytes / (ct_sum * 1000.0), 2) if ct_sum else 0.0,
        "compress_MBps_min": round(ct_lo, 2),
        "compress_MBps_median": round(ct_mid, 2),
        "compress_MBps_max": round(ct_hi, 2),
        "decompress_ms_sum": round(dt_sum, 1),
        "decompress_ms_median": round(
            statistics.median([t for _, t in dt_pairs]) if dt_pairs else 0.0, 3),
        "decompress_MBps": round(dt_bytes / (dt_sum * 1000.0), 2) if dt_sum else 0.0,
        "decompress_MBps_min": round(dt_lo, 2),
        "decompress_MBps_median": round(dt_mid, 2),
        "decompress_MBps_max": round(dt_hi, 2),
        "ct_chunks": len(ct_pairs),
        "dt_chunks": len(dt_pairs),
        "chunk_ratio_min": round(cr_lo, 4),
        "chunk_ratio_median": round(cr_mid, 4),
        "chunk_ratio_max": round(cr_hi, 4),
        "wall_s": round(float(meta.get("wall_s", 0)), 2),
        "verified": verified or "n/a",
        "rc": meta.get("rc", 0),
        "codec_mix": " ".join(f"{k}:{v}" for k, v in mix.most_common()),
        "fields": per_field,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = args[0] if args else "results"
    outdir = root
    if "-o" in sys.argv:
        outdir = sys.argv[sys.argv.index("-o") + 1]
    if not os.path.isdir(root):
        sys.exit(f"no such directory: {root}")

    runs = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            r = load_run(d)
            if r:
                runs.append(r)
    if not runs:
        sys.exit(f"no completed runs found under {root}/")

    def write_csv(path, subset):
        """One CSV for `subset`, with a ratio_ column per field THOSE runs saw.

        Scoping the field columns to the subset is the point of writing a
        per-workload file at all. The physical quantities are disjoint across
        workloads -- LAMMPS has position/velocity/force, Nyx has
        density/xmom/rho_E, VPIC sixteen field variables, WarpX ten -- so a
        combined header is their UNION and every row carries a zero for every
        other workload's fields. That is ~35 mostly-empty columns across the
        four, and a 0.0 there is indistinguishable from "compressed to
        nothing" rather than "this workload has no such field".
        """
        fields = sorted({f for r in subset for f in r["fields"]})
        flat = []
        for r in subset:
            row = {k: v for k, v in r.items() if k != "fields"}
            for f in fields:
                row[f"ratio_{f}"] = r["fields"].get(f, 0.0)
            flat.append(row)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        return path

    # The combined file stays: comparing one workload against another at the
    # same bandwidth is a real use, and it is the only file that supports it.
    csv_p = write_csv(os.path.join(outdir, "summary.csv"), runs)

    # Plus one per workload, which is what a per-workload question should be
    # answered from. Written only when there is more than one workload --
    # otherwise it would duplicate summary.csv byte for byte.
    per_workload = defaultdict(list)
    for r in runs:
        per_workload[r["workload"]].append(r)
    extra_csvs = []
    if len(per_workload) > 1:
        for wl, subset in sorted(per_workload.items()):
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in wl)
            extra_csvs.append(
                write_csv(os.path.join(outdir, f"summary_{safe}.csv"), subset))

    WORKLOADS = {
        "lammps-ljmelt": ("LAMMPS", lambda r: f"LJ melt, {r['atoms']:,} atoms"),
        "nyx-sedov": ("Nyx", lambda r: f"Sedov blast, {r['files']} field file(s)"),
        "vpic-weibel": ("VPIC",
                        lambda r: f"Weibel instability, {r['files']} field file(s)"),
        "warpx-laser": ("WarpX", lambda r: "laser acceleration, IN SITU"),
    }

    # One section per workload. This aggregator is shared, and run_benchmark.sh
    # puts every workload's cells in ONE directory, so describing them all with
    # the first run's label -- which is what the single-workload version did --
    # would file a Nyx run under "LJ melt".
    by_wl = defaultdict(list)
    for r in runs:
        by_wl[r["workload"]].append(r)

    def axes(r):
        """The cell a run occupies, as (mode, cost model, bandwidth)."""
        return (r["mode"], r["cost_model"] or r["config"], r["bw_GBps"])

    lines = ["# Clio-NeuroPress benchmark", ""]
    lines += [
        f"{len(runs)} run(s) across {len(by_wl)} workload(s). Exploration mode "
        f"throughout unless a run's `config` column says otherwise.",
        "",
        "Throughput is bytes over CODEC time (CUDA-event bracket around the "
        "codec call), not wall clock, and covers only the chunks that could be "
        "timed -- `ct_chunks` / `dt_chunks` in `summary.csv` say how many. A "
        "chunk stored raw has nothing to decompress, and a CPU codec in the "
        "action space (brotli, zlib) reports no GPU time at all.",
        "",
    ]

    for wl in sorted(by_wl):
        rs_all = by_wl[wl]
        a = rs_all[0]
        title, describe = WORKLOADS.get(
            wl, (wl, lambda r: f"{r['files'] or r['chunks']} unit(s)"))
        # Only what meta.json actually recorded. Defaulting this is how a
        # float64 LAMMPS run gets published as float32 -- the workload table
        # below exists because the same bug once mislabelled a whole sweep.
        prec = a.get("physics_precision")
        lines += [
            f"## {title}", "",
            f"{describe(a)}{', ' + prec if prec else ''}. "
            f"{a['payload_bytes'] / 2**30:.2f} GiB payload, {a['chunks']} chunks, "
            f"{a['steps']} timesteps, {a['device'].upper()}.",
            "",
            "| mode | cost model | bw GB/s | ratio | stored | cmp MB/s | "
            "dcmp MB/s | cmp s | dcmp s | wall s | verified |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rs_all, key=axes):
            lines.append(
                f"| {r['mode']} | {r['cost_model'] or r['config']} | "
                f"{r['bw_GBps']:g} | **{r['ratio']:.3f}×** | "
                f"{r['stored_bytes'] / 2**30:.3f} GiB | "
                f"{r['compress_MBps']:.1f} | {r['decompress_MBps']:.1f} | "
                f"{r['compress_ms_sum'] / 1000:.1f} | "
                f"{r['decompress_ms_sum'] / 1000:.1f} | "
                f"{r['wall_s']:.1f} | {r['verified']} |")

        # Best and worst, which the aggregate above averages away. The spread
        # between them is the point: it is how the run's behaviour changes as
        # the simulation evolves.
        lines += [
            "", f"### {title}: best and worst chunk", "",
            "| mode | cost model | bw GB/s | ratio min / med / max | "
            "cmp MB/s min / med / max | dcmp MB/s min / med / max | "
            "cmp/dcmp chunks |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rs_all, key=axes):
            lines.append(
                f"| {r['mode']} | {r['cost_model'] or r['config']} | "
                f"{r['bw_GBps']:g} | "
                f"{r['chunk_ratio_min']:.2f} / {r['chunk_ratio_median']:.2f} / "
                f"{r['chunk_ratio_max']:.2f} | "
                f"{r['compress_MBps_min']:.0f} / {r['compress_MBps_median']:.0f} / "
                f"{r['compress_MBps_max']:.0f} | "
                f"{r['decompress_MBps_min']:.0f} / "
                f"{r['decompress_MBps_median']:.0f} / "
                f"{r['decompress_MBps_max']:.0f} | "
                f"{r['ct_chunks']}/{r['dt_chunks']} of {r['chunks']} |")

        # Repeats, grouped. A single run per cell cannot be read on its own:
        # exploration adopts on MEASURED candidate times, so the winner flips
        # between runs on identical input. The min..max column is the run-to-
        # run band, and a bandwidth effect is only real where the bands of two
        # cells do not overlap.
        by_cell = defaultdict(list)
        for r in rs_all:
            by_cell[axes(r)].append(r)
        if any(len(v) > 1 for v in by_cell.values()):
            lines += [
                "", f"### {title}: repeats (mean and run-to-run band)", "",
                "| mode | cost model | bw GB/s | n | ratio mean | ratio min..max "
                "| band % | cmp MB/s mean | dcmp MB/s mean |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for cell in sorted(by_cell):
                rs = by_cell[cell]
                mode, model, bw = cell
                rr = [r["ratio"] for r in rs]
                mean = sum(rr) / len(rr)
                band = (max(rr) - min(rr)) / mean * 100.0 if mean else 0.0
                cm = sum(r["compress_MBps"] for r in rs) / len(rs)
                dm = sum(r["decompress_MBps"] for r in rs) / len(rs)
                lines.append(
                    f"| {mode} | {model} | {bw:g} | {len(rs)} | "
                    f"**{mean:.3f}×** | {min(rr):.3f} .. {max(rr):.3f} | "
                    f"{band:.1f}% | {cm:.1f} | {dm:.1f} |")

        lines += ["", f"### {title}: codec mix", ""]
        for r in sorted(rs_all, key=axes):
            lines.append(f"- `{r['tag']}`: {r['codec_mix']}")
        lines += [""]

    lines += ["Per-run detail: `summary.csv`. Per-chunk detail: "
              "`<run>/blobs.csv`; NeuroPress's own choice per chunk: "
              "`<run>/selection.csv`; every measured candidate: "
              "`<run>/explore.csv`.", ""]

    md_p = os.path.join(outdir, "summary.md")
    open(md_p, "w").write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nwrote {csv_p} and {md_p}")
    for p in extra_csvs:
        print(f"      {p}")


if __name__ == "__main__":
    main()

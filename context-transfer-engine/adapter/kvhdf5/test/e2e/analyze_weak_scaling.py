#!/usr/bin/env python3
"""Analyze the weak-scaling campaign raw log into a CSV and pgfplots figure bodies.

Reads results/weak_scaling_raw.log (one `key=value ...` line per completed point,
tagged with its repeat index) and reduces the repeats at each (bdev, blocks) to a
median with an interquartile spread.

Median + IQR rather than mean +/- stddev: GPU clock/thermal state makes the
run-to-run distribution non-Gaussian and occasionally produces a single slow
outlier, which a mean would absorb into the point estimate and a stddev would
turn into a misleadingly wide symmetric bar.

Error bars are emitted as a SYMMETRIC half-IQR to match the house style of the
paper's existing figures (fig_read.tex etc. use `+-(0,err)`); the true asymmetric
quartiles are preserved in the CSV.

Usage:  uv run analyze_weak_scaling.py [raw_log] [--out-dir DIR]
"""
import argparse
import collections
import pathlib
import statistics
import sys

# Draw order, and the paper's house palette (fig_read.tex / fig_write.tex). The
# LINES ARE THE FOUR SYSTEM ARMS the paper compares, not GPUH5's bdev tiers.
# GPUH5 Sync is the hero (drawn heavier below); the three host-mediated arms are
# the flat baselines. Column 0 is the WS_ARM value emitted on the WSRESULT line.
TRANSPORTS = [
    ("gpuh5",    "wsgpuh5", "1F77B4", "solid",          "mark=*",          "GPUH5 Sync"),
    ("hostclio", "wsclio",  "6A51A3", "densely dotted", "mark=square*",    "Raw+CLIO"),
    ("raw",      "wsraw",   "555555", "dashed",         "mark=triangle*",  "Raw"),
    ("hdf5",     "wshdf5",  "E07B39", "solid",          "mark=diamond*",   "HDF5"),
]


def parse(path):
    """Yield dicts for well-formed result lines, skipping NORESULT rows."""
    rows = []
    for raw in pathlib.Path(path).read_text().splitlines():
        raw = raw.strip()
        if not raw or "STATUS=NORESULT" in raw:
            continue
        rec = {}
        for tok in raw.split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            rec[k] = v
        if "blocks" not in rec or "bw_gbps" not in rec:
            continue
        # A point whose bytes did not land is not a measurement. `na` is the
        # no-op backend, which by definition stores nothing (see the bench).
        if rec.get("verified") == "FAIL":
            print(f"WARN dropping unverified point: {raw}", file=sys.stderr)
            continue
        rows.append(rec)
    return rows


def quartiles(vals):
    vals = sorted(vals)
    n = len(vals)
    med = statistics.median(vals)
    if n < 4:
        return med, min(vals), max(vals)
    lo = statistics.median(vals[: n // 2])
    hi = statistics.median(vals[(n + 1) // 2 :])
    return med, lo, hi


def main():
    ap = argparse.ArgumentParser()
    here = pathlib.Path(__file__).parent
    ap.add_argument("raw_log", nargs="?",
                    default=str(here / "results" / "weak_scaling_raw.log"))
    ap.add_argument("--out-dir", default=str(here / "results"))
    args = ap.parse_args()

    rows = parse(args.raw_log)
    if not rows:
        sys.exit(f"no usable rows in {args.raw_log}")

    agg = collections.defaultdict(lambda: {"bw": [], "lat": []})
    for r in rows:
        key = (r["arm"], int(r["blocks"]))
        agg[key]["bw"].append(float(r["bw_gbps"]))
        agg[key]["lat"].append(float(r.get("submit_med_us", "-1")))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV: full quartiles, the record of what was measured. ----
    csv_path = out_dir / "weak_scaling.csv"
    with csv_path.open("w") as f:
        f.write("arm,blocks,reps,bw_med,bw_q1,bw_q3,lat_med,lat_q1,lat_q3\n")
        for bdev, _, _, _, _, _ in TRANSPORTS:
            for blocks in sorted(b for d, b in agg if d == bdev):
                a = agg[(bdev, blocks)]
                bm, bq1, bq3 = quartiles(a["bw"])
                lm, lq1, lq3 = quartiles(a["lat"])
                f.write(f"{bdev},{blocks},{len(a['bw'])},{bm:.4f},{bq1:.4f},"
                        f"{bq3:.4f},{lm:.3f},{lq1:.3f},{lq3:.3f}\n")

    # ---- pgfplots bodies for both figures. ----
    def emit(metric, ylabel, fname, scale=1.0, ymode=""):
        lines = []
        for bdev, cname, hexc, style, mark, label in TRANSPORTS:
            pts = sorted(b for d, b in agg if d == bdev)
            if not pts:
                continue
            coords = []
            for blocks in pts:
                vals = agg[(bdev, blocks)][metric]
                med, q1, q3 = quartiles(vals)
                err = (q3 - q1) / 2.0
                coords.append(f"({blocks},{med * scale:.4f})+-(0,{err * scale:.4f})")
            wrapped, cur = [], "  coordinates {"
            for c in coords:
                if len(cur) + len(c) > 88:
                    wrapped.append(cur)
                    cur = "               "
                cur += c + " "
            wrapped.append(cur.rstrip() + "};")
            opts = f"color={cname}, {style}, {mark}"
            if mark != "mark=none":
                opts += f", mark options={{fill={cname}}}"
            if bdev == "gpuh5":          # the hero line gets extra weight
                opts += ", line width=1.5pt"
            lines.append(f"\\addplot[{opts}]")
            lines.extend(wrapped)
            lines.append(f"\\addlegendentry{{{label}}}")

        defs = "\n".join(f"\\definecolor{{{c}}}{{HTML}}{{{h}}}"
                         for _, c, h, _, _, _ in TRANSPORTS)
        body = f"""% Weak scaling: {ylabel} vs concurrent GPU producer blocks.
% GENERATED by analyze_weak_scaling.py -- edit the analyzer, not this file.
% Source: results/weak_scaling_raw.log
% Lines are the four system arms; 128 MB per block held constant; RAM sink for
% all; async disabled (I/O bandwidth only). GPUH5 writes device-initiated, one
% PutBlob per block in-kernel; raw/hdf5/hostclio route through one host pipeline
% (D2H then write), so they are flat across block count. x ends at 768 = 6
% resident blocks/SM x 128 SMs, the RTX 4090's resident max at 256 threads.
% Points are medians over repeats; error bars are the half-interquartile range.
{defs}
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=\\columnwidth, height=5.4cm,
  xmode=log, log basis x=2, xtick={{1,4,16,64,256,768}},
  xticklabels={{1,4,16,64,256,768}}, enlarge x limits=0.10,
  xlabel={{concurrent GPU blocks}},
  ylabel={{{ylabel}}},
  {ymode}ymin=0,
  xtick align=outside, ytick align=outside,
  tick label style={{font=\\footnotesize}}, label style={{font=\\footnotesize}},
  legend style={{font=\\scriptsize, at={{(0.5,1.02)}}, anchor=south, legend columns=2,
    draw=none, /tikz/every even column/.append style={{column sep=5pt}}}},
  legend cell align=left,
  grid=major, grid style={{gray!18}},
  every axis plot/.append style={{line width=0.9pt, mark size=2.1pt}},
  error bars/y dir=both, error bars/y explicit,
]
{chr(10).join(lines)}
\\end{{axis}}
\\end{{tikzpicture}}
"""
        (out_dir / fname).write_text(body)
        return out_dir / fname

    f1 = emit("bw", "aggregate I/O bandwidth (GB/s)", "fig_weak_scaling_bw.tex")
    f2 = emit("lat", "median submission latency (ms)",
              "fig_weak_scaling_lat.tex", scale=1e-3, ymode="ymode=log,\n  ")

    reps = sorted({int(r["rep"]) for r in rows if "rep" in r})
    print(f"points={len(rows)} reps={len(reps)} cells={len(agg)}")
    print(f"wrote {csv_path}\nwrote {f1}\nwrote {f2}")

    # Console summary so the shape is visible without opening the figure.
    print("\nmedian bandwidth (GB/s):")
    all_b = sorted({b for _, b in agg})
    print("  blocks: " + " ".join(f"{b:>7}" for b in all_b))
    for arm, _, _, _, _, _ in TRANSPORTS:
        cells = []
        for b in all_b:
            if (arm, b) in agg:
                cells.append(f"{statistics.median(agg[(arm, b)]['bw']):>7.2f}")
            else:
                cells.append("      -")
        print(f"  {arm:>9}: " + " ".join(cells))


if __name__ == "__main__":
    main()

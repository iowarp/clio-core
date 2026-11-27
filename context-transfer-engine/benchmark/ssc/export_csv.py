"""Export the campaign's results as plain CSV files (for sharing, spreadsheets, or backup).

    env -u PYTHONPATH python3 export_csv.py <results dir> [out dir]     (default out: <results>/csv)

Writes:
    grid.csv        every measured grid cell: one row per (node setup, phase, composition, repeat)
    decisions.csv   every offline decision: one row per (scenario, controller, schedule, window),
                    scored with the same metric as the paper (paper_plots.slowdown)
    summary.csv     per (scenario, controller): decisions, accuracy (within 25%), geomean slowdown,
                    median / p90 decision time
    learning.csv    learning-curve decisions: one row per (scenario, controller, subset, window)
    live.csv        live end-to-end runs, one row per window (if any live results exist)
Reads only; the JSONL files stay the source of truth.
"""

import csv
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_plots as pp      # noqa: E402

AXES = ("placement", "cache", "indexer", "compression")


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def grid_rows(res):
    out = []
    for p in sorted(glob.glob(os.path.join(res, "grid", "grid_*.jsonl"))):
        name = os.path.basename(p)[5:-6]
        if "_" in name:           # grid_<node>.jsonl is the merged grid; skip partial / side files
            continue
        for l in open(p):
            r = json.loads(l)
            c = r["composition"]
            out.append([name, r["phase"], *(c[a] for a in AXES), r.get("rep"), r.get("wall_s"),
                        r.get("write_s"), r.get("read_s"), r.get("logical_mb"), r.get("mb_per_s"),
                        r.get("n_puts"), r.get("failed_puts", 0), r.get("failed_gets", 0),
                        r.get("dram_fraction")])
    return out


def main():
    res = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(res, "csv")
    os.makedirs(out, exist_ok=True)

    write(os.path.join(out, "grid.csv"),
          ["node", "phase", *AXES, "rep", "wall_s", "write_s", "read_s", "logical_mb", "mb_per_s",
           "n_puts", "failed_puts", "failed_gets", "dram_fraction"], grid_rows(res))

    grids, dec, summ = {}, [], []
    for (node, app), d in sorted(pp.load_offline(res).items()):
        gp = os.path.join(res, "grid", f"grid_{node}.jsonl")
        if not os.path.exists(gp):
            continue
        grid = grids.setdefault(node, pp.load_grid(gp))
        try:
            pp.add_fixed(d, grid)
        except Exception:         # scenarios without a guards arm have no fixed baselines
            pass
        for v, rows in sorted(d.items()):
            sl = []
            for r in rows:
                try:
                    s = pp.slowdown(r, grid)
                except KeyError:  # composition not in this node's grid
                    continue
                sl.append(s)
                rec = r["recommended"]
                dec.append([node, app, v, r.get("seed"), r.get("step"), r["phase"],
                            *(rec[a] for a in AXES), round(r.get("latency_s", 0.0), 4),
                            bool(r.get("violation")), round(s, 4), s <= pp.ACC_WITHIN])
            if not sl:
                continue
            lat = sorted(r.get("latency_s", 0.0) for r in rows)
            summ.append([node, app, v, len(sl), round(100 * statistics.mean(x <= pp.ACC_WITHIN for x in sl), 1),
                         round(math.exp(statistics.mean(map(math.log, sl))), 4),
                         round(statistics.median(lat), 3), round(lat[int(0.9 * (len(lat) - 1))], 3)])
    write(os.path.join(out, "decisions.csv"),
          ["node", "app", "controller", "seed", "step", "phase", *AXES, "latency_s", "violation",
           "slowdown", "within_25pct"], dec)
    write(os.path.join(out, "summary.csv"),
          ["node", "app", "controller", "decisions", "accuracy_pct", "slowdown_geomean",
           "latency_median_s", "latency_p90_s"], summ)

    lc = []
    for (node, app, v), byn in sorted(pp.load_lc(res).items()):
        for n, subsets in sorted(byn.items()):
            for sub, rows in sorted(subsets.items()):
                for r in rows:
                    rec = r["recommended"]
                    lc.append([node, app, v, n, " ".join(sub), r.get("seed"), r.get("step"), r["phase"],
                               *(rec[a] for a in AXES), round(r["slowdown"], 4), r["slowdown"] <= pp.ACC_WITHIN])
    write(os.path.join(out, "learning.csv"),
          ["node", "app", "controller", "measured_phases", "subset", "seed", "step", "phase", *AXES,
           "slowdown", "within_25pct"], lc)

    live = []
    for p in sorted(glob.glob(os.path.join(res, "live", "*", "*.jsonl"))):
        scen = os.path.basename(os.path.dirname(p))
        arm = os.path.basename(p)[:-6]
        for l in open(p):
            r = json.loads(l)
            live.append([scen, arm, json.dumps(r, sort_keys=True)])
    if live:
        write(os.path.join(out, "live.csv"), ["scenario", "arm_file", "row_json"], live)


if __name__ == "__main__":
    main()

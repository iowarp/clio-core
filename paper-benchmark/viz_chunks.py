#!/usr/bin/env python3
"""Plot a run chunk by chunk: what the model saw, what it said, what happened.

    ./viz_chunks.py --run /tmp/nyx-learn/learn --out /tmp/viz
    ./viz_chunks.py --run /tmp/vpic-learn/learn --out /tmp/viz --field ez --field cbx

Two figures, both with chunk index on the x axis so a feature and the outcome
at the same x are the same chunk:

    chunk_stats.png       the three statistics the model consumes -- Shannon
                          entropy, mean absolute deviation, second derivative.
                          Together with chunk size and error bound these are
                          the model's ENTIRE view of a chunk, so a prediction
                          it gets wrong is wrong from these numbers alone.

    chunk_prediction.png  predicted against actual compression ratio, plus a
                          rug marking the chunks that produced an SGD gradient.

Three panels rather than three lines on one pair of axes: MAD and the second
derivative span several orders of magnitude and are drawn logarithmic, entropy
is bits on a linear scale, and putting two units on one axis is how a chart
lies. Same reason the ratio figure has one y axis -- predicted and actual are
both compression ratio, so they belong together.

`--run DIR` is a run_config.sh store (it reads `selection.csv`, and
`meta.json` if present). Workload-agnostic: it reads both blob-name shapes the
benchmarks produce,

    nyx / vpic   plt00007/fab0000_comp00_density/chunk_0    frame first
    warpx        step00010/E_x/chunk_0                      frame first
    lammps       position/step_140/chunk_0                  field first

so `--field density`, `--field ez` and `--field position` all work. Without
`--field` every chunk is drawn in the order the run produced them, which for a
multi-variable workload is a sawtooth: one chunk per variable per frame. Filter
to a single variable to see that variable's own trajectory.

THE GATE RUG IS ONLY EXACT UNDER RATIO-ONLY COST WEIGHTS. Phase-1 SGD fires on
`error_pct > neuropress_mape_threshold`, where error_pct is the MAPE of the
whole cost -- w_ct*ct + w_dt*dt + bytes/(ratio*bw). With the latency weights
zeroed that reduces to |1 - actual/predicted| on the capped ratios, which is
what this computes from selection.csv. Under the default balanced weights the
two constants dominate and this would overstate the gate badly, so the script
reads `cost_model` out of meta.json and refuses to draw the rug unless it says
`ratio`. Pass --force-gate if you know better than meta.json.
"""
import argparse, csv, json, os, re, sys

FIELD_RE = re.compile(r"^fab\d+_comp\d+_")
# plt00007 (nyx), step00010 (warpx, no separator), step_140 (lammps).
FRAME_RE = re.compile(r"^(?:plt|step_?)(\d+)$")
# Reference categorical palette, slots 1-3 in fixed order -- three series, which
# validates on every adjacent pair in both light and dark.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GATE = "#8046B5"


def field_of(blob):
    """The variable name in a blob path, or None if it is not one of ours."""
    parts = blob.split("/")
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    ma, mb = FRAME_RE.match(a), FRAME_RE.match(b)
    if ma and not mb:
        return FIELD_RE.sub("", b)
    if mb and not ma:
        return FIELD_RE.sub("", a)
    return None


def read_rows(path, want_fields):
    """selection.csv -> the primary row per chunk, in run order.

    `primary` only. Exploration and best mode also log an `adopted` row per
    chunk -- the alternative that actually got stored -- and mixing the two
    would put two points at one x with no way to tell which the model predicted.
    """
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("role") not in (None, "", "primary"):
                continue
            f = field_of(r["blob"])
            if want_fields and f not in want_fields:
                continue
            out.append((f, r))
    return out


def fnum(r, key, default=0.0):
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return default


def cost_model_of(run_dir):
    """What cost weights the run used, from meta.json. None when unrecorded."""
    p = os.path.join(run_dir, "meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh).get("cost_model")
    except (ValueError, OSError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, metavar="DIR",
                   help="a run_config.sh store, or a selection.csv directly")
    p.add_argument("--out", required=True)
    p.add_argument("--field", action="append",
                   help="repeatable; restrict to these variables")
    p.add_argument("--mape", type=float, default=0.30,
                   help="the run's neuropress_mape_threshold (default 0.30)")
    p.add_argument("--cap", type=float, default=100.0,
                   help="the model's ratio cap (default 100)")
    p.add_argument("--force-gate", action="store_true",
                   help="draw the gate rug even when meta.json is not ratio-only")
    a = p.parse_args()

    csvp = a.run if a.run.endswith(".csv") else os.path.join(a.run, "selection.csv")
    if not os.path.exists(csvp):
        sys.exit(f"no selection log at {csvp}")
    run_dir = os.path.dirname(csvp)

    want = set(a.field) if a.field else None
    rows = read_rows(csvp, want)
    if not rows:
        sys.exit(f"no primary rows in {csvp}"
                 + (f" for field(s) {sorted(want)}" if want else ""))

    ent = [fnum(r, "entropy") for _, r in rows]
    mad = [fnum(r, "mad") for _, r in rows]
    dv = [fnum(r, "second_deriv") for _, r in rows]
    pr = [fnum(r, "pred_ratio") for _, r in rows]
    ar = [fnum(r, "actual_ratio") for _, r in rows]
    x = list(range(len(rows)))

    cm = cost_model_of(run_dir)
    draw_gate = a.force_gate or cm == "ratio"
    if not draw_gate:
        print(f"NOTE: cost_model={cm!r} in meta.json, not 'ratio' -- the SGD gate "
              f"is computed from the FULL cost there and cannot be derived from "
              f"selection.csv. Rug omitted; pass --force-gate to draw it anyway.")
    gate = []
    for pv, av in zip(pr, ar):
        pc, ac = min(a.cap, pv), min(a.cap, av)
        gate.append(pc > 0 and ac > 0 and abs(1.0 - ac / pc) > a.mape)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(a.out, exist_ok=True)
    title_bit = ", ".join(sorted(want)) if want else "all variables"

    # ---- figure 1: the model's inputs -------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    panels = [
        (axes[0], ent, "Shannon entropy  (bits)", SERIES[0], False),
        (axes[1], mad, "mean absolute deviation", SERIES[1], True),
        (axes[2], dv, "second derivative", SERIES[2], True),
    ]
    for ax, vals, label, color, logy in panels:
        ax.plot(x, vals, "-", color=color, lw=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        if logy:
            pos = [v for v in vals if v > 0]
            if pos:
                ax.set_yscale("log")
                nz = len(vals) - len(pos)
                if nz:
                    # A log axis drops non-positive points silently. Say so on
                    # the panel rather than letting a gap read as missing data.
                    ax.text(0.995, 0.06, f"{nz} chunk(s) exactly 0, not drawable on log",
                            transform=ax.transAxes, ha="right", fontsize=7.5,
                            color="0.45")
    axes[-1].set_xlabel("chunk index  (the order the run produced them)")
    axes[-1].set_xlim(0, max(1, len(rows) - 1))
    fig.suptitle(f"What the model sees per chunk - {title_bit}", fontsize=11)
    fig.tight_layout()
    out1 = os.path.join(a.out, "chunk_stats.png")
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    print(f"  {out1}")

    # ---- figure 2: prediction against outcome -----------------------------
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(x, ar, "-", color=SERIES[2], lw=1.0, label="actual ratio")
    ax.plot(x, pr, "-", color=SERIES[1], lw=1.0, label="predicted ratio")
    finite = [v for v in ar + pr if v > 0]
    if finite and max(finite) / max(min(finite), 1e-9) > 20:
        ax.set_yscale("log")
    if finite and max(finite) > a.cap:
        ax.axhline(a.cap, color="0.45", ls="--", lw=1.2)
        # Left edge with an opaque box: at the right edge this label lands on
        # top of the densest part of the trace on every workload tried.
        ax.text(len(rows) * 0.008, a.cap, f" {a.cap:g}x model cap ",
                ha="left", va="center", fontsize=8, color="0.25",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    ax.set_ylabel("compression ratio")
    ax.set_xlabel("chunk index")
    ax.set_xlim(0, max(1, len(rows) - 1))
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=SERIES[2], label="actual ratio"),
               Line2D([], [], color=SERIES[1], label="predicted ratio")]
    if draw_gate:
        lo, hi = ax.get_ylim()
        y = lo * 1.06 if ax.get_yscale() == "log" else lo + (hi - lo) * 0.02
        fired = [i for i in x if gate[i]]
        ax.plot(fired, [y] * len(fired), "|", color=GATE, ms=7, mew=0.9)
        handles.append(Line2D([], [], color=GATE, marker="|", ls="none",
                              label=f"SGD gate fired ({len(fired)})"))
        ax.set_ylim(lo, hi)
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.suptitle(f"Predicted against delivered - {title_bit}", fontsize=11, y=1.02)
    fig.tight_layout()
    out2 = os.path.join(a.out, "chunk_prediction.png")
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"  {out2}")

    # ---- the numbers behind the pictures ----------------------------------
    err = [abs(av - pv) / av if av > 0 else 0.0 for pv, av in zip(pr, ar)]
    n = len(rows)
    half = n // 2
    first = sum(err[:half]) / max(1, half)
    second = sum(err[half:]) / max(1, n - half)
    capped = sum(1 for v in pr if v >= a.cap - 1e-3)
    over = sum(1 for v in ar if v > a.cap)
    print(f"chunks {n}  mean ratio MAPE {sum(err)/n:.3f}  "
          f"first half {first:.3f} -> second half {second:.3f} "
          f"({'improved' if second < first else 'no improvement'})")
    print(f"predicted at the {a.cap:g}x cap: {capped} ({100.0*capped/n:.0f}%)   "
          f"ACTUAL above it: {over} ({100.0*over/n:.0f}%)"
          + ("   <- unreachable targets" if over else ""))
    if draw_gate:
        print(f"gate at MAPE > {a.mape:g}: {sum(gate)} of {n} "
              f"({100.0*sum(gate)/n:.1f}%)")
    if not want:
        per = {}
        for (f, _), e in zip(rows, err):
            per.setdefault(f, []).append(e)
        if len(per) > 1:
            print("per variable, first half -> second half:")
            for f in sorted(per, key=lambda k: -sum(per[k]) / len(per[k])):
                v = per[f]
                h = len(v) // 2
                if h == 0:
                    continue
                fa, sb = sum(v[:h]) / h, sum(v[h:]) / (len(v) - h)
                print(f"  {str(f):<12} {fa:7.3f} -> {sb:7.3f}  "
                      f"{100.0*(sb-fa)/fa:+7.1f}%" if fa > 0 else
                      f"  {str(f):<12} {fa:7.3f} -> {sb:7.3f}")


if __name__ == "__main__":
    main()

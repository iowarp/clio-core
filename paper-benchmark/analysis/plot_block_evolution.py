#!/usr/bin/env python3
"""How much does one block change from timestep to timestep, over the run?

    ./plot_block_evolution.py OUT.png nyx=DIR vpic=DIR warpx=DIR ...

DIR is an evolution.py output directory (blocks.csv beside evolution.json).
A BLOCK is a fixed byte range of one field -- the unit Clio actually
compresses -- so this tracks the same bytes, in the same place, forward
through the run.

    E(B_t, B_t+dt)  =  ||B_t+dt - B_t||  /  (||B_t+dt|| + ||B_t|| + eps)

E = 0 when a block is bit-identical to its previous dump, and approaches 1
when the two are unrelated. It is scale free, so blocks of very different
magnitude are comparable, which matters here: WarpX's j fields are 1e13 and
its B fields are 1e1.

Rows:
  1. every block's trajectory, one faint line each, with the median over
     blocks in bold -- the shape of the run.
  2. the same as a distribution early vs late, so "it settles down" or "it
     stays put" is a number rather than an impression.
  3. the fraction of cells bit-identical to the previous dump, which is the
     quantity a delta-coder would actually exploit.

E SATURATES AT 1 ON A ZERO-BACKGROUND FIELD -- momenta and currents sit at
zero outside the active region, so the instant anything appears, the
denominator is as small as the numerator and E jumps to 1 while the block is
barely changing. Those blocks are drawn but excluded from the medians, and
the count is reported: evolution.py's own Nyx example warns about exactly
this.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import plotting as P  # noqa: E402

SAT = 0.99          # at or above this, E is saturated and uninformative


def load(d: str) -> pd.DataFrame:
    b = pd.read_csv(os.path.join(d, "blocks.csv"))
    if "nonfinite" in b.columns:
        b = b[b["nonfinite"] == 0]
    b["key"] = b["field"].astype(str) + "#" + b["block"].astype(str)
    return b


def main(out: str, *specs: str) -> int:
    P._style()
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in specs]
    data = {n: load(d) for n, d in runs}
    fig, ax = plt.subplots(3, len(runs), figsize=(4.6 * len(runs), 9.8),
                           squeeze=False)
    for j, (name, _) in enumerate(runs):
        b = data[name]
        c = P.SERIES[j % len(P.SERIES)]
        unsat = b[b["evolution"] < SAT]
        n_sat_keys = b.loc[b["evolution"] >= SAT, "key"].nunique()

        # ---- row 1: every block's trajectory --------------------------
        a = ax[0][j]
        for _, g in b.groupby("key"):
            g = g.sort_values("step_to")
            a.plot(g["step_to"], g["evolution"], color=c, lw=0.7, alpha=0.22)
        med = unsat.groupby("step_to")["evolution"].median()
        a.plot(med.index, med.values, color=P.INK, lw=2.2,
               label="median (unsaturated)")
        a.set_yscale("log")
        a.set_xlabel("timestep", fontsize=8)
        a.set_ylabel("E  (0 = unchanged, 1 = unrelated)", fontsize=8)
        a.set_title(f"{name} -- each block, through the run", fontsize=9.5)
        a.annotate(f"{b['key'].nunique()} blocks, {b['step_to'].nunique()} "
                   f"dumps\n{n_sat_keys} blocks saturate at E>={SAT}",
                   xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7.5,
                   color=P.INK)
        a.legend(loc="upper right", fontsize=7)
        P._finish(a)

        # ---- row 2: early vs late -------------------------------------
        a = ax[1][j]
        t = unsat["step_to"]
        if len(t):
            lo, hi = t.quantile(0.25), t.quantile(0.75)
            early = unsat[t <= lo]["evolution"]
            late = unsat[t >= hi]["evolution"]
            bins = np.logspace(np.log10(max(unsat["evolution"].min(), 1e-6)),
                               0, 40)
            a.hist(early, bins=bins, color=c, alpha=0.55,
                   label=f"first quarter (med {early.median():.3f})")
            a.hist(late, bins=bins, color=P.INK_2, alpha=0.45,
                   label=f"last quarter (med {late.median():.3f})")
            a.set_xscale("log")
            a.legend(loc="upper left", fontsize=7)
        a.set_xlabel("E", fontsize=8)
        a.set_ylabel("block-pairs", fontsize=8)
        a.set_title(f"{name} -- does it settle?", fontsize=9.5)
        P._finish(a)

        # ---- row 3: what a delta-coder could actually reuse ------------
        a = ax[2][j]
        if "pct_cells_same" in b.columns:
            m = b.groupby("step_to")["pct_cells_same"].median()
            a.plot(m.index, m.values, color=c, lw=2.0)
            a.fill_between(m.index, 0, m.values, color=c, alpha=0.15)
            a.set_ylim(-2, 102)
            a.annotate(f"first dump {m.iloc[0]:.1f}%\n"
                       f"last dump {m.iloc[-1]:.1f}%",
                       xy=(0.97, 0.93), xycoords="axes fraction", ha="right",
                       va="top", fontsize=8, color=P.INK)
        else:
            a.text(0.5, 0.5, "not in this blocks.csv", ha="center",
                   transform=a.transAxes, color=P.MUTED)
        a.set_xlabel("timestep", fontsize=8)
        a.set_ylabel("% of cells identical to previous dump", fontsize=8)
        a.set_title(f"{name} -- reusable without any coding", fontsize=9.5)
        P._finish(a)

    fig.suptitle("Spatial redundancy ACROSS timesteps: the same block, "
                 "tracked forward through the run\n"
                 "E = ||B_t+dt - B_t|| / (||B_t+dt|| + ||B_t||) -- 0 means "
                 "the block did not move", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}\n")

    print(f"{'':<8}{'blocks':>8}{'dumps':>7}{'E first':>10}{'E last':>9}"
          f"{'same first':>12}{'same last':>11}")
    for name, _ in runs:
        b = data[name]
        u = b[b["evolution"] < SAT]
        t = u["step_to"]
        lo, hi = t.quantile(0.25), t.quantile(0.75)
        s = b.groupby("step_to")["pct_cells_same"].median() \
            if "pct_cells_same" in b.columns else None
        print(f"{name:<8}{b['key'].nunique():>8}{b['step_to'].nunique():>7}"
              f"{u[t <= lo]['evolution'].median():>10.4f}"
              f"{u[t >= hi]['evolution'].median():>9.4f}"
              f"{(s.iloc[0] if s is not None else float('nan')):>11.1f}%"
              f"{(s.iloc[-1] if s is not None else float('nan')):>10.1f}%")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], *sys.argv[2:]))

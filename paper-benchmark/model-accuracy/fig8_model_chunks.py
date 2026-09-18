#!/usr/bin/env python3
"""Figure 8's two per-chunk metrics, for every model, not just the deployed one.

    fig8_model_chunks.py --inputs DIR --out CSV [--hc-suffix S] [--floor 5]

Figure 8 asks whether NeuroPress's online model gets better over a run. This
script asks the same question of the baselines, on the same chunks and the same
candidate set, so the answer can be drawn as extra lines in that figure rather
than as a separate table.

Input is `make_table.sh`'s prepared directory (`<inputs>/<workload>/rows.csv`
and `hcompress<suffix>.csv`): 32 measured configurations per chunk, in the
order the run executed them, plus each model's inputs. Nothing is re-measured
and no model is retrained.

TWO FLOORS, AND WHY THEY DIFFER

  ranking (which configuration a model picks)   1 ms   the DEPLOYED policy
  reporting (regret and cost MAPE)              1 ms   --floor, same by default

The ranking floor is the one the selector really applies (`NeuroPressCost`), so
a counterfactual pick has to use it or it is not the pick that system would have
made.

The reporting floor DEFAULTS TO THE SAME 1 ms, which is not what `fig8_trace.py`
does: panels (a)-(c) report at 5 ms, upstream's trace convention. That is the
wrong floor for a comparison BETWEEN models here. Chunks are 8 MiB, so the cost's
I/O term is at most 1.7 ms while a 5 ms floor puts at least 10 ms of constant
underneath it; every model's cost is then pinned near 10 ms and the differences
between them land in the third decimal. At 1 ms the same regrets are roughly 10x
larger and the models separate. `--floor 5` reproduces the panel-(a)-(c)
convention.

THE PICK

  NeuroPress (online)   the configuration the RUN ACTUALLY ADOPTED as primary
                        (`executed`), which is what Figure 8 already scores.
  every other model     argmin of its own predicted deployed cost -- the pick
                        that model would have made, had it been the selector.

The two definitions agree for NeuroPress on the great majority of chunks; the
script prints the agreement rather than assuming it.

Output, one row per (workload, chunk, model): regret_pct, cost_mape_pct, the
picked and cheapest configuration, and the costs behind both.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import SETTINGS, predictions                  # noqa: E402
from models_offline import (NeuroPressNN, XGBoostBaseline,        # noqa: E402
                            PRED_TIME_FLOOR_MS, RATIO_CAP)

BW_BYTES_PER_MS = 5e6
#: Short names for the figure's legend, in draw order. The key is the label
#: `accuracy_table.predictions()` returns.
MODEL_LABEL = {
    "NeuroPress NN + online learning (as deployed)": "NeuroPress (online)",
    "NeuroPress NN": "NeuroPress (static)",
    "XGBoost": "XGBoost",
    "HCompress CCP (+ feedback)": "HCompress CCP (+fb)",
    "HCompress CCP (seed only)": "HCompress CCP (seed)",
}
ONLINE = "NeuroPress NN + online learning (as deployed)"


def cost_at(ct, dt, ratio, nbytes, floor):
    """The deployed cost model at an arbitrary time floor, elementwise."""
    return (np.maximum(floor, ct) + np.maximum(floor, dt)
            + nbytes / (np.minimum(RATIO_CAP, np.maximum(ratio, 0.1)) * BW_BYTES_PER_MS))


def chunk_bounds(chunk: np.ndarray):
    """Start/stop of each run of equal chunk ids; rows.csv is grouped by chunk."""
    starts = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
    return np.r_[starts, len(chunk)]


def one_workload(key, label, rows, hc, nn, xgb, floor):  # noqa: C901
    preds = predictions(rows, hc, nn, xgb, clamp=True)

    mct = rows.ct_ms.to_numpy(dtype=float)
    mdt = rows.dt_ms.to_numpy(dtype=float)
    mratio = rows.ratio.to_numpy(dtype=float)
    nbytes = rows.bytes.to_numpy(dtype=float)
    # A candidate counts only where all three components were measured, so both
    # the optimum and the picked cost are measurements throughout.
    ok = (np.isfinite(mct) & (mct > 0) & np.isfinite(mdt) & (mdt > 0)
          & np.isfinite(mratio) & (mratio > 0))
    meas_report = cost_at(mct, mdt, mratio, nbytes, floor)
    executed = rows.executed.to_numpy(dtype=int) if "executed" in rows else np.zeros(len(rows), int)
    lib = rows.library.to_numpy()

    # Ranking cost (1 ms floor) and reporting cost (--floor) per model.
    rank = {m: preds[m]["cost"] for m in preds}                    # already at 1 ms
    report = {m: cost_at(preds[m]["ct_ms"], preds[m]["dt_ms"], preds[m]["ratio"], nbytes, floor)
              for m in preds}

    bounds = chunk_bounds(rows.chunk.to_numpy())
    models = [m for m in MODEL_LABEL if m in preds]
    recs, dropped, agree, agree_n = [], 0, 0, 0

    for i in range(len(bounds) - 1):
        sl = slice(bounds[i], bounds[i + 1])
        m_ok = ok[sl]
        if m_ok.sum() < 2:
            dropped += 1
            continue
        idx = np.flatnonzero(m_ok)                                 # offsets inside the chunk
        meas = meas_report[sl][idx]
        best = float(meas.min())
        best_lib = lib[sl][idx][int(np.argmin(meas))]
        picks = {}
        for m in models:
            r = rank[m][sl][idx]
            if not np.isfinite(r).any():
                picks = {}
                break
            picks[m] = int(np.argmin(np.where(np.isfinite(r), r, np.inf)))
        if not picks:
            dropped += 1
            continue
        # NeuroPress online is scored on the pick the run actually made.
        ex = np.flatnonzero(executed[sl][idx] == 1)
        if ONLINE in picks and len(ex) == 1:
            agree += int(ex[0] == picks[ONLINE])
            agree_n += 1
            picks[ONLINE] = int(ex[0])
        for m, p in picks.items():
            mc = float(meas[p])
            pc = float(report[m][sl][idx][p])
            recs.append({
                "workload": key, "chunk_index": i, "chunk": rows.chunk.to_numpy()[sl][0],
                "model": MODEL_LABEL[m],
                "regret_pct": 100.0 * (mc / best - 1.0),
                "cost_mape_pct": 100.0 * abs(pc - mc) / mc,
                "pick": lib[sl][idx][p], "best": best_lib,
                "pick_cost_ms": mc, "best_cost_ms": best, "pred_cost_ms": pc,
                # Carried per row so a plot cannot silently label the wrong floor.
                "floor_ms": floor,
            })

    n = len(bounds) - 1 - dropped
    note = (f"  pick == its own argmin on {100.0 * agree / agree_n:.1f}% of {agree_n} chunks"
            if agree_n else "")
    print(f"{label:<7} {n} chunks scored"
          + (f", {dropped} dropped (fewer than 2 fully measured candidates)" if dropped else "")
          + note)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="make_table.sh's inputs/ directory")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--hc-suffix", default="", help="read hcompress<suffix>.csv")
    ap.add_argument("--floor", type=float, default=1.0,
                    help="reporting time floor in ms (1 = the deployed policy; "
                         "5 = panels (a)-(c)'s convention)")
    ap.add_argument("--nnwt", default="/u/imuradli/clio-core/context-transport-primitives/"
                                      "src/compress/model/weights/model.nnwt")
    ap.add_argument("--xgb", default="/u/imuradli/NeuroPress/neural_net/weights/xgb_model.pkl")
    a = ap.parse_args()

    nn, xgb = NeuroPressNN(a.nnwt), XGBoostBaseline(a.xgb)
    recs = []
    for key, label in SETTINGS:
        d = os.path.join(a.inputs, key)
        rp, hp = os.path.join(d, "rows.csv"), os.path.join(d, f"hcompress{a.hc_suffix}.csv")
        if not (os.path.exists(rp) and os.path.exists(hp)):
            print(f"{key}: SKIPPED (missing rows.csv or hcompress{a.hc_suffix}.csv)", file=sys.stderr)
            continue
        recs += one_workload(key, label, pd.read_csv(rp), pd.read_csv(hp), nn, xgb, a.floor)
    if not recs:
        print("nothing to score", file=sys.stderr)
        return 1
    out = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6g")
    print(f"\nwrote {len(out)} rows ({out.model.nunique()} models x "
          f"{out.workload.nunique()} workloads) to {a.out}")
    print(f"ranking floor {PRED_TIME_FLOOR_MS:g} ms (deployed), "
          f"reporting floor {a.floor:g} ms, ratio cap {RATIO_CAP:g}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

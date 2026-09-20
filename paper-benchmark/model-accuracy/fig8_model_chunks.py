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
convention. A floor BELOW the ranking floor is refused: `predictions()` hands
back components that the deployed policy has already floored at 1 ms, so only
the measured side of the MAPE would move and the two sides would no longer be
floored alike.

TIES, AND WHY THEY ARE BROKEN AT RANDOM

The deployed policy floors a predicted time at 1 ms and caps a predicted ratio
at 100x. On these 8 MiB chunks most predictions are BELOW that floor and ABOVE
that cap, so many candidates collapse to an identical clamped cost -- measured
at 20.1 of 32 tied on VPIC and 15.6 on AI for HCompress. `np.argmin` returns the
FIRST of a tied set, which makes column order, not the model, decide the pick,
and hands a model whose ranking is anti-correlated with measurement (Spearman
-0.27 on VPIC) a respectable-looking regret.

So a tie is broken by drawing uniformly among the tied candidates, from a
generator seeded by --tie-seed. `n_tied` is written per row, so how much of any
result is tie-determined stays visible instead of being absorbed into the
number. Run a few seeds to get an interval; --tie-break first restores the old
index-order behaviour.

Each (workload, model) draws from its OWN stream. One shared generator would
have made every model's picks depend on how many ties the models drawn before
it happened to have: swapping only the HCompress input file for its `_k7`
variant moved NeuroPress (static)'s picks on 540 of 3471 chunks and its mean
regret from 112.6% to 113.8%, a number no HCompress file should be able to
touch.

THE PICK

  NeuroPress (online)   the configuration the RUN ACTUALLY ADOPTED as primary
                        (`executed`), which is what Figure 8 already scores.
  every other model     argmin of its own predicted deployed cost -- the pick
                        that model would have made, had it been the selector.

A chunk whose adopted configuration was not fully measured therefore has no
measured cost for the online model at all, and is dropped for EVERY model so
that all five stay scored on one chunk set. This is not rare: 53 of WarpX's 500
chunks and 40 of AI's 656. Scoring those chunks on a recomputed argmin instead
-- which is what this script used to do when the executed row fell out of the
measured set -- flattered the deployed model, by 1.6 points of mean regret on
WarpX and 6.3 on AI.

Output, one row per (workload, chunk, model): regret_pct, cost_mape_pct, the
picked and cheapest configuration, and the costs behind both.
"""
from __future__ import annotations

import argparse
import os
import sys

import datetime as _dt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import BW_BYTES_PER_MS, SETTINGS, predictions   # noqa: E402
from models_offline import (NeuroPressNN, XGBoostBaseline,          # noqa: E402
                            PRED_TIME_FLOOR_MS, RATIO_CAP)

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
WORKLOAD_KEYS = [k for k, _ in SETTINGS]


def cost_at(ct, dt, ratio, nbytes, floor):
    """`accuracy_table.cost` at an arbitrary time floor, elementwise."""
    return (np.maximum(floor, ct) + np.maximum(floor, dt)
            + nbytes / (np.minimum(RATIO_CAP, np.maximum(ratio, 0.1)) * BW_BYTES_PER_MS))


def chunk_bounds(chunk: np.ndarray):
    """Start/stop of each run of equal chunk ids; rows.csv is grouped by chunk."""
    starts = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
    return np.r_[starts, len(chunk)]


def tie_streams(seed: int, key: str, models: list):
    """One draw sequence per (workload, model), independent of every other.

    Keyed by position rather than by name so a stream is reproducible from
    --tie-seed alone, and so adding a model cannot renumber the existing ones.
    """
    w = WORKLOAD_KEYS.index(key)
    return {m: np.random.default_rng([seed, w, i]) for i, m in enumerate(models)}


def one_workload(key, label, rows, hc, nn, xgb, floor, tie_seed, tie_break,
                 say=print):
    preds = predictions(rows, hc, nn, xgb, clamp=True)
    models = [m for m in MODEL_LABEL if m in preds]
    rng = tie_streams(tie_seed, key, models)

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
    chunk_id = rows.chunk.to_numpy()

    # Ranking cost (1 ms floor) and reporting cost (--floor) per model.
    rank = {m: preds[m]["cost"] for m in models}                    # already at 1 ms
    report = {m: cost_at(preds[m]["ct_ms"], preds[m]["dt_ms"], preds[m]["ratio"],
                         nbytes, floor) for m in models}

    bounds = chunk_bounds(chunk_id)
    recs, tie_hist = [], {m: [] for m in models}
    thin = unadopted = unranked = 0
    agree = agree_n = sole = 0

    for i in range(len(bounds) - 1):
        sl = slice(bounds[i], bounds[i + 1])
        idx = np.flatnonzero(ok[sl])                               # offsets inside the chunk
        if len(idx) < 2:
            thin += 1
            continue
        # The online model's pick is the run's own, so a chunk whose adopted
        # configuration was never fully measured cannot be scored for it -- and
        # is dropped for all five, or the models would stop sharing a chunk set.
        ex = np.flatnonzero(executed[sl][idx] == 1)
        if ONLINE in rank and len(ex) != 1:
            unadopted += 1
            continue

        tied = {}
        for m in models:
            r = np.where(np.isfinite(rank[m][sl][idx]), rank[m][sl][idx], np.inf)
            if not np.isfinite(r).any():
                tied = {}
                break
            # Every candidate within a float tick of the minimum is genuinely
            # tied: the clamp made them the same number, so nothing in the
            # model distinguishes them.
            tied[m] = np.flatnonzero(r <= r.min() * (1 + 1e-12))
        if not tied:
            unranked += 1
            continue

        meas = meas_report[sl][idx]
        best = float(meas.min())
        chunk_lib = lib[sl][idx]
        best_lib = chunk_lib[int(np.argmin(meas))]
        # Drawn and recorded only once the chunk is known to be scored, so a
        # chunk dropped above cannot enter the tie histogram or consume a draw
        # that a later chunk's pick then depends on.
        picks = {}
        for m, t in tied.items():
            tie_hist[m].append(len(t))
            picks[m] = int(t[0] if tie_break == "first" or len(t) == 1
                           else rng[m].choice(t))
        if ONLINE in picks:
            # A tie is not a disagreement: the model ranks every tied candidate
            # equally, so the question is whether the run's pick is IN that set,
            # not whether a coin landed on it.
            agree_n += 1
            agree += int(ex[0] in set(tied[ONLINE].tolist()))
            sole += int(len(tied[ONLINE]) == 1 and ex[0] == tied[ONLINE][0])
            picks[ONLINE] = int(ex[0])

        for m, p in picks.items():
            mc = float(meas[p])
            pc = float(report[m][sl][idx][p])
            recs.append({
                "workload": key, "chunk_index": i, "chunk": chunk_id[sl][0],
                "model": MODEL_LABEL[m],
                "regret_pct": 100.0 * (mc / best - 1.0),
                "cost_mape_pct": 100.0 * abs(pc - mc) / mc,
                "pick": chunk_lib[p], "best": best_lib,
                "pick_cost_ms": mc, "best_cost_ms": best, "pred_cost_ms": pc,
                "n_tied": len(tied[m]),
                # Carried per row so a plot cannot silently label the wrong floor.
                "floor_ms": floor,
            })

    n_cand = int(np.diff(bounds).max()) if len(bounds) > 1 else 0
    ties = "  ".join(f"{MODEL_LABEL[m]}: {np.mean(v):.1f}"
                     for m, v in tie_hist.items() if v)
    say(f"  mean candidates tied at the argmin (of {n_cand}) -- {ties}")
    drops = [(thin, "fewer than 2 fully measured candidates"),
             (unadopted, "the adopted configuration was not fully measured"),
             (unranked, "no model could rank them")]
    note = (f"  run's pick within its own argmin set on {100.0 * agree / agree_n:.1f}%"
            f" of {agree_n} chunks (its sole argmin on {100.0 * sole / agree_n:.1f}%)"
            if agree_n else "")
    say(f"{label:<7} {len(bounds) - 1 - thin - unadopted - unranked} chunks scored"
          + "".join(f", {n} dropped ({why})" for n, why in drops if n) + note)
    return recs


# ---------------------------------------------------------------------------
# Default model locations, resolved so this runs on a machine that is not the
# one it was written on.
#
#   model.nnwt     ships IN this repo, so it is found relative to THIS file.
#   xgb_model.pkl  does NOT: it belongs to the upstream NeuroPress checkout.
#                  Set NEUROPRESS_DIR (default ~/NeuroPress) or pass --xgb.
#
# Both defaults used to be absolute paths under one author's home directory,
# which breaks silently on any other machine -- argparse happily accepts the
# path and the load fails later, far from the cause.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NNWT = os.path.normpath(os.path.join(
    _HERE, "..", "..", "context-transport-primitives",
    "src", "compress", "model", "weights", "model.nnwt"))
DEFAULT_XGB = os.path.join(
    os.environ.get("NEUROPRESS_DIR", os.path.expanduser("~/NeuroPress")),
    "neural_net", "weights", "xgb_model.pkl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="make_table.sh's inputs/ directory")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--hc-suffix", default="", help="read hcompress<suffix>.csv")
    ap.add_argument("--floor", type=float, default=1.0,
                    help="reporting time floor in ms (1 = the deployed policy; "
                         "5 = panels (a)-(c)'s convention)")
    ap.add_argument("--nnwt", default=DEFAULT_NNWT)
    ap.add_argument("--xgb", default=DEFAULT_XGB)
    ap.add_argument("--tie-break", default="random", choices=["random", "first"],
                    help="how to choose among candidates the clamp made equal")
    ap.add_argument("--tie-seed", type=int, default=0)
    a = ap.parse_args()
    if a.floor < PRED_TIME_FLOOR_MS:
        ap.error(f"--floor below the deployed {PRED_TIME_FLOOR_MS:g} ms would floor the "
                 "two sides of the MAPE differently: the predictions arrive already "
                 "floored at 1 ms, so only the measured side would move")

    # The per-workload lines below -- chunks scored, chunks dropped and why, the
    # mean tied-at-argmin count, and whether the run's own pick was inside its
    # argmin set -- are the audit trail for every number in the CSV, and none of
    # it can be recovered from the CSV. Keep a log beside the output.
    log_path = os.path.splitext(os.path.abspath(a.out))[0] + ".log"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "w")

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        log.write(line + "\n")

    say(f"== fig8_model_chunks.py {_dt.datetime.now().isoformat(timespec='seconds')}")
    say(f"   inputs={a.inputs} hc-suffix={a.hc_suffix or '<none>'} floor={a.floor} "
        f"tie-break={a.tie_break} tie-seed={a.tie_seed}")
    nn, xgb = NeuroPressNN(a.nnwt), XGBoostBaseline(a.xgb)
    recs = []
    for key, label in SETTINGS:
        d = os.path.join(a.inputs, key)
        rp, hp = os.path.join(d, "rows.csv"), os.path.join(d, f"hcompress{a.hc_suffix}.csv")
        if not (os.path.exists(rp) and os.path.exists(hp)):
            print(f"{key}: SKIPPED (missing rows.csv or hcompress{a.hc_suffix}.csv)", file=sys.stderr)
            continue
        recs += one_workload(key, label, pd.read_csv(rp), pd.read_csv(hp), nn, xgb,
                             a.floor, a.tie_seed, a.tie_break, say)
    if not recs:
        print("nothing to score", file=sys.stderr)
        return 1
    out = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6g")
    say(f"\nwrote {len(out)} rows ({out.model.nunique()} models x "
          f"{out.workload.nunique()} workloads) to {a.out}")
    say(f"tie-break {a.tie_break} (seed {a.tie_seed}, one stream per workload and model)")
    say(f"ranking floor {PRED_TIME_FLOOR_MS:g} ms (deployed), "
          f"reporting floor {a.floor:g} ms, ratio cap {RATIO_CAP:g}x")
    say(f"log -> {log_path}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

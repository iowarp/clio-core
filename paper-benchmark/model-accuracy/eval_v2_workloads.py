#!/usr/bin/env python3
"""Score the new-workloads NeuroPress v2 runs (run_v2_all_workloads.sh).

    eval_v2_workloads.py [DATASET ...] [--runs DIR] [--out PREFIX]

Per dataset there are three runs through Clio on the 4-tier cost model:
static, learnexp and exhaustive. The exhaustive run measured all 44
compressing settings on every chunk (v2_measured.csv); storing raw is the
45th, cost = bytes / tier bandwidth. That is the per-chunk truth.

Per chunk and mode:
  pick / stored / best   v2's pick, the setting finally stored (exploration
                         can replace the pick), the cheapest by the truth
  regret_pick/_stored    truth cost of pick / stored / truth best cost - 1
  ape_ct / ape_dt / ape_r  |predicted - measured| / measured for the pick,
                         prediction made before learning from that chunk,
                         measurement = the same run's own; picks of store, and
                         picks that did not shrink the chunk (stored raw, so
                         unmeasured), are left out and counted
Cost = compress ms + decompress ms + bytes / (ratio * tier bandwidth); a
setting with ratio <= 1 is stored raw: compress ms + bytes / bandwidth.

Per dataset and mode: wall write (stage + compress) and read (get +
decompress) time from stdout.log, summed compress / decompress / explore /
learning ms from phases.csv, overall ratio, MAPE, mean and median regret,
how often the best was stored, and the total truth cost against the
per-chunk best (oracle) and the best single setting for the whole dataset.
Writes PREFIX_chunks.csv and PREFIX_summary.csv.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-work/runs"
# Every workload run in full (no sampling): figures/new-workloads/nn-v2 shows
# only these. Of the new (non-simulation) workloads only gnn-igbh and
# graph-orkut-full are kept (user, 2026-10-05: the others, their runs,
# searches, data and figures were removed; list in
# /mnt/nvme0/v2-work/removed_new_workloads_2026-10-05.log).
FULL_WORKLOADS = [
    "nyx-full", "nyx-multiphase-50g", "gnn-igbh", "graph-orkut-full",
    "ref-lammps-b70-2000", "ref-vpic-126-2000", "vpic-slabs", "ref-warpx-64x64x512-2000"]
WEIGHTS_JSON = os.path.join(HERE, "..", "..", "context-transport-primitives", "src",
                            "compress", "model", "weights", "v2", "model_v2.json")
MODES = ("static", "learn", "learnexp", "exhaustive")


def workload_size(ds):
    """(chunks, GiB) of a staged workload (/mnt/nvme0/v2-work/<ds>/fields)."""
    d = os.path.join(os.path.dirname(RUNS), ds, "fields")
    names = os.listdir(d)
    return len(names), sum(os.path.getsize(os.path.join(d, n)) for n in names) / 2**30


def workload_label(ds, sep="  "):
    """'<ds>  (N chunks, X.X GiB)' for plot labels."""
    n, gib = workload_size(ds)
    return f"{ds}{sep}({n} chunks, {gib:.1f} GiB)"


def total_label(dss):
    """'N workloads, C chunks, X.X GiB in total' over the given workloads."""
    sizes = [workload_size(d) for d in dss]
    return (f"{len(dss)} workloads, {sum(n for n, _ in sizes):,} chunks, "
            f"{sum(g for _, g in sizes):.1f} GiB in total")


def run_finished(run):
    """True when a run's stdout.log holds its final timing line."""
    p = os.path.join(run, "stdout.log")
    return os.path.exists(p) and "stage+compress" in open(p).read()


def settings_list():
    """Setting names in model order and the index of 'store'."""
    names = json.load(open(WEIGHTS_JSON))["outputs"]["settings"]
    store = [i for i, s in enumerate(names) if s.split()[0] == "store"]
    assert len(store) == 1, "expected exactly one store setting"
    return names, store[0]


# The settings that the best single codec and the oracle select from: all 45
# again (user, 2026-10-05, "return add 45 configs of codecs"; it was 33
# earlier the same day: one cascaded / gdeflate variant, at most 3 per codec).
# NeuroPress selects among all 45 as well. Put indices here to leave settings
# out of the best single codec and the oracle.
DROPPED_SETTINGS = ()


def candidate_settings(n_settings=45):
    """The setting indices that the best single codec and the oracle select from.

    @param n_settings number of settings in the model
    @return list of indices, without DROPPED_SETTINGS
    """
    return [s for s in range(n_settings) if s not in DROPPED_SETTINGS]


def for_selection(cost):
    """A copy of a cost matrix with the dropped settings at +inf.

    Use it only for the argmin of the best single codec or the oracle, never to
    score a pick: a pick of a dropped setting keeps its measured cost.

    @param cost cost[chunk, setting] (or one chunk's cost[setting])
    @return the copy, dropped settings' columns set to +inf
    """
    c = np.array(cost, dtype=float)
    c[..., list(DROPPED_SETTINGS)] = np.inf
    return c


def raw_primary(m):
    """Rows logging a primary that did not shrink its chunk and was stored
    raw (no measurement of its setting: ct 0 when exploration deferred the
    store, ratio <= 1 otherwise)."""
    return (m.role == "primary") & ((m.comp_ms <= 0) | (m.ratio <= 1))


def load_truth(run, n_settings, store, w=(1.0, 1.0, 1.0), bw=None):
    """Chunk x setting truth cost from an exhaustive run.

    A setting that does not shrink the chunk (ratio <= 1) is stored raw by
    Clio, so its cost is its compress time plus raw I/O; storing raw outright
    costs the I/O alone. Each term carries its weight:
    w_ct * ct + w_dt * dt + w_io * bytes / (ratio * bw).

    @param run         exhaustive run directory
    @param n_settings  number of settings (45)
    @param store       index of the store setting
    @param w           (w_ct, w_dt, w_io); (1, 1, 1) is the balanced model
    @param bw          one bandwidth in bytes per ms for every chunk, or None
                       for the tier each chunk was written to
    @return (blobs, cost[chunk, setting]) with NaN where unmeasured
    """
    pred = pd.read_csv(os.path.join(run, "v2_pred.csv"))
    meas = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    blobs = pred.blob.tolist()
    idx = {b: i for i, b in enumerate(blobs)}
    cost = np.full((len(blobs), n_settings), np.nan)
    m = meas[~raw_primary(meas)].drop_duplicates(["blob", "setting"])
    # A measurement whose round trip failed (no decompress time) is not
    # truth; its chunk then counts as incomplete.
    failed = m.decomp_ms <= 0
    if failed.any():
        print(f"  {os.path.basename(run)}: dropped {int(failed.sum())} measurement(s) "
              f"with a failed decompress: "
              f"{m[failed][['setting', 'spec', 'ratio']].to_dict('records')[:3]}")
    m = m[~failed]
    w_ct, w_dt, w_io = w
    m_bw = m.tier_bw.to_numpy(float) if bw is None else np.full(len(m), float(bw))
    raw_io = pred.set_index("blob").bytes.reindex(m.blob).to_numpy(float) / m_bw
    kept = m.ratio.to_numpy() > 1
    c = np.where(kept, w_ct * m.comp_ms + w_dt * m.decomp_ms + w_io * raw_io / m.ratio,
                 w_ct * m.comp_ms + w_io * raw_io)
    cost[m.blob.map(idx).to_numpy(), m.setting.to_numpy()] = c
    p_bw = pred.tier_bw.to_numpy(float) if bw is None else float(bw)
    cost[:, store] = w_io * pred.bytes.to_numpy(float) / p_bw
    return blobs, cost


def run_chunks(run, store):
    """Per-chunk pick, stored setting and the pick's predicted vs measured.

    @param run    run directory (any mode)
    @param store  index of the store setting
    @return DataFrame in stream order
    """
    pred = pd.read_csv(os.path.join(run, "v2_pred.csv"))
    meas = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    prim = meas[meas.role == "primary"].drop_duplicates("blob")
    raw = set(prim.blob[raw_primary(prim)])
    prim = prim.set_index("blob")
    adopted = meas[meas.adopted == 1].drop_duplicates("blob", keep="last")
    adopted = adopted.set_index("blob").setting
    pick = prim.setting.reindex(pred.blob).fillna(store).astype(int).to_numpy()
    stored = adopted.reindex(pred.blob).fillna(store).astype(int).to_numpy()
    r = np.arange(len(pred))
    out = pd.DataFrame({"i": r, "blob": pred.blob, "bytes": pred.bytes,
                        "tier_bw": pred.tier_bw, "updates": pred.updates,
                        "pick": pick, "stored": stored,
                        "pick_raw": pred.blob.isin(raw).to_numpy(),
                        # constant-valued chunk: MAD feature at its floor
                        "const": pred.x2.to_numpy() <= -11.9})
    for k, col in (("ct", "comp_ms"), ("dt", "decomp_ms"), ("r", "ratio")):
        p = np.array([pred[f"{k}{s}"].iat[i] for i, s in zip(r, pick)], float)
        m = prim[col].reindex(pred.blob).to_numpy(float).copy()
        m[(pick == store) | out.pick_raw.to_numpy()] = np.nan
        out[f"{k}_pred"], out[f"{k}_meas"] = p, m
        out[f"ape_{k}"] = 100 * np.abs(p - m) / m
    return out


def wall_times(run):
    """(write s, read s) from stdout.log: stage+compress and get+decompress."""
    text = open(os.path.join(run, "stdout.log")).read()
    w = re.search(r"stage\+compress ([0-9.]+) s", text)
    r = re.search(r"READ 1/1.*?get\+decompress: ([0-9.]+) ms", text, re.S)
    return (float(w.group(1)) if w else np.nan,
            float(r.group(1)) / 1e3 if r else np.nan)


def phase_sums(run):
    """Summed per-chunk phase times (ms) and the final stored bytes."""
    ph = pd.read_csv(os.path.join(run, "phases.csv"))
    w, rd = ph[ph.path == "write"], ph[ph.path == "read"]
    return {"compress_ms": w.compress_ms.sum(), "decompress_ms": rd.decompress_ms.sum(),
            "explore_ms": w.explore_ms.sum(), "learn_ms": w.sgd_ms.sum(),
            "select_ms": w.nn_ms.sum(), "convert_ms": w.convert_ms.sum(),
            "ratio": rd.chunk_bytes.sum() / rd.stored_bytes.sum()}


def score_mode(ds, mode, run, names, store, blobs, truth):
    """Per-chunk scores and one summary row for one run."""
    d = run_chunks(run, store)
    row_of = {b: i for i, b in enumerate(blobs)}
    rows = d.blob.map(row_of)
    d = d[rows.notna()].reset_index(drop=True)
    t = truth[rows.dropna().astype(int).to_numpy()]
    r = np.arange(len(d))
    best = np.nanargmin(for_selection(t), axis=1)   # oracle: candidates only
    bc = t[r, best]
    d["best"] = best
    d["regret_pick"] = 100 * (t[r, d.pick] / bc - 1)
    d["regret_stored"] = 100 * (t[r, d.stored] / bc - 1)
    # Best single setting over the chunks where every setting was measured.
    complete = ~np.isnan(t).any(axis=1)
    single = for_selection(t[complete]).sum(axis=0)
    s_best = int(np.argmin(single))
    d["cost_stored"], d["cost_best"] = t[r, d.stored], bc
    d["cost_single"] = t[r, s_best]
    w, rd = wall_times(run)
    total = np.nansum(t[r, d.stored])
    summary = {"dataset": ds, "mode": mode, "chunks": len(d),
               "write_s": w, "read_s": rd, **phase_sums(run),
               "mape_ct": d.ape_ct.mean(), "mape_dt": d.ape_dt.mean(),
               "mape_r": d.ape_r.mean(), "medape_r": d.ape_r.median(),
               "picks_stored_raw": int(d.pick_raw.sum()),
               "regret_mean": d.regret_stored.mean(),
               "regret_median": d.regret_stored.median(),
               "regret_pick_mean": d.regret_pick.mean(),
               "best_stored_pct": 100 * np.mean(d.stored == d.best),
               "cost_ms": total, "oracle_cost_ms": bc.sum(),
               "over_oracle_pct": 100 * (total / bc.sum() - 1),
               "best_single": names[s_best],
               "complete_chunks": int(complete.sum()),
               "best_single_over_oracle_pct":
                   100 * (single[s_best] / bc[complete].sum() - 1),
               "vs_best_single_pct":
                   100 * (1 - t[r, d.stored][complete].sum() / single[s_best]),
               "updates_end": int(d.updates.iloc[-1]) if len(d) else 0}
    for k in ("pick", "stored", "best"):
        d[k] = [names[s] for s in d[k]]
    return d.assign(dataset=ds, mode=mode), summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=os.path.join(RUNS, "v2_workloads"))
    a = ap.parse_args()
    names, store = settings_list()
    datasets = a.datasets or sorted(
        os.path.basename(p)[:-len("_exhaustive")]
        for p in glob.glob(os.path.join(a.runs, "*_exhaustive")))
    chunks, summary = [], []
    for ds in datasets:
        ex = os.path.join(a.runs, f"{ds}_exhaustive")
        if not os.path.exists(os.path.join(ex, "stdout.log")):
            print(f"skip {ds}: no exhaustive run")
            continue
        blobs, truth = load_truth(ex, len(names), store)
        for mode in MODES:
            run = os.path.join(a.runs, f"{ds}_{mode}")
            if not os.path.exists(os.path.join(run, "stdout.log")):
                continue
            d, s = score_mode(ds, mode, run, names, store, blobs, truth)
            chunks.append(d)
            summary.append(s)
    s = pd.DataFrame(summary)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(s.round(2).to_string(index=False))
    s.to_csv(a.out + "_summary.csv", index=False)
    pd.concat(chunks).to_csv(a.out + "_chunks.csv", index=False)
    print("wrote", a.out + "_{summary,chunks}.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Selective-lossy study: does NeuroPress's per-chunk lossy/lossless choice
beat lossless-only and uniform lossy on PERFORMANCE at equal ACHIEVED error?

  selective_analyze.py [--traces DIR] [--inmem DIR] [--out DIR]

Inputs
  <traces>/<wl>-eb<eb>/explore.csv   selective_traces.sh: every chunk x all 32
                                     actions, isolated timings, measured error
  <inmem>/<wl>[-eb<eb>]/runs/...     inmem.sh: measured write + timed reads

Trace side (per workload and bound). Every strategy is a per-chunk action and
is scored on the same measured numbers with the BALANCED cost model NeuroPress
ranks with: sum(max(ct,1ms) + max(dt,1ms) + B / (ratio * 1.2 GB/s)). It is
reported on the isolated timings and on "smoothed" ones (each action's median
within its field), which no per-chunk choice can exploit as noise. Error is
aggregated correctly: RMSE element-weighted, max as a max, PSNR per chunk.

Run side. Each measured run's per-chunk quantize flag is recovered by matching
its stored size against the trace's exact payload of that chunk and codec,
quantized (+56 B header) or not (+24 B), and joined to the trace's measured
error of that chunk at that bound. That join is exact because a chunk's error
depends only on whether it was quantized, which is checked per trace. The
result is MEASURED end-to-end time against ACHIEVED error, for NeuroPress's
real learning-mode decisions.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

import numpy as np
import pandas as pd

import summarize_inmem as si

BW = 1.2e6          # B/ms, the RAM tier's measured per-request speed (inmem.sh NP_BW)
FLOOR_MS = 1.0      # the balanced cost model's time floor (RankKernel min_time_ms)
DAMAGED_DB = 40.0   # a quantized chunk below this PSNR (own range) counts as damaged
SEEDS = 200
HDR_Q, HDR_LL = 56, 24


def load_trace(path):
    """Per-chunk x action matrices, NeuroPress's pick, and measured quality."""
    e = pd.read_csv(path)
    e["cfg"] = (e.lib_name.str.replace("nvcomp-", "") + "|q" + e.quantize.astype(str)
                + "|s" + (e.shuffle > 0).astype(int).astype(str))
    full = e.groupby("blob").cfg.nunique()
    e = e[e.blob.isin(full[full == 32].index)]
    chunks, cfgs = sorted(e.blob.unique()), sorted(e.cfg.unique())
    piv = {k: e.pivot_table(index="blob", columns="cfg", values=k).loc[chunks, cfgs].to_numpy()
           for k in ("ratio", "ct_ms", "dt_ms")}
    first = e.drop_duplicates("blob").set_index("blob").loc[chunks]
    prim = e[e.role == "primary"].drop_duplicates("blob").set_index("blob").loc[chunks]
    q = e[e.quantize == 1]
    # Quality from measured rows only: a quantized action whose output was not
    # worth keeping is stored raw and never decoded (quality_measured 0). The
    # chunk's other quantized actions carry the same error.
    qm = q[q.quality_measured == 1]
    spread = qm.groupby("blob").meas_rmse.agg(lambda s: s.max() - s.min()).max()
    ql = qm.drop_duplicates("blob").set_index("blob").reindex(chunks)
    return {"chunks": chunks, "idx": {c: i for i, c in enumerate(chunks)}, "cfgs": cfgs,
            "file": np.array([c.rsplit("/", 1)[0] for c in chunks]),
            "B": first.chunk_bytes.to_numpy().astype(float), "n": first.chunk_bytes.to_numpy() / 4.0,
            "quant": np.array([c.split("|")[1] == "q1" for c in cfgs]),
            "lib": np.array([c.split("|")[0] for c in cfgs]),
            "np": np.array([cfgs.index(c) for c in prim.cfg]),
            "feat": first[["entropy", "mad", "second_deriv"]].reset_index(drop=True),
            "field": pd.Series(chunks).str.extract(r"(?:fab\d+_comp\d+_)?([^/]+)/chunk")[0].to_numpy(),
            "rmse": ql.meas_rmse.to_numpy(), "maxe": ql.meas_max_error.to_numpy(),
            "psnr": ql.meas_psnr_db.to_numpy(), "ssim": ql.meas_ssim.to_numpy(),
            "measured": bool(ql.meas_rmse.notna().all()), "spread": spread, **piv}


def add_ranges(d, traces_dir, wl):
    """Attach value ranges (field_ranges.py) and derive each chunk's PSNR from
    its measured RMSE and its OWN range. The runtime's meas_psnr_db is not
    used: it reports 120 dB whenever the absolute RMSE is below 1e-10, which
    hides chunks whose values are that small (12 on Nyx). A chunk that
    quantizes exactly (RMSE 0) or is constant (range 0) is infinite.
    """
    fp = os.path.join(traces_dir, f"{wl}_field_ranges.csv")
    cp = os.path.join(traces_dir, f"{wl}_chunk_ranges.csv")
    if os.path.exists(fp):
        fr = pd.read_csv(fp).set_index("file")
        d["ranges"] = (fr["max"] - fr["min"]).astype(float)
    if os.path.exists(cp):
        cr = pd.read_csv(cp).set_index("blob").reindex(d["chunks"])
        rng = (cr["max"] - cr["min"]).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            d["psnr"] = np.where((d["rmse"] > 0) & (rng > 0), 20 * np.log10(rng / d["rmse"]), np.inf)
        d["crange"] = rng
    return d


def smooth(d):
    """A copy whose times are each action's median within the chunk's field."""
    s = dict(d)
    for k in ("ct_ms", "dt_ms"):
        m = d[k].copy()
        for f in np.unique(d["field"]):
            r = d["field"] == f
            m[r] = np.median(d[k][r], axis=0)
        s[k] = m
    return s


def cost_matrix(d, bw=BW, floor=FLOOR_MS):
    """Per chunk x action balanced cost, ms."""
    return (np.maximum(d["ct_ms"], floor) + np.maximum(d["dt_ms"], floor)
            + d["B"][:, None] / (d["ratio"] * bw))


def field_psnr(d, q):
    """Per-field PSNR, the way compression papers report it: per dump file
    (one field at one timestep), MSE over the file against the file's own
    value range, lossless files infinite; then the median over timesteps.

    :return: {field: median PSNR dB}, or {} without the file ranges
    """
    if "ranges" not in d:
        return {}
    mse = pd.Series(np.where(q, d["rmse"] ** 2, 0.0)).groupby(d["file"]).mean()
    rng = d["ranges"].reindex(mse.index)
    with np.errstate(divide="ignore"):
        psnr = np.where(mse > 0, 10 * np.log10(rng ** 2 / mse), np.inf)
    field = pd.Series(mse.index).str.extract(r"(?:fab\d+_comp\d+_)?([^/]+)$")[0].to_numpy()
    med = pd.Series(psnr).groupby(field).median().to_dict()
    med["__worst_file__"] = float(np.min(psnr))
    return med


def quality(d, q):
    """Dataset-level achieved error of a per-chunk quantize mask."""
    n = d["n"]
    pq = np.where(q, d["psnr"], np.inf)
    fp = field_psnr(d, q)
    wf = fp.pop("__worst_file__", np.nan)
    worst = min(fp.values()) if fp else np.nan
    extra = {"worst_field_psnr": None if not fp or np.isinf(worst) else float(worst),
             # the lowest PSNR of any field at any timestep: the median over
             # timesteps above can hide a bad frame
             "worst_file_psnr": None if not fp or np.isinf(wf) else float(wf),
             "fields_lost": int(sum(v < 20 for v in fp.values())) if fp else None}
    return {**extra,"rmse": float(np.sqrt((n * q * d["rmse"] ** 2).sum() / n.sum())),
            "max_err": float((d["maxe"] * q).max()),
            "worst_psnr": None if np.isinf(pq.min()) else float(pq.min()),
            "damaged": int((pq < DAMAGED_DB).sum()),
            "ssim": float((n * np.where(q, d["ssim"], 1.0)).sum() / n.sum()),
            "lossy": float(q.mean())}


def score(d, pick):
    """Performance and achieved error of one strategy (`pick` = action per chunk)."""
    i = np.arange(len(pick))
    ratio, ct, dt, B = d["ratio"][i, pick], d["ct_ms"][i, pick], d["dt_ms"][i, pick], d["B"]
    out = {"cost_s": float(cost_matrix(d)[i, pick].sum() / 1e3),
           "cost_raw_s": float((ct + dt + B / (ratio * BW)).sum() / 1e3),
           "comp_GBs": float(B.sum() / ct.sum() / 1e6), "decomp_GBs": float(B.sum() / dt.sum() / 1e6),
           "CR": float(B.sum() / (B / ratio).sum())}
    out.update(quality(d, d["quant"][pick]))
    return out


def fixed(d, mask):
    """The single action within `mask` with the lowest total balanced cost."""
    tot = cost_matrix(d).sum(axis=0)
    tot[~mask] = np.inf
    return int(np.argmin(tot))


def strategies(d):
    """Lossless-only, uniform lossy, NeuroPress, the oracles and the controls."""
    C, n = cost_matrix(d), len(d["chunks"])
    ll, ly = fixed(d, ~d["quant"]), fixed(d, d["quant"])
    npq = d["quant"][d["np"]]
    s = {"Lossless only, best fixed": np.full(n, ll),
         "Lossless only, per-chunk best codec": np.where(~d["quant"][None, :], C, np.inf).argmin(axis=1),
         "Uniform lossy, best fixed": np.full(n, ly),
         "Uniform lossy, per-chunk best codec": np.where(d["quant"][None, :], C, np.inf).argmin(axis=1),
         "NeuroPress (trace picks)": d["np"],
         "NeuroPress flags, fixed codecs": np.where(npq, ly, ll),
         "Per-chunk best of all 32 actions": C.argmin(axis=1),
         "Uniform lossy, damaged chunks kept lossless": np.where(d["psnr"] < DAMAGED_DB, ll, ly)}
    return s, ll, ly


def frontiers(d, ll, ly):
    """Mixes of the two fixed configs: random (no data awareness) and two
    greedy data-aware orders -- chunks quantized by balanced-cost cut per unit
    of squared error added (greedy_cost: no choice of chunks does better on
    cost vs RMSE), or per unit of SSIM lost (greedy_ssim: protects the chunks
    quantizing flattens, which RMSE barely sees)."""
    C = cost_matrix(d)
    cut, sse, n = C[:, ll] - C[:, ly], d["n"] * d["rmse"] ** 2, len(d["chunks"])
    rng, rows = np.random.default_rng(0), []
    for p in np.linspace(0, 1, 21):
        acc = []
        for _ in range(SEEDS):
            m = np.zeros(n, bool)
            m[rng.choice(n, int(round(p * n)), replace=False)] = True
            acc.append(score(d, np.where(m, ly, ll)))
        rows.append({"policy": "random", "p": p, **pd.DataFrame(acc).mean(numeric_only=True).to_dict()})
    loss = d["n"] * (1.0 - d["ssim"])
    for name, harm in (("greedy_cost", sse), ("greedy_ssim", loss)):
        order = np.lexsort((-cut, -cut / np.maximum(harm, 1e-30)))
        for p in np.linspace(0, 1, 51):
            m = np.zeros(n, bool)
            m[order[:int(round(p * n))]] = True
            rows.append({"policy": name, "p": p, **score(d, np.where(m, ly, ll))})
    return pd.DataFrame(rows)


def selectivity(d):
    """How NeuroPress's lossless chunks differ from its quantized ones."""
    npq = d["quant"][d["np"]]
    df = d["feat"].copy()
    df["np_lossy"], df["psnr_q"], df["rmse_q"], df["field"] = npq, d["psnr"], d["rmse"], d["field"]
    dam = d["psnr"] < DAMAGED_DB
    return {"np_lossy": float(npq.mean()),
            "lossless_by_field": {k: float(1 - v) for k, v in df.groupby("field").np_lossy.mean().items()},
            "median_features": df.groupby("np_lossy")[["entropy", "mad", "second_deriv", "psnr_q", "rmse_q"]]
                                 .median().rename(index={True: "quantized", False: "lossless"}).to_dict("index"),
            "damaged": int(dam.sum()), "damaged_kept_lossless": int((dam & ~npq).sum()),
            "damaged_by_field": {k: int(v) for k, v in pd.Series(d["field"][dam]).value_counts().items()},
            "lossless_zero_error": int(((d["rmse"] == 0) & ~npq).sum()), "lossless": int((~npq).sum())}


def run_flags(run_dir, d):
    """Per-chunk quantize flags of one measured run, and how many were resolved."""
    comp = open(os.path.join(run_dir, "compose.yaml")).read()
    rows = list(csv.DictReader(open(os.path.join(run_dir, "blobs.csv"))))
    q = np.zeros(len(d["chunks"]), bool)
    # A FIXED configuration names its codec; its flag is the config's. Anything
    # else -- NeuroPress with or without online learning, or no compression --
    # is recovered chunk by chunk from the stored sizes below.
    static = re.search(r"neuropress_static_quantize: (\w+)", comp)
    if "neuropress_static_lib:" in comp:
        q[:] = bool(static and static.group(1) == "true")
        return q, len(rows), 0
    resolved = unresolved = 0
    for r in rows:
        i = d["idx"].get(r["blob"])
        if i is None:
            continue
        lib = r["codec"].replace("nvcomp-", "")
        if lib not in set(d["lib"]):  # stored raw ("not beneficial"): exact, so lossless
            q[i] = False
            resolved += 1
            continue
        cand = [(abs(int(r["stored"]) - (round(d["B"][i] / d["ratio"][i, k])
                                        + (HDR_Q if d["quant"][k] else HDR_LL))), d["quant"][k])
                for k in np.where(d["lib"] == lib)[0]]
        best = min(cand)
        q[i] = best[1]
        resolved += best[0] <= 8
        unresolved += best[0] > 8
    return q, resolved, unresolved


def measured_runs(inmem, traces):
    """Every measured in-memory run with a trace for its data and bound."""
    out = []
    for run_json in sorted(glob.glob(os.path.join(inmem, "*", "run.json"))):
        base = os.path.dirname(run_json)
        meta = json.load(open(run_json))
        rel = float(meta.get("rel", 0) or 0)
        key = bound_key(meta["workload"], "rel" if rel > 0 else "eb", rel if rel > 0 else meta["eb"])
        # -replay-<policy> / -np-<variant>: one arm, named by the suffix;
        # -ratio / -balanced: a cost-model campaign, whose arms keep their names.
        m = re.fullmatch(r"(nyx|vpic|lammps|warpx)(-(?:eb|rel)[0-9e.-]+?)?(-(replay|np)-(\w+)|-ratio|-balanced)?",
                         os.path.basename(base))
        if key not in traces or not m:
            continue
        d = traces[key]
        for rd in sorted(glob.glob(os.path.join(base, "runs", "*", "r*"))):
            x = si.parse(os.path.join(rd, "stdout.log"))
            # External codecs quantize their own way; the trace does not cover them.
            ext = re.search(r"neuropress_static_lib: (cusz|cuszp|ndzip)\b",
                            open(os.path.join(rd, "compose.yaml")).read())
            if not x or not x[4] or ext:
                continue
            q, ok, bad = run_flags(rd, d)
            w, reads = x[0], x[2]
            out.append({"workload": meta["workload"], "eb": float(meta["eb"]), "rel": rel,
                        "arm": f"{m.group(4)}_{m.group(5)}" if m.group(4) else rd.split(os.sep)[-2],
                        "rep": os.path.basename(rd),
                        "ratio": x[1], "write_s": w, "read_s": st.median(reads),
                        "e2e_1w1r_s": w + st.median(reads), "e2e_1w5r_s": w + 5 * st.median(reads),
                        "bound": x[5], "flags_resolved": ok, "flags_unresolved": bad,
                        **quality(d, q)})
    return pd.DataFrame(out)


WIRE = {"lz4": 11, "snappy": 12, "zstd": 13, "gdeflate": 14, "deflate": 15, "ans": 16,
        "cascaded": 21, "bitcomp": 22}


def write_replays(path, d, strat):
    """Replay decision files (CLIO_NEUROPRESS_REPLAY_CHOICES) for two ideal
    per-chunk policies, so the in-memory runs can MEASURE them with no
    selection or learning cost:
      oracle   each chunk's best action on the balanced cost (smoothed times)
      protect  uniform lossy, damaged chunks kept lossless (only if any)
      floor40  uniform lossy, chunks whose ANALYTICAL PSNR is under 40 dB kept
               lossless (decidable before compressing)
    """
    e = pd.read_csv(path, usecols=["blob", "lib_name", "quantize", "shuffle", "preset"])
    e["cfg"] = (e.lib_name.str.replace("nvcomp-", "") + "|q" + e.quantize.astype(str)
                + "|s" + (e.shuffle > 0).astype(int).astype(str))
    e = e.drop_duplicates(["blob", "cfg"]).set_index(["blob", "cfg"])
    todo = {"oracle": strat["Per-chunk best of all 32 actions"]}
    if (d["psnr"] < DAMAGED_DB).any():
        todo["protect"] = strat["Uniform lossy, damaged chunks kept lossless"]
    # floor40: what a quality floor can decide BEFORE compressing -- the
    # analytical PSNR from the chunk's range and the bound (the selection
    # log's psnr_db), not the measured one.
    ana = (pd.read_csv(path, usecols=["blob", "quantize", "psnr_db"]).query("quantize == 1")
             .groupby("blob").psnr_db.median().reindex(d["chunks"]).to_numpy())
    ana = np.where(ana < 0, 120.0, ana)   # -1 = constant chunk: quantizes exactly
    todo["floor40"] = np.where(ana < DAMAGED_DB, strat["Lossless only, best fixed"],
                               strat["Uniform lossy, best fixed"])
    for name, pick in todo.items():
        rows = []
        for c, k in zip(d["chunks"], pick):
            r = e.loc[(c, d["cfgs"][k])]
            rows.append({"blob": c, "role": "primary", "wire_lib": WIRE[d["lib"][k]],
                         "quantize": int(r.quantize), "shuffle": int(r.shuffle), "preset": int(r.preset)})
        out = os.path.join(os.path.dirname(path), f"replay_{name}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"   replay {name}: {len(rows)} chunks, {np.mean([r['quantize'] for r in rows]):.1%} quantized -> {out}")


def json_safe(x):
    """Nested dicts/lists with NaN and infinities replaced by None."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def bound_key(wl, kind, value):
    """Trace/run key: (workload, absolute bound) or (workload, "rel<eps>")."""
    return (wl, float(value)) if kind == "eb" else (wl, f"rel{float(value):g}")


def records(df):
    """DataFrame rows as JSON-safe dicts (NaN -> None)."""
    return df.round(6).astype(object).where(df.notna(), None).to_dict("records")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(here, "..", "results", "selective"))
    ap.add_argument("--inmem", default=os.path.join(here, "..", "results", "inmem"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-replays", action="store_true",
                    help="write replay_{oracle,protect}.csv beside each trace")
    a = ap.parse_args()
    out = a.out or a.traces
    traces, report = {}, {"traces": {}}
    for p in sorted(glob.glob(os.path.join(a.traces, "*-*", "explore.csv"))):
        m = re.match(r"(\w+)-(eb|rel)(.+)", os.path.basename(os.path.dirname(p)))
        if not m:
            continue
        wl, kind, eb = m.groups()
        d = add_ranges(load_trace(p), a.traces, wl)
        traces[bound_key(wl, kind, eb)] = d
        sm = smooth(d)
        s, ll, ly = strategies(sm)
        tab = {k: {**score(sm, v), "cost_isolated_s": score(d, v)["cost_s"]} for k, v in s.items()}
        report["traces"][f"{wl}-{kind}{eb}"] = {
            "chunks": len(d["chunks"]), "quality_measured": d["measured"], "error_spread": float(d["spread"]),
            "fixed_lossless": d["cfgs"][ll], "fixed_lossy": d["cfgs"][ly], "strategies": tab,
            "selectivity": selectivity(d),
            "frontiers": records(frontiers(sm, ll, ly).drop(columns=["worst_psnr"], errors="ignore"))}
        print(f"\n== {wl} eb {eb}: {len(d['chunks'])} chunks; error spread across quantized "
              f"actions {d['spread']:.2e}; fixed lossless {d['cfgs'][ll]}, fixed lossy {d['cfgs'][ly]}")
        print(pd.DataFrame(tab).T[["lossy", "cost_s", "cost_isolated_s", "CR", "rmse", "max_err",
                                   "damaged", "ssim"]].to_string(float_format=lambda v: f"{v:.4g}"))
        if a.write_replays:
            write_replays(p, d, s)
    runs = measured_runs(a.inmem, traces)
    if len(runs):
        report["runs"] = records(runs)
        runs.to_csv(os.path.join(out, "measured_runs.csv"), index=False)
        print("\n== measured in-memory runs joined to trace error")
        print(runs.groupby(["workload", "eb", "arm"])[["lossy", "e2e_1w1r_s", "e2e_1w5r_s", "rmse", "ssim",
                                                      "max_err", "flags_unresolved"]]
              .median().to_string(float_format=lambda v: f"{v:.4g}"))
    json.dump(json_safe(report), open(os.path.join(out, "selective.json"), "w"), allow_nan=False)
    print(f"-> {os.path.join(out, 'selective.json')}")


if __name__ == "__main__":
    main()

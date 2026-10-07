#!/usr/bin/env python3
"""Per-chunk ground truth of every workload: one full Clio run per candidate setting (every chunk
with that one setting; run_ground_truth.sh), so each chunk's real compress, write, read and
decompress times and stored bytes are measured for every setting. The workloads are
deterministic, so this is measured once; the best single codec and the per-chunk oracle under any
cost model, tier speed or read count then come from these tables.

    ground_truth.py plan [--reads R]      write GT_DIR/plan.txt (run order) and print the estimate
    ground_truth.py collect               write the tables of the finished runs (no GPU work)

Run order (plan): every workload's settings ranked by how often the exhaustive search makes them
the best one (per chunk and as the single codec) under a set of objectives (end-to-end time at
0.25 / 0.5 / 1.2 / 12 GB/s and the final cost models at 0.5 / 1 GB/s); the workloads take turns,
so that each workload has its most relevant settings first. A setting that is never the best
comes last; 'store' (no compression, the baseline) comes after the first five.

Collect, per workload, into GT_DIR/<workload>/:
  chunks_<workload>.csv    one row per chunk and finished setting: raw and stored bytes, the
                           write's compress / I/O / wall ms and the mean over the reads of the
                           decompress / I/O / wall ms (phases.csv)
  settings_<workload>.csv  one row per finished setting: application time of the run (1 write +
                           R reads), write and mean read-pass time, the time for 1 write + 10
                           reads (write + 10 x mean pass), ratio, read I/O s per stored GB,
                           k-means s, bit-exact and timed checks
  summary_<workload>.txt   the best single codec by measured time, and the per-chunk oracle
                           (each chunk's lowest write wall + 10 x read wall ms) over the finished
                           settings
  exhaustive/v2_measured.csv          the whole exhaustive search (every chunk x setting:
                                      compress ms, decompress ms, ratio, cost)
  oracle_per_chunk_<workload>.csv     from it, per chunk: the best setting and its value under
                                      every objective of the plan (cost models, time at 4 speeds)
  exhaustive_check_<workload>.txt     the new search against the stored baseline: ratios must be
                                      identical (deterministic workloads), times within noise
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import compare_parallel_runs as cp
import eval_v2_workloads as ev
import final_config as fc
import replay_learning as rl

GT_DIR = os.environ.get("GT_DIR", "/mnt/nvme0/v2-work/ground-truth")
# tests: GT_WORKLOADS="Nyx" limits the workloads, GT_DATASETS="Nyx=nyx-multiphase-1g" swaps datasets
DATASETS = dict(x.split("=", 1) for x in os.environ.get("GT_DATASETS", "").split() if "=" in x)


def workloads():
    """@return the workloads of this run (GT_WORKLOADS, default every final workload)."""
    only = os.environ.get("GT_WORKLOADS", "").split()
    return [wl for wl in fc.WORKLOADS if not only or wl in only]


def dataset(wl):
    """@return the dataset of one workload (GT_DATASETS override, else final_config.py)."""
    return DATASETS.get(wl, fc.WORKLOADS[wl][0])
RD = 10   # reads of the final setup, for the time objectives and the extrapolation
# measured on the option A runs (2026-10-07): per workload, s per run outside the timed work
# (preload, setup, untimed read check), mean read pass s of the best static, read s per stored GB
OVERHEAD_S = {"Nyx": 300.0, "VPIC": 130.0, "WarpX": 120.0, "incflo": 135.0}
READ_S_PER_GB = {"Nyx": 1.0, "VPIC": 1.25, "WarpX": 1.08, "incflo": 1.09}
KMEANS_S = {"Nyx": 3.2, "VPIC": 1.6, "WarpX": 1.65, "incflo": 1.5}


def objectives(meas, nb, store):
    """@return {name: (chunks x settings) matrix to minimize}: end-to-end time at 4 storage
    speeds and the final cost models at 0.5 and 1 GB/s; candidates only (others NaN)."""
    st, kept = (lambda r: (np.where(r > 1, nb[:, None] / np.where(r > 1, r, 1.0), nb[:, None]), r > 1))(meas[..., 2])
    st[:, store] = nb
    ct = np.where(np.arange(meas.shape[1]) == store, 0.0, meas[..., 0])
    dt = np.where(kept, meas[..., 1], 0.0)
    dt[:, store] = 0.0
    out = {}
    for s in (0.25e6, 0.5e6, 1.2e6, 12e6):
        out[f"time at {s / 1e6:g} GB/s"] = ct + st / s + RD * (dt + st / s)
    for w in {v[1] for v in fc.WORKLOADS.values()}:
        for bw in (0.5e6, 1e6):
            out[f"cost {fc.model_name(w)} at {bw / 1e6:g} GB/s"] = w[0] * ct + w[1] * dt + w[2] * st / bw
    return {k: ev.for_selection(np.where(np.isfinite(v), v, np.nan)) for k, v in out.items()}


def ranking(wl):
    """@return (setting indices in run order, names, estimated wall s per setting) of one workload."""
    names, store, order, meas, _ = rl.load(dataset(wl))
    nb = order.bytes.to_numpy(float)
    score = np.zeros(len(names))
    for m in objectives(meas, nb, store).values():
        best = np.nanargmin(m, axis=1)
        score += np.bincount(best, minlength=len(names))
        score[int(np.nanargmin(np.nansum(np.where(np.isnan(m), np.inf, m), axis=0)))] += 1e9
    cand = [s for s in ev.candidate_settings() if s != store]
    ranked = sorted(cand, key=lambda s: -score[s])
    ranked.insert(min(5, len(ranked)), store)
    # wall estimate: overhead + write + (reads + untimed check) x pass
    r = meas[..., 2]
    st = np.where(r > 1, nb[:, None] / np.where(r > 1, r, 1.0), nb[:, None])
    st[:, store] = nb
    ct = np.nan_to_num(meas[..., 0]).sum(0) / 1e3
    dt = np.nan_to_num(meas[..., 1]).sum(0) / 1e3
    gb = st.sum(0) / 1e9
    est = {s: (ct[s], dt[s], gb[s]) for s in ranked}
    return ranked, names, est


def plan(reads):
    """Write GT_DIR/plan.txt ('WORKLOAD DATASET SETTING NAME' per line, run order) and print the
    estimated time."""
    per = {wl: ranking(wl) for wl in workloads()}
    lines, total = [], 0.0
    for k in range(max(len(p[0]) for p in per.values())):
        for wl in [x for x in ("WarpX", "incflo", "VPIC", "Nyx") if x in per]:
            ranked, names, est = per[wl]
            if k >= len(ranked):
                continue
            s = ranked[k]
            ct, dt, gb = est[s]
            pass_s = dt + gb * READ_S_PER_GB[wl] + KMEANS_S[wl]
            wall = OVERHEAD_S[wl] + ct + gb + (reads + 1) * pass_s
            total += wall
            lines.append(f"{wl} {dataset(wl)} {s} {names[s].replace(' ', '_')} {wall:.0f} {total:.0f}")
    os.makedirs(GT_DIR, exist_ok=True)
    with open(os.path.join(GT_DIR, "plan.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} runs, estimated {total / 3600:.1f} h with {reads} reads; first runs:")
    for x in lines[:12]:
        wl, _, s, name, wall, cum = x.split()
        print(f"  {wl:6s} setting {s:>2s} {name:45s} ~{int(wall) / 60:4.1f} min (done after {int(cum) / 3600:4.1f} h)")
    for h in (4, 8, 12):
        n = sum(1 for x in lines if int(x.split()[5]) <= h * 3600)
        print(f"  after {h:2d} h: {n} runs")


def exhaustive_meas(path, order, n_settings):
    """@return the (chunks x settings x [compress ms, decompress ms, ratio]) matrix of one
    exhaustive search folder, in the order of `order` (as replay_learning.load)."""
    m = pd.read_csv(os.path.join(path, "v2_measured.csv"))
    m = m[~ev.raw_primary(m) & (m.decomp_ms > 0)].drop_duplicates(["blob", "setting"])
    idx = {b: i for i, b in enumerate(order.blob)}
    meas = np.full((len(order), n_settings, 3), np.nan)
    r = m.blob.map(idx)
    k = r.notna()
    for j, col in enumerate(("comp_ms", "decomp_ms", "ratio")):
        meas[r[k].astype(int), m.setting[k], j] = m[col][k]
    return meas


def exhaustive_report(wl):
    """Write oracle_per_chunk_<wl>.csv and exhaustive_check_<wl>.txt from the new exhaustive
    search; @return the report lines (or [] without the search)."""
    d = os.path.join(GT_DIR, wl.lower())
    path = os.path.join(d, "exhaustive")
    if not os.path.exists(os.path.join(path, "v2_measured.csv")):
        return []
    names, store, order, base, _ = rl.load(dataset(wl))
    new = exhaustive_meas(path, order, len(names))
    rows = np.isfinite(new[..., 2]).any(axis=1)   # the chunks of the new search
    n_all = len(order)
    order, new, base = order[rows].reset_index(drop=True), new[rows], base[rows]
    nb = order.bytes.to_numpy(float)
    both = np.isfinite(new[..., 2]) & np.isfinite(base[..., 2])
    same_ratio = np.isclose(new[..., 2][both], base[..., 2][both], rtol=1e-9, atol=0).mean()
    rel = lambda j: np.nanmedian(np.abs(new[..., j][both] / base[..., j][both] - 1))
    tot = lambda m, j: np.nansum(m[..., j], axis=0)
    corr = [np.corrcoef(tot(new, j)[tot(base, j) > 0], tot(base, j)[tot(base, j) > 0])[0, 1] for j in (0, 1)]
    out = {"blob": order.blob, "raw_bytes": nb}
    lines = [f"{wl} exhaustive search: {np.isfinite(new[..., 2]).sum()} chunk x setting results "
             f"({len(nb)} of {n_all} chunks)",
             f"  vs the stored baseline: identical ratio on {100 * same_ratio:.3f} % of {both.sum()} pairs; median "
             f"per-pair time change compress {100 * rel(0):.1f} %, decompress {100 * rel(1):.1f} %; per-setting "
             f"total time correlation compress {corr[0]:.4f}, decompress {corr[1]:.4f}"]
    for obj, m in objectives(new, nb, store).items():
        best = np.nanargmin(m, axis=1)
        out[f"best: {obj}"] = [names[b] for b in best]
        out[f"value: {obj}"] = m[np.arange(len(nb)), best]
        single = int(np.nanargmin(np.nansum(np.where(np.isnan(m), np.inf, m), axis=0)))
        lines.append(f"  {obj}: best single {names[single]}; per-chunk oracle saves "
                     f"{100 * (1 - np.nansum(m[np.arange(len(nb)), best]) / np.nansum(m[:, single])):.1f} % against it")
    pd.DataFrame(out).to_csv(os.path.join(d, f"oracle_per_chunk_{wl.lower()}.csv"), index=False)
    with open(os.path.join(d, f"exhaustive_check_{wl.lower()}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def phases(run):
    """@return (write rows, timed read rows) of one run's phases.csv: the reads inside the timed
    read passes only (the untimed bit-exact check reads every chunk once more, without a page
    cache drop before it)."""
    cols = ["seq", "chunk_id", "path", "chunk_bytes", "stored_bytes", "compress_ms", "decompress_ms", "io_ms",
            "wall_ms"]
    p = pd.read_csv(os.path.join(run, "phases.csv"), usecols=cols)
    w = p[p.path == "write"]
    rd = p[p.path != "write"].sort_values("seq")
    # the read rows have no timestamp; the timed passes come first, the untimed check last
    timed = len(cp.parse_proc(run)["passes_ns"]) * len(w)
    if len(rd) not in (timed, timed + len(w)):
        raise SystemExit(f"{run}: {len(rd)} read rows for {timed // max(len(w), 1)} timed passes of {len(w)} chunks")
    return w, rd.iloc[:timed]


def chunk_table(run, name, setting):
    """@return one row per chunk of one finished run (write and mean timed-read columns)."""
    w, rd = phases(run)
    w = w.set_index("chunk_id")
    r = rd.groupby("chunk_id")[["decompress_ms", "io_ms", "wall_ms"]].mean()
    n = rd.groupby("chunk_id").size()
    t = pd.DataFrame({"setting": setting, "name": name, "raw_bytes": w.chunk_bytes, "stored_bytes": w.stored_bytes,
                      "write_compress_ms": w.compress_ms, "write_io_ms": w.io_ms, "write_wall_ms": w.wall_ms,
                      "read_decompress_ms": r.decompress_ms, "read_io_ms": r.io_ms, "read_wall_ms": r.wall_ms,
                      "reads": n})
    return t.rename_axis("chunk").reset_index()


def codec_check(run, setting, exh):
    """@return (chunks whose stored ratio differs from the exhaustive search's for this setting,
    chunks stored raw by a fallback): a codec that fails now and then shows here (2026-10-07:
    spspeed shuffle=bit stored one VPIC chunk raw and wrote one corrupt incflo chunk)."""
    if exh is None:
        return np.nan, np.nan
    b = pd.read_csv(os.path.join(run, "blobs.csv"), usecols=["blob", "ratio"])
    e = b.blob.map(exh[exh.setting == setting].drop_duplicates("blob").set_index("blob").ratio)
    if e.notna().sum() == 0:   # 'store' has no exhaustive result
        return np.nan, np.nan
    known = e.notna()
    return int((~np.isclose(b.ratio[known], e[known], rtol=1e-3)).sum()), int((b.ratio[known] == 0).sum())


def setting_row(run, name, setting, exh=None):
    """@return the run-level numbers of one finished run; exh: the workload's exhaustive search."""
    x = cp.parse_proc(run)
    log = open(os.path.join(run, "stdout.log")).read()
    km = sum(float(v) for v in re.findall(r"consumer \(k-means\): ([\d.]+) ms", log)) / 1e3
    passes = [(b - a) / 1e9 for a, b in x["passes_ns"]]
    write = (x["end_ns"] - x["start_ns"]) / 1e9
    w, rd = phases(run)
    gb = w.stored_bytes.sum() / 1e9
    return {"setting": setting, "name": name, "app_s": write + sum(passes), "write_s": write,
            "read_pass_s": float(np.mean(passes)), "app_10reads_s": write + RD * float(np.mean(passes)),
            "reads": len(passes), "ratio": x["raw"] / x["stored"], "stored_GB": gb,
            "read_io_s_per_GB": rd.io_ms.sum() / 1e3 / gb / max(len(passes), 1),
            "kmeans_s": km, "digest_ok": x["digest_ok"], "timed_ok": x["timed_ok"],
            **dict(zip(("ratio_mismatch_chunks", "raw_fallback_chunks"), codec_check(run, setting, exh))), "run": run}


def collect():
    """Write the per-workload tables of every finished run in GT_DIR."""
    for wl in workloads():
        print("\n".join(exhaustive_report(wl)))
        d = os.path.join(GT_DIR, wl.lower())
        runs = sorted(x for x in (os.listdir(d) if os.path.isdir(d) else []) if re.match(r"s\d+_", x))
        if not runs:
            continue
        rows, chunks = [], []
        ef = os.path.join(d, "exhaustive", "v2_measured.csv")
        exh = pd.read_csv(ef, usecols=["blob", "setting", "ratio"]) if os.path.exists(ef) else None
        for x in runs:
            setting, name = int(x[1:].split("_", 1)[0]), x.split("_", 1)[1]
            rows.append(setting_row(os.path.join(d, x), name, setting, exh))
            chunks.append(chunk_table(os.path.join(d, x), name, setting))
        s = pd.DataFrame(rows).sort_values("app_10reads_s")
        s.to_csv(os.path.join(d, f"settings_{wl.lower()}.csv"), index=False)
        c = pd.concat(chunks, ignore_index=True)
        c.to_csv(os.path.join(d, f"chunks_{wl.lower()}.csv"), index=False)
        c["e2e_ms"] = c.write_wall_ms + RD * c.read_wall_ms
        best = c.loc[c.groupby("chunk").e2e_ms.idxmin()]
        ok = s[s.digest_ok & s.timed_ok]
        odd = s[s.ratio_mismatch_chunks.fillna(0) > 0]
        lines = [f"{wl}: {len(s)} finished settings ({len(ok)} bit-exact and timed) of {len(ev.candidate_settings())}",
                 "settings with chunks whose ratio differs from the exhaustive search: "
                 + (", ".join(f"{r['name']} ({int(r.ratio_mismatch_chunks)}, raw fallback {int(r.raw_fallback_chunks)})"
                              for _, r in odd.iterrows()) or "none"),
                 f"best single by measured time (1 write + {RD} reads, write + {RD} x mean read pass): "
                 f"{ok.iloc[0]['name']} {ok.iloc[0].app_10reads_s:.1f} s, ratio {ok.iloc[0].ratio:.2f}",
                 f"per-chunk oracle over the finished settings (each chunk's lowest write wall + {RD} x read wall): "
                 f"sum {best.e2e_ms.sum() / 1e3:.1f} s of chunk time, ratio {best.raw_bytes.sum() / best.stored_bytes.sum():.2f}, "
                 f"settings used: {best.name.value_counts().head(6).to_dict()}"]
        with open(os.path.join(d, f"summary_{wl.lower()}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["plan", "collect"])
    ap.add_argument("--reads", type=int, default=int(os.environ.get("GT_READS", 2)))
    a = ap.parse_args()
    plan(a.reads) if a.what == "plan" else collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())

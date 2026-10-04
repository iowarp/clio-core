#!/usr/bin/env python3
"""Summarise temporal_sweep CSVs: what lossless look-ahead coding buys over
independent per-frame coding, per codec setting and per workload.

    temporal_summary.py CSV [CSV ...] [--out DIR]

For every (workload, window T, codec setting) the baseline is the `indep`
row of the same group; a mode's gain is its compressed bytes relative to
that baseline over the same groups (sum of bytes, so large chunks weigh as
they do on disk). Times are the medians the sweep recorded, summed over the
window: codec compress + residual kernel, codec decompress + reconstruction.

Prints, per workload and T:
  - the best setting per mode (by total ratio) and its gain vs indep
  - the gain of each mode for every setting, as a table
  - the time overhead of the residual / reconstruction kernels
and writes <out>/<wl>_T<T>_summary.csv with every (mode, setting) row.
"""
import argparse
import os
import re
import sys

import pandas as pd


def load(paths):
    frames = []
    for p in paths:
        m = re.search(r"np-temporal/([^/]+)/temporal_T(\d+)\.csv$", os.path.abspath(p))
        wl = m.group(1) if m else os.path.basename(os.path.dirname(p))
        T = int(m.group(2)) if m else 0
        d = pd.read_csv(p)
        d["wl"] = wl
        d["T"] = T if T else d["frames"]
        d["setting"] = d["algorithm"] + " " + d["settings"].fillna("")
        d["setting"] = d["setting"].str.strip()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def summarise(d):
    ok = d[d.ok == 1]
    rows = []
    for (wl, T, setting), g in ok.groupby(["wl", "T", "setting"]):
        base = g[g["mode"] == "indep"].set_index("group")
        if base.empty:
            continue
        for mode, gm in g.groupby("mode"):
            gm = gm.set_index("group")
            common = base.index.intersection(gm.index)
            if len(common) == 0:
                continue
            b, m = base.loc[common], gm.loc[common]
            bytes_in = m["bytes_in"].sum()
            rows.append({
                "wl": wl, "T": T, "setting": setting, "mode": mode,
                "groups": len(common),
                "ratio": bytes_in / m["comp_bytes"].sum(),
                "indep_ratio": bytes_in / b["comp_bytes"].sum(),
                "bytes_vs_indep": m["comp_bytes"].sum() / b["comp_bytes"].sum(),
                "comp_ms": (m["comp_ms"] + m["resid_ms"]).sum(),
                "decomp_ms": (m["decomp_ms"] + m["recon_ms"]).sum(),
                "indep_comp_ms": b["comp_ms"].sum(),
                "indep_decomp_ms": b["decomp_ms"].sum(),
                "resid_ms": m["resid_ms"].sum(),
                "recon_ms": m["recon_ms"].sum(),
                "comp_GBps": bytes_in / 1e6 / (m["comp_ms"] + m["resid_ms"]).sum(),
                "decomp_GBps": bytes_in / 1e6 / (m["decomp_ms"] + m["recon_ms"]).sum(),
            })
    return pd.DataFrame(rows)


def report(s, out):
    pd.set_option("display.width", 200)
    for (wl, T), g in s.groupby(["wl", "T"]):
        print("=" * 100)
        print(f"{wl}  window T={T}  ({int(g.groups.max())} groups)")
        print("-- best setting per mode (total ratio over the sampled groups)")
        for mode, gm in g.groupby("mode"):
            r = gm.sort_values("ratio", ascending=False).iloc[0]
            print(f"   {mode:12s} {r.setting:40s} ratio {r.ratio:7.3f}  "
                  f"vs indep x{1 / r.bytes_vs_indep:6.3f}  "
                  f"comp {r.comp_GBps:6.2f} GB/s  decomp {r.decomp_GBps:6.2f} GB/s")
        print("-- bytes relative to indep (1.00 = same; < 1 = smaller), by setting")
        piv = g.pivot_table(index="setting", columns="mode", values="bytes_vs_indep")
        piv["indep_ratio"] = g[g["mode"] == "indep"].set_index("setting")["ratio"]
        print(piv.round(3).sort_values("indep_ratio", ascending=False).to_string())
        print("-- kernel overhead: residual / reconstruction ms per GB of input")
        # simpler: average per-row kernel time share
        kg = g[g["mode"] != "indep"].copy()
        kg["resid_share"] = kg.resid_ms / kg.comp_ms
        kg["recon_share"] = kg.recon_ms / kg.decomp_ms
        print(kg.groupby("mode")[["resid_share", "recon_share"]].mean().round(3).to_string())
        if out:
            os.makedirs(out, exist_ok=True)
            g.to_csv(os.path.join(out, f"{wl}_T{T}_summary.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = load(a.csv)
    bad = d[d.ok != 1]
    if len(bad):
        print(f"{len(bad)} failed rows:", bad.note.value_counts().to_dict(), file=sys.stderr)
    report(summarise(d), a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does one lossless codec win everywhere? Reads run_new_sweep.sh output.

    analyze_new_sweep.py [--root ~/np-newsweep] [--out DIR]

Per (dataset, product): ratio = sampled input bytes / compressed bytes of
each setting (failed rows excluded; a setting must succeed on every sampled
chunk of the product to qualify). Reports
  - the best setting per product, its family and ratio, and the runner-up
    FAMILY (best setting of a different codec);
  - the GLOBAL setting: the one with the highest geometric-mean ratio over
    all products, and each product's loss when forced to use it;
  - per-chunk winners (oracle) per dataset.
Writes per_product.csv, global_loss.csv and chunk_winners.csv to --out.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def family(alg, settings):
    s = str(settings) if isinstance(settings, str) else ""
    sh = " +bitshuf" if ("shuffle=bit" in s or "bitshuffle=msb" in s) else \
         (" +byteshuf" if "shuffle=byte" in s else "")
    return alg + sh


def load(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*", "configs.csv"))):
        ds = os.path.basename(os.path.dirname(d))
        c = pd.read_csv(d)
        extra = os.path.join(os.path.dirname(d), "configs_shuffle.csv")
        if os.path.exists(extra):  # the shuffled-variant pass, same chunks
            c = pd.concat([c, pd.read_csv(extra)], ignore_index=True)
        m = pd.read_csv(os.path.join(os.path.dirname(d), "map.csv"), header=None,
                        names=["file", "product"])
        c = c.merge(m, on="file", how="left")
        c["dataset"] = ds
        rows.append(c)
    d = pd.concat(rows, ignore_index=True)
    d["settings"] = d["settings"].fillna("")
    d["setting"] = (d.algorithm + " " + d.settings).str.strip()
    d["family"] = [family(a, s) for a, s in zip(d.algorithm, d.settings)]
    return d


def per_product(d):
    nchunks = d.groupby(["dataset", "product"]).file.nunique()
    ok = d[d.ok == 1]
    g = ok.groupby(["dataset", "product", "setting", "family"]).agg(
        n=("file", "nunique"), b=("bytes", "sum"), c=("comp_bytes", "sum"),
        ct=("comp_ms", "sum"), dt=("decomp_ms", "sum")).reset_index()
    g = g[g.n == g.set_index(["dataset", "product"]).index.map(nchunks)]
    g["ratio"] = g.b / g.c
    g["comp_GBps"] = g.b / 1e6 / g.ct
    g["decomp_GBps"] = g.b / 1e6 / g.dt
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/np-newsweep"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = load(a.root)
    g = per_product(d)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)

    rows = []
    for (ds, p), gp in g.groupby(["dataset", "product"]):
        gp = gp.sort_values("ratio", ascending=False)
        b = gp.iloc[0]
        alg_b = b.setting.split()[0]
        rest = gp[gp.setting.str.split().str[0] != alg_b]
        r2 = rest.iloc[0] if len(rest) else None
        zs = gp[gp.setting == "zstd shuffle=byte"]
        rows.append({"dataset": ds, "product": p, "best": b.setting, "ratio": b.ratio,
                     "comp_GBps": b.comp_GBps, "decomp_GBps": b.decomp_GBps,
                     "next_codec": r2.setting if r2 is not None else "",
                     "next_ratio": r2.ratio if r2 is not None else np.nan,
                     "zstd_byteshuf_ratio": zs.ratio.iloc[0] if len(zs) else np.nan})
    pp = pd.DataFrame(rows)
    pp["best_vs_zstdshuf"] = pp.ratio / pp.zstd_byteshuf_ratio
    print("== best setting per product (ratio), and the best of a DIFFERENT codec")
    print(pp.round(3).to_string(index=False))

    # global setting: highest geomean ratio over products where it qualified everywhere
    piv = g.pivot_table(index=["dataset", "product"], columns="setting", values="ratio")
    full = piv.dropna(axis=1)
    gm = np.exp(np.log(full).mean()).sort_values(ascending=False)
    glob_set = gm.index[0]
    print(f"\n== global single best setting (geomean over {len(full)} products): "
          f"{glob_set} ({gm.iloc[0]:.3f}x); next: "
          + ", ".join(f"{s} {v:.3f}x" for s, v in gm.iloc[1:4].items()))
    # each product's TRUE best: any setting that succeeded on all its chunks,
    # not only the settings that qualified on every product
    loss = (piv.max(axis=1) / full[glob_set]).rename("best_over_global")
    gl = pd.concat([full[glob_set].rename("global_ratio"), piv.max(axis=1).rename("best_ratio"),
                    piv.idxmax(axis=1).rename("best_setting"), loss], axis=1).reset_index()
    print(gl.sort_values("best_over_global", ascending=False).round(3).to_string(index=False))

    # per-chunk oracle winners by codec (first token of the setting)
    ok = d[d.ok == 1]
    win = ok.loc[ok.groupby(["dataset", "file"]).comp_bytes.idxmin()]
    win = win.assign(codec=win.setting.str.split().str[0])
    cw = win.groupby("dataset").codec.value_counts(normalize=True).rename("share").reset_index()
    print("\n== per-chunk winning codec share by dataset")
    print(cw.pivot(index="dataset", columns="codec", values="share").fillna(0).round(2).to_string())
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        pp.to_csv(os.path.join(a.out, "per_product.csv"), index=False)
        gl.to_csv(os.path.join(a.out, "global_loss.csv"), index=False)
        cw.to_csv(os.path.join(a.out, "chunk_winners.csv"), index=False)


if __name__ == "__main__":
    main()

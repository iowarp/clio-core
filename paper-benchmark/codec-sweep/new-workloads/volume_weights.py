"""Volume weights for sampled chunks: a sampled chunk of a product stands for
(product's chunks in the dataset) / (product's sampled chunks) real chunks."""
import os, re, subprocess, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))


def product_totals(src):
    """@return {product: number of 4 MiB chunks in the whole dataset}"""
    out = subprocess.run([sys.executable, os.path.join(HERE, "sample_chunks.py"), "--dir", src,
                          "--fraction", "1", "--map", "/dev/stdout"], capture_output=True, text=True).stdout
    tot = {}
    for line in out.splitlines():
        if "," in line and "#" in line:
            tot[line.rsplit(",", 1)[1]] = tot.get(line.rsplit(",", 1)[1], 0) + 1
    return tot


def chunk_weights(sweep_dir, src):
    """@return DataFrame file, chunk, product, weight"""
    m = pd.read_csv(os.path.join(sweep_dir, "map.csv"), header=None, names=["k", "product"])
    m[["file", "chunk"]] = m["k"].str.rsplit("#", n=1, expand=True)
    m["chunk"] = m["chunk"].astype(int)
    tot = product_totals(src)
    n = m.groupby("product")["k"].transform("count")
    m["weight"] = m["product"].map(tot) / n
    return m[["file", "chunk", "product", "weight"]]


def apply(df, w):
    """Scale every per-chunk quantity by its chunk's weight (costs are linear)."""
    d = df.merge(w, on=["file", "chunk"])
    for c in ("bytes", "comp_bytes", "comp_ms", "decomp_ms"):
        d[c] = d[c] * d["weight"]
    return d

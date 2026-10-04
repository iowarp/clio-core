#!/usr/bin/env python3
"""A cost model on a four-tier hierarchy: every codec vs per-chunk choice.

The data is spread over the hierarchy by share of input bytes

    10% DRAM, 30% node-local NVMe, 30% SSD (/work/nvme), 30% HDD (/work/hdd)

by dealing the 4 MiB chunks round-robin in that 1:3:3:3 ratio (in file and
chunk order), so every tier holds a representative sample of the run instead
of whichever dumps came first. Each chunk is charged its own tier's measured
write bandwidth (probe_tiers.sh) in the chosen setting's cost
(cost_model_per_chunk.OBJECTIVES), e.g. balanced:

    cost = compress_ms + decompress_ms + compressed_bytes / bandwidth(tier)

or compression-focused (--model compression): compressed_bytes / bandwidth only.

A single codec is used for every chunk on every tier; the per-chunk optimal
(an oracle on measured values) takes the cheapest codec for each chunk knowing
the tier it lands on, so it can pick differently on DRAM than on disk.

Writes figures/cost-model-tiers/per-chunk/hierarchy_<model>.png in the same
layout as the single-tier figures (cost_model_per_chunk.py).

  cost_model_hierarchy.py [PROBE_CSV] [--model balanced|compression|speed]
                          [--bw-method M | --bw TIER=GBPS,...] [--tag T]
                          [--sweep DIR] [--out DIR]

--bw-method takes every tier's bandwidth from one probe method, e.g.
ior_write_n1 from ior_tiers.sh's ior_bw.csv. --bw sets them by hand instead
(hypothetical tiers, no probe CSV needed), e.g.
dram=12,nvme=1,burst_buffer=0.512,lustre=0.25. --tag is appended to the file
name.
"""
import argparse
import os

import pandas as pd

import cost_model_per_chunk as pc
import cost_model_tiers as cm

# tier -> (label, share of the input bytes it holds, in tenths)
HIERARCHY = {"dram": ("DRAM", 1), "nvme": ("NVMe", 3),
             "burst_buffer": ("SSD", 3), "lustre": ("HDD", 3)}


def place(df):
    """Assign every chunk to a tier, round-robin in the HIERARCHY ratio.

    @param df per-chunk measurements (cost_model_tiers.load_workload)
    @return Series aligned with df: the tier of each row's chunk
    """
    pattern = [t for t, (_, tenths) in HIERARCHY.items() for _ in range(tenths)]
    keys = (df[["file", "chunk"]].drop_duplicates()
            .sort_values(["file", "chunk"]).reset_index(drop=True))
    keys["tier"] = [pattern[i % len(pattern)] for i in range(len(keys))]
    return df.merge(keys, on=["file", "chunk"], how="left")["tier"].set_axis(
        df.index)


def place_weighted(df):
    """Put every chunk on every tier, weighted by the tier's HIERARCHY share.

    Each copy's bytes, compressed bytes and times are scaled by the share, so
    every cost (linear in them) is the expectation over the placement and the
    input still totals its real size. The oracle picks per (chunk, tier).
    Unlike place, this fills every tier however few chunks were sampled.

    @param df per-chunk measurements
    @return (expanded DataFrame, Series aligned with it: each row's tier)
    """
    parts = []
    for t, (_, tenths) in HIERARCHY.items():
        w = tenths / sum(n for _, n in HIERARCHY.values())
        p = df.copy()
        p["file"] = p["file"].astype(str) + "@" + t
        for col in ("bytes", "comp_bytes", "comp_ms", "decomp_ms"):
            p[col] = p[col] * w
        p["tier"] = t
        p["weight"] = w
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    return out.drop(columns="tier"), out["tier"]


def byte_shares(df, tier):
    """Share of input bytes each tier received, for the record.

    @param df   per-chunk measurements
    @param tier output of place
    @return dict tier -> share of bytes
    """
    chunks = df.assign(tier=tier).groupby(["file", "chunk"]).first()
    return (chunks.groupby("tier")["bytes"].sum() / chunks["bytes"].sum()).to_dict()


def main():
    """Place the chunks, score every codec and the oracle, draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_csv", nargs="?",
                    help="tier_bw.csv (probe_tiers.sh) or ior_bw.csv (ior_tiers.sh)")
    ap.add_argument("--sweep", default=cm.DEFAULT_SWEEP)
    ap.add_argument("--float32", help="run_float32_sweep.sh results dir")
    ap.add_argument("--out", default=pc.DEFAULT_OUT)
    ap.add_argument("--model", default="balanced", choices=list(pc.OBJECTIVES))
    ap.add_argument("--bw-method", default=None,
                    help="probe method for every tier, e.g. ior_write_n1")
    ap.add_argument("--bw", default=None,
                    help="hand-set GB/s per tier, e.g. dram=12,nvme=1,...")
    ap.add_argument("--tag", default="", help="suffix for the output name")
    ap.add_argument("--placement", default="deal", choices=["deal", "weighted"],
                    help="deal chunks round-robin 1:3:3:3, or put each on "
                         "every tier weighted by its share")
    ap.add_argument("--cpu", help="run_float32_sweep.sh ENGINE=cpu results "
                    "dir, measured on the same chunks as --float32")
    ap.add_argument("--engines", default=None, choices=["gpu", "cpu", "both"],
                    help="which codecs to draw (default: both with --cpu, "
                         "else gpu); adds _cpu / _gpu_cpu to the file name")
    ap.add_argument("--ratio-line", action="store_true",
                    help="overlay each bar's compression ratio as a line; "
                         "writes hierarchy_<model><tag>_ratio.png")
    a = ap.parse_args()
    if a.engines is None:
        a.engines = "both" if a.cpu else "gpu"
    if a.bw:
        bw = {k: float(v) for k, v in (kv.split("=") for kv in a.bw.split(","))}
        if set(bw) != set(HIERARCHY):
            ap.error(f"--bw needs exactly {', '.join(HIERARCHY)}")
    elif a.probe_csv:
        bw = cm.tier_bandwidths(a.probe_csv, a.bw_method)
    else:
        fixed = {t: g for t, (_, g) in cm.FIXED_TIERS.items()}
        bw = {"dram": fixed["dram"], "nvme": fixed["nvme"],
              "burst_buffer": fixed["ssd"], "lustre": fixed["hdd"]}
    res, rows = {}, []
    for w, df in pc.load_data(a).items():
        if a.placement == "weighted":
            df, tier = place_weighted(df)
        else:
            tier = place(df)
        res[w] = pc.evaluate(df, tier.map(bw), a.model)
        shares = byte_shares(df, tier)
        rows.append({"workload": w, "best": pc.label(res[w]["best"]),
                     "best_ms": res[w]["fixed"][res[w]["best"]],
                     "oracle_ms": res[w]["oracle"], "raw_ms": res[w]["raw"],
                     "gain_pct": 100 * res[w]["gain"],
                     **{f"{HIERARCHY[t][0]}_bytes": shares.get(t, 0.0)
                        for t in HIERARCHY},
                     "mix": pc.mix_text(res[w]["mix_variant"])[7:]})
    os.makedirs(a.out, exist_ok=True)
    tiers = ", ".join(f"{10 * n}% {name} {bw[t]:.2f} GB/s"
                      for t, (name, n) in HIERARCHY.items())
    title = (f"{pc.OBJECTIVES[a.model][0]} cost model on a four-tier hierarchy "
             f"({tiers}): every codec alone vs per-chunk selection")
    if a.bw:
        src = "hypothetical bandwidths."
    elif a.bw_method and a.bw_method.startswith("ior_"):
        _, op, n = a.bw_method.split("_")
        src = f"IOR 1 MiB {op}, {n[1:]} process{'es' if n != 'n1' else ''}."
    else:
        src = "chunks dealt 1:3:3:3."
    if a.bw or a.probe_csv:
        note = f" SSD = /work/nvme, HDD = /work/hdd; {src}"
    else:
        note = (" Each chunk on every tier, weighted 1:3:3:3"
                if a.placement == "weighted" else " Chunks dealt 1:3:3:3")
        note += "; codec bar = its best setting." if a.float32 else "."
    cpu_note = (" CPU codecs on 2x EPYC 7763, data in DRAM, each bar the best"
                " of 1 thread and 128 threads.")
    gpu_note = (" GPU codecs on an A100, data already in GPU memory (no PCIe"
                " copy counted).")
    note += {"gpu": gpu_note, "cpu": cpu_note,
             "both": cpu_note + gpu_note}[a.engines]
    if a.engines == "cpu":
        title = title.replace("every codec", "every CPU codec")
    elif a.engines == "gpu" and a.cpu:
        title = title.replace("every codec", "every GPU codec")
    engines = {"gpu": "", "cpu": "_cpu", "both": "_gpu_cpu"}[a.engines]
    out = os.path.join(a.out, f"hierarchy_{a.model}{a.tag}{engines}"
                              f"{'_ratio' if a.ratio_line else ''}.png")
    pc.plot_figure(title, note, a.model, res, out, a.ratio_line)
    with pd.option_context("display.width", 220, "display.precision", 3,
                           "display.max_colwidth", 60):
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

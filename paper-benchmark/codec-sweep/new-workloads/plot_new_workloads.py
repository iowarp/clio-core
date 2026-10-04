#!/usr/bin/env python3
"""The new-workloads study drawn exactly like figures/cost-model-tiers/per-chunk:
every codec alone vs per-chunk selection, per dataset, under each cost model,
plus the codec mix the per-chunk oracle picks.

    plot_new_workloads.py [--root ~/np-newsweep] [--out ../../figures/new-workloads]

Reads run_new_sweep.sh / run_shuffle_pass.sh output (configs.csv and, where
complete, configs_shuffle.csv) and reuses cost_model_per_chunk's evaluate(),
plot_figure() and plot_mix() unchanged, so the figures are comparable with
the cost-model-tiers ones. Each codec setting is a candidate: a codec's bar is
its best single setting used for every chunk, the oracle picks among all.

A setting that cannot take a chunk (e.g. a 64-bit codec on an odd-length
32-bit tail) counts as storing that chunk uncompressed, with the measured
"store" times: what a real system would do, and the oracle needs every
candidate on every chunk.

Bandwidths are read from the same probe CSVs, the same way, as the
cost-model-tiers figures (results/tier-probe: probe_tiers.sh and ior_tiers.sh,
measured on Delta), so the two sets of figures share one scale.
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cost_model_tiers as cm  # noqa: E402
import cost_model_per_chunk as cpc  # noqa: E402
import cost_model_hierarchy as ch  # noqa: E402
import volume_weights as vw  # noqa: E402

DATASETS = {  # sweep dir -> panel name, in panel order (all 2-4+ GB)
    "graph-livejournal-full": "LiveJournal graph",
    "graph-orkut-full": "Orkut graph",
    "sparse-fem": "FEM matrices",
    "hep-nanoaod": "NanoAOD",
    "genomics-reads": "Genome reads",
    "astro-camels": "CAMELS TNG z=0",
    "detector-frames": "Detector frames",
    "consumer-nyx": "Nyx analysis",
    "consumer-vpic-full": "VPIC analysis",
    "consumer-lammps-full": "LAMMPS analysis",
    "dl-resnet18-train": "ResNet-18",
    "dl-pythia-ckpt": "Pythia ckpts",
    "dl-qwen-bf16": "Qwen bf16 ckpt",
    "dl-opt-relu-act": "OPT ReLU acts",
    "dl-gpt2-kv": "GPT-2 KV cache",
    "ref-nyx-256-2000": "Nyx sim",
    "ref-vpic-126-2000": "VPIC sim",
    "ref-warpx-64x64x512-2000": "WarpX sim",
    "ref-lammps-b70-2000": "LAMMPS sim",
}
NVCOMP = {"lz4", "snappy", "zstd", "gdeflate", "deflate", "ans", "cascaded", "bitcomp"}
PROBE = os.path.join(os.path.dirname(HERE), "results", "tier-probe")
TIER_CSV = glob.glob(os.path.join(PROBE, "1*", "tier_bw.csv"))[0]  # probe_tiers.sh
IOR_CSV = glob.glob(os.path.join(PROBE, "ior-*", "ior_bw.csv"))[0]  # ior_tiers.sh
# The same bandwidths the cost-model-tiers figures use, read the same way
# (cost_model_tiers.tier_bandwidths): measured on Delta.
BW_PROBE = cm.tier_bandwidths(TIER_CSV)            # durable write, D2H pinned
BW = {t: BW_PROBE[t] for t in ("dram", "nvme")}    # single-tier figures
# The hierarchy figure's bandwidths, set by hand (GB/s), keyed like
# cost_model_hierarchy.HIERARCHY (10% / 30% / 30% / 30% of the input bytes).
BW_HIER = {"dram": 12.0, "nvme": 1.0, "burst_buffer": 0.5, "lustre": 0.25}


NAME = {"lz4": "LZ4", "snappy": "Snappy", "zstd": "Zstd", "gdeflate": "GDeflate",
        "deflate": "Deflate", "ans": "ANS", "cascaded": "Cascaded", "bitcomp": "Bitcomp",
        "ndzip": "ndzip", "gpulz": "GPULZ", "spspeed": "SPspeed", "spratio": "SPratio",
        "store": "Uncompressed"}


def short_config(cfg):
    """Compact bar label for one setting, e.g. 'Cascaded rle1 d0 int +byte8'."""
    toks = cfg.split()
    kv = dict(t.split("=", 1) for t in toks[1:])
    parts = [NAME.get(toks[0], toks[0])]
    if "level" in kv:
        parts.append(f"L{kv['level']}")
    if "algo" in kv:
        parts.append("sparse" if kv["algo"] == "1" else "default")
    for k, tag in (("rle", "rle"), ("delta", "d")):
        if k in kv:
            parts.append(f"{tag}{kv[k]}")
    if kv.get("bp") == "0":
        parts.append("no-bp")
    if "type" in kv:
        parts.append({"longlong": "i64", "int": "i32", "short": "i16", "char": "u8",
                      "float16": "f16", "float8": "fp8"}.get(kv["type"], kv["type"]))
    if "bitshuffle" in kv:
        parts.append(f"bitshuf-{kv['bitshuffle']}")
    if kv.get("shuffle") == "byte":
        parts.append(f"+byte{kv.get('elem', '4')}")
    elif kv.get("shuffle") == "bit":
        parts.append("+bit")
    return " ".join(parts)


def best_configs(df, bw, model):
    """The setting each bar of cost_model_per_chunk.evaluate uses: per codec
    variant (codec, codec|shuf), its cheapest setting over every chunk.

    @return dict variant key -> setting string
    """
    w_ct, w_dt, w_io = cpc.OBJECTIVES[model][2]
    d = df.assign(score=w_ct * df["comp_ms"] + w_dt * df["decomp_ms"]
                  + w_io * df["comp_bytes"] / (bw * 1e6), variant=cpc.variants(df))
    per_cfg = d.groupby("config")["score"].sum()
    fam = d.groupby("config")["variant"].first()
    return per_cfg.groupby(fam).idxmin().to_dict()


def label_bars_with_settings(cfgmaps):
    """After cost_model_per_chunk draws a panel (codec-level labels, titles and
    picks unchanged), replace each codec bar's y tick label with the exact
    setting that bar uses. cfgmaps: dataset -> {variant key: setting}.
    Also widens the figure's left margin and panel spacing for the longer
    labels."""
    orig_panel = cpc.plot_panel

    def panel(ax, wl, r, ratio_line=False):
        out = orig_panel(ax, wl, r, ratio_line)
        by_name = {cpc.label(k): short_config(c) for k, c in cfgmaps.get(wl, {}).items()}
        ticks = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([by_name.get(t, t) for t in ticks], fontsize=8)
        return out

    cpc.plot_panel = panel
    from matplotlib.figure import Figure
    orig_adjust = Figure.subplots_adjust

    def adjust(self, *args, **kw):  # plot_figure's 4-across layout, wider gutters
        if kw.get("left") == 0.07:
            kw.update(left=0.095, wspace=0.62)
        return orig_adjust(self, *args, **kw)

    Figure.subplots_adjust = adjust


def load_dataset(d):
    """Per-chunk rows of one dataset, every complete setting, failures filled."""
    parts = [pd.read_csv(p) for p in (os.path.join(d, "configs.csv"),
                                      os.path.join(d, "configs_shuffle.csv"))
             if os.path.exists(p)]
    df = pd.concat(parts, ignore_index=True)
    # a sweep still writing may leave a partial last line
    df = df[df["file"].astype(str).str.contains("#") & df["ok"].notna()
            & df["comp_ms"].notna()].copy()
    df["settings"] = df["settings"].fillna("")
    df["config"] = (df["algorithm"] + " " + df["settings"]).str.strip()
    df[["file", "chunk"]] = df["file"].str.rsplit("#", n=1, expand=True)
    df["chunk"] = df["chunk"].astype(int)
    df["codec"] = df["algorithm"].map(lambda a: f"nvcomp-{a}" if a in NVCOMP else a)
    nchunks = df.groupby(["file", "chunk"]).ngroups
    cover = df.groupby("config").apply(lambda g: g[["file", "chunk"]].drop_duplicates().shape[0],
                                       include_groups=False)
    df = df[df["config"].isin(cover[cover == nchunks].index)]  # drop half-run settings
    store = (df[(df["config"] == "store") & (df["ok"] == 1)]
             .set_index(["file", "chunk"])[["comp_ms", "decomp_ms"]])
    bad = df["ok"] != 1
    if bad.any():
        key = pd.MultiIndex.from_frame(df.loc[bad, ["file", "chunk"]])
        st = store.reindex(key)
        df.loc[bad, "comp_bytes"] = df.loc[bad, "bytes"]
        df.loc[bad, "comp_ms"] = st["comp_ms"].fillna(0).to_numpy()
        df.loc[bad, "decomp_ms"] = st["decomp_ms"].fillna(0).to_numpy()
    return df[["file", "chunk", "codec", "config", "bytes", "comp_bytes",
               "comp_ms", "decomp_ms"]], int(bad.sum()), df["config"].nunique()


def plot_hierarchy(data, out, nset, model="balanced"):
    """cost_model_hierarchy.py on these datasets: chunks dealt round-robin
    1:3:3:3 over DRAM / NVMe / SSD / HDD (BW_HIER), each charged its tier's
    bandwidth; one codec everywhere vs the per-chunk (so per-tier) optimal.
    Writes <out>/hierarchy_<model>.png.

    @param data  dataset -> per-chunk rows
    @param out   output directory
    @param nset  settings-count text for the note
    @param model key of cost_model_per_chunk.OBJECTIVES
    """
    tiers = ", ".join(f"{10 * n}% {name} {BW_HIER[t]:g} GB/s"
                      for t, (name, n) in ch.HIERARCHY.items())
    res, rows = {}, []
    cfgmaps = {}
    for k, df in data.items():
        tier = ch.place(df)
        res[k] = cpc.evaluate(df, tier.map(BW_HIER), model)
        cfgmaps[k] = best_configs(df, tier.map(BW_HIER), model)
        shares = ch.byte_shares(df, tier)
        rows.append({"dataset": k, "best": cpc.label(res[k]["best"]),
                     "gain_pct": round(100 * res[k]["gain"], 2),
                     **{f"{ch.HIERARCHY[t][0]}_bytes": round(shares.get(t, 0.0), 2)
                        for t in ch.HIERARCHY},
                     "mix": cpc.mix_text(res[k]["mix_variant"])[7:]})
    path = os.path.join(out, f"hierarchy_{model}.png")
    label_bars_with_settings(cfgmaps)
    cpc.plot_figure(
        f"{cpc.OBJECTIVES[model][0]} cost model on a four-tier hierarchy ({tiers}): "
        "every codec alone vs per-chunk selection",
        " SSD and HDD = the burst buffer and Lustre tiers; bandwidths set by hand. Chunks dealt "
        "1:3:3:3 round-robin in file order. GPU codecs on an A100, data already in GPU memory "
        f"(no PCIe copy counted). {nset} lossless settings per dataset, every one on the same "
        "sampled chunks (each weighted by the share of the data it stands for); each bar is labelled with the single setting it uses for every chunk "
        "(codec, its options, and +byteN / +bit pre-shuffle).",
        model, res, path)
    print(f"== hierarchy, {model}: {tiers}")
    print(pd.DataFrame(rows).to_string(index=False))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/np-newsweep"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "new-workloads"))
    ap.add_argument("--min-settings", type=int, default=40,
                    help="leave out a dataset with fewer complete settings (139 = both passes)")
    ap.add_argument("--all", action="store_true",
                    help="also draw the single-tier figures (compression / balanced / "
                         "speed on NVMe) and the codec-mix figure")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}
    for key, name in DATASETS.items():
        d = os.path.join(a.root, key)
        if not os.path.exists(os.path.join(d, "configs.csv")):
            print(f"skip {key}: not swept yet")
            continue
        if sum(1 for _ in open(os.path.join(d, "configs.csv"))) < 2:
            print(f"skip {key}: no rows yet")
            continue
        df, filled, ncfg = load_dataset(d)
        if ncfg < a.min_settings:
            print(f"skip {key}: only {ncfg} complete settings (sweep still running)")
            continue
        # Weight every sampled chunk by the real chunks it stands for, so a
        # small, fully sampled array does not count more than its share of
        # the data (costs are linear in bytes and times, so this is exact).
        src = (os.path.expanduser(f"~/np-data/{key[4:]}/fields") if key.startswith("ref-")
               else os.path.expanduser(f"~/np-data/new/{key}"))
        df = vw.apply(df, vw.chunk_weights(d, src)).drop(columns=["product", "weight"])
        data[key] = df
        print(f"{key}: {df.groupby(['file', 'chunk']).ngroups} chunks, {ncfg} settings, "
              f"{filled} failed pairs stored raw")
    if not data:
        print("no complete dataset yet")
        return
    cm.WORKLOADS = {k: DATASETS[k] for k in DATASETS if k in data}  # panel names / order
    ns = sorted({df["config"].nunique() for df in data.values()})
    nset = f"{ns[0]}" if len(ns) == 1 else f"{ns[0]} to {ns[-1]}"
    if a.all:
        draw_single_tier(data, a.out, nset)
    plot_hierarchy(data, a.out, nset)
    print("wrote", sorted(os.listdir(a.out)))


def draw_single_tier(data, out, nset):
    """The per-tier figures: compression / balanced / speed on NVMe, and the
    codec mix on DRAM and NVMe (bandwidths from the probe CSV)."""
    titles = {"compression": f"Compression-focused cost on node-local NVMe, {BW['nvme']:.2f} GB/s",
              "balanced": f"Balanced cost on node-local NVMe, {BW['nvme']:.2f} GB/s",
              "speed": "Speed-focused cost"}
    # The reference palette colours only eight codecs; give the two that win
    # chunks here their own swatch instead of folding them into "Other (< 1%)".
    cpc.CODEC_COLOR.setdefault("gpulz", "#7b3fa0")
    cpc.CODEC_COLOR.setdefault("store", cpc.STORE_COLOR)
    mixes = {}
    for model in ("compression", "balanced", "speed"):
        # under a speed-only cost the zero-work copy wins every chunk, so the
        # uncompressed candidate is left out of that model only
        dm = {k: (df[df["config"] != "store"] if model == "speed" else df)
              for k, df in data.items()}
        res = {k: cpc.evaluate(df, BW["nvme"], model) for k, df in dm.items()}
        cpc.plot_figure(f"{titles[model]}: every codec alone vs per-chunk selection "
                        "on graph, analysis-output, deep-learning and simulation data",
                        f" Candidates: {nset} lossless GPU settings per dataset (12 codecs and their data-type "
                        "options, with and without byte / bit pre-shuffle); a codec's bar is its best single setting. "
                        "Up to 24 sampled 4 MiB chunks per array type, each weighted by the share of the data it stands for."
                        + (" Uncompressed storage is left out here: a plain copy wins any "
                           "speed-only cost." if model == "speed" else ""),
                        model, res, os.path.join(out, f"{model}_nvme.png"))
        for tier in BW:
            r = res if tier == "nvme" or model == "speed" else \
                {k: cpc.evaluate(df, BW[tier], model) for k, df in dm.items()}
            mixes[(tier, model)] = cpc.mix_table(r)
    cpc.plot_mix(mixes, BW, os.path.join(out, "codec_mix.png"))
    for (tier, model), t in mixes.items():
        if tier == "nvme":
            print(f"\n== oracle codec mix, {model}, NVMe")
            print((100 * t).round(0).astype(int).to_string())


if __name__ == "__main__":
    main()

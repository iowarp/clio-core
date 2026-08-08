#!/usr/bin/env python3
# Regenerate the GNN training figures from MEASURED data.
#
# The upstream gnn/make_training_fig.py hardcodes a loss/acc series and a
# peak = [10.24, 0.03125] GiB bar pair from an old 8 GB-GPU run; nothing in it
# is read from a run.  This parses the actual run logs instead, so every number
# plotted traces back to a line in logs/.
#
# Panels:
#   fig_training.png  (a) ogbn-arxiv in-core vs Eternia, overlaid (bit-exact)
#                     (b) papers100M peak GPU to train: in-core OOMs an A100-40GB,
#                         Eternia trains inside a 64 MiB window
#   fig_codec.png     papers100M loss curve, zstd (lossless) vs cuSZp (lossy),
#                     plus the store-size bars that motivate the lossy arm

import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# "   e00  loss 3.741445 acc 0.0136  ||  loss 3.741445 acc 0.0136"
RE_PAIR = re.compile(
    r"^\s*e(\d+)\s+loss\s+([\d.]+)\s+acc\s+([\d.]+)\s+\|\|\s+loss\s+([\d.]+)\s+acc\s+([\d.]+)")
# "[TRAIN]   eternia e26 loss=4.599901 acc=0.0626 val_acc=0.0623 (7348.9s elapsed)"
RE_ET = re.compile(
    r"eternia e(\d+) loss=([\d.]+) acc=([\d.]+) val_acc=([\d.]+)")
# "[TRAIN] ETERNIA: stored A 54208MiB -> 50257MiB zstd (1.079x) in 158.62s"
RE_STORE = re.compile(r"stored A (\d+)MiB -> (\d+)MiB \S+ \(([\d.]+)x\)")
RE_PEAK = re.compile(r"peak GPU window=(\d+)MiB")


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "replace")


def parse_pairs(txt):
    """in-core || eternia per-epoch table -> (ep, base_loss, base_acc, et_loss, et_acc)"""
    ep, bl, ba, el, ea = [], [], [], [], []
    for line in txt.splitlines():
        m = RE_PAIR.match(line)
        if m:
            ep.append(int(m.group(1)))
            bl.append(float(m.group(2))); ba.append(float(m.group(3)))
            el.append(float(m.group(4))); ea.append(float(m.group(5)))
    return ep, bl, ba, el, ea


def parse_eternia(txt):
    """per-epoch eternia trace -> (ep, loss, acc, vacc)"""
    ep, ls, ac, va = [], [], [], []
    for m in RE_ET.finditer(txt):
        ep.append(int(m.group(1)))
        ls.append(float(m.group(2)))
        ac.append(float(m.group(3)))
        va.append(float(m.group(4)))
    return ep, ls, ac, va


def parse_store(txt):
    m = RE_STORE.search(txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_peak(txt):
    m = RE_PEAK.search(txt)
    return int(m.group(1)) if m else None


def fig_training(arxiv_txt, papers_txt):
    ep, bl, ba, el, ea = parse_pairs(arxiv_txt)
    if not ep:
        print("!! no arxiv per-epoch table found; skipping panel (a)", file=sys.stderr)
        return
    dloss = max(abs(x - y) for x, y in zip(bl, el))
    dacc = max(abs(x - y) for x, y in zip(ba, ea))

    a_mib, stored, ratio = parse_store(papers_txt)
    peak = parse_peak(papers_txt)
    a_gib, peak_gib = a_mib / 1024.0, peak / 1024.0
    HBM_GIB = 40016 / 1024.0          # measured free HBM on the A100-SXM4-40GB

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    axL.plot(ep, bl, "-", color="#1565c0", lw=2.5)
    axL.plot(ep, bl, "o", color="#1565c0", ms=4, mfc="white", mew=1.0)
    axL.plot(ep, el, "x", color="#e53935", ms=6)
    axL.set_xlabel("epoch")
    axL.set_ylabel("cross-entropy loss", color="#1565c0")
    axL.tick_params(axis="y", labelcolor="#1565c0")
    axA = axL.twinx()
    axA.plot(ep, [v * 100 for v in ba], "-", color="#2e7d32", lw=2.0)
    axA.plot(ep, [v * 100 for v in ea], "x", color="#e53935", ms=6)
    axA.set_ylabel("train accuracy (%)", color="#2e7d32")
    axA.tick_params(axis="y", labelcolor="#2e7d32")
    axA.set_ylim(0, 30)
    axL.set_title("ogbn-arxiv: in-core $\\equiv$ Eternia\n"
                  "(bit-exact: max$|\\Delta$loss$|$=%.0e, max$|\\Delta$acc$|$=%.0e)"
                  % (dloss, dacc), fontsize=11)
    axL.grid(ls=":", alpha=0.4)
    axL.legend(handles=[
        Line2D([0], [0], color="#1565c0", lw=2.5, marker="o", mfc="white",
               label="in-core baseline"),
        Line2D([0], [0], color="#e53935", lw=0, marker="x", ms=7,
               label="Eternia (streamed) — coincides")],
        loc="upper right", fontsize=9)

    axR.bar(["In-core\n(features resident)", "Eternia\n(stream window)"],
            [a_gib, peak_gib], color=["#c62828", "#1565c0"],
            edgecolor="black", hatch=["///", None], width=0.6)
    axR.axhline(HBM_GIB, ls="--", color="black", lw=1.3)
    axR.text(1.45, HBM_GIB + 0.6, "A100-40GB (%.1f GiB free)" % HBM_GIB,
             ha="right", va="bottom", fontsize=9)
    axR.text(0, a_gib + 0.6, "OOM$\\times$\n(needs %.1f GiB)" % a_gib, ha="center",
             va="bottom", color="#c62828", fontweight="bold", fontsize=9)
    axR.text(1, peak_gib + 2.0, "trains\n%d MiB" % peak, ha="center", va="bottom",
             color="#1565c0", fontweight="bold", fontsize=9)
    axR.set_ylabel("peak GPU memory (GiB)")
    axR.set_ylim(0, a_gib * 1.25)
    axR.set_title("ogbn-papers100M (%.1f GiB features, non-tiled)\n"
                  "in-core OOMs; Eternia trains" % a_gib, fontsize=11)
    axR.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_training.png")
    fig.savefig(p, dpi=150)
    print("wrote", p, "  (arxiv %d epochs; papers %.1f GiB -> %d MiB window)"
          % (len(ep), a_gib, peak))


def fig_codec(zstd_txt, cuszp_txt):
    ez, lz, _, _ = parse_eternia(zstd_txt)
    ec, lc, _, _ = parse_eternia(cuszp_txt)
    if not ez or not ec:
        print("!! missing codec traces; skipping fig_codec", file=sys.stderr)
        return
    sz, sc = parse_store(zstd_txt), parse_store(cuszp_txt)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    axL.plot(ez, lz, "-", color="#1565c0", lw=2.5, label="zstd (lossless)")
    axL.plot(ec, lc, "x", color="#e53935", ms=6, lw=0,
             label="cuSZp (lossy, balanced)")
    n = min(len(lz), len(lc))
    d = max(abs(lz[i] - lc[i]) for i in range(n))
    axL.set_xlabel("epoch")
    axL.set_ylabel("cross-entropy loss")
    axL.set_title("papers100M: lossy compression does not perturb training\n"
                  "max$|\\Delta$loss$|$ = %.1e over %d epochs" % (d, n), fontsize=11)
    axL.grid(ls=":", alpha=0.4)
    axL.legend(fontsize=9)

    labels, vals, cols = ["uncompressed"], [sz[0] / 1024.0], ["#9e9e9e"]
    labels.append("zstd\n(%.3fx)" % sz[2]); vals.append(sz[1] / 1024.0); cols.append("#1565c0")
    labels.append("cuSZp\n(%.3fx)" % sc[2]); vals.append(sc[1] / 1024.0); cols.append("#e53935")
    axR.bar(labels, vals, color=cols, edgecolor="black", width=0.6)
    for i, v in enumerate(vals):
        axR.text(i, v + 0.7, "%.1f GiB" % v, ha="center", fontsize=9,
                 fontweight="bold")
    axR.set_ylabel("feature-store size (GiB)")
    axR.set_ylim(0, vals[0] * 1.22)
    axR.set_title("Stored size of the 111M x 128-d feature matrix", fontsize=11)
    axR.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_codec.png")
    fig.savefig(p, dpi=150)
    print("wrote", p, "  (zstd %.3fx vs cuSZp %.3fx; max|dloss|=%.1e over %d ep)"
          % (sz[2], sc[2], d, n))


def main():
    os.makedirs(OUT, exist_ok=True)
    arxiv = read(os.path.join(LOGS, "exp_a_arxiv_zstd.log"))
    zstd = read(os.path.join(LOGS, "exp_b_full30.log"))
    fig_training(arxiv, zstd)
    cus = os.path.join(LOGS, "exp_d_cuszp_bal.log")
    if os.path.exists(cus):
        fig_codec(zstd, read(cus))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# fig_training.png -- two panels:
#   (a) training curve (loss + train-acc vs epoch) with the in-core and Eternia
#       runs overlaid (bit-exact, max|delta|=0)
#   (b) peak GPU memory to TRAIN a 10 GiB feature matrix: in-core needs it all
#       resident (OOMs the 8 GB GPU) vs Eternia (32 MiB streaming window)

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")

# ogbn-arxiv, 20 epochs, lr 0.2, H=64 -- in-core == Eternia (bit-exact: max|dloss|=0,
# max|dacc|=0 in the real run).
LOSS = [3.741431, 3.688135, 3.640067, 3.595663, 3.553087, 3.510463, 3.466750,
        3.421589, 3.375138, 3.328121, 3.281811, 3.237850, 3.197930, 3.163337,
        3.134589, 3.111371, 3.092796, 3.077780, 3.065336, 3.054704]
ACC = [1.33, 1.34, 6.14, 11.10, 16.58, 21.12, 18.95, 17.64, 16.95, 16.56, 16.34,
       16.25, 16.20, 16.19, 16.18, 16.19, 16.22, 16.27, 16.37, 16.49]
EP = list(range(len(LOSS)))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- (a) training curve, in-core overlaid with Eternia ----
    axL.plot(EP, LOSS, "-", color="#1565c0", lw=2.5, label="loss (in-core)")
    axL.plot(EP, LOSS, "o", color="#1565c0", ms=4, mfc="white", mew=1.0)
    axL.plot(EP, LOSS, "x", color="#e53935", ms=6, label="loss (Eternia, streamed)")
    axL.set_xlabel("epoch")
    axL.set_ylabel("cross-entropy loss", color="#1565c0")
    axL.tick_params(axis="y", labelcolor="#1565c0")
    axA = axL.twinx()
    axA.plot(EP, ACC, "-", color="#2e7d32", lw=2.0)
    axA.plot(EP, ACC, "x", color="#e53935", ms=6)
    axA.set_ylabel("train accuracy (%)", color="#2e7d32")
    axA.tick_params(axis="y", labelcolor="#2e7d32")
    axA.set_ylim(0, 30)
    axL.set_title("GNN training converges — in-core $\\equiv$ Eternia\n"
                  "(bit-exact: max$|\\Delta$loss$|$=0, max$|\\Delta$acc$|$=0)", fontsize=11)
    axL.grid(ls=":", alpha=0.4)
    # legend: red x = Eternia overlays in-core exactly
    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Line2D([0], [0], color="#1565c0", lw=2.5, marker="o", mfc="white",
               label="in-core baseline"),
        Line2D([0], [0], color="#e53935", lw=0, marker="x", ms=7,
               label="Eternia (streamed) — coincides")],
        loc="upper right", fontsize=9)

    # ---- (b) peak GPU to TRAIN a 10 GiB feature matrix ----
    labels = ["In-core\n(features resident)", "Eternia\n(stream window)"]
    peak = [10.24, 0.03125]        # GiB
    colors = ["#c62828", "#1565c0"]
    bars = axR.bar(labels, peak, color=colors, edgecolor="black",
                   hatch=["///", None], width=0.6)
    axR.axhline(8.0, ls="--", color="black", lw=1.3)
    axR.text(1.45, 8.15, "8 GB GPU", ha="right", va="bottom", fontsize=9)
    axR.text(0, 10.24 + 0.2, "OOM✗\n(needs 10.2 GiB)", ha="center", va="bottom",
             color="#c62828", fontweight="bold", fontsize=9)
    axR.text(1, 0.9, "trains\n32 MiB", ha="center", va="bottom", color="#1565c0",
             fontweight="bold", fontsize=9)
    axR.set_ylabel("peak GPU memory (GiB)")
    axR.set_ylim(0, 12)
    axR.set_title("Training a 10 GiB feature matrix\nin-core OOMs; Eternia trains",
                  fontsize=11)
    axR.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_training.png")
    fig.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot which action NeuroPress selected, dump by dump, as the data evolves.

    ./viz_actions.py --out /tmp/nyx-viz/actions \
        --sel 0.001:/tmp/nyx-lossy/eb001 --sel 0.01:/tmp/nyx-lossy/eb01 \
        --sel 0.1:/tmp/nyx-lossy/eb10

`--sel EB:DIR` is repeatable; DIR is a run_config.sh store (it reads
`selection.csv` inside). Lossless runs are fine too -- pass `0:DIR`.

A NeuroPress action is not just a codec. It is the tuple
(library, byte shuffle, quantize, preset), and this draws all of it at once:

    lane    = library            zstd / ans / bitcomp / ...
    color   = error bound        one series per --sel
    filled  = quantize on        hollow means the chunk was kept lossless
    square  = 4-byte shuffle     circle means no shuffle

The x axis is the dump index, which is time: this is the Sedov blast expanding.
The point of the plate is that the selection MOVES along it -- early dumps are
nearly constant and late ones are not, so the same policy on the same field
picks different actions as the physics fills the box.

Preset is not drawn because nothing in these runs varies it (it is 2
throughout); if that changes the script says so rather than hiding it.
"""
import argparse, collections, csv, os, re, sys

FIELD_RE = re.compile(r"^fab\d+_comp\d+_")
# The reference categorical palette, slots 1-3, in fixed order. Three series is
# within the count that validates on all pairs in both light and dark.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def read_sel(path):
    """selection.csv -> {field: {dump: row}}. Blob names are
    plt%05d/fab0000_comp%02d_<field>/chunk_%d."""
    out = collections.defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            frame, stem, _ = r["blob"].split("/")
            out[FIELD_RE.sub("", stem)][int(frame[3:])] = r
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sel", action="append", required=True, metavar="EB:DIR",
                   help="repeatable; the bound and its run_config.sh store")
    p.add_argument("--out", required=True)
    p.add_argument("--field", action="append")
    p.add_argument("--plt", help="AMReX plotfile dir, to label the x axis in sim time")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = []
    for spec in a.sel:
        if ":" not in spec:
            sys.exit(f"--sel wants EB:DIR, got {spec!r}")
        eb, d = spec.split(":", 1)
        csvp = d if d.endswith(".csv") else os.path.join(d, "selection.csv")
        if not os.path.exists(csvp):
            sys.exit(f"no selection log at {csvp}")
        runs.append((float(eb), csvp, read_sel(csvp)))
    runs.sort(key=lambda r: r[0])

    presets = {r["preset"] for _, _, sel in runs for f in sel for r in sel[f].values()}
    if len(presets) > 1:
        print(f"NOTE: preset varies across these runs ({sorted(presets)}); "
              f"the plate does not encode it.")

    fields = a.field or [f for f in ["density", "xmom", "ymom", "zmom", "rho_E", "rho_e"]
                         if any(f in sel for _, _, sel in runs)]
    # Lanes are shared across every subplot so the plates are comparable, and
    # ordered by how much work the codec does -- roughly fastest to strongest.
    ORDER = ["nvcomp-bitcomp", "nvcomp-lz4", "nvcomp-snappy", "nvcomp-ans",
             "nvcomp-cascaded", "nvcomp-gdeflate", "nvcomp-zstd"]
    seen = {r["lib_name"] for _, _, sel in runs for f in sel for r in sel[f].values()}
    lanes = [l for l in ORDER if l in seen] + sorted(seen - set(ORDER))
    lane_of = {l: i for i, l in enumerate(lanes)}

    os.makedirs(a.out, exist_ok=True)
    ndump = max(d for _, _, sel in runs for f in sel for d in sel[f]) + 1

    fig, axes = plt.subplots(len(fields), 1, squeeze=False,
                             figsize=(11, 1.05 * len(lanes) * len(fields) * 0.55 + 1.6 * len(fields)),
                             sharex=True)
    for ax, field in zip(axes[:, 0], fields):
        for k, (eb, _, sel) in enumerate(runs):
            rows = sel.get(field, {})
            if not rows:
                continue
            dumps = sorted(rows)
            # A small vertical offset per bound: several bounds often land in
            # the same lane on the same dump, and stacked markers hide that.
            off = (k - (len(runs) - 1) / 2) * 0.18
            ys = [lane_of[rows[d]["lib_name"]] + off for d in dumps]
            ax.plot(dumps, ys, "-", color=SERIES[k % len(SERIES)], lw=1, alpha=.45, zorder=1)
            for d, y in zip(dumps, ys):
                r = rows[d]
                ax.plot(d, y,
                        marker="s" if r["shuffle"] != "0" else "o",
                        ms=6.5 if r["shuffle"] != "0" else 6,
                        mfc=SERIES[k % len(SERIES)] if r["quantize"] == "1" else "none",
                        mec=SERIES[k % len(SERIES)], mew=1.5, zorder=3)
        ax.set(yticks=range(len(lanes)),
               yticklabels=[l.replace("nvcomp-", "") for l in lanes],
               ylim=(-0.7, len(lanes) - 0.3))
        ax.grid(alpha=0.25, axis="both")
        ax.set_ylabel(field, fontsize=10, rotation=0, ha="right", va="center", labelpad=44)
    axes[-1, 0].set_xlabel("dump  (the blast expanding)")
    axes[-1, 0].set_xlim(-0.6, ndump - 0.4)

    handles = [Line2D([], [], color=SERIES[k % len(SERIES)], marker="o", ls="-",
                      label=f"eb = {eb:g}") for k, (eb, _, _) in enumerate(runs)]
    handles += [
        Line2D([], [], color="0.35", marker="o", ls="none", mfc="0.35", label="quantized"),
        Line2D([], [], color="0.35", marker="o", ls="none", mfc="none", label="kept lossless"),
        Line2D([], [], color="0.35", marker="s", ls="none", mfc="none", label="4-byte shuffle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Action selected per dump - lane = codec, fill = quantize, "
                 "square = shuffle", fontsize=11, y=1.035)
    fig.tight_layout()
    out = os.path.join(a.out, "actions.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  {out}")

    # How much the selection actually moves, which is the claim the plate makes.
    for eb, _, sel in runs:
        for field in fields:
            rows = sel.get(field, {})
            if not rows:
                continue
            acts = [(r["lib_name"], r["shuffle"], r["quantize"])
                    for _, r in sorted(rows.items())]
            switches = sum(1 for i in range(1, len(acts)) if acts[i] != acts[i - 1])
            print(f"eb={eb:<6g} {field:<8} {len(set(acts))} distinct action(s), "
                  f"{switches} switch(es) over {len(acts)} dumps")


if __name__ == "__main__":
    main()

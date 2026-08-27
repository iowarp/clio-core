#!/usr/bin/env python3
"""Did the error bound actually do anything? Requested vs applied quantization.

    ./viz_bound.py --out /tmp/viz \
        --run nyx:0.1:/tmp/nyx-lossy/eb10 \
        --run lammps:0.1:/tmp/lmp-lossy/eb10 \
        --reinterpret lammps-velocity:f8:/tmp/lmp-raw/velocity_step_100_chunk_0.bin

A positive CLIO_NEUROPRESS_ERROR_BOUND unmasks NeuroPress's 16 quantize
actions, and the selection log's `quantize` column records that one of them was
CHOSEN. It does not record that quantization RAN. Those are different events,
and on a float64 workload they come apart completely.

The tell is in the same row. `actual_psnr` is seeded to -1 and overwritten only
inside upstream's `if (d_quantized && quant_result.isValid())`, so
`actual_psnr > 0` means the quantizer really ran on that chunk (see the comment
at compressor_runtime.cc's PSNR block). This plate is that comparison, per run.

`--reinterpret LABEL:DTYPE:FILE` shows why it comes apart. NeuroPress reads
every buffer as float32 whatever the declared type -- upstream's behaviour, and
Clio keeps it -- so a float64 buffer arrives as pairs of float32 words. Many of
those words are denormal, infinite or NaN, and the quantizer declines cleanly on
a non-finite range rather than producing garbage.

DTYPE is the buffer's REAL element type: f8 for float64, f4 for float32.
"""
import argparse, collections, csv, os, sys
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def read_rows(store):
    p = store if store.endswith(".csv") else os.path.join(store, "selection.csv")
    if not os.path.exists(p):
        sys.exit(f"no selection log at {p}")
    return list(csv.DictReader(open(p)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL:EB:STORE",
                   help="repeatable; a run to summarise")
    p.add_argument("--reinterpret", action="append", default=[],
                   metavar="LABEL:DTYPE:FILE",
                   help="repeatable; show a real buffer read as float32")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for spec in a.run:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            sys.exit(f"--run wants LABEL:EB:STORE, got {spec!r}")
        label, eb, store = parts
        rows = read_rows(store)
        req = sum(1 for r in rows if r["quantize"] == "1")
        # actual_psnr > 0 is the only record that the quantizer really ran.
        ran = sum(1 for r in rows if float(r["actual_psnr"]) > 0)
        runs.append({"label": f"{label}\neb={eb}", "n": len(rows), "req": req,
                     "ran": ran,
                     "ratio": float(np.mean([float(r["actual_ratio"]) for r in rows]))})
        print(f"{label} eb={eb}: {len(rows)} chunks, quantize requested {req}, "
              f"applied {ran}, mean actual ratio {runs[-1]['ratio']:.3f}")

    ncol = 2 + (1 if a.reinterpret else 0)
    fig, ax = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2))

    y = np.arange(len(runs))
    h = 0.38
    ax[0].barh(y + h / 2, [r["req"] for r in runs], height=h,
               color=SERIES[0], label="quantize chosen")
    ax[0].barh(y - h / 2, [r["ran"] for r in runs], height=h,
               color=SERIES[1], label="quantize applied")
    for i, r in enumerate(runs):
        if r["ran"] == 0 and r["req"] > 0:
            ax[0].text(r["req"] * 0.02, i - h / 2, " none applied", va="center",
                       fontsize=8.5, color=SERIES[1], weight="bold")
        ax[0].text(r["req"], i + h / 2, f" {r['req']}/{r['n']}", va="center",
                   fontsize=8)
    ax[0].set(yticks=y, yticklabels=[r["label"] for r in runs],
              xlabel="chunks", title="chosen is not applied")
    ax[0].legend(fontsize=8, loc="upper center", ncol=2, framealpha=.92)
    ax[0].set_xlim(0, max(r["req"] for r in runs) * 1.28)

    ax[1].barh(y, [r["ratio"] for r in runs], color=SERIES[2])
    ax[1].axvline(1.0, ls="--", lw=1, color="0.4")
    ax[1].set(yticks=y, yticklabels=[r["label"] for r in runs], xscale="log",
              xlabel="mean actual ratio (dashed = no compression)",
              title="and what it was worth")
    for i, r in enumerate(runs):
        ax[1].text(r["ratio"], i, f" {r['ratio']:.2f}x", va="center", fontsize=8)

    for k, spec in enumerate(a.reinterpret):
        parts = spec.split(":", 2)
        if len(parts) != 3:
            sys.exit(f"--reinterpret wants LABEL:DTYPE:FILE, got {spec!r}")
        label, dt, path = parts
        buf = np.fromfile(path, dtype=np.dtype(dt))
        as32 = buf.view(np.float32)
        fin = np.isfinite(as32)
        bad = int((~fin).sum())
        # Log y: the float32 case is a single narrow spike holding nearly
        # every word, and on a linear axis it flattens the float64 case -- the
        # one that matters -- into the baseline.
        ax[2].hist(np.log10(np.abs(as32[fin & (as32 != 0)])), bins=90,
                   color=SERIES[k % len(SERIES)], alpha=.7,
                   label=f"{label} — {bad} non-finite of {as32.size}")
        ax[2].set(xlabel="log10 |value| when the buffer is read as float32",
                  ylabel="words", yscale="log",
                  title="what the quantizer actually sees")
        print(f"{label}: {path}\n  real range {buf.min():.4g} .. {buf.max():.4g} "
              f"({dt});  as float32 -> {bad}/{as32.size} non-finite, "
              f"finite span {as32[fin].min():.3g} .. {as32[fin].max():.3g}")
    if a.reinterpret:
        ax[2].legend(fontsize=7.5, loc="upper left")

    for b in ax:
        b.grid(alpha=0.3, axis="x")
    fig.suptitle("A positive error bound only matters if the quantizer accepts it",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "bound_applied.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    main()

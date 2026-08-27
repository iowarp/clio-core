#!/usr/bin/env python3
"""Compare the original Nyx fields against what came back from a lossy tier.

    ./viz_lossy.py --orig /tmp/nyx-quick --out /tmp/nyx-viz/lossy \
                   --compare 0.01:/tmp/nyx-decomp/eb01 \
                   --compare 0.1:/tmp/nyx-decomp/eb10

`--compare EB:DIR` is repeatable and names one bound and the directory of
`--dump-decompressed` output produced at it: one .bin per blob, named
<frame>_<stem>_chunk_<i>.bin. This assumes ONE CHUNK PER FIELD FILE, so run the
sweep with --chunk equal to the per-component size (ncell^3 x 4) or the files
here are chunk fragments and will not reshape.

The plate is original on top, then one decompressed row per bound, then one
|error| row per bound. Every field row shares ONE color scale so the rows are
comparable; each error row is scaled to ITS OWN bound, so what it shows is the
fraction of that budget each cell spent -- across bounds compare the
decompressed rows, not the error rows.

Two things worth knowing before reading the pictures:

  The bound is ABSOLUTE. |orig - decoded| <= eb, the same eb everywhere, so
  what it costs you depends entirely on the local magnitude. On Sedov density,
  ambient is 1.0 and the evacuated interior falls to ~0.002, so one bound is
  simultaneously invisible outside the shock and worth ~80% of the value
  inside it. The |error| row is on a fixed 0..eb scale for exactly that reason.

  The bound is a CEILING, not an instruction. NeuroPress decides per chunk
  whether quantizing is worth it, so a "lossy" run is lossy only where the
  selector took the offer -- on this data it quantizes every density chunk and
  leaves rho_E and rho_e bit-exact.
"""
import argparse, glob, os, re, sys
import numpy as np
import importlib.util as iu

_here = os.path.dirname(os.path.abspath(__file__))
_spec = iu.spec_from_file_location("viz", os.path.join(_here, "viz_fields.py"))
viz = iu.module_from_spec(_spec); _spec.loader.exec_module(viz)

FIELDS = ["density", "xmom", "ymom", "zmom", "rho_E", "rho_e"]


def decomp_path(dec_dir, frame_dir, stem):
    """--dump-decompressed flattens the blob name, '/' -> '_'."""
    return os.path.join(dec_dir, f"{frame_dir}_{stem}_chunk_0.bin")


def pairs(orig_dir, dec_dir, field):
    """(original, decompressed) per frame, in frame order, skipping any frame
    whose decompressed blob is missing rather than silently misaligning."""
    out = []
    for src in sorted(glob.glob(os.path.join(orig_dir, "plt*",
                                             f"fab0000_comp*_{field}.f32"))):
        frame = os.path.basename(os.path.dirname(src))
        stem = os.path.basename(src)[:-4]
        dec = decomp_path(dec_dir, frame, stem)
        if os.path.exists(dec):
            out.append((frame, src, dec))
    return out


def load_bin(path, n):
    a = np.fromfile(path, dtype=np.float32)
    if len(a) != n ** 3:
        sys.exit(f"{path}: {len(a)} floats, expected {n**3} -- was the sweep "
                 f"run with --chunk {n**3 * 4}?")
    return a.reshape(n, n, n, order="F")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", action="append", metavar="EB:DIR", default=[],
                   help="repeatable; the bound and its --dump-decompressed dir")
    p.add_argument("--decomp", help="single-bound form; use with --eb")
    p.add_argument("--eb", type=float, help="bound for the single --decomp form")
    p.add_argument("--field", action="append")
    p.add_argument("--plt", help="AMReX plotfile dir, for sim times")
    p.add_argument("--cols", type=int, default=5, help="frames per plate")
    p.add_argument("--no-error-rows", action="store_true",
                   help="reconstructions only; drop the |error| rows")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    runs = []
    for spec in a.compare:
        if ":" not in spec:
            sys.exit(f"--compare wants EB:DIR, got {spec!r}")
        eb, d = spec.split(":", 1)
        runs.append((float(eb), d))
    if a.decomp:
        if a.eb is None:
            sys.exit("--decomp needs --eb")
        runs.append((a.eb, a.decomp))
    if not runs:
        sys.exit("give at least one --compare EB:DIR")
    runs.sort(key=lambda r: r[0])
    for eb, d in runs:
        if not os.path.isdir(d):
            sys.exit(f"no such decompressed directory: {d}")

    os.makedirs(a.out, exist_ok=True)
    times = viz.plot_times(a.plt)
    fields = a.field or FIELDS
    summary = []

    for field in fields:
        base = pairs(a.orig, runs[0][1], field)
        if not base:
            print(f"{field}: no matched pairs, skipping"); continue
        n = round(len(np.fromfile(base[0][1], dtype=np.float32)) ** (1 / 3))
        O = [viz.load(src) for _, src, _ in base]

        # One decompressed series per bound, aligned to the same frame list, so
        # a bound whose readback is short is caught rather than silently
        # shifting the comparison by a frame.
        Ds, Es, worsts = [], [], []
        for eb, d in runs:
            pr = pairs(a.orig, d, field)
            if len(pr) != len(base):
                sys.exit(f"{field}: {d} has {len(pr)} frames, {runs[0][1]} has "
                         f"{len(base)} -- the two runs do not line up")
            D = [load_bin(dp, n) for _, _, dp in pr]
            E = [np.abs(o - x) for o, x in zip(O, D)]
            Ds.append(D); Es.append(E); worsts.append(max(float(e.max()) for e in E))

        for (eb, _), E, worst in zip(runs, Es, worsts):
            summary.append((field, eb, worst, sum(1 for e in E if e.max() > 0),
                            len(E), float(min(v.min() for v in O)),
                            float(max(v.max() for v in O))))
            print(f"{field} @ eb={eb}: {sum(1 for e in E if e.max() > 0)}"
                  f"/{len(E)} frames altered, worst |err| = {worst:.3e}")
        if max(worsts) == 0:
            print(f"  {field}: left BIT-EXACT at every bound -- no plate drawn")
            continue

        # Same scale for original and decompressed, or the comparison is a lie.
        lo = min(float(v.min()) for v in O); hi = max(float(v.max()) for v in O)
        pool = np.concatenate([v.ravel()[::7] for v in O])
        if lo > 0 and hi / lo > 50:
            norm, cmap = LogNorm(lo, hi), "inferno"
        elif lo < 0:
            m = float(np.percentile(np.abs(pool), 99.9)) or max(abs(lo), hi)
            norm, cmap = Normalize(-m, m), "RdBu_r"
        else:
            norm, cmap = Normalize(lo, float(np.percentile(pool, 99.9))), "inferno"
        # Error on a FIXED 0..eb scale: the question is how much of the budget
        # each cell spent, and a per-frame autoscale would hide that.
        enorm = Normalize(0, a.eb)

        idx = np.unique(np.linspace(0, len(O) - 1, min(a.cols, len(O))).astype(int))
        mid = n // 2
        # original, each decompressed, and (unless suppressed) each error
        nrow = 1 + len(runs) * (1 if a.no_error_rows else 2)
        fig, axes = plt.subplots(nrow, len(idx),
                                 figsize=(2.15 * len(idx), 2.3 * nrow), squeeze=False)
        data_rows, err_rows = [0], []
        for c, i in enumerate(idx):
            lab = (f"t={times[i]:.4g}" if times and i < len(times) else f"dump {i}")
            im0 = axes[0][c].imshow(O[i][:, :, mid].T, origin="lower", norm=norm, cmap=cmap)
            axes[0][c].set_title(lab, fontsize=8)
            for k, (eb, _) in enumerate(runs):
                r = 1 + k
                axes[r][c].imshow(Ds[k][i][:, :, mid].T, origin="lower", norm=norm, cmap=cmap)
                if c == 0: data_rows.append(r)
            for k, (eb, _) in enumerate([] if a.no_error_rows else runs):
                r = 1 + len(runs) + k
                # Each error row against ITS OWN bound: the panel reads as the
                # fraction of that budget spent. A shared scale would render the
                # tighter bound near-black and say nothing about its structure.
                im2 = axes[r][c].imshow(Es[k][i][:, :, mid].T, origin="lower",
                                        norm=Normalize(0, eb), cmap="magma")
                axes[r][c].set_xlabel(f"{Es[k][i].max():.1e}", fontsize=6.5)
                if c == 0: err_rows.append(r)
            for r in range(nrow):
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        labels = ["original"] + [f"eb = {eb:g}" for eb, _ in runs]
        if not a.no_error_rows:
            labels += [f"|err| / {eb:g}" for eb, _ in runs]
        for r, t in enumerate(labels):
            axes[r][0].set_ylabel(t, fontsize=9)
        fig.suptitle(f"{field} - original vs NeuroPress lossy at "
                     + " and ".join(f"eb={eb:g}" for eb, _ in runs), fontsize=12)
        fig.colorbar(im0, ax=axes[:1 + len(runs), :].ravel().tolist(),
                     shrink=0.5, label=field)
        if not a.no_error_rows:
            # Each error row is normalised to a DIFFERENT bound, so one shared
            # colorbar can only be honest in units of the bound itself --
            # labelling it with either row's absolute values would misread the
            # other.
            fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), "magma"),
                         ax=axes[1 + len(runs):, :].ravel().tolist(),
                         shrink=0.7, label="|error| as a fraction of its bound")
        out = os.path.join(a.out, f"{field}_sidebyside.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  {out}")

    # One plate for the whole run: which fields each bound actually reached.
    # Colors are the reference categorical palette's slots 1-2, in fixed order.
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    if summary:
        by_eb = {}
        for f, eb, worst, alt, tot, lo, hi in summary:
            by_eb.setdefault(eb, []).append((f, worst, alt, tot, lo, hi))
        names = [r[0] for r in by_eb[runs[0][0]]]
        y = np.arange(len(names))
        h = 0.8 / len(runs)
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 0.62 * len(names) * len(runs) + 2.2))

        for k, (eb, _) in enumerate(runs):
            off = (k - (len(runs) - 1) / 2) * h
            rows = by_eb[eb]
            # Zero is not plottable on a log axis and "bit-exact" is the most
            # important outcome here, so it gets a label rather than a bar.
            # LINEAR, deliberately. On a log axis every bar runs from an
            # arbitrary floor and 0.0095 vs 0.095 look the same length, which
            # is the one comparison this panel exists to make.
            ax[0].barh(y + off, [r[1] for r in rows], height=h * 0.92,
                       color=SERIES[k % len(SERIES)], label=f"eb = {eb:g}")
            ax[0].axvline(eb, ls="--", lw=1, color=SERIES[k % len(SERIES)])
            for i, r in enumerate(rows):
                if r[1] == 0:
                    ax[0].text(max(e for e, _ in runs) * 0.012, y[i] + off, "bit-exact",
                               va="center", fontsize=7.5,
                               color=SERIES[k % len(SERIES)], weight="bold")
            ax[1].barh(y + off, [100 * r[2] / r[3] for r in rows], height=h * 0.92,
                       color=SERIES[k % len(SERIES)], label=f"eb = {eb:g}")

        ax[0].set(yticks=y, yticklabels=names,
                  xlabel="worst |error| over the run  (dashed = the bound asked for)",
                  title="what each bound actually cost")
        ax[0].set_xlim(0, max(eb for eb, _ in runs) * 1.12)
        ax[1].set(yticks=y, yticklabels=names, xlabel="% of dumps altered",
                  title="how much of the run it touched", xlim=(0, 105))
        for b in ax:
            b.grid(alpha=0.3, axis="x"); b.legend(fontsize=8, loc="lower right")
        fig.suptitle("Absolute error bounds across the Nyx hydro state", fontsize=12)
        fig.tight_layout()
        out = os.path.join(a.out, "lossy_summary.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Original vs decompressed, at the beginning, middle and last frame.

    ./figure_lossy.py --orig-source f32 --orig-dir DUMPS \
                      --decomp-dir DECOMP --eb 0.01 --workload nyx --out-dir FIGS

Section 6 of the benchmark spec. Writes three files, named as section 7 asks:

    beginning_original_vs_decompressed.png
    middle_original_vs_decompressed.png
    last_original_vs_decompressed.png

Each is three panels: the original, what came back through the tier, and
|original - decoded|.

THE FIRST TWO PANELS SHARE ONE COLOR SCALE, computed over both, so the
comparison is of the data and not of two independent autoscalings. THE ERROR
PANEL IS ON ITS OWN 0..eb SCALE, deliberately: what it shows is the fraction
of the error budget each cell actually spent, which is the question worth
asking. Do not compare error panels ACROSS bounds -- each is normalised to a
different eb; compare the decompressed panels instead.

THE BOUND IS ABSOLUTE, so what it costs depends entirely on local magnitude.
On Sedov density the ambient is 1.0 and the evacuated interior reaches ~0.002,
so one bound is simultaneously invisible outside the shock and worth most of
the value inside it.

THE BOUND IS ALSO A CEILING, NOT AN INSTRUCTION. NeuroPress decides per chunk
whether quantizing is worth it, so a "lossy" run is lossy only where the
selector took the offer. An error panel that is entirely zero means that chunk
was stored bit-exact, not that the figure is broken -- the caption says which.

DECOMPRESSED BLOBS ARE PER CHUNK and have to be concatenated in chunk order to
rebuild a field: <frame>_<stem>_chunk_<i>.bin. Sorting those names as strings
puts chunk_10 before chunk_2, so the index is parsed as an integer.
"""
import argparse, glob, importlib.util as iu, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

_here = os.path.dirname(os.path.abspath(__file__))
_spec = iu.spec_from_file_location("ev", os.path.join(_here, "evolution.py"))
ev = iu.module_from_spec(_spec); _spec.loader.exec_module(ev)
_fspec = iu.spec_from_file_location("fe", os.path.join(_here, "figure_evolution.py"))
fe = iu.module_from_spec(_fspec); _fspec.loader.exec_module(fe)


def decompressed_frames(d, field, dtype):
    """-> {frame_key: 1-D array}, chunks concatenated in numeric order.

    TWO NAMINGS, because the drivers name blobs differently and the dump just
    flattens whatever the blob was called:

        plt00000_fab0000_comp00_density_chunk_0.bin   replay (Nyx)
        force_step_0_chunk_0.bin                      library / in situ

    Sorting these as strings puts chunk_10 before chunk_2 and step_100 before
    step_40, so both the chunk index and the frame key are parsed as integers.
    """
    out = {}
    for p in glob.glob(os.path.join(d, "*.bin")):
        b = os.path.basename(p)
        if "_chunk_" not in b:
            continue
        stem, _, tail = b.rpartition("_chunk_")
        try:
            cidx = int(tail[:-4])
        except ValueError:
            continue
        m = re.match(r"^(?P<field>.+)_step_(?P<step>\d+)$", stem)
        if m:                                   # library / in-situ naming
            fname, frame = m["field"], int(m["step"])
        else:
            m = re.match(r"^(?P<frame>plt\d+)_(?P<rest>.+)$", stem)
            if not m:
                continue
            frame = int(m["frame"][3:])         # dump index, numerically
            fname = m["rest"]
        # Exact suffix match: rho_e must not also collect rho_E's chunks, and
        # a substring test would let it.
        if fname != field and not fname.endswith("_" + field):
            continue
        out.setdefault(frame, []).append((cidx, p))
    return {k: np.concatenate([np.fromfile(q, dtype=dtype) for _, q in sorted(v)])
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-source", required=True, choices=sorted(ev.SOURCES))
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--decomp-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eb", type=float, required=True)
    ap.add_argument("--workload", default="")
    ap.add_argument("--field", default="")
    ap.add_argument("--pick-from", default="",
                    help="results.csv: plot the field the selector quantized most")
    ap.add_argument("--axis", default="z", choices=["x", "y", "z"])
    ap.add_argument("--shape", default="")
    ap.add_argument("--f64", action="store_true")
    ap.add_argument("--atoms", action="store_true")
    a = ap.parse_args()

    # PICK THE FIELD THE SELECTOR ACTUALLY QUANTIZED. A lossy run is lossy only
    # where the model took the offer, so plotting a fixed default can publish
    # three identical panels and an empty error map from a working run --
    # measured: at eb=0.01 Nyx quantized rho_E and rho_e 6/6 chunks each and
    # left density and all three momenta bit-exact.
    field = a.field
    if not field and a.pick_from and os.path.exists(a.pick_from):
        import csv, collections
        n = collections.Counter()
        for row in csv.DictReader(open(a.pick_from)):
            if row.get("quantize") == "1" and row.get("field"):
                n[row["field"]] += 1
        if n:
            field = n.most_common(1)[0][0]
    field = field or fe.DEFAULT_FIELD.get(a.workload, "")
    dtype = np.float64 if a.f64 else np.float32
    want = [field] if field else None
    if a.orig_source == "openpmd" and field:
        m = re.fullmatch(r"([EBj])([xyz])", field)
        want = [f"{m[1]}/{m[2]}"] if m else [field]

    orig = list(ev.SOURCES[a.orig_source](a.orig_dir, dtype, want))
    dec = decompressed_frames(a.decomp_dir, field, dtype)
    if not orig or not dec:
        sys.exit(f"nothing to compare: {len(orig)} original frame(s), "
                 f"{len(dec)} decompressed frame(s) for field {field!r}")

    keys = sorted(dec)                       # frame keys, in dump order
    n = min(len(orig), len(keys))
    pick = [(0, "beginning"), (n // 2, "middle"), (n - 1, "last")]
    shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else None
    os.makedirs(a.out_dir, exist_ok=True)

    for i, label in pick:
        o = orig[i][1][field].astype(np.float64)
        d = dec[keys[i]].astype(np.float64)
        if len(d) != len(o):
            print(f"   {label}: size mismatch {len(o)} vs {len(d)}, skipped")
            continue
        err = np.abs(o - d)

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
        if a.atoms:
            po, pd = o.reshape(-1, 3), d.reshape(-1, 3)
            axes[0].scatter(po[:, 0], po[:, 1], s=0.4, c="tab:blue", linewidths=0)
            axes[1].scatter(pd[:, 0], pd[:, 1], s=0.4, c="tab:orange", linewidths=0)
            e = np.linalg.norm(po - pd, axis=1)
            im = axes[2].scatter(po[:, 0], po[:, 1], s=0.4, c=e, cmap="magma",
                                 vmin=0, vmax=a.eb, linewidths=0)
            for ax in axes:
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=axes[2], fraction=0.046)
        else:
            so = fe.slice_of(o, a.axis, shape)
            sd = fe.slice_of(d, a.axis, shape)
            se = fe.slice_of(err, a.axis, shape)
            if so is None or sd is None:
                sys.exit(f"{len(o)} elements does not match "
                         f"{'shape ' + a.shape if shape else 'a cube'}; use --shape")
            so, sd, se = so[0], sd[0], se[0]
            both = np.concatenate([so.ravel(), sd.ravel()])
            fin = both[np.isfinite(both)]
            lo, hi = np.percentile(fin, [1, 99]) if fin.size else (0, 1)
            if lo == hi:
                lo, hi = lo - 1e-12, hi + 1e-12
            if fin.min() > 0 and hi / max(lo, 1e-300) > 50:
                norm, cmap = LogNorm(vmin=max(lo, fin.min()), vmax=hi), "inferno"
            elif fin.min() < 0:
                m = max(abs(lo), abs(hi)); norm, cmap = Normalize(-m, m), "RdBu_r"
            else:
                norm, cmap = Normalize(lo, hi), "inferno"
            im0 = axes[0].imshow(so.T, origin="lower", norm=norm, cmap=cmap)
            axes[1].imshow(sd.T, origin="lower", norm=norm, cmap=cmap)
            im2 = axes[2].imshow(se.T, origin="lower", vmin=0, vmax=a.eb, cmap="magma")
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im0, ax=axes[:2], fraction=0.02, pad=0.02)
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

        worst = float(err.max())
        axes[0].set_title("original")
        axes[1].set_title("decompressed")
        # Say outright whether the selector actually quantized this frame; a
        # blank error panel otherwise reads as a broken figure.
        axes[2].set_title(f"|error|, scale 0..{a.eb:g}"
                          + ("  (stored bit-exact)" if worst == 0 else ""))
        fig.suptitle(f"{a.workload or ''} {field} — {label} frame, eb={a.eb:g}, "
                     f"worst |error| {worst:.4g}"
                     f"{'  EXCEEDS BOUND' if worst > a.eb else ''}", y=0.99)
        out = os.path.join(a.out_dir, f"{label}_original_vs_decompressed.png")
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"   {out}  worst |err| {worst:.4g}"
              f"{'  EXCEEDS eb' if worst > a.eb else ''}")


if __name__ == "__main__":
    main()

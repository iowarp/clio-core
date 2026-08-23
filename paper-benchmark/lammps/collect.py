#!/usr/bin/env python3
"""Aggregate a paper-benchmark sweep into summary.csv and summary.md.

Reads each run directory's meta.json (what was configured) and blobs.csv (what
happened to every chunk) and reports the numbers a paper needs: compression
ratio, bytes stored, how many chunks the codec actually shrank, the codec mix,
per-field ratios, and time.

Usage: ./collect.py results/ [-o OUTDIR]
"""
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

FIELDS = ("position", "velocity", "force")


def load_run(d):
    """One run directory -> a dict of aggregates, or None if unusable."""
    meta_p, blobs_p = os.path.join(d, "meta.json"), os.path.join(d, "blobs.csv")
    if not (os.path.isfile(meta_p) and os.path.isfile(blobs_p)):
        return None
    with open(meta_p) as f:
        meta = json.load(f)
    rows = list(csv.DictReader(open(blobs_p)))
    if not rows:
        return None

    total_in = sum(int(r["bytes"]) for r in rows)
    # `stored` already accounts for a chunk the codec could not shrink: the
    # compressor stores the caller's bytes and reports lib=0, so stored ==
    # bytes there. Summing it is the true tier footprint.
    total_stored = sum(int(r["stored"]) for r in rows)
    compressed = [r for r in rows if r["lib"] != "0"]

    per_field = {}
    for f in FIELDS:
        sel = [r for r in rows if r["blob"].startswith(f + "/")]
        if sel:
            i = sum(int(r["bytes"]) for r in sel)
            s = sum(int(r["stored"]) for r in sel)
            per_field[f] = round(i / s, 4) if s else 0.0

    mix = Counter(r["codec"] if r["lib"] != "0" else "raw" for r in rows)
    ms = [float(r["compress_ms"]) for r in rows]

    verified = None
    out_p = os.path.join(d, "stdout.log")
    if os.path.isfile(out_p):
        txt = open(out_p, errors="replace").read()
        if "VERIFIED:" in txt:
            verified = "pass"
        elif "FAILED:" in txt:
            verified = "FAIL"

    # Physics that differs from the deck's defaults. Part of the identity of a
    # run: two runs of the same config at different density or temperature are
    # different experiments and must not be averaged together.
    phys = meta.get("physics", {}) or {}
    phys_set = {k: v for k, v in phys.items() if v not in ("deck", "", None)}
    phys_key = " ".join(f"{k}={v}" for k, v in sorted(phys_set.items()))

    return {
        "tag": meta.get("tag", os.path.basename(d)),
        "config": meta.get("config", ""),
        "physics": phys_key or "deck defaults",
        "device": meta.get("device", ""),
        "atoms": meta.get("atoms", 0),
        "frames": meta.get("frames", 0),
        "chunks": len(rows),
        "payload_bytes": total_in,
        "stored_bytes": total_stored,
        "ratio": round(total_in / total_stored, 4) if total_stored else 0.0,
        "compressed_chunks": len(compressed),
        "raw_chunks": len(rows) - len(compressed),
        "compress_ms_sum": round(sum(ms), 1),
        "compress_ms_median": round(statistics.median(ms), 3),
        "wall_s": round(float(meta.get("wall_s", 0)), 2),
        "verified": verified or "n/a",
        "rc": meta.get("rc", 0),
        "codec_mix": " ".join(f"{k}:{v}" for k, v in mix.most_common()),
        **{f"ratio_{f}": per_field.get(f, 0.0) for f in FIELDS},
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = args[0] if args else "results"
    outdir = root
    if "-o" in sys.argv:
        outdir = sys.argv[sys.argv.index("-o") + 1]
    if not os.path.isdir(root):
        sys.exit(f"no such directory: {root}")

    runs = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            r = load_run(d)
            if r:
                runs.append(r)
    if not runs:
        sys.exit(f"no completed runs found under {root}/")

    cols = list(runs[0].keys())
    csv_p = os.path.join(outdir, "summary.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(runs)

    # Average repeats -- but only of runs that are the same experiment.
    # Config AND physics both have to match, or a sweep repeated at a second
    # temperature would silently collapse into the first one's numbers.
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r["config"] or r["tag"], r["physics"])].append(r)
    multi_physics = len({r["physics"] for r in runs}) > 1

    lines = ["# LAMMPS paper benchmark", ""]
    a = runs[0]
    lines += [
        f"Workload: LJ melt, {a['atoms']:,} atoms, {a['frames']} frames, "
        f"{a['payload_bytes'] / 1048576:.1f} MiB float64, {a['chunks']} chunks, "
        f"{a['device'].upper()}."
        + (f" Physics: {a['physics']}."
           if not multi_physics
           else " Runs below differ in physics -- see the column."),
        "",
        "| config |" + (" physics |" if multi_physics else "") +
        " ratio | stored | compressed/raw | position | velocity | force "
        "| compress ms | wall s | verified |",
        "|---|" + ("---|" if multi_physics else "") +
        "---|---|---|---|---|---|---|---|---|",
    ]
    for (cfg, phys), rs in by_cfg.items():
        n = len(rs)
        avg = lambda k: sum(r[k] for r in rs) / n  # noqa: E731
        ver = "pass" if all(r["verified"] == "pass" for r in rs) else \
              ("FAIL" if any(r["verified"] == "FAIL" for r in rs) else "n/a")
        note = f" (n={n})" if n > 1 else ""
        physcol = f" {phys} |" if multi_physics else ""
        lines.append(
            f"| `{cfg}`{note} |{physcol} **{avg('ratio'):.3f}×** | "
            f"{avg('stored_bytes') / 1048576:.1f} MiB | "
            f"{avg('compressed_chunks'):.0f} / {avg('raw_chunks'):.0f} | "
            f"{avg('ratio_position'):.3f}× | {avg('ratio_velocity'):.3f}× | "
            f"{avg('ratio_force'):.3f}× | {avg('compress_ms_sum'):.0f} | "
            f"{avg('wall_s'):.1f} | {ver} |")
    lines += ["", "## Codec mix", ""]
    for (cfg, phys), rs in by_cfg.items():
        label = f"`{cfg}`" + (f" ({phys})" if multi_physics else "")
        lines.append(f"- {label}: {rs[0]['codec_mix']}")
    lines += ["", f"Per-run detail: `summary.csv`. Per-chunk detail: "
                  f"`<run>/blobs.csv`; NeuroPress's own choice per chunk: "
                  f"`<run>/selection.csv`; every measured candidate (exploration "
                  f"only): `<run>/explore.csv`.", ""]

    md_p = os.path.join(outdir, "summary.md")
    open(md_p, "w").write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nwrote {csv_p} and {md_p}")


if __name__ == "__main__":
    main()

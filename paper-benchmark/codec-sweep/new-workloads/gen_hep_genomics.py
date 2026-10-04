#!/usr/bin/env python3
"""Particle-physics and genomics data fields, as the columnar arrays their
analysis frameworks hold, one flat little-endian file per (array kind, batch).

    gen_hep_genomics.py nanoaod --input PATH_OR_URL --out DIR [--events-per-batch 200000]
    gen_hep_genomics.py bam     --input URL --out DIR --regions 1,2 [--reads-per-batch 2000000]

nanoaod  a CMS NanoAOD ROOT file's Events tree, decompressed, with every
         branch kept contiguous (as a columnar engine holds it) and grouped
         by what it is:
           jagged_f32  per-object floats (Muon_pt, Jet_eta, ...), flattened
           counts_u32  per-event object counts (nMuon, nJet, ...)
           flags_u8    per-event booleans (HLT_*, Flag_*)
           event_f32   per-event floats (MET_pt, fixedGridRhoFastjet, ...)
           ints        every other integer branch, at its own width
         -> <kind>_b<NN>.bin for event batch NN
bam      aligned reads of the given regions of a BAM (fetched remotely with
         its index), batch by batch in coordinate order:
           pos.i32 mapq.u8 flag.u16 tlen.i32 matepos.i32 rlen.u16
           qual.u8 (base qualities, concatenated) seq.u8 (bases, ASCII)
         -> <array>_b<NN>.<dtype>
"""
import argparse
import json
import os

import numpy as np


def cmd_nanoaod(a):
    import awkward as ak
    import uproot
    t = uproot.open(a.input)["Events"]
    groups = {"jagged_f32": [], "counts_u32": [], "flags_u8": [], "event_f32": [], "ints": []}
    for name, br in t.items():
        typ = str(br.typename)
        if "[]" in typ and "float" in typ:
            groups["jagged_f32"].append(name)
        elif name.startswith("n") and "[]" not in typ and "int" in typ:
            groups["counts_u32"].append(name)
        elif typ == "bool":
            groups["flags_u8"].append(name)
        elif typ in ("float", "double") and "[]" not in typ:
            groups["event_f32"].append(name)
        elif "int" in typ or "short" in typ or "char" in typ:
            groups["ints"].append(name)
    print({k: len(v) for k, v in groups.items()}, flush=True)
    for b, start in enumerate(range(0, t.num_entries, a.events_per_batch)):
        stop = min(start + a.events_per_batch, t.num_entries)
        index = {}
        for kind, names in groups.items():
            if not names:
                continue
            arrs = t.arrays(names, entry_start=start, entry_stop=stop, library="ak")
            parts, pos = [], 0
            for n in names:
                x = ak.to_numpy(ak.flatten(arrs[n], axis=None))
                if kind == "jagged_f32" or kind == "event_f32":
                    x = x.astype(np.float32)
                elif kind == "counts_u32":
                    x = x.astype(np.uint32)
                elif kind == "flags_u8":
                    x = x.astype(np.uint8)
                parts.append(x.view(np.uint8))
                index[n] = {"file": f"{kind}_b{b:02d}.bin", "dtype": x.dtype.str,
                            "offset": pos, "nbytes": int(x.nbytes)}
                pos += int(x.nbytes)
            np.concatenate(parts).tofile(os.path.join(a.out, f"{kind}_b{b:02d}.bin"))
        with open(os.path.join(a.out, f"index_b{b:02d}.json"), "w") as fh:
            json.dump(index, fh)
        print(f"batch {b}: events {start}-{stop}", flush=True)


def cmd_bam(a):
    import pysam
    # a remote BAM needs its index URL; pysam caches the .bai in the cwd
    bam = pysam.AlignmentFile(a.input, "rb", index_filename=a.input + ".bai")
    cols = {k: [] for k in ("pos", "mapq", "flag", "tlen", "matepos", "rlen", "qual", "seq")}
    dt = {"pos": np.int32, "mapq": np.uint8, "flag": np.uint16, "tlen": np.int32,
          "matepos": np.int32, "rlen": np.uint16}
    ext = {"pos": "i32", "mapq": "u8", "flag": "u16", "tlen": "i32", "matepos": "i32",
           "rlen": "u16", "qual": "u8", "seq": "u8"}
    b, n = 0, 0

    def flush():
        nonlocal b, n
        for k, v in cols.items():
            arr = (np.frombuffer(b"".join(v), dtype=np.uint8) if k in ("qual", "seq")
                   else np.asarray(v, dtype=dt[k]))
            arr.tofile(os.path.join(a.out, f"{k}_b{b:02d}.{ext[k]}"))
            v.clear()
        print(f"batch {b}: {n} reads", flush=True)
        b, n = b + 1, 0

    for region in a.regions.split(","):
        for r in bam.fetch(region):
            if r.is_secondary or r.is_supplementary or r.query_sequence is None:
                continue
            cols["pos"].append(r.reference_start)
            cols["mapq"].append(r.mapping_quality)
            cols["flag"].append(r.flag)
            cols["tlen"].append(r.template_length)
            cols["matepos"].append(r.next_reference_start)
            cols["rlen"].append(r.query_length)
            q = r.query_qualities
            cols["qual"].append(bytes(q) if q is not None else b"")
            cols["seq"].append(r.query_sequence.encode())
            n += 1
            if n >= a.reads_per_batch:
                flush()
    if n:
        flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nanoaod", "bam"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--events-per-batch", type=int, default=200_000)
    ap.add_argument("--regions", default="1")
    ap.add_argument("--reads-per-batch", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"nanoaod": cmd_nanoaod, "bam": cmd_bam}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does storing four timesteps chunk by chunk leave ratio on the table that
temporal dedup, grouping or deltas would recover?

    temporal_compression.py --root FIELDS --frames 50 51 52 53 --out CSV \
        [--blocks 16 64 128]

Every strategy stores the SAME bytes (all fields, T0..T3) losslessly and is
verified by decompressing every frame and comparing it byte for byte. A
"chunk" is one cubic block of one field; 128 on a 128^3 grid is the whole
field file, the pipeline's own 8 MiB chunk.

  independent      each chunk of each timestep compressed on its own
  dedup_index      same, but a chunk byte-identical to the SAME index at the
                   previous timestep is stored as a 16-byte reference
  dedup_content    a chunk whose bytes were already stored ANYWHERE (any
                   index, any earlier timestep or earlier in this one) is a
                   16-byte reference
  group_time       the four versions of each chunk index, T0..T3 back to
                   back, compressed as ONE frame (the codec sees the history)
  xor_delta        T0 compressed; T1..T3 stored as the bitwise XOR with the
                   previous timestep, each compressed (identical -> all zeros)

Codecs are CPU stand-ins for the pipeline's GPU ones: zstd level 3 and LZ4,
each plain and after a 4-byte shuffle (bytes of equal significance grouped,
as nvCOMP Bitcomp / our byte-shuffle preprocessor do).
"""
import argparse
import os
import time

import lz4.frame
import numpy as np
import pandas as pd
import xxhash
import zstandard

import classify_temporal as C

REF_BYTES = 16
ZC = zstandard.ZstdCompressor(level=3)
ZD = zstandard.ZstdDecompressor()


def shuffle(b):
    """4-byte shuffle: byte k of every float32, then byte k+1, ..."""
    return np.frombuffer(b, np.uint8).reshape(-1, 4).T.tobytes()


def unshuffle(b):
    return np.frombuffer(b, np.uint8).reshape(4, -1).T.tobytes()


def make_codecs():
    """name -> (compress, decompress) over bytes, verified by the caller."""
    z = (ZC.compress, lambda c: ZD.decompress(c))
    l4 = (lz4.frame.compress, lz4.frame.decompress)
    out = {}
    for name, (c, d) in (("zstd-3", z), ("lz4", l4)):
        out[name] = (c, d)
        out[name + "+shuffle"] = (lambda b, c=c: c(shuffle(b)),
                                  lambda x, d=d: unshuffle(d(x)))
    return out


def sized(codec, data):
    """Compressed size of `data`, after checking it decodes back exactly."""
    comp, dec = codec
    frame = comp(data)
    if dec(frame) != data:
        raise AssertionError("lossless round trip failed")
    return len(frame)


def strategies(rows, codec):
    """Stored bytes of every strategy for one field.

    @param rows [T0..T3] block rows (nblocks, block volume) float32
    @return dict strategy -> bytes
    """
    nb = rows[0].shape[0]
    raw = [[r[i].tobytes() for i in range(nb)] for r in rows]
    size = {t: [sized(codec, raw[t][i]) for i in range(nb)] for t in range(4)}
    out = {"independent": sum(sum(size[t]) for t in range(4))}
    out["dedup_index"] = sum(size[0]) + sum(
        REF_BYTES if raw[t][i] == raw[t - 1][i] else size[t][i]
        for t in range(1, 4) for i in range(nb))
    seen, total = set(), 0
    for t in range(4):
        for i in range(nb):
            h = xxhash.xxh3_128_digest(raw[t][i])
            total += REF_BYTES if h in seen else size[t][i]
            seen.add(h)
    out["dedup_content"] = total
    out["group_time"] = sum(sized(codec, b"".join(raw[t][i] for t in range(4)))
                            for i in range(nb))
    u = [r.view(np.uint32) for r in rows]
    out["xor_delta"] = sum(size[0]) + sum(
        sized(codec, (u[t][i] ^ u[t - 1][i]).tobytes())
        for t in range(1, 4) for i in range(nb))
    return out


def run(source, blocks, codecs):
    """Every block size x codec x strategy, summed over the fields."""
    recs = []
    for field in source.fields():
        grids = [source.load(field, t) for t in range(4)]
        for b in blocks:
            rows = [C.to_blocks(g, (b, b, b))[0] for g in grids]
            raw = 4 * grids[0].nbytes
            for name, codec in codecs.items():
                t0 = time.perf_counter()
                for s, v in strategies(rows, codec).items():
                    recs.append(dict(field=field, block=b, codec=name, strategy=s,
                                     raw_bytes=raw, stored_bytes=v))
                print(f"  {field:8s} {b:3d}³ {name:13s} {time.perf_counter() - t0:5.1f} s",
                      flush=True)
    df = pd.DataFrame(recs)
    tot = df.groupby(["block", "codec", "strategy"], as_index=False)[
        ["raw_bytes", "stored_bytes"]].sum()
    tot["ratio"] = tot["raw_bytes"] / tot["stored_bytes"]
    return df, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", nargs=4, required=True)
    ap.add_argument("--blocks", nargs="+", type=int, default=[16, 64, 128])
    ap.add_argument("--out", required=True, help="CSV of totals (per-field beside it)")
    a = ap.parse_args()
    frames = [f"plt{int(f):05d}" if f.isdigit() else f for f in a.frames]
    source = C.RawF32Source(a.root, frames)
    df, tot = run(source, a.blocks, make_codecs())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tot.to_csv(a.out, index=False)
    df.to_csv(a.out.replace(".csv", "_per_field.csv"), index=False)
    piv = tot.pivot_table(index=["block", "codec"], columns="strategy", values="ratio")
    print(piv[["independent", "dedup_index", "dedup_content", "group_time",
               "xor_delta"]].round(3).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Deterministic, compressible data generator for DTSchedule WfBench runs.

Stock WfBench fills every file with ``os.urandom`` bytes, so no compression
system can do anything with its output. This module replaces that with
data whose compressibility resembles real scientific files. The class of
data is chosen from the file extension (see ``EXT_TABLE``), and every file
is seeded from its own name so re-runs reproduce identical bytes.

Data classes
------------
``float_field``
    float32 2-D field: a sum of Gaussian bumps and sinusoids plus Gaussian
    noise (amplitude ``DT_DATA_NOISE``, default 0.05). Lossy-compressible,
    moderately lossless-compressible -- like simulation output.
``text_seq``
    FASTQ-like 4-line records (ACGT reads of 100-150 bases, phred-like
    quality strings with runs). Lossless only.
``text_table``
    Tab-separated rows with repeated chromosome ids, increasing positions
    and small alphabets (VCF/TSV-like). Lossless only.
``mixed_binary``
    50 % float_field, 50 % incompressible bytes, striped in 64 KiB runs so
    already-compressed formats (.gz/.bam/.jpg) stay hard.

Performance: one 8 MiB base block is generated per (class, seed) and
cached; every further block is a cheap rotation (line-aligned for text,
element-aligned for floats) with a small additive drift for floats, so
the steady-state rate is memcpy-bound (>= 500 MB/s).

Environment overrides
---------------------
``DT_DATA_CLASS``  force one class for every file (``auto`` = use table).
``DT_DATA_NOISE``  noise amplitude in [0, 1] for ``float_field``.

CLI
---
``dt_datagen.py --selftest DIR [--size-mb N]`` writes one file per class
and prints generation rate plus zstd/lz4 ratios (if those modules exist).
``dt_datagen.py --gen NAME --nbytes N --out FILE`` writes one file.
``dt_datagen.py --classify NAME`` prints the class a name maps to.
"""
import argparse
import os
import sys
import time
import zlib
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np

BLOCK = 8 << 20
"""Base block size in bytes; generation and perturbation work per block."""

STRIPE = 64 << 10
"""Stripe length for interleaving the halves of ``mixed_binary``."""

CLASSES = ("float_field", "text_seq", "text_table", "mixed_binary")
"""All data classes, in the order the self-test reports them."""

DEFAULT_NOISE = 0.05
"""Default Gaussian noise amplitude for ``float_field``."""

QUANTUM = np.float32(1.0 / 512.0)
"""float_field values are snapped to this grid (like data stored with
limited precision), which is what gives lossless codecs something to
find while keeping the field smooth for lossy codecs."""

EXT_TABLE: Dict[str, str] = {
    # float fields: images, traces, meshes, generic binary dumps
    ".fits": "float_field", ".nc": "float_field", ".h5": "float_field",
    ".hdf5": "float_field", ".bp": "float_field", ".sac": "float_field",
    ".dat": "float_field", ".stf": "float_field", ".lht": "float_field",
    ".bin": "float_field", ".raw": "float_field", ".f32": "float_field",
    # sequence text
    ".fastq": "text_seq", ".fq": "text_seq", ".sfq": "text_seq",
    ".fa": "text_seq", ".fna": "text_seq", ".fasta": "text_seq",
    ".bfq": "text_seq",
    # tabular text
    ".vcf": "text_table", ".txt": "text_table", ".tbl": "text_table",
    ".csv": "text_table", ".map": "text_table", ".pileup": "text_table",
    ".out": "text_table", ".sam": "text_table", ".hdr": "text_table",
    ".err": "text_table", ".list": "text_table", ".idx": "text_table",
    ".soil": "text_table", ".weather": "text_table",
    ".operation": "text_table", ".ctrl": "text_table",
    # already-compressed / index formats
    ".gz": "mixed_binary", ".zip": "mixed_binary", ".bam": "mixed_binary",
    ".bai": "mixed_binary", ".bwt": "mixed_binary", ".sa": "mixed_binary",
    ".pac": "mixed_binary", ".amb": "mixed_binary", ".ann": "mixed_binary",
    ".bt2": "mixed_binary", ".jpg": "mixed_binary", ".png": "mixed_binary",
    ".gif": "mixed_binary", ".bfa": "mixed_binary", ".dict": "mixed_binary",
    ".fai": "mixed_binary",
}
"""File extension (lower-case, with dot) to data class. Unknown -> mixed."""

CLASS_EXAMPLE_EXT = {
    "float_field": ".fits", "text_seq": ".fastq",
    "text_table": ".vcf", "mixed_binary": ".gz",
}
"""One representative extension per class, used by ``--selftest``."""

_CACHE: "OrderedDict[Tuple[str, int, float], np.ndarray]" = OrderedDict()
_CACHE_MAX = 8
_NEWLINES: Dict[int, np.ndarray] = {}
"""id(base block) -> positions of newlines, for line-aligned rotation."""


def seed_for(name: str) -> int:
    """Derive the deterministic seed for a file from its base name.

    :param name: File path or bare name; only the base name is hashed.
    :return: CRC32 of the base name (unsigned 32-bit).
    """
    base = os.path.basename(str(name).replace("clio::", ""))
    return zlib.crc32(base.encode("utf-8")) & 0xFFFFFFFF


def classify(name: str, data_class: Optional[str] = None) -> str:
    """Pick the data class for a file.

    :param name: File path or bare name; the extension selects the class.
    :param data_class: Explicit class (``auto``/None = consult env, then
        the extension table).
    :return: One of ``CLASSES``.
    """
    forced = data_class or os.environ.get("DT_DATA_CLASS", "")
    if forced and forced != "auto":
        if forced not in CLASSES:
            raise ValueError(f"unknown data class '{forced}', "
                             f"choices: {CLASSES}")
        return forced
    base = os.path.basename(str(name).replace("clio::", "")).lower()
    _, ext = os.path.splitext(base)
    return EXT_TABLE.get(ext, "mixed_binary")


def noise_level(noise: Optional[float] = None) -> float:
    """Resolve the float_field noise amplitude.

    :param noise: Explicit amplitude in [0, 1]; None reads ``DT_DATA_NOISE``.
    :return: Noise amplitude clipped to [0, 1].
    """
    if noise is None:
        noise = float(os.environ.get("DT_DATA_NOISE", DEFAULT_NOISE))
    return float(min(max(noise, 0.0), 1.0))


def _float_field(seed: int, noise: float, nbytes: int = BLOCK) -> np.ndarray:
    """Build a float32 2-D field of ``nbytes`` bytes.

    The field is a sum of 8 Gaussian bumps and 3 sinusoids over a unit
    square plus Gaussian noise scaled by the field's own std.

    :param seed: RNG seed.
    :param noise: Noise amplitude relative to the field std.
    :param nbytes: Output size in bytes (multiple of 4).
    :return: uint8 view of the float32 field.
    """
    rng = np.random.default_rng(seed)
    n_el = nbytes // 4
    width = 2048
    height = max(1, n_el // width)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    field = np.zeros((height, width), dtype=np.float32)
    for _ in range(8):
        cx, cy = rng.random(2, dtype=np.float32)
        sigma = np.float32(0.03 + 0.12 * rng.random())
        amp = np.float32(0.5 + 1.5 * rng.random())
        field += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2)
                              / (2.0 * sigma * sigma))
    for _ in range(3):
        fx, fy = rng.integers(1, 12, size=2)
        phase = np.float32(rng.random() * 2.0 * np.pi)
        amp = np.float32(0.1 + 0.4 * rng.random())
        field += amp * np.sin(2.0 * np.pi * (fx * xs + fy * ys) + phase)
    std = float(field.std()) or 1.0
    if noise > 0.0:
        field += (noise * std) * rng.standard_normal(
            field.shape, dtype=np.float32)
    field = np.round(field / QUANTUM) * QUANTUM
    out = field.reshape(-1).view(np.uint8)
    if out.size < nbytes:
        out = np.concatenate([out, out[: nbytes - out.size]])
    return np.ascontiguousarray(out[:nbytes])


def _text_seq(seed: int, nbytes: int = BLOCK) -> np.ndarray:
    """Build FASTQ-like records totalling ``nbytes`` bytes.

    :param seed: RNG seed.
    :param nbytes: Output size in bytes; the last record may be cut.
    :return: uint8 array of ASCII text.
    """
    rng = np.random.default_rng(seed)
    n_rec = nbytes // 200 + 64
    max_len = 150
    # weighted alphabet: A 30 %, C 20 %, G 20 %, T 30 %
    bases = np.frombuffer(b"AAACCGGTTT", dtype=np.uint8)
    seqs = bases[rng.integers(0, 10, size=(n_rec, max_len), dtype=np.int8)]
    steps = rng.integers(-2, 3, size=(n_rec, max_len), dtype=np.int8)
    quals = np.clip(np.cumsum(steps, axis=1, dtype=np.int16) + 36, 2, 40)
    quals = (quals + 33).astype(np.uint8)
    lengths = rng.integers(100, max_len + 1, size=n_rec).tolist()
    tag = f"{seed:08x}"
    parts = []
    total = 0
    seq_rows = seqs.tobytes()
    qual_rows = quals.tobytes()
    for i, ln in enumerate(lengths):
        base = i * max_len
        rec = (f"@SEQ_{tag}_{i:07d} len={ln}\n".encode()
               + seq_rows[base: base + ln] + b"\n+\n"
               + qual_rows[base: base + ln] + b"\n")
        parts.append(rec)
        total += len(rec)
        if total >= nbytes:
            break
    buf = np.frombuffer(b"".join(parts), dtype=np.uint8)
    return np.ascontiguousarray(buf[:nbytes])


def _text_table(seed: int, nbytes: int = BLOCK) -> np.ndarray:
    """Build VCF/TSV-like tab-separated rows totalling ``nbytes`` bytes.

    :param seed: RNG seed.
    :param nbytes: Output size in bytes; the last row may be cut.
    :return: uint8 array of ASCII text.
    """
    rng = np.random.default_rng(seed)
    n_rows = nbytes // 40 + 64
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    per_chrom = max(1, n_rows // len(chroms) + 1)
    chrom_col = np.repeat(np.arange(len(chroms)), per_chrom)[:n_rows]
    chrom_names = [chroms[c] for c in chrom_col.tolist()]
    pos = np.cumsum(rng.integers(1, 400, size=n_rows)).tolist()
    bases = "ACGT"
    ref = [bases[b] for b in rng.integers(0, 4, size=n_rows).tolist()]
    alt = [bases[b] for b in rng.integers(0, 4, size=n_rows).tolist()]
    qual = np.round(rng.gamma(4.0, 12.0, size=n_rows), 1).tolist()
    filters = ["PASS", "PASS", "PASS", "LowQual", "."]
    filt = [filters[f] for f in rng.integers(0, 5, size=n_rows).tolist()]
    depth = rng.integers(5, 120, size=n_rows).tolist()
    af = np.round(rng.beta(0.5, 2.0, size=n_rows), 3).tolist()
    header = ("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    lines = [header]
    total = len(header)
    cols = zip(chrom_names, pos, ref, alt, qual, filt, depth, af)
    for ch, po, rf, al, qu, fi, dp, fr in cols:
        row = (f"{ch}\t{po}\t.\t{rf}\t{al}\t{qu}\t{fi}\t"
               f"DP={dp};AF={fr}\n")
        lines.append(row)
        total += len(row)
        if total >= nbytes:
            break
    buf = np.frombuffer("".join(lines).encode("ascii"), dtype=np.uint8)
    return np.ascontiguousarray(buf[:nbytes])


def _mixed_binary(seed: int, noise: float, nbytes: int = BLOCK) -> np.ndarray:
    """Build a block that is half float field, half incompressible bytes.

    The halves are interleaved in ``STRIPE``-byte runs so any chunk of
    the file carries both kinds of data.

    :param seed: RNG seed.
    :param noise: Noise amplitude for the float half.
    :param nbytes: Output size in bytes.
    :return: uint8 array.
    """
    rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    half = (nbytes // 2 // STRIPE) * STRIPE or nbytes // 2
    floats = _float_field(seed, noise, max(4, (half // 4) * 4))
    floats = floats[:half]
    rand = np.frombuffer(rng.bytes(half), dtype=np.uint8)
    n_stripes = half // STRIPE
    if n_stripes == 0:
        out = np.concatenate([floats, rand])
    else:
        f_s = floats[: n_stripes * STRIPE].reshape(n_stripes, STRIPE)
        r_s = rand[: n_stripes * STRIPE].reshape(n_stripes, STRIPE)
        out = np.stack([f_s, r_s], axis=1).reshape(-1)
        out = np.concatenate([out, floats[n_stripes * STRIPE:],
                              rand[n_stripes * STRIPE:]])
    if out.size < nbytes:
        out = np.concatenate([out, rand[: nbytes - out.size]])
    return np.ascontiguousarray(out[:nbytes])


def base_block(data_class: str, seed: int, noise: float) -> np.ndarray:
    """Return (and cache) the 8 MiB base block for a class and seed.

    :param data_class: One of ``CLASSES``.
    :param seed: RNG seed.
    :param noise: Noise amplitude (only used by float classes).
    :return: uint8 array of ``BLOCK`` bytes; treat as read-only.
    """
    key = (data_class, int(seed), round(noise, 4))
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    if data_class == "float_field":
        blk = _float_field(seed, noise)
    elif data_class == "text_seq":
        blk = _text_seq(seed)
    elif data_class == "text_table":
        blk = _text_table(seed)
    elif data_class == "mixed_binary":
        blk = _mixed_binary(seed, noise)
    else:
        raise ValueError(f"unknown data class '{data_class}'")
    if blk.size < BLOCK:
        reps = BLOCK // blk.size + 1
        blk = np.ascontiguousarray(np.tile(blk, reps)[:BLOCK])
    _CACHE[key] = blk
    while len(_CACHE) > _CACHE_MAX:
        _, old = _CACHE.popitem(last=False)
        _NEWLINES.pop(id(old), None)
    return blk


def _line_shift(base: np.ndarray, k: int) -> int:
    """Pick a line-aligned rotation for text block ``k``.

    :param base: The base text block.
    :param k: Block index (0 = unrotated).
    :return: Byte offset of a newline to rotate to.
    """
    if k == 0:
        return 0
    nl = _NEWLINES.get(id(base))
    if nl is None:
        nl = np.flatnonzero(base == 10)
        _NEWLINES[id(base)] = nl
    if nl.size == 0:
        return 0
    return int(nl[(k * 7919) % nl.size]) + 1


def perturbed_block(data_class: str, seed: int, noise: float,
                    k: int) -> np.ndarray:
    """Return block ``k`` of a file: the base block cheaply perturbed.

    Text classes rotate at a line boundary; float classes rotate by a
    whole number of float32 elements and add a small per-block drift;
    mixed_binary rotates by a whole stripe pair so the halves stay put.

    :param data_class: One of ``CLASSES``.
    :param seed: Seed of the file.
    :param noise: Noise amplitude.
    :param k: Block index within the file.
    :return: uint8 array of ``BLOCK`` bytes (new array for k > 0).
    """
    base = base_block(data_class, seed, noise)
    if k == 0:
        return base
    if data_class == "float_field":
        f32 = base.view(np.float32)
        shift = (k * 4099) % f32.size
        out = np.roll(f32, shift)
        out += np.float32(k) * QUANTUM
        return out.view(np.uint8)
    if data_class == "mixed_binary":
        shift = (k * 2 * STRIPE) % base.size
        return np.roll(base, shift)
    return np.roll(base, -_line_shift(base, k))


def generate_array(path_or_name: str, nbytes: int, seed: Optional[int] = None,
                   data_class: Optional[str] = None,
                   noise: Optional[float] = None,
                   offset: int = 0) -> np.ndarray:
    """Generate ``nbytes`` bytes of a file starting at ``offset`` as uint8.

    Bytes at a given offset are a pure function of (name, seed, class,
    noise), so a file can be produced in any chunking and re-runs match.

    :param path_or_name: File path or name; picks the class and seed.
    :param nbytes: Number of bytes to return.
    :param seed: Seed override (default CRC32 of the base name).
    :param data_class: Class override (default by extension / env).
    :param noise: Noise amplitude override (default env / 0.05).
    :param offset: Byte offset of the first returned byte within the file.
    :return: Contiguous uint8 array of length ``nbytes`` (may alias the
        cached base block when it is exactly block 0; do not modify).
    """
    if nbytes <= 0:
        return np.empty(0, dtype=np.uint8)
    cls = classify(path_or_name, data_class)
    sd = seed_for(path_or_name) if seed is None else int(seed)
    nz = noise_level(noise)
    first = offset // BLOCK
    last = (offset + nbytes - 1) // BLOCK
    if first == last:
        blk = perturbed_block(cls, sd, nz, first)
        lo = offset - first * BLOCK
        return blk[lo: lo + nbytes]
    out = np.empty(nbytes, dtype=np.uint8)
    pos = 0
    for k in range(first, last + 1):
        blk = perturbed_block(cls, sd, nz, k)
        lo = offset - k * BLOCK if k == first else 0
        hi = (offset + nbytes) - k * BLOCK if k == last else BLOCK
        out[pos: pos + hi - lo] = blk[lo:hi]
        pos += hi - lo
    return out


def generate(path_or_name: str, nbytes: int, seed: Optional[int] = None,
             data_class: Optional[str] = None, noise: Optional[float] = None,
             offset: int = 0) -> bytes:
    """Generate ``nbytes`` bytes of a file starting at ``offset``.

    Same contract as :func:`generate_array` but returns ``bytes``.

    :param path_or_name: File path or name; picks the class and seed.
    :param nbytes: Number of bytes to return.
    :param seed: Seed override (default CRC32 of the base name).
    :param data_class: Class override (default by extension / env).
    :param noise: Noise amplitude override (default env / 0.05).
    :param offset: Byte offset of the first returned byte within the file.
    :return: ``bytes`` of length ``nbytes``.
    """
    return generate_array(path_or_name, nbytes, seed=seed,
                          data_class=data_class, noise=noise,
                          offset=offset).tobytes()


def write_file(fp, name: str, nbytes: int, chunk: int = BLOCK,
               offset: int = 0, seed: Optional[int] = None,
               data_class: Optional[str] = None,
               noise: Optional[float] = None) -> int:
    """Stream ``nbytes`` of generated data for ``name`` into ``fp``.

    :param fp: Binary file object opened for writing/appending.
    :param name: File name used for class/seed selection.
    :param nbytes: Total bytes to write.
    :param chunk: Bytes per ``fp.write`` call.
    :param offset: Logical offset of the first byte (pass the number of
        bytes already written when appending).
    :param seed: Seed override.
    :param data_class: Class override.
    :param noise: Noise amplitude override.
    :return: Number of bytes written.
    """
    done = 0
    chunk = max(1, int(chunk))
    while done < nbytes:
        n = min(chunk, nbytes - done)
        buf = generate_array(name, n, seed=seed, data_class=data_class,
                             noise=noise, offset=offset + done)
        view = memoryview(buf).cast("B")
        while view:
            wrote = fp.write(view)
            if wrote is None:
                wrote = len(view)
            view = view[wrote:]
        done += n
    return done


def _ratio(data: bytes) -> Dict[str, float]:
    """Compute compression ratios with whichever codecs are importable.

    :param data: Raw bytes to compress.
    :return: Mapping codec name -> ratio (raw / compressed).
    """
    out = {}
    try:
        import zstandard
        c = zstandard.ZstdCompressor(level=3).compress(data)
        out["zstd3"] = len(data) / max(1, len(c))
    except ImportError:
        pass
    try:
        import lz4.frame
        c = lz4.frame.compress(data)
        out["lz4"] = len(data) / max(1, len(c))
    except ImportError:
        pass
    return out


def selftest(out_dir: str, size_mb: int = 32) -> int:
    """Write one file per class, report rates, ratios and determinism.

    ``base_ms`` is the one-time cost of the file's base block,
    ``gen_MB/s`` the steady in-memory generation rate, ``write_ms`` the
    time to stream the file to ``out_dir`` (storage-bound).

    :param out_dir: Directory to write ``selftest.<ext>`` files into.
    :param size_mb: Size of each file in MiB.
    :return: 0 on success, 1 if a determinism check fails.
    """
    os.makedirs(out_dir, exist_ok=True)
    nbytes = size_mb << 20
    rc = 0
    print(f"{'class':<13}{'file':<20}{'MiB':>5}{'base_ms':>9}{'gen_MB/s':>10}"
          f"{'write_ms':>10}  ratios")
    for cls in CLASSES:
        name = f"selftest{CLASS_EXAMPLE_EXT[cls]}"
        path = os.path.join(out_dir, name)
        t0 = time.perf_counter()
        generate_array(name, 1)
        base_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        generate_array(name, nbytes)
        gen_rate = nbytes / 1e6 / max(1e-6, time.perf_counter() - t0)
        t0 = time.perf_counter()
        with open(path, "wb") as fp:
            write_file(fp, name, nbytes)
        write_ms = (time.perf_counter() - t0) * 1000.0
        with open(path, "rb") as fp:
            data = fp.read()
        again = generate(name, min(nbytes, 2 * BLOCK + 123), offset=0)
        if data[: len(again)] != again:
            print(f"DETERMINISM FAILURE for {name}", file=sys.stderr)
            rc = 1
        rat = "  ".join(f"{k}={v:.2f}x" for k, v in _ratio(data).items())
        print(f"{cls:<13}{name:<20}{size_mb:>5}{base_ms:>9.0f}{gen_rate:>10.0f}"
              f"{write_ms:>10.0f}  {rat or '(no zstandard/lz4 module)'}")
    return rc


def main(argv=None) -> int:
    """CLI entry point.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Process exit code.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", metavar="DIR",
                   help="write one file per class into DIR and report")
    p.add_argument("--size-mb", type=int, default=32,
                   help="per-file size for --selftest (MiB)")
    p.add_argument("--classify", metavar="NAME",
                   help="print the data class NAME maps to")
    p.add_argument("--gen", metavar="NAME", help="generate data for NAME")
    p.add_argument("--nbytes", type=int, default=BLOCK,
                   help="bytes to generate with --gen")
    p.add_argument("--out", metavar="FILE",
                   help="output path for --gen (default: NAME in cwd)")
    p.add_argument("--data-class", default=None, choices=CLASSES,
                   help="force a data class")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest(args.selftest, args.size_mb)
    if args.classify:
        print(classify(args.classify, args.data_class))
        return 0
    if args.gen:
        out = args.out or os.path.basename(args.gen)
        with open(out, "wb") as fp:
            write_file(fp, args.gen, args.nbytes, data_class=args.data_class)
        print(f"wrote {args.nbytes} bytes to {out}")
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

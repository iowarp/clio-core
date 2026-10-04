#!/usr/bin/env python3
"""NeuroPress error-bounded quantization, as a standalone host script.

This is a host-side copy of QuantizeDevice / DequantizeDevice
(context-transport-primitives/src/compress/preprocess/data_stats_gpu_kernels.cu).
It is not a codec: it only turns float32/float64 values into a packed integer
grid so a lossless codec can run on fewer bits. The guarantee is

    |original - reconstructed| <= error_bound   (or bit-exact, if escaped)

per element. Arithmetic, in order:

  1. Range of the finite values: lo, hi. Non-finite values never set the
     range (they later escape).
  2. Optional relative bound: eb = rel * (hi - lo), matching
     CLIO_NEUROPRESS_REL_BOUND. A constant chunk keeps the absolute eb.
  3. Grid step delta, chosen so a reconstructed value lands within 0.95*eb
     after rounding and the float32/float64 cast:
         float32  delta = max(2*(0.95*eb - ULP/2), 0.95*eb)
         float64  delta = 2*(0.95*eb - 12*ULP)
     scale = 1/delta, offset = lo.
  4. Index q = round((x - lo) * scale)  (half away from zero, like C round).
     Reconstruct with one fma:  z = (float)fma(q, delta, lo).
  5. Width: 8 / 16 / 32 bits from ceil((hi-lo)/delta); 64 if float64 needs it.
  6. If any element is outside that width, is non-finite, or fails the bound
     check, the whole chunk uses escape mode: native-width slots plus a
     bitmap. Escaped elements keep their original bits.

Packed size is num_elements * width_bytes [+ bitmap], rounded up to 8 bytes
(nvcomp-bitcomp reads 8-byte words).

  neuropress_quantize.py INPUT.f32 --eb 1e-3 [--rel 1e-3] [--out PREFIX]
  neuropress_quantize.py --self-test

INPUT is raw little-endian float32 (default) or float64 (--f64). --out writes
PREFIX.q (packed bytes) and PREFIX.recon (reconstructed floats). Without
--out it prints a one-line report.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

INT64_MAX_D = 9223372036854774784.0  # largest double below 2^63
F32_MAX = float(np.finfo(np.float32).max)


def half_ulp_f32(x: float) -> float:
    """Half the float32 spacing at |x|; 0 when x is outside the float32 range.

    @param x magnitude used for the reserve in MakeGrid
    @return 0.5 * (nextafterf(|x|) - |x|)
    """
    ax = abs(x)
    if not (ax <= F32_MAX):
        return 0.0
    f = np.float32(ax)
    if not np.isfinite(f):
        return 0.0
    nxt = np.nextafter(f, np.float32(np.inf))
    return 0.5 * (float(nxt) - float(f))


def ulp_f64(x: float) -> float:
    """Full float64 spacing at |x|; inf when x is not finite."""
    ax = abs(x)
    if not math.isfinite(ax):
        return math.inf
    return float(np.nextafter(ax, np.inf) - ax)


def make_grid(lo: float, hi: float, eb: float, f64: bool):
    """Grid whose reconstructions land within 0.95*eb of the source.

    @param lo, hi  min and max of the finite values
    @param eb      absolute error bound (> 0)
    @param f64     True for float64 input
    @return dict offset/delta/scale/inv_scale, or None if no positive step
    """
    m = min(max(abs(lo), abs(hi)) + eb, sys.float_info.max)
    if hi - lo == 0.0:
        d = 1.0
    elif not math.isfinite(eb):
        d = 1e300
    elif f64:
        d = 2.0 * (0.95 * eb - 12.0 * ulp_f64(m))
    else:
        d = max(2.0 * (0.95 * eb - half_ulp_f32(m)), 0.95 * eb)
    if not (d > 0.0):
        return None
    d = min(d, 1e300)
    s = 1.0 / d
    inv = 1.0 / s
    if not (s > 0.0 and math.isfinite(s) and inv > 0.0 and math.isfinite(inv)):
        return None
    return {"offset": lo, "delta": d, "scale": s, "inv_scale": inv}


def precision_for(qmax: float, f64: bool) -> int:
    """Bits per index from the largest index the range needs.

    @param qmax ceil((hi-lo)/delta)
    @param f64  True allows a 64-bit width
    @return 8, 16, 32, 64, or 0 if no integer width holds qmax
    """
    if qmax <= 127.0:
        return 8
    if qmax <= 32767.0:
        return 16
    if qmax <= 2147483647.0:
        return 32
    if f64 and qmax <= INT64_MAX_D:
        return 64
    return 0


def round_away(x: np.ndarray) -> np.ndarray:
    """C/CUDA round: half away from zero, not banker's rounding."""
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def pack_indices(q: np.ndarray, bits: int) -> bytes:
    """Little-endian packed signed indices.

    @param q    integer indices
    @param bits 8, 16, 32 or 64
    @return packed bytes
    """
    dt = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[bits]
    return np.ascontiguousarray(q.astype(dt, copy=False)).tobytes()


def pad8(buf: bytes) -> bytes:
    """@return buf extended with zeros to a multiple of 8 bytes."""
    n = (len(buf) + 7) & ~7
    return buf if len(buf) == n else buf + b"\x00" * (n - len(buf))


def decode(q: np.ndarray, inv_scale: float, offset: float, dtype):
    """One fused multiply-add then the element type, matching Decode<T>.

    numpy may not expose fma; float64 multiply-add is used then, which is
    the same value except possibly 1 ulp, and the bound check uses eb.
    """
    z = np.add(np.multiply(q.astype(np.float64), inv_scale, dtype=np.float64),
               offset, dtype=np.float64)
    return z.astype(dtype, copy=False)


class Quantized:
    """Packed NeuroPress quantization of one buffer."""

    def __init__(self):
        self.packed = b""
        self.error_bound = 0.0
        self.effective_error_bound = 0.0
        self.scale = 0.0
        self.data_min = 0.0
        self.data_max = 0.0
        self.precision = 0
        self.elem_bytes = 4
        self.escapes = False
        self.escape_count = 0
        self.num_elements = 0
        self.dtype = np.float32


def _finite_range(x: np.ndarray):
    """@return (lo, hi, any_nonfinite) of the finite values; (0,0,True) if none."""
    finite = np.isfinite(x)
    skipped = bool((~finite).any())
    if not finite.any():
        return 0.0, 0.0, True
    xf = x[finite].astype(np.float64, copy=False)
    return float(xf.min()), float(xf.max()), skipped


def _try_grid(x, g, bits, eb, dtype) -> tuple | None:
    """Grid-encode x; None if any element misses the width or the bound.

    @return (indices as int64, reconstruction) or None
    """
    xd = x.astype(np.float64, copy=False)
    q = round_away((xd - g["offset"]) * g["scale"])
    lo_i = {8: -128, 16: -32768, 32: -2147483648, 64: -2**63}[bits]
    hi_i = {8: 127, 16: 32767, 32: 2147483647, 64: INT64_MAX_D}[bits]
    if not np.all(np.isfinite(q)) or np.any(q < lo_i) or np.any(q > hi_i):
        return None
    z = decode(q, g["inv_scale"], g["offset"], dtype)
    if np.any(np.abs(z.astype(np.float64) - xd) > eb):
        return None
    return q.astype(np.int64), z


def _escape_encode(x, g, eb, dtype, limit) -> tuple:
    """Native-width slots plus a bitmap of escaped elements.

    @return (packed bytes, escape count)
    """
    n = x.size
    elem = int(np.dtype(dtype).itemsize)
    slot_bits = 64 if elem == 8 else 32
    xd = x.astype(np.float64, copy=False)
    q = round_away((xd - g["offset"]) * g["scale"])
    lo_i = -2**63 if elem == 8 else -2147483648
    hi_i = INT64_MAX_D if elem == 8 else 2147483647
    chk = eb
    z = decode(q, g["inv_scale"], g["offset"], dtype)
    ok = (np.abs(xd) < limit) & np.isfinite(q) & (q >= lo_i) & (q <= hi_i)
    with np.errstate(invalid="ignore"):
        ok &= np.abs(z.astype(np.float64) - xd) <= chk
    bitmap = np.packbits((~ok).astype(np.uint8), bitorder="little")
    if bitmap.size < (n + 7) // 8:
        bitmap = np.pad(bitmap, (0, (n + 7) // 8 - bitmap.size))
    slots = np.empty(n, dtype=np.int64 if elem == 8 else np.int32)
    slots[ok] = q[ok]
    raw = np.ascontiguousarray(x).view(np.int64 if elem == 8 else np.int32)
    slots[~ok] = raw[~ok]
    packed = pack_indices(slots, slot_bits) + bitmap.tobytes()
    return pad8(packed), int((~ok).sum())


def quantize(x: np.ndarray, error_bound: float, rel: float = 0.0) -> Quantized:
    """Quantize a 1-D float32 or float64 array the way QuantizeDevice does.

    @param x            contiguous float32 or float64 values
    @param error_bound  absolute bound; ignored per-chunk when rel > 0 except
                        on a constant chunk
    @param rel          if > 0, eb = rel * (hi - lo)  (CLIO_NEUROPRESS_REL_BOUND)
    @return Quantized (packed bytes and the parameters Dequantize needs)
    """
    x = np.ascontiguousarray(x)
    if x.dtype not in (np.float32, np.float64):
        raise TypeError("quantize: float32 or float64 array")
    if not (error_bound > 0.0) and not (rel > 0.0):
        raise ValueError("quantize: need a positive --eb or --rel")
    f64 = x.dtype == np.float64
    out = Quantized()
    out.dtype, out.elem_bytes = x.dtype, int(x.dtype.itemsize)
    out.num_elements = int(x.size)
    lo, hi, skipped = _finite_range(x)
    eb = float(error_bound)
    if rel > 0.0 and hi > lo:
        eb = rel * (hi - lo)
    out.error_bound, out.data_max = eb, hi
    g = make_grid(lo, hi, eb, f64) if math.isfinite(lo) else None
    bits = 0
    if g is not None:
        bits = precision_for(math.ceil((hi - lo) / g["delta"]), f64)
    grid_ok = None
    if bits and not skipped:
        grid_ok = _try_grid(x, g, bits, eb, x.dtype)
    if grid_ok is not None:
        q, _ = grid_ok
        out.packed = pad8(pack_indices(q, bits))
        out.precision, out.escapes = bits, False
    else:
        limit = math.inf
        if bits == 0 and math.isfinite(eb) and eb > 0.0:
            limit = math.ldexp(1.0, math.floor(math.log2(eb)) + (47 if f64 else 24))
            lo, hi, _ = _finite_range(x[np.abs(x) < limit])
            g = make_grid(lo, hi, eb, f64) or {
                "offset": lo, "delta": 1.0, "scale": 1.0, "inv_scale": 1.0}
        if g is None:
            g = {"offset": lo, "delta": 1.0, "scale": 1.0, "inv_scale": 1.0}
        out.packed, out.escape_count = _escape_encode(x, g, eb, x.dtype, limit)
        out.precision, out.escapes = (64 if f64 else 32), True
    out.scale = g["scale"]
    out.data_min = g["offset"]
    out.effective_error_bound = 0.5 * g["delta"]
    return out


def dequantize(q: Quantized) -> np.ndarray:
    """Inverse of quantize: indices through fma, escaped slots bit-exact.

    @param q result of quantize
    @return array of q.dtype, q.num_elements long
    """
    n, bits, elem = q.num_elements, q.precision, q.elem_bytes
    width = bits // 8
    idx_dt = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[width]
    raw = np.frombuffer(q.packed, dtype=np.uint8)
    qv = np.frombuffer(raw[: n * width], dtype=idx_dt)
    z = decode(qv.astype(np.float64), 1.0 / q.scale, q.data_min, q.dtype)
    if not q.escapes:
        return z
    bitmap = raw[n * width: n * width + (n + 7) // 8]
    bits_a = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8),
                           bitorder="little")[:n].astype(bool)
    orig = np.frombuffer(raw[: n * elem],
                         dtype=np.int64 if elem == 8 else np.int32)
    z[bits_a] = orig[bits_a].view(q.dtype)
    return z


def report(x: np.ndarray, q: Quantized, z: np.ndarray) -> dict:
    """Error and size numbers for one round trip.

    @return dict of printable fields
    """
    xd, zd = x.astype(np.float64), z.astype(np.float64)
    finite = np.isfinite(xd)
    err = np.abs(zd[finite] - xd[finite]) if finite.any() else np.array([0.0])
    mx = float(err.max()) if err.size else 0.0
    mse = float(np.mean((zd[finite] - xd[finite]) ** 2)) if finite.any() else 0.0
    peak = float(np.max(np.abs(xd[finite]))) if finite.any() else 0.0
    psnr = 10.0 * math.log10(peak * peak / mse) if mse > 0 and peak > 0 else math.inf
    in_b = x.nbytes
    return {
        "n": q.num_elements, "in_bytes": in_b, "out_bytes": len(q.packed),
        "ratio": in_b / len(q.packed) if q.packed else math.inf,
        "precision": q.precision, "escapes": int(q.escapes),
        "escape_count": q.escape_count, "eb": q.error_bound,
        "eb_eff": q.effective_error_bound, "max_err": mx,
        "within_bound": mx <= q.error_bound + 0.0,
        "psnr_db": psnr, "mse": mse,
    }


def self_test() -> int:
    """A few synthetic round trips; prints one line per case. @return 0/1."""
    rng = np.random.default_rng(1)
    cases = [
        ("const", np.full(4096, 3.14, np.float32), 1e-3, 0.0),
        ("sine", np.sin(np.linspace(0, 8, 65536, dtype=np.float32)), 1e-3, 0.0),
        ("rel", rng.standard_normal(8192).astype(np.float32) * 50, 1.0, 1e-3),
        ("nan", np.array([1.0, np.nan, 2.0, np.inf], np.float32), 0.1, 0.0),
    ]
    fail = 0
    for name, x, eb, rel in cases:
        q = quantize(x, eb, rel)
        z = dequantize(q)
        r = report(x, q, z)
        finite = np.isfinite(x)
        ok = np.all(np.abs(z[finite].astype(np.float64) - x[finite])
                    <= q.error_bound)
        ok = ok and np.array_equal(z[~finite].view(np.uint32),
                                   x[~finite].view(np.uint32))
        print(f"  {name:8s} bits={r['precision']:<2d} esc={r['escape_count']:<5d} "
              f"ratio={r['ratio']:.2f} max_err={r['max_err']:.3g} "
              f"eb={r['eb']:.3g} {'ok' if ok else 'FAIL'}")
        fail += not ok
    return 0 if fail == 0 else 1


def load_raw(path: str, f64: bool) -> np.ndarray:
    """@return 1-D array of little-endian floats from a raw file."""
    dt = np.float64 if f64 else np.float32
    return np.fromfile(path, dtype=dt)


def main() -> int:
    """CLI: --self-test, or quantize a raw float file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", nargs="?", help="raw little-endian float file")
    ap.add_argument("--eb", type=float, default=1e-3,
                    help="absolute error bound (default 1e-3)")
    ap.add_argument("--rel", type=float, default=0.0,
                    help="value-range relative bound; 0 = absolute (default)")
    ap.add_argument("--f64", action="store_true",
                    help="read the file as float64 (default float32)")
    ap.add_argument("--out", help="write PREFIX.q and PREFIX.recon")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.input:
        ap.error("input file or --self-test")
    x = load_raw(a.input, a.f64)
    q = quantize(x, a.eb, a.rel)
    z = dequantize(q)
    r = report(x, q, z)
    print(f"n={r['n']} in={r['in_bytes']} out={r['out_bytes']} "
          f"ratio={r['ratio']:.3f} bits={r['precision']} "
          f"escapes={r['escape_count']} eb={r['eb']:.4g} "
          f"eb_eff={r['eb_eff']:.4g} max_err={r['max_err']:.4g} "
          f"psnr={r['psnr_db']:.2f} dB "
          f"{'within bound' if r['within_bound'] else 'OVER BOUND'}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out + ".q", "wb").write(q.packed)
        z.tofile(a.out + ".recon")
        print(f"wrote {a.out}.q and {a.out}.recon")
    return 0 if r["within_bound"] else 2


if __name__ == "__main__":
    sys.exit(main())

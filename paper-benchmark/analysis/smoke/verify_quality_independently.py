#!/usr/bin/env python3
"""Independent reimplementation of the quantize/dequantize/quality chain.

Written from the equations in the CUDA sources, in numpy, touching no project
code. If explore.csv agrees with this, two separate implementations of the same
arithmetic agree -- which is the one thing a self-check cannot give you.

  usage: verify_quality_independently.py <run_dir> <fields_dir> [error_bound]

Tolerances are PER METRIC because the metrics are differently conditioned.
max_error is a max-reduction with no accumulation and is held to 1e-9. rmse,
psnr and ssim are well-conditioned sums, 1e-5. ssim_deviation is a difference
of nearly equal float32 quantities -- dvar cancels ~4 digits on a chunk whose
quantization error is mostly bias -- so a 1e-7 accumulator perturbation moves
it ~1e-3. Its 5e-2 tolerance reflects that conditioning, not a weaker standard;
quality_metrics.h:57 documents the same float32 order-dependence upstream.

  quantize      data_stats_gpu_kernels.cu:717   q  = round((v-offset)*scale)
  dequantize    data_stats_gpu_kernels.cu:732   v' = q*inv_scale + offset
  eff. bound    data_stats_gpu_kernels.cu:880
  precision     quantization.h:100
  accumulators  quality_metrics_gpu_kernels.cu:60
  derivation    quality_metrics.h:144
"""
import csv, glob, sys
import numpy as np

# usage: verify_quality_independently.py <run_dir> <fields_dir> [eb]


def required_precision(data_range, eb):           # quantization.h:100
    if eb <= 0.0:
        return 32
    nb = (data_range / (2.0 * eb)) * 1.1
    return 8 if nb <= 127.0 else (16 if nb <= 32767.0 else 32)


def roundtrip(a, error_bound):                    # the two kernels + eff. bound
    dmin, dmax = float(a.min()), float(a.max())
    data_range = dmax - dmin
    if data_range <= 0.0:                         # constant-data substitution
        data_range = 1.0
    max_abs = max(abs(dmin), abs(dmax))
    float_repr_error = max_abs * 2.4e-7
    available = error_bound - float_repr_error - error_bound * 0.05
    min_eb_i32 = data_range / 4.0e9
    if available <= 0.0:
        eff = max(min_eb_i32, float_repr_error * 0.1)
        achievable = False
    else:
        eff, achievable = available, True
    eff = max(eff, min_eb_i32)
    scale = 1.0 / (2.0 * eff)
    prec = required_precision(data_range, eff)
    lo, hi = {8: (-128.0, 127.0), 16: (-32768.0, 32767.0),
              32: (-2147483648.0, 2147483647.0)}[prec]

    q = np.round((a.astype(np.float64) - dmin) * scale)
    q = np.clip(q, lo, hi)
    rec = (q * (1.0 / scale) + dmin).astype(np.float32)
    return rec, dict(eff=eff, prec=prec, achievable=achievable, scale=scale)


def quality(x, y):                                # accumulators + derivation
    n = x.size
    shift = np.float32(x[0])                      # kernel uses d_orig[0]
    # The kernel forms d, dx and dy in FLOAT32 (quality_metrics_gpu_kernels.cu
    # :70-72) before accumulating. Differencing in float64 here would make the
    # comparison test numpy's arithmetic rather than the pipeline's.
    d = (x - y).astype(np.float32)
    dx = (x - shift).astype(np.float32)
    dy = (y - shift).astype(np.float32)
    d, dx, dy = d.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64)
    sq_err = float((d * d).sum())
    max_abs_err = float(np.abs(d).max())
    min_x, max_x = float(x.min()), float(x.max())
    sum_x, sum_xx = float(dx.sum()), float((dx * dx).sum())
    sum_y, sum_yy = float(dy.sum()), float((dy * dy).sum())
    sum_xy = float((dx * dy).sum())

    rmse = np.sqrt(sq_err / n)
    data_range = max_x - min_x
    dr = data_range if data_range > 0.0 else 1.0
    psnr = 120.0 if rmse < 1e-10 else min(120.0, 20.0 * np.log10(dr / rmse))

    dmu_x, dmu_y = sum_x / n, sum_y / n
    mu_x, mu_y = dmu_x + shift, dmu_y + shift
    var_x = sum_xx / n - dmu_x * dmu_x
    var_y = sum_yy / n - dmu_y * dmu_y
    cov = sum_xy / n - dmu_x * dmu_y
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    ssim = max(-1.0, min(1.0, num / den)) if den > 0.0 else 1.0

    dmu = dmu_x - dmu_y
    dvar = max(0.0, sq_err / n - dmu * dmu)
    aa = mu_x * mu_x + mu_y * mu_y + c1
    cc = var_x + var_y + c2
    dev = max(0.0, (aa * dvar + dmu * dmu * cc - dmu * dmu * dvar) / (aa * cc)) \
        if aa * cc > 0.0 else 0.0
    return dict(rmse=rmse, max_error=max_abs_err, psnr_db=psnr, ssim=ssim,
                ssim_deviation=dev, data_range=data_range)


def rel(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m


#: per-metric, by conditioning -- see the module docstring
TOL = {"max_error": 1e-9, "rmse": 1e-5, "psnr_db": 1e-5,
       "ssim": 1e-5, "ssim_deviation": 5e-2}


def main(run_dir, fields_dir, eb="0.01"):
    eb = float(eb)
    run_dir = run_dir.rstrip("/") + "/"

    # one row per chunk is enough: the codec and shuffle are lossless, so all
    # 16 quantized rows of a chunk carry the same measurement
    logged = {}
    for r in csv.DictReader(open(run_dir + "explore.csv")):
        if r["quantize"] == "1" and r["blob"] not in logged:
            logged[r["blob"]] = r
    if not logged:
        print("no quantized rows in explore.csv")
        return 1

    worst = {k: 0.0 for k in TOL}
    fails = []
    seen = nonzero = 0
    for b in sorted(logged):
        frame, field, _ = b.split("/")
        g = glob.glob(f"{fields_dir}/{frame}/{field}.f32")
        if not g:
            continue
        a = np.fromfile(g[0], dtype=np.float32)
        rec, _ = roundtrip(a, eb)
        mine = quality(a, rec)
        row = logged[b]
        seen += 1
        if mine["rmse"] > 0:
            nonzero += 1
        for k, col in (("rmse", "meas_rmse"), ("max_error", "meas_max_error"),
                       ("psnr_db", "meas_psnr_db"), ("ssim", "meas_ssim"),
                       ("ssim_deviation", "meas_ssim_deviation")):
            if row[col] in ("NA", "", "-1"):
                fails.append((b, col, row[col], mine[k], float("nan")))
                continue
            d = rel(float(row[col]), mine[k])
            worst[k] = max(worst[k], d)
            if d > TOL[k]:
                fails.append((b, k, float(row[col]), mine[k], d))

    print("independently recomputed %d chunks "
          "(%d with real error, %d constant)\n" % (seen, nonzero, seen - nonzero))
    print("%-16s %12s %12s   %s" % ("metric", "worst rel", "tolerance", "verdict"))
    for k in ("max_error", "rmse", "psnr_db", "ssim", "ssim_deviation"):
        print("%-16s %12.2e %12.0e   %s" % (
            k, worst[k], TOL[k], "MATCH" if worst[k] <= TOL[k] else "MISMATCH"))
    if fails:
        print("\n%d disagreement(s):" % len(fails))
        for f in fails[:10]:
            print("   ", f)
        return 1
    print("\nAGREES: two independent implementations, %d chunks" % seen)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file quality_metrics.h
 * @brief MEASURED reconstruction quality: RMSE, max error, PSNR, SSIM.
 *
 * Computed by comparing the decoded bytes against the original ones. That is
 * the distinction this file exists to make: the `actual_psnr` NeuroPress
 * already logs is ANALYTICAL -- 10*log10(range^2 / (eb^2/3)), derived from the
 * requested bound and the data range, capped at 120 dB, and therefore blind to
 * whether the bound was actually met. Every bound-violating chunk measured so
 * far reported a perfectly healthy analytical PSNR.
 *
 * The NN emits eight heads, four of which are quality (psnr, rmse, max_error,
 * ssim, plus mae). Those are PREDICTIONS. Nothing here predicts anything.
 *
 * PARITY WITH UPSTREAM IS THE POINT, so the arithmetic below is upstream
 * NeuroPress's `traceQualityKernel` + `computeTraceQuality`
 * (src/hdf5/H5VLgpucompress.cu) restated, not re-derived:
 *
 *   nine accumulators over the element pairs (x = original, y = decoded)
 *     [0] sum (x-y)^2   [1] max |x-y|   [2] min x   [3] max x
 *     [4] sum x   [5] sum x^2   [6] sum y   [7] sum y^2   [8] sum x*y
 *
 *   rmse       = sqrt([0]/n)
 *   max_error  = [1]
 *   dr         = ([3]-[2]) > 0 ? [3]-[2] : 1          // range of the ORIGINAL
 *   psnr       = rmse < 1e-10 ? 120 : min(120, 20*log10(dr/rmse))
 *   mu_x,mu_y  = [4]/n, [6]/n
 *   var_x      = [5]/n - mu_x^2      var_y = [7]/n - mu_y^2
 *   cov        = [8]/n - mu_x*mu_y
 *   C1,C2      = (0.01*dr)^2, (0.03*dr)^2
 *   ssim       = clamp( ((2 mu_x mu_y + C1)(2 cov + C2)) /
 *                       ((mu_x^2 + mu_y^2 + C1)(var_x + var_y + C2)), -1, 1 )
 *
 * MAE IS DELIBERATELY ABSENT. Upstream's kernel does not accumulate sum|x-y|,
 * so it measures four quantities and predicts five; adding a fifth here would
 * mean Clio reporting a "measured MAE" that has nothing to compare against.
 *
 * THERE IS NO HOST IMPLEMENTATION, AND THAT IS THE POINT. These metrics are
 * computed on the GPU or not at all. A host path would be reachable by a
 * host-resident chunk and would then produce plausible numbers off the CPU --
 * invisible in the results, since only the timings would move. That is the
 * same reasoning as CLIO_NEUROPRESS_REQUIRE_DEVICE, which refuses a
 * host-resident chunk at the compressor rather than quietly computing on the
 * CPU. ComputeQualityDevice REFUSES a host pointer; callers must treat false
 * as "no measurement", never as "measure it somewhere else".
 *
 * What remains here is QualityFromAccumulators, which is arithmetic on the
 * NINE NUMBERS THE GPU REDUCTION ALREADY PRODUCED -- about twenty flops, no
 * access to the data. Upstream finishes the same way (computeTraceQuality is
 * a host function). Sharing it is what keeps the derivation from drifting.
 *
 * ACCUMULATION WIDTH: the reduction is FLOAT, matching upstream, and finishes
 * with a cross-block atomicAdd, so the result is order-dependent and moves in
 * the low bits between identical runs -- upstream included. Compare on
 * relative difference, never bit equality.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_QUALITY_METRICS_H
#define CLIO_CTP_COMPRESS_PREPROCESS_QUALITY_METRICS_H

#include <cmath>
#include <cstddef>

namespace ctp::compress::preprocess {

/** The four measured quantities, plus the inputs a reader needs to check them. */
struct QualityMetrics {
  double rmse = 0.0;
  double max_error = 0.0;
  double psnr_db = 120.0;   ///< capped at 120, upstream's convention
  double ssim = 1.0;        ///< in [-1, 1]; 1.0 for a bit-exact round trip
  /**
   * 1 - ssim, computed WITHOUT the subtraction, because ssim saturates.
   *
   * For a good reconstruction ssim approaches 1 and every digit that matters
   * is in the tail: rho_E at eb=0.1 has a true deviation near 1e-14, which
   * prints as "1" at any sane precision and reads as a perfect match on data
   * that was quantized. The deviation is the informative quantity and it
   * cannot survive being recovered as 1 - ssim afterwards -- double rounding
   * near 1 destroys anything below ~1e-16, and the direct form is already
   * polluted well above that.
   *
   * Derived from an exact rearrangement rather than measured separately:
   *   A = mu_x^2+mu_y^2+C1   B = 2 mu_x mu_y + C1
   *   C = var_x+var_y+C2     D = 2 cov + C2
   *   A - B = (mu_x-mu_y)^2 = dmu^2      C - D = var_x+var_y-2cov = var(x-y)
   *   1 - BD/AC = [A*dvar + dmu^2*C - dmu^2*dvar] / (A*C)
   * and both differences come from accumulators the reduction already has:
   * dmu from slots 4 and 6, dvar from slot 0. No extra pass, no new slot.
   */
  double ssim_deviation = 0.0;
  double data_range = 0.0;  ///< max(orig) - min(orig); PSNR/SSIM scale
  std::size_t n = 0;        ///< elements compared
};

/**
 * @brief The nine sums the reduction produces, in upstream's slot order.
 *
 * SLOTS 4-8 ARE SHIFTED BY `shift`, and that is the one place this deviates
 * from upstream. Upstream accumulates raw sum(x), sum(x^2), ... and then forms
 * the variance as E[x^2] - E[x]^2 -- the textbook catastrophically-cancelling
 * expression. Measured on a constant field of 3.0 with uniform noise of
 * 1e-3: the true variance is 3.333e-07, float32 holds ~7 significant digits,
 * so 9.0000003 - 9.0000003 collapses to ZERO and upstream reports a flawless
 * SSIM of exactly 1.0 for data that is demonstrably not identical. Its own
 * float reduction cannot see the difference.
 *
 * Subtracting a shift first is algebraically identical -- variance and
 * covariance are both shift-invariant -- and conditions the subtraction so the
 * cancellation happens between small numbers instead of large ones. The mean
 * is recovered exactly by adding the shift back.
 *
 * THIS MATTERS FOR THE WORKLOAD, not just for a synthetic case: Nyx's early
 * frames are ambient density 1.0 across nearly the whole box, which is exactly
 * the regime that cancels.
 */
struct QualityAccumulators {
  double sq_err = 0.0;      ///< [0] sum (x-y)^2
  double max_abs_err = 0.0; ///< [1] max |x-y|
  double min_x = 0.0;       ///< [2]
  double max_x = 0.0;       ///< [3]
  double sum_x = 0.0;       ///< [4] sum (x - shift)
  double sum_xx = 0.0;      ///< [5] sum (x - shift)^2
  double sum_y = 0.0;       ///< [6] sum (y - shift)
  double sum_yy = 0.0;      ///< [7] sum (y - shift)^2
  double sum_xy = 0.0;      ///< [8] sum (x - shift)(y - shift)
  /** Origin slots 4-8 are measured from. Any value inside the data works;
   *  both paths use the first original element. 0 reproduces upstream. */
  double shift = 0.0;
};

/**
 * @brief Turn the nine accumulators into the four metrics.
 *
 * Split out from the reduction so the host and device paths share ONE copy of
 * the arithmetic: a divergence here would be invisible in the results (both
 * would report plausible numbers) and is exactly what the parity test cannot
 * catch if each side derives its own.
 */
inline QualityMetrics QualityFromAccumulators(const QualityAccumulators &a,
                                              std::size_t n) {
  QualityMetrics m;
  m.n = n;
  if (n == 0) return m;
  const double dn = static_cast<double>(n);

  m.rmse = std::sqrt(a.sq_err / dn);
  m.max_error = a.max_abs_err;
  m.data_range = a.max_x - a.min_x;

  // Upstream substitutes 1.0 for a degenerate range rather than dividing by
  // zero: a constant field has no range, and every error on it is then
  // reported against unity.
  const double dr = (m.data_range > 0.0) ? m.data_range : 1.0;

  // 1e-10 is upstream's threshold, not an epsilon of convenience: below it the
  // round trip is called bit-exact and PSNR saturates rather than diverging.
  m.psnr_db = (m.rmse < 1e-10)
                  ? 120.0
                  : std::fmin(120.0, 20.0 * std::log10(dr / m.rmse));

  // Means and second moments ABOUT THE SHIFT, then un-shift. Variance and
  // covariance are shift-invariant, so only the means need the origin added
  // back -- and the means never cancelled in the first place.
  const double dmu_x = a.sum_x / dn;
  const double dmu_y = a.sum_y / dn;
  const double mu_x = dmu_x + a.shift;
  const double mu_y = dmu_y + a.shift;
  const double var_x = a.sum_xx / dn - dmu_x * dmu_x;
  const double var_y = a.sum_yy / dn - dmu_y * dmu_y;
  const double cov = a.sum_xy / dn - dmu_x * dmu_y;
  const double c1 = (0.01 * dr) * (0.01 * dr);
  const double c2 = (0.03 * dr) * (0.03 * dr);
  const double num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2);
  const double den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);
  m.ssim = (den > 0.0) ? std::fmax(-1.0, std::fmin(1.0, num / den)) : 1.0;

  // 1 - ssim, from the rearrangement documented on the field. dmu is exact
  // (the shift cancels in the difference of the two shifted sums) and dvar is
  // the error variance, straight from the squared-error slot.
  const double dmu = dmu_x - dmu_y;
  const double dvar = std::fmax(0.0, a.sq_err / dn - dmu * dmu);
  const double aa = mu_x * mu_x + mu_y * mu_y + c1;
  const double cc = var_x + var_y + c2;
  m.ssim_deviation =
      (aa * cc > 0.0)
          ? std::fmax(0.0, (aa * dvar + dmu * dmu * cc - dmu * dmu * dvar) /
                               (aa * cc))
          : 0.0;
  return m;
}

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_QUALITY_METRICS_H

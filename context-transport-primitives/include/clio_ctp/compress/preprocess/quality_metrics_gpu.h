/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file quality_metrics_gpu.h
 * @brief Device entry point for the measured quality metrics.
 *
 * Same arithmetic as the host reference in quality_metrics.h -- both finish
 * through QualityFromAccumulators, so the two cannot drift apart in the
 * derivation, only in the reduction.
 *
 * THE REDUCTION IS FLOAT, MATCHING UPSTREAM, AND THAT IS DELIBERATE. Upstream
 * NeuroPress reduces its nine accumulators in float and combines blocks with
 * atomicAdd, so its result depends on block completion order and moves in the
 * low bits between identical runs. Reducing in double here would be more
 * accurate and would make a parity comparison meaningless -- the difference
 * would be Clio's better arithmetic rather than a real disagreement. Use the
 * host reference when accuracy matters; use this when comparability does.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_QUALITY_METRICS_GPU_H
#define CLIO_CTP_COMPRESS_PREPROCESS_QUALITY_METRICS_GPU_H

#include <cstddef>

#include "clio_ctp/compress/preprocess/quality_metrics.h"

namespace ctp::compress::preprocess {

/**
 * @brief Measure reconstruction quality over two DEVICE-resident float arrays.
 *
 * @param d_orig     device pointer, n floats, the original data
 * @param d_decoded  device pointer, n floats, the round-tripped data
 * @param n          element count (floats, not bytes)
 * @param stream     CUDA stream (cudaStream_t as void*), may be null
 * @param out        filled on success
 * REFUSES A HOST POINTER. Either argument residing in host memory returns
 * false and measures nothing -- there is deliberately no staging path and no
 * host implementation to fall back to. A quality number computed off the CPU
 * would be indistinguishable in the output from one computed on the GPU,
 * which is exactly the failure CLIO_NEUROPRESS_REQUIRE_DEVICE exists to
 * prevent elsewhere in this compressor.
 *
 * @return false if n == 0, a pointer is null or host-resident, or a CUDA call
 *         failed; `out` is left untouched rather than filled with plausible
 *         zeros. Callers must treat false as "not measured".
 */
bool ComputeQualityDevice(const void *d_orig, const void *d_decoded,
                          std::size_t n, void *stream, QualityMetrics *out);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_QUALITY_METRICS_GPU_H

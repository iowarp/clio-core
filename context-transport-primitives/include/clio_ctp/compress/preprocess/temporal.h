/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file temporal.h
 * @brief Device primitives for temporal (look-ahead) prediction of float32
 *        frames: the prediction, residuals, and the closed-loop bound check.
 *
 * Used by ctp::LookaheadCodec (compress/lookahead.h). Every function works on
 * device buffers, launches on `stream` (a cudaStream_t passed as an opaque
 * pointer, nullptr = default stream) and synchronizes it before returning.
 * On a build without CUDA every function returns false.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_TEMPORAL_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_TEMPORAL_H_

#include <cstddef>
#include <cstdint>

namespace ctp::compress::preprocess {

/**
 * @brief Two-sided temporal prediction: out[i] = lo[i] + w * (hi[i] - lo[i]).
 *
 * With w = 0 the prediction is lo itself (one-sided).
 *
 * @param lo     Earlier reconstructed frame.
 * @param hi     Later reconstructed frame (may equal lo when w = 0).
 * @param w      Weight of hi, in [0, 1].
 * @param out    Prediction, n floats.
 * @param n      Element count.
 * @param stream cudaStream_t as an opaque pointer.
 * @return false on a CUDA failure or without CUDA.
 */
bool TemporalPredictDevice(const float *lo, const float *hi, float w,
                           float *out, size_t n, void *stream);

/**
 * @brief Element-wise out[i] = a[i] + sign * b[i], sign = +1 or -1.
 *
 * Residuals (sign -1) and reconstructions (sign +1) use this one kernel so
 * the encoder and the decoder round identically.
 *
 * @return false on a CUDA failure or without CUDA.
 */
bool AddDevice(const float *a, const float *b, float sign, float *out,
               size_t n, void *stream);

/**
 * @brief Closed-loop reconstruction with an exact error-bound check.
 *
 * recon[i] = pred[i] + dec[i] (or dec[i] when pred is null), computed exactly
 * as AddDevice does. Every element whose reconstruction is not strictly
 * closer to x[i] than the largest float32 not above eb (NaN included) is
 * replaced by x[i] and appended to the escape list (index, value), so every
 * element of recon is within eb of x or bit-exact.
 *
 * @param x       Original frame.
 * @param pred    Prediction, or nullptr.
 * @param dec     Decoded residual (or decoded frame when pred is null).
 * @param eb      Absolute error bound.
 * @param recon   Output reconstruction, n floats (may alias dec).
 * @param esc_idx Escape indices, capacity n.
 * @param esc_val Escape values, capacity n.
 * @param esc_count Host output: number of escapes written.
 * @param n       Element count (must fit in 32 bits).
 * @param stream  cudaStream_t as an opaque pointer.
 * @return false on a CUDA failure, n >= 2^32, or without CUDA.
 */
bool BoundCheckDevice(const float *x, const float *pred, const float *dec,
                      double eb, float *recon, uint32_t *esc_idx,
                      float *esc_val, uint64_t *esc_count, size_t n,
                      void *stream);

/**
 * @brief out[idx[j]] = val[j] for j < m (applies an escape list).
 * @return false on a CUDA failure or without CUDA.
 */
bool ScatterDevice(const uint32_t *idx, const float *val, uint64_t m,
                   float *out, void *stream);

/**
 * @brief Cheap estimate of what look-ahead and spatial coding would store
 *        for the middle of three consecutive frames of one chunk.
 *
 * Both leftovers are quantized to the bound, q = rint(r / (2 eb)), and
 * costed at log2(2|q| + 1) bits per value (the width of a zigzag code):
 *   temporal: r = mid - (prev + next) / 2, what look-ahead stores;
 *   spatial:  r = mid - Lorenzo(mid), the neighbour prediction spatial
 *             compressors use (3-D on an nx*ny*nz grid; 1-D when ny = nz = 1).
 */
struct TemporalProbeResult {
  double temporal_bits = 0.0;  /**< mean estimated bits, temporal leftover */
  double spatial_bits = 0.0;   /**< mean estimated bits, spatial leftover */
  double temporal_zero = 0.0;  /**< share of temporal leftovers that round to 0 */
  double spatial_zero = 0.0;   /**< share of spatial leftovers that round to 0 */
};

/** @brief True when the probe expects look-ahead to store less. */
inline bool LookaheadPays(const TemporalProbeResult &r) {
  return r.temporal_bits < r.spatial_bits;
}

/**
 * @brief Runs the probe on three device frames of nx * ny * nz floats
 *        (x fastest).
 *
 * @param prev, mid, next Three consecutive frames of the same chunk.
 * @param nx, ny, nz      Grid; use ny = nz = 1 for flat data.
 * @param eb              Absolute error bound.
 * @param out             Host result.
 * @param stream          cudaStream_t as an opaque pointer.
 * @return false on a CUDA failure, an empty grid, or without CUDA.
 */
bool TemporalProbeDevice(const float *prev, const float *mid,
                         const float *next, size_t nx, size_t ny, size_t nz,
                         double eb, TemporalProbeResult *out, void *stream);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_TEMPORAL_H_

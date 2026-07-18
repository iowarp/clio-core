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

#ifndef SAFE_BDEV_EC_REED_SOLOMON_H_
#define SAFE_BDEV_EC_REED_SOLOMON_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gf256.h"

/**
 * Systematic Reed-Solomon erasure coding over GF(2^8).
 *
 * A code is configured with `k` data shards and a maximum of `m_max` parity
 * shards. The (k + m_max) x k generator matrix has the identity on top (so the
 * data shards are stored verbatim — systematic) and a Cauchy matrix on the
 * bottom for the parity rows.
 *
 * Two properties make this the right fit for safe_bdev's incremental-resilience
 * requirement:
 *   1. Each parity row is an INDEPENDENT linear combination of the data shards.
 *      Raising fault tolerance from c -> c+1 means computing exactly one more
 *      parity shard (EncodeParityShard(c, ...)); existing data and parity are
 *      never touched.
 *   2. Cauchy construction keeps every k x k submatrix invertible, so the code
 *      stays MDS ("any k of n reconstructs") at every intermediate parity level.
 */

namespace clio::run::safe_bdev::ec {

class ReedSolomon {
 public:
  /**
   * @param k      number of data shards
   * @param m_max  maximum number of parity shards (max simultaneous failures)
   */
  ReedSolomon(int k, int m_max) : k_(k), m_max_(m_max) {
    // Cauchy element for parity row r, data column c:
    //   1 / (x_r XOR y_c), with x_r = k + r and y_c = c.
    // x_r and y_c are drawn from disjoint ranges so x_r XOR y_c is never 0.
    // Requires k + m_max <= 256.
    cauchy_.assign(static_cast<size_t>(m_max_) * k_, 0);
    for (int r = 0; r < m_max_; ++r) {
      const uint8_t x = static_cast<uint8_t>(k_ + r);
      for (int c = 0; c < k_; ++c) {
        const uint8_t y = static_cast<uint8_t>(c);
        cauchy_[static_cast<size_t>(r) * k_ + c] = GfInv(GfAdd(x, y));
      }
    }
  }

  int k() const { return k_; }
  int m_max() const { return m_max_; }

  /** Coefficient applied to data column `c` when forming parity row `r`. */
  uint8_t CauchyCoeff(int r, int c) const {
    return cauchy_[static_cast<size_t>(r) * k_ + c];
  }

  /**
   * Compute a single parity shard (parity row `r`) from the k data shards.
   * This is the incremental primitive: it touches only the data, never other
   * parity shards.
   *
   * @param r     parity row index in [0, m_max)
   * @param data  k pointers to data shards, each `len` bytes
   * @param len   shard length in bytes
   * @param out   output buffer of `len` bytes (overwritten)
   */
  void EncodeParityShard(int r, const std::vector<const uint8_t *> &data,
                         size_t len, uint8_t *out) const {
    std::memset(out, 0, len);
    for (int c = 0; c < k_; ++c) {
      GfMulAddRegion(out, data[c], CauchyCoeff(r, c), len);
    }
  }

  /**
   * Compute the first `m` parity shards from the k data shards.
   * @param m  number of parity shards to produce, in [0, m_max]
   */
  void Encode(const std::vector<const uint8_t *> &data, size_t len, int m,
              std::vector<std::vector<uint8_t>> *parity_out) const {
    parity_out->assign(static_cast<size_t>(m), std::vector<uint8_t>(len, 0));
    for (int r = 0; r < m; ++r) {
      EncodeParityShard(r, data, len, (*parity_out)[r].data());
    }
  }

  /**
   * Reconstruct ALL k data shards from any k surviving shards.
   *
   * Each survivor is identified by a global shard index: data shards are
   * 0..k-1, parity shards are k..k+m_max-1. At least k survivors must be
   * supplied; the first k are used.
   *
   * @param survivor_index  global indices of the survivors (size >= k)
   * @param survivor_shard  survivor shard buffers (parallel to survivor_index)
   * @param len             shard length in bytes
   * @param data_out        receives the k reconstructed data shards
   * @return true on success, false if fewer than k survivors or singular system
   */
  bool DecodeData(const std::vector<int> &survivor_index,
                  const std::vector<const uint8_t *> &survivor_shard,
                  size_t len,
                  std::vector<std::vector<uint8_t>> *data_out) const {
    if (static_cast<int>(survivor_index.size()) < k_) {
      return false;
    }
    // Build the k x k matrix A whose row i is the generator-matrix row of the
    // i-th survivor, plus invert it. A_inv maps survivors -> data shards.
    std::vector<uint8_t> a(static_cast<size_t>(k_) * k_, 0);
    for (int i = 0; i < k_; ++i) {
      const int g = survivor_index[i];
      if (g < k_) {
        // Data survivor: identity row e_g.
        a[static_cast<size_t>(i) * k_ + g] = 1;
      } else {
        // Parity survivor: Cauchy row (g - k).
        const int r = g - k_;
        for (int c = 0; c < k_; ++c) {
          a[static_cast<size_t>(i) * k_ + c] = CauchyCoeff(r, c);
        }
      }
    }
    std::vector<uint8_t> a_inv;
    if (!InvertMatrix(a, k_, &a_inv)) {
      return false;
    }
    // data_i = sum_j A_inv[i][j] * survivor_j  (over the first k survivors).
    data_out->assign(static_cast<size_t>(k_), std::vector<uint8_t>(len, 0));
    for (int i = 0; i < k_; ++i) {
      uint8_t *out = (*data_out)[i].data();
      for (int j = 0; j < k_; ++j) {
        GfMulAddRegion(out, survivor_shard[j],
                       a_inv[static_cast<size_t>(i) * k_ + j], len);
      }
    }
    return true;
  }

  /**
   * Invert an n x n matrix over GF(2^8) via Gauss-Jordan elimination.
   * @return false if the matrix is singular.
   */
  static bool InvertMatrix(const std::vector<uint8_t> &in, int n,
                           std::vector<uint8_t> *out) {
    std::vector<uint8_t> m = in;          // working copy
    out->assign(static_cast<size_t>(n) * n, 0);
    for (int i = 0; i < n; ++i) {
      (*out)[static_cast<size_t>(i) * n + i] = 1;  // identity
    }
    for (int col = 0; col < n; ++col) {
      // Find a pivot row at or below `col` with a non-zero entry in `col`.
      int pivot = -1;
      for (int row = col; row < n; ++row) {
        if (m[static_cast<size_t>(row) * n + col] != 0) {
          pivot = row;
          break;
        }
      }
      if (pivot < 0) {
        return false;  // singular
      }
      if (pivot != col) {
        SwapRows(&m, n, pivot, col);
        SwapRows(out, n, pivot, col);
      }
      // Scale the pivot row so the pivot becomes 1.
      const uint8_t inv = GfInv(m[static_cast<size_t>(col) * n + col]);
      ScaleRow(&m, n, col, inv);
      ScaleRow(out, n, col, inv);
      // Eliminate the pivot column from every other row.
      for (int row = 0; row < n; ++row) {
        if (row == col) {
          continue;
        }
        const uint8_t factor = m[static_cast<size_t>(row) * n + col];
        if (factor == 0) {
          continue;
        }
        AddScaledRow(&m, n, col, row, factor);
        AddScaledRow(out, n, col, row, factor);
      }
    }
    return true;
  }

 private:
  static void SwapRows(std::vector<uint8_t> *m, int n, int a, int b) {
    for (int c = 0; c < n; ++c) {
      std::swap((*m)[static_cast<size_t>(a) * n + c],
                (*m)[static_cast<size_t>(b) * n + c]);
    }
  }

  static void ScaleRow(std::vector<uint8_t> *m, int n, int row, uint8_t s) {
    for (int c = 0; c < n; ++c) {
      (*m)[static_cast<size_t>(row) * n + c] =
          GfMul((*m)[static_cast<size_t>(row) * n + c], s);
    }
  }

  // dst_row ^= factor * src_row
  static void AddScaledRow(std::vector<uint8_t> *m, int n, int src_row,
                           int dst_row, uint8_t factor) {
    for (int c = 0; c < n; ++c) {
      (*m)[static_cast<size_t>(dst_row) * n + c] ^=
          GfMul((*m)[static_cast<size_t>(src_row) * n + c], factor);
    }
  }

  int k_;
  int m_max_;
  std::vector<uint8_t> cauchy_;  // m_max x k, row-major
};

}  // namespace clio::run::safe_bdev::ec

#endif  // SAFE_BDEV_EC_REED_SOLOMON_H_

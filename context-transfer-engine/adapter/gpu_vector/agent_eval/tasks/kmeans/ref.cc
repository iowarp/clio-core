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
 * @file ref.cc
 * @brief Independent CPU reference for the agent-eval k-means task.
 *
 * Shares no code with any GPU implementation: it regenerates the point set
 * from the task's PointVal formula and runs Lloyd's algorithm on the host,
 * accumulating centroid sums in double. Prints one line:
 *   REF counts=<n0>,...,<n(k-1)> csum=<%.6f>
 *
 * Usage: ref <data_mb> <dims> <k> <iters>
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

/**
 * @brief The task's synthetic coordinate for global element `idx`.
 * @param idx element index (point * dims + dim)
 * @param dims coordinates per point
 * @param k number of clusters
 * @return the coordinate value
 */
static float PointVal(unsigned long long idx, unsigned dims, unsigned k) {
  const unsigned long long point = idx / dims;
  const unsigned long long dim = idx % dims;
  const unsigned long long cluster = point % k;
  const float centre = static_cast<float>(cluster) * 8.0f;
  const unsigned long long h =
      (point * 6364136223846793005ull + dim * 1442695040888963407ull);
  const float jitter =
      static_cast<float>(static_cast<unsigned>(h >> 40)) * (2.0f / 16777216.0f) -
      1.0f;
  return centre + jitter;
}

/**
 * @brief Runs the reference and prints the REF line.
 * @param argc argument count (5 expected)
 * @param argv data_mb dims k iters
 * @return 0 on success, 2 on bad usage
 */
int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s data_mb dims k iters\n", argv[0]);
    return 2;
  }
  const unsigned long long data_mb = std::strtoull(argv[1], nullptr, 10);
  const unsigned dims = static_cast<unsigned>(std::atoi(argv[2]));
  const unsigned k = static_cast<unsigned>(std::atoi(argv[3]));
  const int iters = std::atoi(argv[4]);
  const unsigned long long npts = data_mb * 1048576ull / (4ull * dims);

  std::vector<float> cent(static_cast<size_t>(k) * dims);
  for (unsigned c = 0; c < k; ++c)
    for (unsigned i = 0; i < dims; ++i)
      cent[c * dims + i] = PointVal(static_cast<unsigned long long>(c) * dims + i, dims, k);

  std::vector<unsigned long long> counts(k);
  for (int it = 0; it < iters; ++it) {
    std::vector<double> sums(static_cast<size_t>(k) * dims, 0.0);
    std::fill(counts.begin(), counts.end(), 0ull);
#pragma omp parallel
    {
      std::vector<double> ls(static_cast<size_t>(k) * dims, 0.0);
      std::vector<unsigned long long> lc(k, 0ull);
      std::vector<float> pt(dims);
#pragma omp for schedule(static)
      for (long long p = 0; p < static_cast<long long>(npts); ++p) {
        for (unsigned i = 0; i < dims; ++i)
          pt[i] = PointVal(static_cast<unsigned long long>(p) * dims + i, dims, k);
        float best = 3.4e38f;
        unsigned bk = 0;
        for (unsigned c = 0; c < k; ++c) {
          float d = 0.0f;
          for (unsigned i = 0; i < dims; ++i) {
            const float x = pt[i] - cent[c * dims + i];
            d += x * x;
          }
          if (d < best) { best = d; bk = c; }
        }
        for (unsigned i = 0; i < dims; ++i) ls[bk * dims + i] += pt[i];
        ++lc[bk];
      }
#pragma omp critical
      {
        for (size_t i = 0; i < ls.size(); ++i) sums[i] += ls[i];
        for (unsigned c = 0; c < k; ++c) counts[c] += lc[c];
      }
    }
    for (unsigned c = 0; c < k; ++c) {
      if (counts[c] == 0) continue;
      for (unsigned i = 0; i < dims; ++i)
        cent[c * dims + i] = static_cast<float>(sums[c * dims + i] / counts[c]);
    }
  }
  double csum = 0.0;
  for (float v : cent) csum += v;
  std::printf("REF counts=");
  for (unsigned c = 0; c < k; ++c) std::printf(c ? ",%llu" : "%llu", counts[c]);
  std::printf(" csum=%.6f\n", csum);
  return 0;
}

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
 * @file kmeans_consumer.cu
 * @brief The GPU side of the k-means consumer (kmeans_consumer.h): one kernel
 * that assigns points to centroids and accumulates per-cluster sums, and the
 * host-side KMeans state.
 */

#include "kmeans_consumer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace kmeans_consumer {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

/**
 * Grid-stride assignment of DIM-value points of element type T: each thread
 * finds the nearest centroid of its points and adds them to block-shared sums
 * and counts, which the block then adds to the global accumulators. One value
 * per point uses |v - c| (the distance of the first, 1-D version, so its
 * results do not change); more values use the squared Euclidean distance.
 */
template <typename T, int DIM>
__global__ void AssignKernel(const T *data, size_t points, const double *c,
                             int k, double *sums, unsigned long long *counts) {
  __shared__ double s_sum[kMaxClusters * kMaxDim];
  __shared__ unsigned long long s_cnt[kMaxClusters];
  __shared__ double s_c[kMaxClusters * kMaxDim];
  for (int j = threadIdx.x; j < k * DIM; j += blockDim.x) {
    s_sum[j] = 0.0;
    s_c[j] = c[j];
  }
  for (int j = threadIdx.x; j < k; j += blockDim.x) s_cnt[j] = 0ULL;
  __syncthreads();
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < points; i += stride) {
    double v[DIM];
    bool finite = true;
    for (int d = 0; d < DIM; ++d) {
      v[d] = static_cast<double>(data[i * DIM + d]);
      finite = finite && isfinite(v[d]);
    }
    if (!finite) continue;
    int best = 0;
    double best_d = 0.0;
    for (int j = 0; j < k; ++j) {
      double dist = 0.0;
      if (DIM == 1) {
        dist = fabs(v[0] - s_c[j]);
      } else {
        for (int d = 0; d < DIM; ++d) {
          const double e = v[d] - s_c[j * DIM + d];
          dist += e * e;
        }
      }
      if (j == 0 || dist < best_d) { best_d = dist; best = j; }
    }
    for (int d = 0; d < DIM; ++d) atomicAdd(&s_sum[best * DIM + d], v[d]);
    atomicAdd(&s_cnt[best], 1ULL);
  }
  __syncthreads();
  for (int j = threadIdx.x; j < k * DIM; j += blockDim.x) atomicAdd(&sums[j], s_sum[j]);
  for (int j = threadIdx.x; j < k; j += blockDim.x) atomicAdd(&counts[j], s_cnt[j]);
}

/** Device scratch kept for the life of the process (one consumer thread). */
struct Scratch {
  double *c = nullptr;
  double *sums = nullptr;
  unsigned long long *counts = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  bool ready = false;
};

/** @return the process's scratch, allocated on first use; null on error. */
Scratch *GetScratch() {
  static Scratch s;
  if (s.ready) return &s;
  const size_t vals = kMaxClusters * kMaxDim * sizeof(double);
  if (cudaMalloc(&s.c, vals) != cudaSuccess ||
      cudaMalloc(&s.sums, vals) != cudaSuccess ||
      cudaMalloc(&s.counts, kMaxClusters * sizeof(unsigned long long)) != cudaSuccess ||
      cudaEventCreate(&s.start) != cudaSuccess ||
      cudaEventCreate(&s.stop) != cudaSuccess) {
    return nullptr;
  }
  s.ready = true;
  return &s;
}

/**
 * Launches the kernel for one element type and point size.
 * @return false for an unsupported point size
 */
template <typename T>
bool Launch(const void *data, size_t points, int dim, int blocks, Scratch *s,
            int k) {
  const T *p = static_cast<const T *>(data);
  switch (dim) {
    case 1: AssignKernel<T, 1><<<blocks, kThreads>>>(p, points, s->c, k, s->sums, s->counts); return true;
    case 2: AssignKernel<T, 2><<<blocks, kThreads>>>(p, points, s->c, k, s->sums, s->counts); return true;
    case 3: AssignKernel<T, 3><<<blocks, kThreads>>>(p, points, s->c, k, s->sums, s->counts); return true;
    default: return false;
  }
}

}  // namespace

double AccumulateChunk(const void *data, size_t n, bool f64, int dim,
                       size_t skip, const double *centroids, int k,
                       double *sums, uint64_t *counts) {
  if (k < 1 || k > kMaxClusters || dim < 1 || dim > kMaxDim || data == nullptr) {
    return -1.0;
  }
  Scratch *s = GetScratch();
  if (s == nullptr) return -1.0;
  const size_t points = n > skip ? (n - skip) / dim : 0;
  const size_t elem = f64 ? sizeof(double) : sizeof(float);
  const void *first = static_cast<const char *>(data) + skip * elem;
  double h_sums[kMaxClusters * kMaxDim];
  unsigned long long h_counts[kMaxClusters];
  cudaEventRecord(s->start);
  cudaMemcpy(s->c, centroids, k * dim * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(s->sums, 0, k * dim * sizeof(double));
  cudaMemset(s->counts, 0, k * sizeof(unsigned long long));
  const size_t want = (points + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < kMaxBlocks ? (want > 0 ? want : 1) : kMaxBlocks);
  const bool ok = f64 ? Launch<double>(first, points, dim, blocks, s, k)
                      : Launch<float>(first, points, dim, blocks, s, k);
  cudaMemcpy(h_sums, s->sums, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_counts, s->counts, k * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaEventRecord(s->stop);
  cudaEventSynchronize(s->stop);
  if (!ok || cudaGetLastError() != cudaSuccess) return -1.0;
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s->start, s->stop);
  for (int j = 0; j < k * dim; ++j) sums[j] += h_sums[j];
  for (int j = 0; j < k; ++j) counts[j] += h_counts[j];
  return static_cast<double>(ms);
}

KMeans::KMeans(int k, int dim)
    : k_(k), dim_(dim), centroids_(k * dim, 0.0), sums_(k * dim, 0.0),
      counts_(k, 0) {}

bool KMeans::Seed(std::vector<double> sample) {
  // Keep only whole points whose values are all finite.
  std::vector<double> pts;
  for (size_t i = 0; i + dim_ <= sample.size(); i += dim_) {
    bool finite = true;
    for (int d = 0; d < dim_; ++d) finite = finite && std::isfinite(sample[i + d]);
    if (finite) pts.insert(pts.end(), sample.begin() + i, sample.begin() + i + dim_);
  }
  const size_t n = pts.size() / dim_;
  if (n == 0) return false;
  if (dim_ == 1) std::sort(pts.begin(), pts.end());   // quantiles
  for (int j = 0; j < k_; ++j) {
    const size_t at = std::min(static_cast<size_t>((j + 0.5) / k_ * n), n - 1);
    for (int d = 0; d < dim_; ++d) centroids_[j * dim_ + d] = pts[at * dim_ + d];
  }
  return true;
}

void KMeans::BeginPass() {
  std::fill(sums_.begin(), sums_.end(), 0.0);
  std::fill(counts_.begin(), counts_.end(), 0);
  points_ = 0;
  chunks_ = 0;
  wall_ms_ = 0.0;
  gpu_ms_ = 0.0;
}

bool KMeans::AddChunk(const void *data, size_t n, bool f64, size_t skip) {
  const auto t0 = std::chrono::steady_clock::now();
  const double ms = AccumulateChunk(data, n, f64, dim_, skip, centroids_.data(),
                                    k_, sums_.data(), counts_.data());
  wall_ms_ += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
  if (ms < 0.0) return false;
  gpu_ms_ += ms;
  ++chunks_;
  return true;
}

double KMeans::EndPass() {
  double shift = 0.0;
  for (int j = 0; j < k_; ++j) {
    points_ += counts_[j];
    if (counts_[j] == 0) continue;
    for (int d = 0; d < dim_; ++d) {
      const double c = sums_[j * dim_ + d] / static_cast<double>(counts_[j]);
      shift = std::max(shift, std::fabs(c - centroids_[j * dim_ + d]));
      centroids_[j * dim_ + d] = c;
    }
  }
  return shift;
}

}  // namespace kmeans_consumer

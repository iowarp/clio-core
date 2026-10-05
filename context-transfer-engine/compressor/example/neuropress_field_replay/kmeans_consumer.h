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
 * @file kmeans_consumer.h
 * @brief A k-means clustering consumer for the field replay: one Lloyd
 * iteration per read pass over the data held in storage.
 *
 * An out-of-core k-means reads the whole dataset once per iteration: each
 * chunk is read back (and decompressed) into GPU memory, every point is
 * assigned to the nearest of K centroids and added to that cluster's sum and
 * count; after the pass the centroids move to the cluster means. With
 * --read-repeat R the replay therefore does R iterations, which is the
 * "1 write, R reads" consumer the cost model describes.
 *
 * A point is one value (a grid field: VPIC, Nyx, WarpX) or DIM consecutive
 * values (a per-atom vector stored x0 y0 z0 x1 ...: LAMMPS, DIM = 3). Each
 * physical field is clustered on its own (the driver keeps one KMeans per
 * field). Elements are float32 or float64.
 */

#ifndef NEUROPRESS_FIELD_REPLAY_KMEANS_CONSUMER_H_
#define NEUROPRESS_FIELD_REPLAY_KMEANS_CONSUMER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kmeans_consumer {

/** Most clusters supported (per-block shared accumulators). */
constexpr int kMaxClusters = 32;
/** Most values per point. */
constexpr int kMaxDim = 3;

/**
 * @brief One chunk of one Lloyd iteration on the GPU.
 *
 * Assigns every complete point of the chunk to the nearest of the k
 * centroids (Euclidean; a point with a NaN or infinite value is skipped) and
 * adds it to that cluster's running sum and count.
 *
 * @param data      device pointer to n elements
 * @param n         number of elements
 * @param f64       true for float64 elements, false for float32
 * @param dim       values per point (1 .. kMaxDim)
 * @param skip      elements before the first complete point (a point split
 *                  by the chunk boundary is skipped); a trailing partial
 *                  point is skipped too
 * @param centroids host array of k * dim centroid values
 * @param k         number of clusters (1 .. kMaxClusters)
 * @param sums      host array of k * dim sums, accumulated into
 * @param counts    host array of k counts, accumulated into
 * @return GPU time of the step in ms (CUDA events, from the centroid upload
 *         to the result download), or a negative value on a CUDA error
 */
double AccumulateChunk(const void *data, size_t n, bool f64, int dim,
                       size_t skip, const double *centroids, int k,
                       double *sums, uint64_t *counts);

/**
 * @brief Out-of-core k-means of one field: the centroids, and the sums and
 * counts of the iteration in progress.
 */
class KMeans {
 public:
  /**
   * @param k   number of clusters (1 .. kMaxClusters)
   * @param dim values per point (1 .. kMaxDim)
   */
  KMeans(int k, int dim);

  /**
   * @brief Seeds the centroids from a sample of points.
   *
   * One value per point: k evenly spaced quantiles of the finite values.
   * More: k points evenly spaced through the sample (finite points only).
   *
   * @param sample points drawn from the data, dim values each (by value)
   * @return false when the sample has no finite point
   */
  bool Seed(std::vector<double> sample);

  /** @brief Starts one Lloyd iteration: clears the sums, counts and timers. */
  void BeginPass();

  /**
   * @brief Adds one device-resident chunk to the iteration in progress.
   * @param data device pointer to n elements
   * @param n    number of elements
   * @param f64  true for float64 elements, false for float32
   * @param skip elements before the chunk's first complete point
   * @return false on a CUDA error
   */
  bool AddChunk(const void *data, size_t n, bool f64, size_t skip);

  /**
   * @brief Ends the iteration: moves every non-empty cluster's centroid to
   * the mean of its points (an empty cluster keeps its centroid).
   * @return the largest move of a centroid value in the iteration
   */
  double EndPass();

  /** @return the current centroids, k * dim values, cluster by cluster */
  const std::vector<double> &centroids() const { return centroids_; }
  /** @return points clustered in the last finished iteration */
  uint64_t points() const { return points_; }
  /** @return chunks clustered in the iteration */
  size_t chunks() const { return chunks_; }
  /** @return host wall time of AddChunk in the iteration, ms */
  double wall_ms() const { return wall_ms_; }
  /** @return GPU time of AddChunk in the iteration, ms (CUDA events) */
  double gpu_ms() const { return gpu_ms_; }

 private:
  int k_;
  int dim_;
  std::vector<double> centroids_;
  std::vector<double> sums_;
  std::vector<uint64_t> counts_;
  uint64_t points_ = 0;
  size_t chunks_ = 0;
  double wall_ms_ = 0.0;
  double gpu_ms_ = 0.0;
};

}  // namespace kmeans_consumer

#endif  // NEUROPRESS_FIELD_REPLAY_KMEANS_CONSUMER_H_

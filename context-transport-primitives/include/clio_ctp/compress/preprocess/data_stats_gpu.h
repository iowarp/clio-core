/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef HERMES_SHM_DATA_STATS_GPU_H
#define HERMES_SHM_DATA_STATS_GPU_H

#include "clio_ctp/compress/preprocess/data_stats.h"
#include "clio_ctp/util/gpu_api.h"

namespace ctp {

/**
 * Device-native counterpart to DataStatisticsFactory: computes the same
 * three selection features (Shannon entropy, MAD, second-derivative mean)
 * directly from a buffer that already lives in GPU device memory, so a
 * caller resolving a CUDA-IPC-backed ShmPtr never has to stage the buffer
 * through host memory just to feed the NeuroPress predictor.
 *
 * Only a 256-bin histogram and a handful of scalars ever cross back to the
 * host -- the buffer itself (which can be orders of magnitude larger) never
 * does. Numerically matches DataStatisticsFactory's host implementation
 * (same byte histogram for entropy, same two-pass mean/MAD, same
 * neighbor-difference second derivative).
 *
 * @param device_data Pointer to data already resident in GPU device memory.
 * @param num_elements Number of elements (NOT bytes) of `type`.
 * @param type Element type (drives the typed mean/MAD/second-derivative
 *   passes; entropy always operates on raw bytes regardless of type).
 * @return false if CUDA support isn't compiled in or `type` is unrecognized.
 */
bool ComputeDeviceStats(const void *device_data, size_t num_elements,
                         DataType type, double *out_entropy, double *out_mad,
                         double *out_second_derivative);

/**
 * The three selection features, held in DEVICE memory.
 *
 * Clio's counterpart of NeuroPress's AutoStatsGPU (src/stats/auto_stats_gpu.h),
 * and it exists for the same reason: upstream's inference path never brings
 * the statistics to the host at all. runStatsKernelsNoSync returns an
 * AutoStatsGPU* and gpucompress_infer_gpu hands that device pointer straight
 * to the NN -- "Stats remain on GPU -- NN inference reads d_stats_ptr directly
 * on device" (gpucompress_compress.cpp).
 *
 * ComputeDeviceStats() above cannot express that: it returns host doubles, so
 * feeding NeuroPress meant copying the histogram and scalars down, computing
 * the entropy sum in a host loop, and uploading a feature matrix back. The
 * bytes were small; the SYNCHRONIZATION was not -- it split one asynchronous
 * pipeline into two host-serialized halves, twice (once for the mean between
 * the two stats passes, once for the features).
 */
struct DeviceFeatureStats {
  double entropy;
  double mad;
  double second_derivative;
};

/**
 * The stream the device stats + inference path runs on, one per thread.
 *
 * Upstream runs each compression on its CompContext's own stream
 * (ctx->stream, gpucompress_pool.cpp) and synchronizes only that stream.
 * Clio has no CompContext to attach to, so the stream is per-thread -- the
 * same scoping choice, and for the same reason, as the per-flow EMA buffer in
 * neuropress_nn_gpu_kernels.cu.
 *
 * @return An opaque cudaStream_t, or nullptr without CUDA (callers then pass
 *   it through unchanged and every entry point below fails cleanly).
 */
void *DeviceStatsStream();

/**
 * Compute the three features into device memory, launching asynchronously.
 *
 * Direct counterpart of NeuroPress's runStatsKernelsNoSync (stats_kernel.cu)
 * including its lifetime contract: the returned buffer is reused by the next
 * call on the same thread, exactly as upstream's belongs to a pool slot and
 * stays valid only until that slot is reused.
 *
 * NOTHING is copied to the host, and the kernels are not synchronized -- the
 * mean needed between the two passes is read on-device (mirroring
 * madPass2Kernel's `stats->sum / stats->num_elements`, stats_kernel.cu)
 * and the entropy sum is reduced on-device (mirroring
 * entropyFromHistogramKernel, entropy_kernel.cu). The caller is expected
 * to enqueue its inference on the SAME stream and synchronize once, at the
 * end, as gpucompress_infer_gpu does.
 *
 * @return Device pointer to a DeviceFeatureStats, or nullptr on failure.
 */
const void *ComputeDeviceStatsResident(const void *device_data,
                                        size_t num_elements, DataType type,
                                        void *stream);

/**
 * Copy the three features out to the host (24 bytes) and synchronize.
 *
 * Deliberately separate from the compute above so it can be called AFTER the
 * inference has been enqueued rather than between the two, which is the whole
 * point. The runtime needs these numbers for the selection log and for
 * Train()'s samples; upstream reads its own the same way when it needs them
 * on the host (nn_gpu.cu's debug path, and the VOL's per-chunk diagnostics).
 */
bool ReadDeviceFeatureStats(const void *device_stats, double *out_entropy,
                            double *out_mad, double *out_second_derivative,
                            void *stream);

/**
 * Computes the same three selection features either on-device (via
 * ComputeDeviceStats, when `chunk` is GPU-resident) or on-host (via
 * DataStatisticsFactory, otherwise) -- the exact dispatch
 * EstCompressionStats uses to feed NeuroPress. Factored out here so the
 * dispatch itself is unit-testable without pulling in the compressor
 * chimod's task-dispatch machinery (compression/PutBlob are a separate,
 * larger device-memory gap this dispatch does not attempt to close).
 */
inline bool ComputeCompressionFeatures(const void *chunk, size_t num_elements,
                                        DataType type, double *out_entropy,
                                        double *out_mad,
                                        double *out_second_derivative) {
  const bool on_device = IsDevicePointer(chunk);
  if (on_device) {
    if (ComputeDeviceStats(chunk, num_elements, type, out_entropy, out_mad,
                            out_second_derivative)) {
      return true;
    }
    // The host routines below dereference `chunk` directly. For a device
    // pointer that is not a wrong answer, it is a segfault -- so report the
    // failure instead of falling through to code that cannot run. Reached
    // when the device-stats path is unavailable or fails; the features are
    // left untouched and the caller decides (feeding NeuroPress zeros would
    // silently skew every prediction for the chunk).
    *out_entropy = 0.0;
    *out_mad = 0.0;
    *out_second_derivative = 0.0;
    return false;
  }
  *out_entropy =
      DataStatisticsFactory::CalculateShannonEntropy(chunk, num_elements, type);
  *out_mad = DataStatisticsFactory::CalculateMAD(chunk, num_elements, type);
  *out_second_derivative =
      DataStatisticsFactory::CalculateSecondDerivative(chunk, num_elements, type);
  return true;
}

}  // namespace ctp

#endif  // HERMES_SHM_DATA_STATS_GPU_H

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef HERMES_SHM_DATA_STATS_GPU_H
#define HERMES_SHM_DATA_STATS_GPU_H

#include <vector>

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
/**
 * Device-resident float64 selection features: converts the chunk to float32 on
 * the GPU and measures that, so a double chunk yields the same features
 * whether it was written from host or device memory. Returns the same opaque
 * stats handle ComputeDeviceStatsResident returns, or nullptr on failure.
 */
const void *ComputeDeviceStatsResidentF32From64(const void *device_data,
                                                  size_t num_doubles,
                                                  void *stream);

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


/**
 * Selection features for the NeuroPress model, honouring the element type.
 *
 * The model was normalised on FLOAT32 statistics, so everything it is shown
 * has to arrive on that scale. Getting there by REINTERPRETING a float64
 * buffer as float32 does not do that -- it relabels the bytes instead of
 * converting the numbers. Each double is then read as two float32 words, and
 * the low word is pure mantissa: about one time in 256 its top bits land on
 * IEEE-754's reserved exponent==255, which IS the NaN encoding. NaN
 * propagates through the mean, so MAD and the second derivative came back NaN
 * for every chunk of every float64 dataset, and the model -- handed NaN for
 * two of its three inputs -- returned a constant prediction. Measured on a
 * 4 MiB LAMMPS chunk: 2212 NaN words of 1048576, and a predicted ratio
 * identical to five significant figures across 249 chunks whose real ratios
 * spanned 0.995x to 2.80x.
 *
 * A float64 buffer is therefore CONVERTED: the values are downcast to float32
 * and the statistics computed on those, which is the same physical data on the
 * scale the model expects. Entropy is unaffected either way (it is a byte
 * histogram and never interprets a float), but it is computed from the same
 * downcast array so all three features describe one representation.
 *
 * @param chunk       host- or device-resident chunk
 * @param chunk_bytes size of the chunk in BYTES (not elements)
 * @param data_type   Context::data_type_ -- 1 float32, 2 float64, else opaque
 */
inline bool ComputeNeuroPressFeatures(const void *chunk, size_t chunk_bytes,
                                        int data_type, double *out_entropy,
                                        double *out_mad,
                                        double *out_second_derivative) {
  if (chunk == nullptr || chunk_bytes == 0) return false;

  /* Float64 on the GPU: convert there and read the result back, so a double
     chunk gets the same features wherever it lives. */
  if (data_type == 2 && IsDevicePointer(chunk)) {
    const size_t n = chunk_bytes / sizeof(double);
    if (n == 0) return false;
    const void *stats = ComputeDeviceStatsResidentF32From64(chunk, n, nullptr);
    if (stats == nullptr) return false;
    return ReadDeviceFeatureStats(stats, out_entropy, out_mad,
                                  out_second_derivative, nullptr);
  }

  /* Float64, host-resident: convert, then measure. */
  if (data_type == 2 && !IsDevicePointer(chunk)) {
    const size_t n = chunk_bytes / sizeof(double);
    if (n == 0) return false;
    const double *src = static_cast<const double *>(chunk);
    std::vector<float> narrowed(n);
    for (size_t i = 0; i < n; ++i) {
      narrowed[i] = static_cast<float>(src[i]);
    }
    return ComputeCompressionFeatures(narrowed.data(), n, DataType::FLOAT32,
                                      out_entropy, out_mad,
                                      out_second_derivative);
  }

  const size_t n = chunk_bytes / sizeof(float);
  if (n == 0) return false;
  return ComputeCompressionFeatures(chunk, n, DataType::FLOAT32, out_entropy,
                                    out_mad, out_second_derivative);
}

}  // namespace ctp

#endif  // HERMES_SHM_DATA_STATS_GPU_H

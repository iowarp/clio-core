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

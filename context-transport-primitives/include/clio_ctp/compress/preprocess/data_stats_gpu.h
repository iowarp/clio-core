/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef HERMES_SHM_DATA_STATS_GPU_H
#define HERMES_SHM_DATA_STATS_GPU_H

#include "clio_ctp/compress/preprocess/data_stats.h"

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

}  // namespace ctp

#endif  // HERMES_SHM_DATA_STATS_GPU_H

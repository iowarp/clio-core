/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file clio_path.h
 * @brief Drives Clio-NeuroPress over one GPU-resident chunk, recording a
 *        callback trace in the same taxonomy as the native side.
 */
#ifndef CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_CLIO_PATH_H_
#define CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_CLIO_PATH_H_

#include <cstddef>
#include <string>

#include "callback_trace.h"

namespace npeq {

/** @brief Mirror of NativeChunkResult, for Clio's stage decomposition. */
struct ClioChunkResult {
  bool ok = false;
  std::string error;

  int action = -1;
  int algo = -1;
  bool quantize_selected = false;
  bool shuffle_selected = false;

  const void *quantized_device = nullptr;
  size_t quantized_bytes = 0;

  /** Round trip through Clio's own dequantizer; see NativeChunkResult. */
  const void *dequantized_device = nullptr;
  size_t dequantized_bytes = 0;
  bool bound_checked = false;
  unsigned long long bound_violations = 0;
  const void *shuffled_device = nullptr;
  size_t shuffled_bytes = 0;

  /** Clio's codec output. Its container differs from native's by design
   *  (D15-1); this is the raw codec payload, which is what is comparable. */
  const void *payload_device = nullptr;
  size_t payload_bytes = 0;

  double entropy = 0.0;
  double mad = 0.0;
  double second_derivative = 0.0;
};

/**
 * @brief Load Clio's NeuroPress predictor. Idempotent.
 * @param weights_dir Directory holding model.nnwt.
 */
bool ClioInitPredictor(const std::string &weights_dir, std::string *error);
void ClioShutdownPredictor();

ClioChunkResult ClioRunChunk(const void *device_input, size_t bytes,
                             double error_bound, int chunk_id);

void ClioReleaseChunk(ClioChunkResult *result);

}  // namespace npeq

#endif  // CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_CLIO_PATH_H_

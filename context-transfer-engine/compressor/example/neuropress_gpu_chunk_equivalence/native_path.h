/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file native_path.h
 * @brief Drives native NeuroPress over one GPU-resident chunk, recording a
 *        callback trace.
 *
 * Kept in its own translation unit with a header that names no NeuroPress
 * type. NeuroPress's internals (`api/internal.hpp`) pull in nvcomp and its own
 * compression factory; Clio's headers pull in theirs. Mixing them in one TU
 * buys nothing and risks a collision that would be tedious to diagnose, so the
 * two implementations meet only through the neutral types below.
 */
#ifndef CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_NATIVE_PATH_H_
#define CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_NATIVE_PATH_H_

#include <cstddef>
#include <string>

#include "callback_trace.h"

namespace npeq {

/**
 * @brief What the native run produced, and the device buffers to compare.
 *
 * The buffers stay ON the device. They are owned by this module and released
 * by NativeReleaseChunk().
 */
struct NativeChunkResult {
  bool ok = false;
  std::string error;

  int action = -1;
  int algo = -1;
  bool quantize_selected = false;
  bool shuffle_selected = false;

  /** Stage-probe outputs, device-resident, or nullptr when not selected. */
  const void *quantized_device = nullptr;
  size_t quantized_bytes = 0;

  /**
   * The quantized buffer sent back through upstream's own dequantizer, and the
   * count of elements whose reconstruction falls outside the REQUESTED bound.
   *
   * Quantization being byte-identical to Clio's says the two agree; it does not
   * say either of them honours the guarantee the caller asked for. That needs
   * the round trip, so the harness does it.
   */
  const void *dequantized_device = nullptr;
  size_t dequantized_bytes = 0;
  bool bound_checked = false;
  unsigned long long bound_violations = 0;
  const void *shuffled_device = nullptr;
  size_t shuffled_bytes = 0;

  /** The production compress call's own output, header included. */
  const void *compressed_device = nullptr;
  size_t compressed_bytes = 0;
  /** Payload only: compressed_device + 64, the blob minus native's header. */
  const void *payload_device = nullptr;
  size_t payload_bytes = 0;

  double entropy = 0.0;
  double mad = 0.0;
  double second_derivative = 0.0;
};

/**
 * @brief Load the library and the network. Idempotent.
 *
 * @param weights_path Path to model.nnwt.
 */
bool NativeInit(const std::string &weights_path, std::string *error);
void NativeShutdown();

/**
 * @brief Run one chunk through native NeuroPress, tracing every stage.
 *
 * Entries are appended to the Recorder's current chunk, which the caller has
 * already opened with BeginChunk(). The input must be device-resident; that is
 * checked, not assumed.
 */
NativeChunkResult NativeRunChunk(const void *device_input, size_t bytes,
                                 double error_bound, int chunk_id);

void NativeReleaseChunk(NativeChunkResult *result);

}  // namespace npeq

#endif  // CLIO_CTE_COMPRESSOR_EXAMPLE_NP_EQUIV_NATIVE_PATH_H_

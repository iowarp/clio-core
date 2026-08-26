/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_path_trace.h
 * @brief One definition of the "[np-path]" end-to-end trace, shared by the
 *        compressor runtime and the selection half.
 *
 * COMPILE-TIME, off unless -DCLIO_NEUROPRESS_PATH_TRACE. Undefined it expands
 * to nothing, so a normal build pays neither a branch nor a format string.
 * Lived in compressor_runtime.cc until the selection half needed it too;
 * copying it a third time would have let the two drift.
 *
 * The trace names the SEVEN steps one chunk passes through, in order:
 *
 *   1 stats    feature extraction -- entropy, MAD, 2nd derivative
 *   2 infer    the NN, ONE forward pass PER ACTION over the action space
 *   3 rank     cost model + sort -> the primary pick
 *   4 primary  the picked codec actually run, and measured
 *   5 gate     error_pct vs the exploration threshold
 *   6 explore  each alternative actually run, and measured
 *   7 adopt    re-scored on measured values; a winner replaces the primary
 *
 * Steps 1-3 are PREDICTION and happen once. Steps 6-7 are MEASUREMENT and do
 * not re-run the network -- the alternatives carry the predictions step 2
 * already made for them, which is why a row can show a prediction beside the
 * measurement of the same action.
 *
 * NpWhere() is what makes a silent host fallback visible: every routine that
 * could change hardware without changing its result prints the residency of
 * the buffer it just took.
 */

#ifndef CLIO_CTE_COMPRESSOR_NEUROPRESS_PATH_TRACE_H_
#define CLIO_CTE_COMPRESSOR_NEUROPRESS_PATH_TRACE_H_

#include <cstdio>

#include <clio_ctp/util/gpu_api.h>

#ifdef CLIO_NEUROPRESS_PATH_TRACE
#define CLIO_PATH_TRACE(...)                        \
  do {                                              \
    std::fprintf(stderr, "[np-path] " __VA_ARGS__); \
    std::fprintf(stderr, "\n");                     \
    std::fflush(stderr);                            \
  } while (0)
#else
#define CLIO_PATH_TRACE(...) ((void)0)
#endif

namespace clio::cte::compressor {

/** "[GPU]" or "[CPU]" for the buffer at `p`.
 *
 *  A null pointer is "[---]", not "[CPU]": absent is not host-resident, and
 *  printing it as host would read as a fallback that never happened. */
inline const char *NpWhere(const void *p) {
  if (p == nullptr) return "[---]";
  return ctp::IsDevicePointer(p) ? "[GPU]" : "[CPU]";
}

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_PATH_TRACE_H_

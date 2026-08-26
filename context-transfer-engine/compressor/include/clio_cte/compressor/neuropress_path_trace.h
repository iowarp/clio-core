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
 * RUNTIME, off unless CLIO_NEUROPRESS_PATH_TRACE is set in the environment.
 * It used to be a compile-time define, which meant the only way to turn the
 * trace on was to hand-edit a generated flags.make and rebuild -- and CMake
 * silently threw that edit away on its next regenerate, so a trace that had
 * been working stopped for no visible reason. The env var cannot be lost that
 * way and needs no rebuild.
 *
 * The check is a function-local static read once per process, so a disabled
 * trace costs one predictable branch at each site, and the sites are per
 * CHUNK (megabytes apart), never per element. Build with
 * -DCLIO_NEUROPRESS_PATH_TRACE_OFF to compile the sites out entirely if even
 * that is unwanted.
 *
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
#include <cstdlib>

#include <clio_ctp/util/gpu_api.h>

namespace clio::cte::compressor {

/** True when CLIO_NEUROPRESS_PATH_TRACE asks for the trace.
 *
 *  Read ONCE per process, not once per chunk: this is consulted from worker
 *  threads, and a getenv per call would be both a syscall on the hot path and
 *  a data race against anyone calling setenv. "0" and the empty string are
 *  off; any other value is on. */
inline bool NpTraceEnabled() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_PATH_TRACE");
    if (e == nullptr || *e == '\0') return false;
    return !(e[0] == '0' && e[1] == '\0');
  }();
  return on;
}

}  // namespace clio::cte::compressor

/* Fully qualified so the macro works from any namespace, including the two
 * .cc files that expand it from inside clio::cte::compressor. Arguments are
 * evaluated ONLY when the trace is on, which is what keeps NpWhere()'s
 * cudaPointerGetAttributes call off a disabled path. */
#ifdef CLIO_NEUROPRESS_PATH_TRACE_OFF
/* Compiled out, but the arguments are still PARSED. `if (false)` keeps every
 * expression referenced, so the locals the trace sites hoist do not turn into
 * unused-variable warnings (six of them, enough to fail a -Werror build), and
 * the format strings stay type-checked in a configuration nobody exercises.
 * The branch is dead and eliminated; nothing is evaluated at runtime. */
#define CLIO_PATH_TRACE(...)                          \
  do {                                                \
    if (false) {                                      \
      std::fprintf(stderr, "[np-path] " __VA_ARGS__); \
    }                                                 \
  } while (0)
#else
#define CLIO_PATH_TRACE(...)                          \
  do {                                                \
    if (::clio::cte::compressor::NpTraceEnabled()) {  \
      std::fprintf(stderr, "[np-path] " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                    \
      std::fflush(stderr);                            \
    }                                                 \
  } while (0)
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

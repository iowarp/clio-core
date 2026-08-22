/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_diag_parity_globals.cc
 * @brief The upstream globals gpucompress_diagnostics.cpp needs, at upstream's
 *        own defaults. Linking gpucompress_api.cpp to reach four floats would
 *        drag in the CompContext pool and nvcomp managers.
 */

#include <cstdio>
#include <cstdlib>

float g_rank_w0 = 1.0f;                   // weight on compression time
float g_rank_w1 = 1.0f;                   // weight on decompression time
float g_rank_w2 = 1.0f;                   // weight on I/O cost
float g_measured_bw_bytes_per_ms = 5e6f;  // 5 GB/s, upstream's default
bool g_debug_nn = false;                  // stderr ranking dump, off

namespace gpucompress {

/** Referenced by gpucompress_flush_manager_cache(), never called here.
 *  Aborts rather than silently doing nothing: reaching it would mean the test
 *  wandered into the compression path. */
void flushCompManagerCache() {
  std::fprintf(stderr,
               "flushCompManagerCache() reached in the diag parity harness: "
               "this test does not link the compression pool\n");
  std::abort();
}

}  // namespace gpucompress

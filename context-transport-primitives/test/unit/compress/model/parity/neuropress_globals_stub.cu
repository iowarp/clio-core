/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_globals_stub.cu
 * @brief The handful of upstream globals nn_gpu.cu references.
 *
 * The parity harness links NeuroPress's nn_gpu.cu directly (it is
 * self-contained pure CUDA) rather than building the whole project, which
 * would drag in nvcomp, cuSZ and ndzip. That leaves a few symbols owned by
 * gpucompress_api.cpp and gpucompress_pool.cpp undefined.
 *
 * Values here are copied EXACTLY from src/api/gpucompress_api.cpp -- the
 * ranking weights and bandwidth feed the cost model the comparison is
 * checking, so a wrong default here would silently invalidate the test.
 *
 * The two pool functions are no-ops on purpose: they walk the CompContext
 * pool, and the harness creates none.
 */

#include <cuda_runtime.h>
#include <atomic>

// gpucompress_api.cpp:62-71 -- verbatim defaults.
cudaEvent_t g_sgd_done = nullptr;
std::atomic<bool> g_sgd_ever_fired{false};
float g_rank_w0 = 1.0f;                  // weight on compression time
float g_rank_w1 = 1.0f;                  // weight on decompression time
float g_rank_w2 = 1.0f;                  // weight on I/O cost
float g_min_psnr_db = 0.0f;              // PSNR floor (0 = no filtering)
float g_measured_bw_bytes_per_ms = 5e6f;  // 5 GB/s
cudaStream_t g_sgd_stream = nullptr;      // nn_gpu.cu creates it on demand
bool g_debug_nn = false;                  // stderr ranking dump, off

struct NNDebugPerConfig;

namespace gpucompress {
// gpucompress_pool.cpp -- no CompContext pool exists in this harness.
void syncAllCompContextStreams() {}
void resetAllSGDEMABuffers() {}
// gpucompress_diagnostics.cpp -- only reachable when g_debug_nn is on.
void printNNDebugRanking(const NNDebugPerConfig *, int, float, float, float) {}
}  // namespace gpucompress

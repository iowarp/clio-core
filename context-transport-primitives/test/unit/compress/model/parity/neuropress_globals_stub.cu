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
 * syncAllCompContextStreams() is a no-op on purpose: it walks the CompContext
 * pool, and no harness builds one. resetAllSGDEMABuffers() cannot be, because
 * the flow harness DOES create contexts and the reset-on-reload behavior is
 * one of the things it checks -- see g_harness_ctx_grad_buffers below.
 */

#include <cuda_runtime.h>
#include <atomic>
#include <vector>

#include "nn/nn_weights.h"  // NN_SGD_GRAD_REGION

// gpucompress_api.cpp -- verbatim defaults.
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

// Stands in for gpucompress_pool.cpp's g_comp_pool: a harness that creates
// CompContexts registers each one's d_sgd_grad_buffer here. Harnesses that
// create none leave it empty, and the reset below stays the no-op it was.
std::vector<float *> g_harness_ctx_grad_buffers;

namespace gpucompress {
// gpucompress_pool.cpp -- no CompContext pool exists in this harness.
void syncAllCompContextStreams() {}

// Mirrors gpucompress_pool.cpp: zero region 2 (the EMA gradient) of
// every live context's buffer, leaving regions 0 and 1 alone. Called from
// loadNNFromBinary (nn_gpu.cu), so a reload restarts every flow's
// gradient history, not just the reloading thread's.
void resetAllSGDEMABuffers() {
  for (float *p : g_harness_ctx_grad_buffers) {
    if (p) {
      cudaMemset(p + 2 * NN_SGD_GRAD_REGION, 0,
                 NN_SGD_GRAD_REGION * sizeof(float));
    }
  }
}
// gpucompress_diagnostics.cpp -- only reachable when g_debug_nn is on.
void printNNDebugRanking(const NNDebugPerConfig *, int, float, float, float) {}
}  // namespace gpucompress

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_MATVEC_H_
#define CLIO_CTE_GPU_VECTOR_MATVEC_H_

#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/types.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_cte/gpu_vector/gpu_vector_kernels.h>
#include <clio_cte/gpu_vector/gpu_vector_view.h>

#if CTP_IS_GPU_COMPILER

namespace clio::cte::gpu_vector {

/**
 * ============================================================================
 * Weight-application transaction: y = W * x  (the mat-vec step of inference)
 * ============================================================================
 *
 * This is the CTE-vector analogue of llama.cpp's `mul_mat_vec` kernel
 * (ggml/src/ggml-cuda/mmvq.cu). It expresses the ONE place model weights are
 * consumed during decode — the matrix-vector product of a weight matrix `W`
 * against a resident activation `x` — as a *transaction over the paged
 * Vector<T>* rather than a read of a VRAM-resident weight buffer.
 *
 * Why this is the right shape for the two-tier cache:
 *
 *   - In llama.cpp the weight matrix is read EXACTLY ONCE per token, strictly
 *     sequentially within each row (kbx increases monotonically over the
 *     quant blocks of the row). There is no temporal reuse of a weight within
 *     a mat-vec — it is pure streaming, bandwidth-bound.
 *
 *   - `dev::vector::read_range(lo, hi, consume)` is precisely a
 *     "stream this contiguous element range once, in order" transaction. Its
 *     built-in rescore lookahead pushes {page_idx, score} hints for the next
 *     kLookahead pages into the per-block rescore_q, so the async cache
 *     manager (Phase 4 prefetch) pulls the upcoming weight pages from CTE
 *     blob storage into the DRAM/HBM tiers *ahead of* the dot product that
 *     consumes them. The sequential access pattern is self-describing — the
 *     manager never has to guess.
 *
 * Layout / mapping to the Vector's per-block cache model
 * ------------------------------------------------------
 * The Vector partitions its element space into `nblocks` independent streams,
 * one page table per CUDA block (GetBlock(view, blockIdx.x)). We therefore map
 * a *contiguous stripe of output rows* to each block so every element a block
 * touches lands in that block's own cache:
 *
 *     rows_per_block = n_rows / nblocks           (n_rows % nblocks == 0)
 *     block b owns rows [b*rows_per_block, (b+1)*rows_per_block)
 *     row r occupies the contiguous element range [r*n_cols, (r+1)*n_cols)
 *
 * so block b streams the single contiguous element range
 *     [ b*rows_per_block*n_cols , (b+1)*rows_per_block*n_cols )
 * exactly matching the Vector's "each block streams its own contiguous
 * region" assumption. Within that stripe we issue one read_range per row.
 *
 * The activation `x` (length n_cols) is tiny and reused across every row, so
 * it stays resident in device global memory — mirroring llama.cpp, where the
 * activation is resident and only the weights stream.
 *
 * T is the page element type. Here T = float for clarity (an F32/F16 weight
 * matrix). For a quantized weight matrix set T = block_q4_0 / block_q4_K etc.
 * and unpack inside `consume` exactly as vec_dot_q4_0_q8_1 does — the
 * streaming transaction is identical; only the per-element math changes.
 */
struct WeightMatVecParams {
  const float *x;              /**< Resident activation, length n_cols. */
  float *y;                    /**< Output vector, length n_rows (device). */
  clio::run::u64 n_cols;       /**< Reduction dim K (elements per weight row). */
  clio::run::u64 rows_per_block; /**< Output rows handled by each CUDA block. */
};

/**
 * The transaction kernel. Launch as <<<nblocks, 32>>> (one warp per block, to
 * match the Vector's warp-cooperative read_range contract and the gpu2cpu
 * threadIdx.x==0 Send producer contract).
 *
 * Per row the warp:
 *   1. issues read_range over the row's K weights — a streaming transaction
 *      that faults/prefetches the row's pages through the two-tier cache;
 *   2. each lane accumulates a partial dot product w[i] * x[i] into a
 *      register (lanes stride by 32, coalesced);
 *   3. warp-reduces the partial sums and lane 0 writes y[row].
 */
inline __global__ void WeightMatVecTxnKernel(
    ::clio::run::IpcManagerGpuInfo info,
    DeviceView<float> weights,
    WeightMatVecParams p) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  ::cte::gpu::dev::vector<float> W(weights, g_ipc_manager_ptr);

  const clio::run::u32 lane = threadIdx.x & 31;
  const clio::run::u64 row_lo =
      static_cast<clio::run::u64>(blockIdx.x) * p.rows_per_block;
  const clio::run::u64 row_hi = row_lo + p.rows_per_block;

  for (clio::run::u64 row = row_lo; row < row_hi; ++row) {
    const clio::run::u64 lo = row * p.n_cols;
    const clio::run::u64 hi = lo + p.n_cols;

    // Per-lane partial dot product. The consume lambda runs on every lane
    // with its (global index i, weight value w); x is indexed by the
    // in-row offset (i - lo).
    float acc = 0.0f;
    const float *x = p.x;
    W.read_range(lo, hi, [&acc, x, lo](clio::run::u64 i, float w) {
      acc += w * x[i - lo];
    });

    // Warp-reduce the K partial sums to a single dot product.
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
      p.y[row] = acc;
    }
    __syncwarp();
  }
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS
/**
 * Host launcher. `W` is a Vector<float> whose element space is the row-major
 * weight matrix [n_rows, n_cols] with n_rows == W.nblocks * rows_per_block.
 * `dx` (length n_cols) and `dy` (length n_rows) are device pointers. The
 * weight pages are streamed out of CTE storage by the transaction; nothing
 * about the matrix needs to be VRAM-resident up front.
 *
 * Caller is responsible for having populated the weight blobs (e.g. via
 * write_range during load) and for choosing rows_per_block so that
 * nblocks * rows_per_block == n_rows.
 */
inline void LaunchWeightMatVec(Vector<float> &W, clio::run::u32 nblocks,
                               clio::run::u32 gpu_id, const float *dx, float *dy,
                               clio::run::u64 n_cols,
                               clio::run::u64 rows_per_block,
                               void *cuda_stream) {
  auto *ipc = CLIO_CPU_IPC;
  ::clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  WeightMatVecParams p{dx, dy, n_cols, rows_per_block};
  WeightMatVecTxnKernel<<<nblocks, 32, 0,
                          static_cast<cudaStream_t>(cuda_stream)>>>(
      gpu_info, W.Device(), p);
}
#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CTP_IS_GPU_COMPILER

#endif  // CLIO_CTE_GPU_VECTOR_MATVEC_H_

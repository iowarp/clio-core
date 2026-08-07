/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_GEMV_H_
#define CLIO_CTE_GPU_VECTOR_GEMV_H_

#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/types.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_cte/gpu_vector/gpu_vector_kernels.h>
#include <clio_cte/gpu_vector/gpu_vector_view.h>

#if CTP_IS_GPU_COMPILER

#include <cuda_fp16.h>

namespace clio::cte::gpu_vector {

/**
 * ============================================================================
 * F16 weight-application GEMV:  y = W * x   (decode / batch-1 linear layer)
 * ============================================================================
 *
 * The real transaction-driven weight kernel (successor to the F32 proof of
 * concept in gpu_vector_matvec.h). The weight matrix W[n_rows, n_cols] is
 * stored half-precision in a Vector<__half> (paged HBM -> pinned DRAM -> CTE
 * blob storage). Each row is streamed through `dev::vector::read_range` — the
 * transaction — so weight residency is handled entirely by the vector's
 * GPU-resident cache and its on-GPU cache-manager (rescore/prefetch), never
 * by a CPU-mediated UVM fault. Cache hits stay on the GPU; only genuine cold
 * misses to CTE storage involve the host.
 *
 * Math:  y[r] = sum_c  float(W[r, c]) * x[c]
 *   - W is __half in the vector (streamed).
 *   - x is a resident F32 activation of length n_cols (reused across rows,
 *     tiny relative to W — mirrors llama.cpp keeping activations resident).
 *   - y is F32 output of length n_rows.
 *
 * Layout / block mapping (identical to gpu_vector_matvec.h): the vector gives
 * each CUDA block its own page table, so a contiguous stripe of output rows
 * maps to each block —
 *     rows_per_block = n_rows / nblocks           (n_rows % nblocks == 0)
 *     block b owns rows [b*rpb, (b+1)*rpb), i.e. contiguous element range
 *         [ b*rpb*n_cols , (b+1)*rpb*n_cols )
 * so every element block b touches lands in block b's own cache.
 *
 * Quantized extension point: to support Q4_K/Q6_K, set the vector element type
 * to the block struct and replace the `__half2float(w)` in the consume lambda
 * with a block unpack + partial dot (as in vec_dot_q4_K_q8_1). The streaming
 * transaction is unchanged; only the per-element math differs.
 */
struct GemvParams {
  const float *x;                /**< Resident activation, length n_cols. */
  float *y;                      /**< Output, length n_rows (device). */
  clio::run::u64 n_cols;         /**< Reduction dim K (weights per row). */
  clio::run::u64 rows_per_block; /**< Output rows handled by each CUDA block. */
};

/**
 * Launch as <<<nblocks, 32>>> (one warp per block: matches read_range's
 * warp-cooperative contract and the gpu2cpu threadIdx.x==0 Send contract).
 */
inline __global__ void WeightGemvF16TxnKernel(
    ::clio::run::IpcManagerGpuInfo info,
    DeviceView<__half> weights,
    GemvParams p) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  ::cte::gpu::dev::vector<__half> W(weights, g_ipc_manager_ptr);

  const clio::run::u32 lane = threadIdx.x & 31;
  const clio::run::u64 row_lo =
      static_cast<clio::run::u64>(blockIdx.x) * p.rows_per_block;
  const clio::run::u64 row_hi = row_lo + p.rows_per_block;

  for (clio::run::u64 row = row_lo; row < row_hi; ++row) {
    const clio::run::u64 lo = row * p.n_cols;
    const clio::run::u64 hi = lo + p.n_cols;

    // Per-lane partial dot product. read_range streams the row's half weights
    // through the vector's cache; each lane dequantizes and multiplies by the
    // matching activation element (indexed by in-row offset i - lo).
    float acc = 0.0f;
    const float *x = p.x;
    W.read_range(lo, hi, [&acc, x, lo](clio::run::u64 i, __half w) {
      acc += __half2float(w) * x[i - lo];
    });

    // Warp-reduce the K partial sums into the row's dot product.
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
 * Host launcher. `W` is a Vector<__half> whose element space is the row-major
 * weight matrix [n_rows, n_cols] with n_rows == nblocks * rows_per_block.
 * `dx` (length n_cols) and `dy` (length n_rows) are device pointers.
 */
inline void LaunchWeightGemvF16(Vector<__half> &W, clio::run::u32 nblocks,
                                clio::run::u32 gpu_id, const float *dx,
                                float *dy, clio::run::u64 n_cols,
                                clio::run::u64 rows_per_block,
                                void *cuda_stream) {
  auto *ipc = CLIO_CPU_IPC;
  ::clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  GemvParams p{dx, dy, n_cols, rows_per_block};
  WeightGemvF16TxnKernel<<<nblocks, 32, 0,
                           static_cast<cudaStream_t>(cuda_stream)>>>(
      gpu_info, W.Device(), p);
}
#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CTP_IS_GPU_COMPILER

#endif  // CLIO_CTE_GPU_VECTOR_GEMV_H_

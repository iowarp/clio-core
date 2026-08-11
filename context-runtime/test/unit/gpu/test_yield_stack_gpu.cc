/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Lane-divided continuation stack.
 *
 * What has to be true, and none of it is implied by "the kernel finished":
 *
 *   1. locals are PER LANE -- lane 5's `acc` is lane 5's across every suspend,
 *      not the block's and not a neighbour's
 *   2. a nested __device__ function can yield, with the yield travelling out
 *      through its caller and back in on resume
 *   3. that function reaches the stack through shared memory, taking NO
 *      parameter for it
 *   4. frames pop: when the inner call finishes, its space is reusable and the
 *      caller carries on with its own locals intact
 *
 * The yield itself stays block-collective, so control flow AT a yield must be
 * uniform across the block. Per-lane VALUES differ freely; per-lane loop
 * counts around a yield would diverge __syncthreads and are not allowed.
 */

#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <cstdio>
#include <vector>

#include "simple_test.h"

namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u32 kBlocks = 4;
constexpr u32 kThreads = 32;
constexpr u32 kLaneBytes = 512;
}  // namespace

/**
 * Nested yieldable function. Takes no stack, no view, no frame -- it finds all
 * of that through shared memory, which is the point of publishing it there.
 */
__device__ void InnerAccumulate(u32 iters, u64 *acc_out) {
  CLIO_YFRAME();
  // Per-lane locals. `k` is the loop counter that must survive; `acc` proves
  // the value is this lane's and no one else's.
  CLIO_YLOCAL_INIT(u32, k, 0);
  CLIO_YLOCAL_INIT(u64, acc, 0);

  CLIO_YBEGIN();
  // `iters` is uniform across the block, so every lane reaches the yield.
  for (; k < iters; ++k) {
    acc += threadIdx.x + k;
    CLIO_YIELD_STACK();
  }
  acc_out[threadIdx.x] = acc;
  CLIO_YEND();
}

__global__ void NestKernel(gy::YieldableView<> v, gy::YieldStackView stack,
                           const u32 *iters_per_block, u64 *acc_out,
                           u32 *lane_tag_out, u32 *entries) {
  CLIO_YKERNEL_ENTER(v, stack);

  if (threadIdx.x == 0) {
    atomicAdd(&entries[v.Block()], 1u);
  }

  CLIO_YFRAME();
  // Distinct per lane, written once, read after the nested call has suspended
  // and resumed many times: catches a frame that got shared or clobbered.
  CLIO_YLOCAL_INIT(u32, lane_tag, 0xA5000000u | (v.Block() << 8) | threadIdx.x);

  CLIO_YBEGIN();

  CLIO_YCALL(InnerAccumulate(iters_per_block[v.Block()],
                             acc_out + v.Block() * blockDim.x));

  // Reached only after the inner call ran to completion.
  lane_tag_out[v.Block() * blockDim.x + threadIdx.x] = lane_tag;

  CLIO_YEND();
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("YieldStack - per-lane locals survive a nested yield",
          "[gpu][yieldable][stack]") {
  gy::Yieldable<> y(kBlocks, kThreads);
  gy::YieldStack stack(kBlocks, kThreads, kLaneBytes);

  // Block b runs b+1 inner iterations: uniform within a block, different
  // across blocks, so blocks finish on different rounds.
  std::vector<u32> h_iters(kBlocks);
  for (u32 b = 0; b < kBlocks; ++b) {
    h_iters[b] = b + 1;
  }
  u32 *d_iters = nullptr;
  REQUIRE(cudaMalloc(&d_iters, kBlocks * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_iters, h_iters.data(), kBlocks * sizeof(u32),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  u64 *d_acc = nullptr;
  u32 *d_tag = nullptr;
  u32 *d_entries = nullptr;
  const size_t nlane = static_cast<size_t>(kBlocks) * kThreads;
  REQUIRE(cudaMalloc(&d_acc, nlane * sizeof(u64)) == cudaSuccess);
  REQUIRE(cudaMemset(d_acc, 0, nlane * sizeof(u64)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_tag, nlane * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemset(d_tag, 0, nlane * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_entries, kBlocks * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemset(d_entries, 0, kBlocks * sizeof(u32)) == cudaSuccess);

  std::vector<u32> pending_trace;
  const u32 rounds = y.RunToCompletion(
      [&](dim3 grid, dim3 block, gy::YieldableView<> view) {
        pending_trace.push_back(view.num_pending_);
        NestKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
            view, stack.View(), d_iters, d_acc, d_tag, d_entries);
      },
      [&]() {}, /*max_rounds=*/64);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<u64> acc(nlane, 0);
  std::vector<u32> tag(nlane, 0);
  std::vector<u32> entries(kBlocks, 0);
  REQUIRE(cudaMemcpy(acc.data(), d_acc, nlane * sizeof(u64),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(tag.data(), d_tag, nlane * sizeof(u32),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(entries.data(), d_entries, kBlocks * sizeof(u32),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  std::fprintf(stderr, "[yield-stack] rounds=%u pending:", rounds);
  for (u32 p : pending_trace) std::fprintf(stderr, " %u", p);
  std::fprintf(stderr, "\n");

  REQUIRE(y.NumPending() == 0);

  for (u32 b = 0; b < kBlocks; ++b) {
    const u32 iters = h_iters[b];
    // Entered once per inner yield, plus the entry that finishes the block.
    REQUIRE(entries[b] == iters + 1);
    for (u32 t = 0; t < kThreads; ++t) {
      const size_t i = static_cast<size_t>(b) * kThreads + t;
      // (1) per-lane accumulator: sum over k of (lane + k)
      u64 want = 0;
      for (u32 k = 0; k < iters; ++k) {
        want += t + k;
      }
      REQUIRE(acc[i] == want);
      // (4) the CALLER's local was untouched by the nested call's frame
      REQUIRE(tag[i] == (0xA5000000u | (b << 8) | t));
    }
  }
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

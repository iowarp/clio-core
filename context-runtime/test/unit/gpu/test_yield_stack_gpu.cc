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

/**
 * Models a page-fault wait: `need` is how many service rounds this LANE still
 * requires. Only some lanes (and some whole warps) ever need any.
 */
__global__ void FaultKernel(gy::YieldableView<> v, gy::YieldStackView stack,
                            const u32 *need, u32 *out, u32 *entries) {
  CLIO_YKERNEL_ENTER(v, stack);
  if (threadIdx.x == 0) {
    atomicAdd(&entries[v.Block()], 1u);
  }

  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u32, tag, 0x5A000000u | (v.Block() << 8) | threadIdx.x);

  CLIO_YBEGIN();

  // A per-thread condition turned into a per-block decision. A lane with
  // nothing to wait for votes 0 and rides along with the ones that do.
  CLIO_YIELD_IF(need[v.Block() * blockDim.x + threadIdx.x] > 0);

  // Reached only when NO lane in the block is still waiting.
  out[v.Block() * blockDim.x + threadIdx.x] = tag;

  CLIO_YEND();
}

/** Stands in for the host servicing faults between rounds. */
__global__ void ServiceFaultsKernel(u32 *need, u32 n) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && need[i] > 0) {
    need[i] -= 1;
  }
}

/**
 * Instrumented to answer one question exactly: on resume, does execution land
 * at the yield, or is anything before it run again?
 *
 * Each counter marks a distinct program point. Straight-line code that already
 * ran must show 1; the loop body must show exactly one execution per
 * iteration; anything deliberately re-evaluated must show one per entry.
 */
__device__ void ProbeInner(u32 iters, u32 *c) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u32, k, 0);
  CLIO_YBEGIN();
  if (threadIdx.x == 0) atomicAdd(&c[0], 1u);  // callee prologue
  for (; k < iters; ++k) {
    if (threadIdx.x == 0) atomicAdd(&c[1], 1u);  // loop body, before yield
    CLIO_YIELD_STACK();
    if (threadIdx.x == 0) atomicAdd(&c[2], 1u);  // loop body, after yield
  }
  CLIO_YEND();
}

/** Sits in a CLIO_YCALL argument, to see whether arguments are re-evaluated. */
__device__ u32 ProbeArg(u32 *c, u32 v) {
  if (threadIdx.x == 0) atomicAdd(&c[3], 1u);
  return v;
}

__global__ void ProbeKernel(gy::YieldableView<> v, gy::YieldStackView stack,
                            u32 iters, u32 *c) {
  CLIO_YKERNEL_ENTER(v, stack);
  if (threadIdx.x == 0) atomicAdd(&c[4], 1u);  // BEFORE the switch: every entry
  CLIO_YFRAME();
  CLIO_YBEGIN();
  if (threadIdx.x == 0) atomicAdd(&c[5], 1u);  // inside switch, before the call
  CLIO_YCALL(ProbeInner(ProbeArg(c, iters), c));
  if (threadIdx.x == 0) atomicAdd(&c[6], 1u);  // after the call returns
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

TEST_CASE("YieldStack - YIELD_IF suspends the block if ANY lane must wait",
          "[gpu][yieldable][stack]") {
  constexpr u32 kB = 4;
  constexpr u32 kT = 64;  // two warps, so a whole warp can fault alone
  gy::Yieldable<> y(kB, kT);
  gy::YieldStack stack(kB, kT, kLaneBytes);

  const size_t n = static_cast<size_t>(kB) * kT;
  std::vector<u32> h_need(n, 0);
  // block 0: nobody waits           -> falls straight through, 1 round
  // block 1: only the SECOND warp   -> 2 rounds
  // block 2: a single lane, 3 waits -> 4 rounds
  // block 3: every odd lane, 2 waits-> 3 rounds
  for (u32 t = 32; t < kT; ++t) h_need[1 * kT + t] = 1;
  h_need[2 * kT + 5] = 3;
  for (u32 t = 1; t < kT; t += 2) h_need[3 * kT + t] = 2;

  u32 *d_need = nullptr;
  u32 *d_out = nullptr;
  u32 *d_entries = nullptr;
  REQUIRE(cudaMalloc(&d_need, n * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_need, h_need.data(), n * sizeof(u32),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_out, n * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemset(d_out, 0, n * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_entries, kB * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemset(d_entries, 0, kB * sizeof(u32)) == cudaSuccess);

  std::vector<u32> pending_trace;
  u32 services = 0;
  y.RunToCompletion(
      [&](dim3 grid, dim3 block, gy::YieldableView<> view) {
        pending_trace.push_back(view.num_pending_);
        FaultKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(
            view, stack.View(), d_need, d_out, d_entries);
      },
      [&]() {
        // The service itself is a KERNEL, which is only possible because no
        // kernel is resident between rounds.
        ++services;
        ServiceFaultsKernel<<<(n + 127) / 128, 128>>>(d_need,
                                                      static_cast<u32>(n));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      },
      /*max_rounds=*/32);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<u32> out(n, 0);
  std::vector<u32> entries(kB, 0);
  REQUIRE(cudaMemcpy(out.data(), d_out, n * sizeof(u32),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(entries.data(), d_entries, kB * sizeof(u32),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  std::fprintf(stderr, "[yield-if] services=%u entries=%u %u %u %u pending:",
               services, entries[0], entries[1], entries[2], entries[3]);
  for (u32 p : pending_trace) std::fprintf(stderr, " %u", p);
  std::fprintf(stderr, "\n");

  REQUIRE(y.NumPending() == 0);

  // A block whose lanes all vote 0 never suspends: one entry, no yield.
  REQUIRE(entries[0] == 1);
  // Otherwise a block is entered once per wait it could not satisfy, plus the
  // entry that gets through -- driven by its SLOWEST lane, not its average.
  REQUIRE(entries[1] == 2);
  REQUIRE(entries[2] == 4);
  REQUIRE(entries[3] == 3);

  // Blocks retire as their slowest lane clears.
  REQUIRE(pending_trace.size() == 4);
  REQUIRE(pending_trace[0] == 4);
  REQUIRE(pending_trace[1] == 3);
  REQUIRE(pending_trace[2] == 2);
  REQUIRE(pending_trace[3] == 1);

  // Every lane got through, including the ones that never waited, and the
  // caller local survived however many suspends its block took.
  for (u32 b = 0; b < kB; ++b) {
    for (u32 t = 0; t < kT; ++t) {
      REQUIRE(out[b * kT + t] == (0x5A000000u | (b << 8) | t));
    }
  }

  cudaFree(d_need);
  cudaFree(d_out);
  cudaFree(d_entries);
}

TEST_CASE("YieldStack - resume lands AT the yield, it does not re-run work",
          "[gpu][yieldable][stack]") {
  constexpr u32 kIters = 3;
  gy::Yieldable<> y(1, 32);
  gy::YieldStack stack(1, 32, kLaneBytes);

  u32 *d_c = nullptr;
  REQUIRE(cudaMalloc(&d_c, 8 * sizeof(u32)) == cudaSuccess);
  REQUIRE(cudaMemset(d_c, 0, 8 * sizeof(u32)) == cudaSuccess);

  y.RunToCompletion(
      [&](dim3 grid, dim3 block, gy::YieldableView<> view) {
        ProbeKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(view, stack.View(),
                                                            kIters, d_c);
      },
      [&]() {}, /*max_rounds=*/32);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  u32 c[8] = {0};
  REQUIRE(cudaMemcpy(c, d_c, 8 * sizeof(u32), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  std::fprintf(stderr,
               "[resume] callee_prologue=%u loop_before=%u loop_after=%u "
               "ycall_arg=%u kernel_prologue=%u switch_head=%u after_call=%u\n",
               c[0], c[1], c[2], c[3], c[4], c[5], c[6]);

  // 3 yields, so 4 entries into the kernel.
  const u32 kEntries = kIters + 1;

  // NOT re-run: straight-line code the switch jumps over.
  REQUIRE(c[5] == 1);  // between YBEGIN and the call
  REQUIRE(c[0] == 1);  // the callee's own prologue
  REQUIRE(c[6] == 1);  // after the call, reached once at the end

  // Run exactly once per iteration -- the loop continues, it does not restart.
  REQUIRE(c[1] == kIters);
  REQUIRE(c[2] == kIters);

  // DELIBERATELY re-run once per entry.
  REQUIRE(c[4] == kEntries);  // everything before CLIO_YBEGIN
  REQUIRE(c[3] == kEntries);  // CLIO_YCALL argument expressions

  cudaFree(d_c);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Yieldable (cooperative) CUDA kernels.
 *
 * The properties worth asserting are not "it produced the right number" -- a
 * kernel that ignored yielding entirely and ran straight through would also do
 * that. They are:
 *
 *   1. state survives a yield (the loop counter continues, it does not restart)
 *   2. a block's LOGICAL id is stable across resumes, even though blockIdx.x is
 *      not, because the driver relaunches a compacted set of pending blocks
 *   3. blocks finish independently -- a block that needs fewer yields stops
 *      being relaunched while its neighbours continue
 *   4. every thread of the block participates at every resume
 *   5. THE POINT: between rounds the device is free, so the host can launch
 *      kernels. That is exactly what a spinning kernel makes impossible.
 */

#include <clio_runtime/gpu/yieldable.h>

#include <cstdio>
#include <vector>

#include "simple_test.h"

namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u32 kBlocks = 8;
constexpr u32 kThreads = 32;
}  // namespace

/** Per-block state. Anything live across a yield MUST live here. */
struct CountState {
  u64 iters_;       // how many times this block should yield (input)
  u64 i_;           // loop counter -- the thing that must survive
  u64 entries_;     // kernel entries observed by this block
  u32 block_seen_;  // logical id the block observed, to catch id drift
  u32 pad_;
};

/**
 * Yields once per loop iteration. `i_` lives in state precisely because a
 * local would not survive: the resume jumps back into the loop body.
 */
__global__ void CountKernel(gy::YieldableView<CountState> v,
                            unsigned long long *work) {
  // BEFORE the yieldable body: this runs on EVERY entry, because the switch
  // inside CLIO_YIELDABLE_BEGIN jumps past anything after it.
  if (threadIdx.x == 0) {
    v.State()->entries_ += 1;
    v.State()->block_seen_ = v.Block();
  }

  CLIO_YIELDABLE_BEGIN(v);

  for (v.State()->i_ = 0; v.State()->i_ < v.State()->iters_;
       ++v.State()->i_) {
    // Every thread contributes, every iteration: proves the whole block is
    // really resumed, not just lane 0.
    atomicAdd(&work[v.Block()], 1ull);
    CLIO_YIELD(v);
  }

  CLIO_YIELDABLE_END();
}

/** Trivial kernel the host launches BETWEEN rounds. */
__global__ void ServiceKernel(unsigned long long *flag) {
  if (threadIdx.x == 0) {
    atomicAdd(flag, 1ull);
  }
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("Yieldable - blocks suspend, resume, and finish independently",
          "[gpu][yieldable]") {
  gy::Yieldable<CountState> y(kBlocks, kThreads);

  // Block b yields b+1 times, so blocks finish at staggered rounds.
  std::vector<CountState> init(kBlocks);
  for (u32 b = 0; b < kBlocks; ++b) {
    init[b].iters_ = b + 1;
    init[b].i_ = 0;
    init[b].entries_ = 0;
    init[b].block_seen_ = 0xFFFFFFFFu;
    init[b].pad_ = 0;
  }
  REQUIRE(cudaMemcpy(y.DeviceState(), init.data(),
                     kBlocks * sizeof(CountState),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  unsigned long long *d_work = nullptr;
  unsigned long long *d_service = nullptr;
  REQUIRE(cudaMalloc(&d_work, kBlocks * sizeof(unsigned long long)) ==
          cudaSuccess);
  REQUIRE(cudaMemset(d_work, 0, kBlocks * sizeof(unsigned long long)) ==
          cudaSuccess);
  REQUIRE(cudaMalloc(&d_service, sizeof(unsigned long long)) == cudaSuccess);
  REQUIRE(cudaMemset(d_service, 0, sizeof(unsigned long long)) == cudaSuccess);

  // Observed pending counts, to prove blocks drop out as they finish.
  std::vector<u32> pending_trace;
  u32 service_calls = 0;

  const u32 rounds = y.RunToCompletion(
      [&](dim3 grid, dim3 block, gy::YieldableView<CountState> view) {
        pending_trace.push_back(view.num_pending_);
        CountKernel<<<grid, block>>>(view, d_work);
      },
      [&]() {
        // Between rounds NO kernel is resident, so this one can run. Under a
        // spinning-wait design it could not: an indefinitely resident kernel
        // blocks every later launch in the context.
        ++service_calls;
        ServiceKernel<<<1, 32>>>(d_service);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      },
      /*max_rounds=*/64);

  std::vector<CountState> out(kBlocks);
  REQUIRE(cudaMemcpy(out.data(), y.DeviceState(),
                     kBlocks * sizeof(CountState),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<unsigned long long> work(kBlocks, 0);
  REQUIRE(cudaMemcpy(work.data(), d_work,
                     kBlocks * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  unsigned long long service_ran = 0;
  REQUIRE(cudaMemcpy(&service_ran, d_service, sizeof(service_ran),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  std::fprintf(stderr, "[yieldable] rounds=%u service_calls=%u service_ran=%llu\n",
               rounds, service_calls, service_ran);
  std::fprintf(stderr, "[yieldable] pending per round:");
  for (u32 p : pending_trace) std::fprintf(stderr, " %u", p);
  std::fprintf(stderr, "\n");

  REQUIRE(y.NumPending() == 0);

  for (u32 b = 0; b < kBlocks; ++b) {
    const u64 want = b + 1;
    // (1) the counter continued across yields rather than restarting
    REQUIRE(out[b].i_ == want);
    // (2) the logical id never drifted, though blockIdx.x did
    REQUIRE(out[b].block_seen_ == b);
    // (4) every thread ran every iteration
    REQUIRE(work[b] == want * kThreads);
    // the block was entered once per yield, plus the final entry that ends it
    REQUIRE(out[b].entries_ == want + 1);
  }

  // (3) the pending set shrinks one block at a time: 8 8 7 6 5 4 3 2 1.
  //
  // Rounds 1 and 2 BOTH run 8 blocks, and that is correct rather than a
  // missed drop-out: block 0 yields once, so after round 1 every block is
  // suspended. Block 0 finishes on round 2, which is the first round anyone
  // can drop out. Thereafter exactly one block finishes per round because
  // block b yields b+1 times.
  REQUIRE(pending_trace.size() == kBlocks + 1);
  REQUIRE(pending_trace[0] == kBlocks);
  REQUIRE(pending_trace[1] == kBlocks);
  REQUIRE(pending_trace.back() == 1);
  for (size_t i = 1; i < pending_trace.size(); ++i) {
    REQUIRE(pending_trace[i] <= pending_trace[i - 1]);
  }
  for (size_t i = 2; i < pending_trace.size(); ++i) {
    REQUIRE(pending_trace[i] == pending_trace[i - 1] - 1);
  }

  // (5) the host really did get the device between rounds
  REQUIRE(service_calls > 0);
  REQUIRE(service_ran == service_calls);

  cudaFree(d_work);
  cudaFree(d_service);
}

TEST_CASE("Yieldable - a kernel that never yields completes in one round",
          "[gpu][yieldable]") {
  gy::Yieldable<CountState> y(kBlocks, kThreads);
  std::vector<CountState> init(kBlocks);
  for (u32 b = 0; b < kBlocks; ++b) {
    init[b].iters_ = 0;  // loop body never runs, so no yield is reached
    init[b].i_ = 0;
    init[b].entries_ = 0;
    init[b].block_seen_ = 0xFFFFFFFFu;
    init[b].pad_ = 0;
  }
  REQUIRE(cudaMemcpy(y.DeviceState(), init.data(),
                     kBlocks * sizeof(CountState),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  unsigned long long *d_work = nullptr;
  REQUIRE(cudaMalloc(&d_work, kBlocks * sizeof(unsigned long long)) ==
          cudaSuccess);
  REQUIRE(cudaMemset(d_work, 0, kBlocks * sizeof(unsigned long long)) ==
          cudaSuccess);

  u32 launches = 0;
  y.RunToCompletion(
      [&](dim3 grid, dim3 block, gy::YieldableView<CountState> view) {
        ++launches;
        CountKernel<<<grid, block>>>(view, d_work);
      },
      [&]() {}, /*max_rounds=*/8);

  // One launch, everything done: yielding costs nothing when unused.
  REQUIRE(launches == 1);
  REQUIRE(y.NumPending() == 0);
  cudaFree(d_work);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

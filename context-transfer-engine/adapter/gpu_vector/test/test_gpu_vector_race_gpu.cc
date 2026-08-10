/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Race stress for the paging protocol: many blocks concurrently probe
 * (lock-free TryHoldRawConst), fault (FetchPagesBatched), async-prefetch
 * (FetchPagesBatchedAsync), and evict the SAME small page space, verifying
 * every read byte against its seeded pattern.
 *
 * Exists because the cross-block claim-visibility race (fetching stored
 * with only a BLOCK-scoped fence before page_num) served the EVICTED page's
 * bytes to lock-free probes in other blocks — which surfaced end-to-end as
 * intermittent garbage llama output after ~3 minutes of model load, and
 * took hours to attribute. This reproduces that class of bug in seconds:
 * a paging bug shows up as wrong bytes far more often than as a crash.
 *
 * Page->table affinity mirrors the MoE kernel: block_override_ = pn % nblocks,
 * so every block touching page pn uses the SAME table — probe/claim/evict
 * collide across blocks by construction.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;

namespace {
constexpr clio::run::u64 kPageBytes = 4096;
constexpr clio::run::u64 kPageElems = kPageBytes / sizeof(clio::run::u32);

/** Seed value for element i. */
CTP_INLINE_CROSS_FUN clio::run::u32 Seed(clio::run::u64 i) {
  return static_cast<clio::run::u32>(i * 2654435761u + 17u);
}
}  // namespace

__global__ void RaceSeedKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<clio::run::u32> v,
                               clio::run::u64 per) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  const clio::run::u64 base = static_cast<clio::run::u64>(blockIdx.x) * per;
  for (clio::run::u64 i = 0; i < per;) {
    const clio::run::u64 run_i = v.HoldPage(base + i, per - i);
    for (clio::run::u64 k_i = 0; k_i < run_i; ++k_i, ++i) {
      v[base + i] = Seed(base + i);
    }
  }
  v.BeginFlush(base, per);
  v.WaitFlush(base, per);
}

/**
 * err[0]  mismatch count
 * err[1]  first bad page
 * err[2]  got (element 0 of the bad read)
 * err[3]  want
 */
__global__ void RaceStressKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<clio::run::u32> v,
                                 clio::run::u64 total_pages, int iters,
                                 unsigned long long *err) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;

  __shared__ gv::VecHeader s_hdr;
  {
    const unsigned long long *hsrc = (const unsigned long long *) v.h_;
    unsigned long long *hdst = (unsigned long long *) &s_hdr;
    constexpr unsigned kWords =
        (unsigned) (sizeof(gv::VecHeader) / sizeof(unsigned long long));
    for (unsigned w = 0; w < kWords; ++w) hdst[w] = hsrc[w];
    for (unsigned b = kWords * 8u; b < sizeof(gv::VecHeader); ++b)
      ((char *) &s_hdr)[b] = ((const char *) v.h_)[b];
  }
  v.h_ = &s_hdr;

  // Per-block LCG stream so blocks collide on pages but not in lockstep.
  unsigned int rng = 1234567u + 97u * blockIdx.x;
  for (int it = 0; it < iters; ++it) {
    rng = rng * 1664525u + 1013904223u;
    const clio::run::u64 pn = rng % total_pages;
    v.block_override_ = (clio::run::u32) (pn % s_hdr.nblocks_);

    // Async prefetch of an unrelated page range every few iterations —
    // the long in-flight window is what widens claim races enough to fire.
    if ((it & 3) == 0) {
      rng = rng * 1664525u + 1013904223u;
      const clio::run::u64 pf = rng % total_pages;
      v.block_override_ = (clio::run::u32) (pf % s_hdr.nblocks_);
      v.FetchPagesBatchedAsync(pf, 2);
      v.block_override_ = (clio::run::u32) (pn % s_hdr.nblocks_);
    }

    clio::run::u64 run = 0;
    const clio::run::u32 *q =
        v.TryHoldRawConst(pn * kPageElems, kPageElems, &run);
    if (q == nullptr) {
      v.FetchPagesBatched(pn, 1);
      q = v.HoldRawConst(pn * kPageElems, kPageElems, &run);
    }
    if (q == nullptr) continue;  // window pinned; not a data error
    // Verify a sample of the page (first, middle, last of the run).
    const clio::run::u64 idx[3] = {0, run / 2, run - 1};
    for (int k = 0; k < 3; ++k) {
      const clio::run::u64 el = pn * kPageElems + idx[k];
      const clio::run::u32 want = Seed(el);
      const clio::run::u32 got = q[idx[k]];
      if (got != want) {
        if (atomicAdd(err, 1ull) == 0) {
          err[1] = pn;
          err[2] = got;
          err[3] = want;
        }
      }
    }
  }
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: concurrent probe/fault/prefetch/evict serves exact bytes",
          "[gpu_vector][race]") {
  {
    std::ofstream cfg("gpu_vector_race.yaml");
    REQUIRE(cfg.is_open());
    cfg << "networking:\n  port: 9433\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "gpu:\n  queue_depth: 4096\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_race.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // 24 seed blocks x 8 pages = 192 pages of truth; the stress phase runs 24
  // blocks with only 8 cache slots per table — heavy oversubscription, so
  // claim/evict runs constantly against the lock-free probes.
  constexpr unsigned kBlocks = 24;
  constexpr clio::run::u64 kPagesPerSlice = 8;
  constexpr clio::run::u64 kTotalPages = kBlocks * kPagesPerSlice;
  constexpr int kIters = 512;
  const clio::run::u64 n = kTotalPages * kPageElems;

  gv::Vector<clio::run::u32> vec("gv_race", {0}, kPageBytes, kBlocks,
                                 /*pages_per_block=*/8, n);

  RaceSeedKernel<<<kBlocks, 32>>>(gpu_info, vec.GetDevice(0),
                                  kPagesPerSlice * kPageElems);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  unsigned long long *err = nullptr;
  REQUIRE(cudaMalloc(&err, 4 * sizeof(unsigned long long)) == cudaSuccess);
  REQUIRE(cudaMemset(err, 0, 4 * sizeof(unsigned long long)) == cudaSuccess);

  RaceStressKernel<<<kBlocks, 32>>>(gpu_info, vec.GetDevice(0), kTotalPages,
                                    kIters, err);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  unsigned long long h[4] = {0};
  REQUIRE(cudaMemcpy(h, err, sizeof(h), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  if (h[0] != 0) {
    fprintf(stderr,
            "race: %llu mismatches; first bad page=%llu got=0x%llx "
            "want=0x%llx\n",
            h[0], h[1], h[2], h[3]);
  }
  REQUIRE(h[0] == 0);
  cudaFree(err);
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

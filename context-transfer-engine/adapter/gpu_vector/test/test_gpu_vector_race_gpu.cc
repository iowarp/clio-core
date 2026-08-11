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

#include <cstdlib>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;

namespace {
// 512 KiB pages — the MoE workload's granularity. The 4 KiB first cut
// PASSED on known-broken code: its DMA lands in microseconds, so the
// claim->land window the race needs never opened. Window length is the
// amplification lever, not iteration count.
constexpr clio::run::u64 kPageBytes = 128ull << 10;
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
    // Flush per page: the slice is larger than the cache, and an evicted
    // dirty page is DROPPED, not written back (weights-path assumption) —
    // deferring the flush to the end seeds cross-page garbage.
    v.BeginFlush(base, i);
    v.WaitFlush(base, i);
  }
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
  // SM SATURATION: idle warps stay RESIDENT (camped on a shared flag) for
  // the kernel's whole life — the llama wedge ingredient a thread0-only
  // grid lacks. With SMs empty, UVM migrations of the managed gpu2cpu
  // queue schedule freely and the wedge never fires; saturated, a stalled
  // migration freezes the copy engines (the captured mechanism).
  __shared__ volatile int s_done;
  if (threadIdx.x == 0) s_done = 0;
  __syncthreads();
  if (threadIdx.x != 0) {
    while (!s_done) {
    }
    return;
  }

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
    {
      rng = rng * 1664525u + 1013904223u;
      const clio::run::u64 pf = rng % total_pages;
      v.block_override_ = (clio::run::u32) (pf % s_hdr.nblocks_);
      v.FetchPagesBatchedAsync(pf, 4);
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
    // LONG READ: hold the raw pointer across ~50 us of "compute" before
    // verifying — the MoE row loop's shape. Nothing pins a held page, so
    // another block's claim (sync fault or async prefetch) can evict and
    // recycle this slot mid-read; the verify below then sees another
    // page's bytes.
    {
      const long long t0 = clock64();
      while (clock64() - t0 < 100000) {
      }
    }
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
  s_done = 1;   // release the camped warps
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: concurrent probe/fault/prefetch/evict serves exact bytes",
          "[gpu_vector][race]") {
  {
    std::ofstream cfg("gpu_vector_race.yaml");
    REQUIRE(cfg.is_open());
    cfg << "networking:\n  port: 9433\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "gpu:\n  queue_depth: 4096\n\n"
        // The seeded image is ~1.2 GB; the default core's target is far
        // smaller (host puts failed ENOMEM at page 800 without this).
        // Compose the DEFAULT core pool (name+id the client binds to) with
        // a big tier, so both the host client and the vector use it.
        << "compose:\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: clio_cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        // CLIO_RACE_TIER=hbm turns this into the sub-minute kHbm wedge
        // repro (managed-queue migration ping-pong under resident faulting
        // kernels): same machinery as the 4-minute llama runs, verdict in
        // the test's own runtime (~30 s; a 60 s timeout = wedge).
        << "      - path: \"" << (std::getenv("CLIO_RACE_TIER")
                                       ? std::getenv("CLIO_RACE_TIER")
                                       : "ram")
        << "::gv_race_tier\"\n"
        << "        bdev_type: \"" << (std::getenv("CLIO_RACE_TIER")
                                            ? std::getenv("CLIO_RACE_TIER")
                                            : "ram")
        << "\"\n        capacity_limit: \"3GB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
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
  constexpr unsigned kBlocks = 48;
  // pages_per_block must exceed 128 so multi_per_block >= 3 and the ASYNC
  // batch slots exist at all (ceil(ppb/64) - 1 of them); 384 pages walked
  // through 192 slots keeps claim/evict hot.
  constexpr clio::run::u64 kPagesPerSlice = 480;
  constexpr clio::run::u64 kTotalPages = kBlocks * kPagesPerSlice;
  // Pressure high enough to model llama's sustained fault storms — the
  // managed-queue migration wedge needs continuous device pushes + CPU
  // drains under resident kernels, not a short burst. CLIO_RACE_ITERS
  // overrides for a quick byte-exactness verdict (the full 8192 runs many
  // minutes; correctness bugs historically reproduced within hundreds).
  int kIters = 8192;
  if (const char *it_env = std::getenv("CLIO_RACE_ITERS")) {
    kIters = std::atoi(it_env);
    if (kIters <= 0) kIters = 8192;
  }
  const clio::run::u64 n = kTotalPages * kPageElems;

  gv::Vector<clio::run::u32> vec("gv_race", {0}, kPageBytes, kBlocks,
                                 /*pages_per_block=*/192, n);

  // HOST-side seed: put every page's pattern straight through the CTE
  // client. The GPU flush path is deliberately NOT used — a walk larger
  // than the cache DROPS dirty evictions by design (weights are read-only
  // in the paged kernels), so a GPU-seeded oversubscribed image is missing
  // pages before the stress even starts. Host puts make the storage state
  // exact, so any stress mismatch indicts the DEVICE READ path alone.
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    ctp::ipc::FullPtr<char> shm = CLIO_CPU_IPC->AllocateBuffer(kPageBytes);
    REQUIRE(shm.ptr_ != nullptr);
    for (clio::run::u64 pg = 0; pg < kTotalPages; ++pg) {
    if ((pg & 0x7FF) == 0) std::fprintf(stderr, "[race-progress] page %llu/%llu\n", (unsigned long long) pg, (unsigned long long) kTotalPages);
      auto *w = reinterpret_cast<clio::run::u32 *>(shm.ptr_);
      for (clio::run::u64 i = 0; i < kPageElems; ++i) {
        w[i] = Seed(pg * kPageElems + i);
      }
      char name[32];
      gv::PageBlobName(pg, name);
      auto f = core.AsyncPutBlob(vec.TagId(), std::string(name), 0,
                                 kPageBytes, shm.Cast<void>().shm_);
      f.Wait();
      if (f->GetReturnCode() != 0) {
        std::fprintf(stderr, "seed put %s rc=%d\n", name,
                     f->GetReturnCode());
      }
      REQUIRE(f->GetReturnCode() == 0);
    }
  }

  // Split write-path from read-path corruption BEFORE the stress phase:
  // verify the seeded blobs from the HOST. A wrong byte here means the
  // batched flush already stored another page's data; a clean pass here
  // followed by a stress failure indicts the device read path.
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<clio::run::u32> buf(kPageElems, 0u);
    clio::run::u64 bad_pages = 0;
    for (clio::run::u64 pg = 0; pg < kTotalPages; pg += 7) {
      char name[32];
      gv::PageBlobName(pg, name);
      auto f = core.AsyncGetBlob(vec.TagId(), std::string(name), 0, kPageBytes,
                                 0, reinterpret_cast<char *>(buf.data()));
      f.Wait();
      if (f->GetReturnCode() != 0) {
        std::fprintf(stderr, "seed-verify: host read %s rc=%d\n", name,
                     f->GetReturnCode());
        ++bad_pages;
        continue;
      }
      for (clio::run::u64 i = 0; i < kPageElems; i += kPageElems / 4) {
        const clio::run::u64 el = pg * kPageElems + i;
        if (buf[i] != Seed(el)) {
          std::fprintf(stderr,
                       "seed-verify: page %llu el %llu got=0x%x want=0x%x\n",
                       (unsigned long long) pg, (unsigned long long) el,
                       buf[i], Seed(el));
          ++bad_pages;
          break;
        }
      }
    }
    if (bad_pages != 0) {
      std::fprintf(stderr, "seed-verify: %llu corrupt pages BEFORE stress\n",
                   (unsigned long long) bad_pages);
    }
    REQUIRE(bad_pages == 0);
  }

  unsigned long long *err = nullptr;
  REQUIRE(cudaMalloc(&err, 4 * sizeof(unsigned long long)) == cudaSuccess);
  REQUIRE(cudaMemset(err, 0, 4 * sizeof(unsigned long long)) == cudaSuccess);

  RaceStressKernel<<<kBlocks, 256>>>(gpu_info, vec.GetDevice(0), kTotalPages,
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

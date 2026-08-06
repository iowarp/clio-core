/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * ON-DEVICE FAULTING under oversubscription, UNCOMPRESSED.
 *
 * Every other gpu_vector test avoids on-device faulting when the dataset
 * exceeds the HBM cache: the capacity test streams host-orchestrated chunks,
 * the transaction test is "host-orchestrated ... no on-device fault", and the
 * smoke test sizes its logical extent to exactly equal the cache. So the
 * configuration this file covers is untested:
 *
 *     nblocks == 1, cache == 1/2 the logical extent, NO compressor,
 *     reads served purely by the cold-miss fault path (allow_cold_miss_fault)
 *     plus read_range's own rescore/prefetch hints.
 *
 * This matters because the documented deadlock that motivated the
 * host-orchestrated workarounds is COMPRESSION-specific: a cuSZp GetBlob is
 * serviced by launching a decompression kernel, so a spin-waiting on-device
 * fault and the decompressor contend for the same GPU. With an uncompressed
 * kRam bdev the fault is serviced by the HOST over IPC and needs no GPU, so
 * faulting should work. "Should" is the reason this test exists.
 *
 * Each case populates the tag by host-side PutBlob (see PopulateViaPutBlob for
 * why that, and not a write kernel), then opens a FRESH vector on the same tag
 * with a cold cache -- so every page must be faulted in on demand. Vector stats
 * are asserted to prove the faults actually happened rather than the data
 * having been resident already.
 *
 * The three cases form a bisect. CONTROL uses a full-size cache: cold, so every
 * page still faults, but nothing is ever evicted. The PrefetchWindowSync case
 * brings the same pages in over the known-good HOST path instead of the device
 * fault. OVERSUB halves the cache so reads must fault AND evict.
 *
 * What this caught: OVERSUB deadlocked on the 9th page of 16 -- the first fault
 * needing an eviction -- while CONTROL's 16 faults passed. The cause was not in
 * gpu_vector at all. gpu_vector preallocates one PodGetBlobTask per cache slot
 * and reuses it, and a reused task carried the ROUTED/STARTED bits of its
 * previous execution in its host-side RunContext, so the runtime never ran it
 * and the kernel spun forever in gpu::Future::Wait(). Fixed in
 * IpcGpu2Cpu::RecvIn. Keep this test at a cache SMALLER than the extent: a
 * full-size cache uses each task slot exactly once and cannot see the bug.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

// 4 KiB pages of u32 -> 1024 elements per page.
constexpr clio::run::u64 kPageSizeBytes = 4096;
constexpr clio::run::u32 kLogicalPages  = 16;
// THE point of this test: the cache holds exactly half the logical extent.
constexpr clio::run::u32 kCachePages    = kLogicalPages / 2;

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  std::fprintf(stderr, "[INIT] Starting Clio server (oversubscribe test)\n");
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  // Uncompressed kRam target: GetBlob is serviced by the host, no GPU needed.
  const clio::run::u64 kRamCapacity = 1ULL << 30;
  clio::run::PoolId bdev_pool_id(970, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("gpu_vector_oversub_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "gpu_vector_oversub_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  g_initialized = true;
#endif
}

}  // namespace

/** Read [0,total) back through the VECTOR API only. With a half-size cold
 *  cache this cannot be served without faulting. */
__global__ void OversubReadKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceView<clio::run::u32> view,
                                  clio::run::u32 *result,
                                  clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<clio::run::u32> v(view, g_ipc_manager_ptr);
  if (blockIdx.x != 0) return;
  v.read_range(0, total, [result] (clio::run::u64 i, clio::run::u32 val) {
    result[i] = val;
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

clio::run::u64 ElemsPerPage() { return kPageSizeBytes / sizeof(clio::run::u32); }

/**
 * Populate `tag` the way the CAE assimilator does: host-side PutBlob of
 * bare-stem "b0_pi<page>" pages (tag travels as tag_id_ only -- prefixed
 * names overflowed chi::string SSO). This is the ONLY way to get real
 * blobs into the store right now, because Vector::FlushAllSync() is a no-op
 * (see the defect notes in this file's header) and the cache-manager thread
 * does not drain dirty pages in kLegacy single-tier mode. It is also exactly
 * how the llama.cpp weight path is populated, so testing against it measures
 * the configuration that actually ships.
 */
void PopulateViaPutBlob(const std::string &tag, clio::run::u32 logical_pages) {
  gv::Vector<clio::run::u32> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                                 /*gpu_pages_per_block=*/logical_pages,
                                 /*host_pages_per_block=*/0, kPageSizeBytes,
                                 /*cache_period_us=*/20000,
                                 gv::CacheMode::kLegacy,
                                 /*manager_threads_per_block=*/32,
                                 /*allow_cold_miss_fault=*/true);
  auto tag_id = vec.TagId();
  auto *cte = CLIO_CTE_CLIENT;
  const clio::run::u64 epp = ElemsPerPage();

  for (clio::run::u32 p = 0; p < logical_pages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *vals = reinterpret_cast<clio::run::u32 *>(buf.ptr_);
    for (clio::run::u64 j = 0; j < epp; ++j) {
      vals[j] = static_cast<clio::run::u32>((p * epp + j) * 2u);
    }
    std::string blob_name = "b0_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob_name, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
  }
  std::fprintf(stderr, "[PUTBLOB] wrote %u pages to '%s'\n", logical_pages,
               tag.c_str());
}

/** Read `tag` back through a FRESH (cold) cache of `cache_pages`, faulting
 *  only -- no FaultAllSync / PrefetchWindowSync. Verifies every element. */
void ReadBackAndVerify(const std::string &label, const std::string &tag,
                       clio::run::u32 logical_pages,
                       clio::run::u32 cache_pages,
                       clio::run::IpcManagerGpuInfo gpu_info,
                       clio::run::u64 total) {
  auto *result = ctp::GpuApi::MallocHost<clio::run::u32>(
      total * sizeof(clio::run::u32));
  REQUIRE(result != nullptr);
  std::memset(result, 0, total * sizeof(clio::run::u32));

  std::fprintf(stderr,
               "[%s] logical=%u pages, cache=%u pages (%.2fx oversubscribed)"
               " -- reading\n",
               label.c_str(), logical_pages, cache_pages,
               static_cast<double>(logical_pages) / cache_pages);
  {
    gv::Vector<clio::run::u32> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                                   cache_pages,
                                   /*host_pages_per_block=*/0, kPageSizeBytes,
                                   /*cache_period_us=*/20000,
                                   gv::CacheMode::kLegacy,
                                   /*manager_threads_per_block=*/32,
                                   /*allow_cold_miss_fault=*/true);
    OversubReadKernel<<<1, 32>>>(gpu_info, vec.Device(), result, total);

    // The stats block is PINNED host memory written with system-scope atomics,
    // so it is readable while the kernel is still running. Watch it advance so
    // that a hang reports WHERE it stopped instead of just timing out: a stuck
    // resolve_total is the signature of the fault-path deadlock this test was
    // written to catch. Bail out of the watch loop once progress stops --
    // either the kernel is done (Synchronize returns immediately) or it is
    // wedged (Synchronize hangs, but we have already printed the counters).
    unsigned long long last_resolve = ~0ULL;
    int stagnant = 0;
    for (int t = 0; t < 80 && stagnant < 8; ++t) {
      std::this_thread::sleep_for(250ms);
      auto s = vec.StatsSnapshot();
      stagnant = (s.resolve_total == last_resolve) ? stagnant + 1 : 0;
      if (stagnant == 0) {
        std::fprintf(stderr,
                     "[%s] t=%5dms resolve=%llu cold=%llu fault=%llu spin=%llu\n",
                     label.c_str(), t * 250, s.resolve_total,
                     s.resolve_cold_miss, s.resolve_fault_get,
                     s.resolve_spin_iters);
      }
      last_resolve = s.resolve_total;
    }
    ctp::GpuApi::Synchronize();
    std::fprintf(stderr, "[%s] read kernel returned\n", label.c_str());

    auto stats = vec.StatsSnapshot();
    std::fprintf(stderr,
                 "[%s] resolve_total=%llu cold_miss=%llu fault_get=%llu spin=%llu\n",
                 label.c_str(), stats.resolve_total, stats.resolve_cold_miss,
                 stats.resolve_fault_get, stats.resolve_spin_iters);
    // NO-PREFETCH CONTRACT. A sequential read_range resolves exactly once per
    // page, and nothing else populates pages in this configuration, so every
    // page must arrive via a cold-miss fault:
    //   fault_get == cold_miss == logical_pages, and hits == 0.
    // If a prefetcher ever starts serving pages, hits goes up and fault_get
    // drops below logical_pages -- which would silently turn this into a much
    // weaker test, so assert it rather than just "some fault happened".
    REQUIRE(stats.resolve_total == logical_pages);
    REQUIRE(stats.resolve_cold_miss == logical_pages);
    REQUIRE(stats.resolve_fault_get == logical_pages);
    REQUIRE(stats.resolve_hits == 0);
    // Every fault past the first cache-full must have evicted a page.
    if (logical_pages > cache_pages) {
      std::fprintf(stderr, "[%s] evictions forced = %u\n", label.c_str(),
                   logical_pages - cache_pages);
    }
  }

  clio::run::u64 bad = 0;
  for (clio::run::u64 i = 0; i < total; ++i) {
    if (result[i] != static_cast<clio::run::u32>(i * 2u)) {
      if (bad < 8) {
        std::fprintf(stderr, "[%s] mismatch at %llu: got %u want %u\n",
                     label.c_str(), (unsigned long long) i, result[i],
                     static_cast<clio::run::u32>(i * 2u));
      }
      ++bad;
    }
  }
  std::fprintf(stderr, "[%s] mismatches=%llu / %llu\n", label.c_str(),
               (unsigned long long) bad, (unsigned long long) total);
  REQUIRE(bad == 0);
  ctp::GpuApi::FreeHost(result);
}

}  // namespace

/** CONTROL: cold cache sized to the FULL extent. Every page must still be
 *  faulted in on first touch, but no page is ever evicted. If this passes and
 *  the oversubscribed case below hangs, the defect is in EVICTION, not in the
 *  fault/GetBlob path. If this hangs too, faulting itself is broken. */
TEST_CASE("gpu_vector: cold on-device faulting, full-size cache (control)",
          "[gpu_vector][cte][fault]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  const clio::run::u64 total = kLogicalPages * ElemsPerPage();

  PopulateViaPutBlob("gpu_vector_oversub_ctl", kLogicalPages);
  std::fprintf(stderr, "[CONTROL] populated + flushed\n");
  ReadBackAndVerify("CONTROL", "gpu_vector_oversub_ctl", kLogicalPages,
                    kLogicalPages, gpu_info, total);
}

/** DISCRIMINATOR: same cold full-size cache as CONTROL, but the pages are
 *  brought in by the HOST-side PrefetchWindowSync (a known-good GetBlob path)
 *  instead of by the device fault. If this returns correct data then the blobs
 *  persisted correctly and the defect is specifically in the ON-DEVICE fault
 *  delivery; if it also returns zeros, FlushAllSync never persisted them. */
TEST_CASE("gpu_vector: cold read via host PrefetchWindowSync (discriminator)",
          "[gpu_vector][cte][fault]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  const clio::run::u64 total = kLogicalPages * ElemsPerPage();

  PopulateViaPutBlob("gpu_vector_oversub_pf", kLogicalPages);

  auto *result = ctp::GpuApi::MallocHost<clio::run::u32>(
      total * sizeof(clio::run::u32));
  std::memset(result, 0, total * sizeof(clio::run::u32));
  {
    gv::Vector<clio::run::u32> vec("gpu_vector_oversub_pf", /*nblocks=*/1,
                                   /*gpu_id=*/0, kLogicalPages,
                                   /*host_pages_per_block=*/0, kPageSizeBytes,
                                   /*cache_period_us=*/20000,
                                   gv::CacheMode::kLegacy,
                                   /*manager_threads_per_block=*/32,
                                   /*allow_cold_miss_fault=*/true);
    vec.PrefetchWindowSync(0);  // host-driven; binds slot s -> page s
    OversubReadKernel<<<1, 32>>>(gpu_info, vec.Device(), result, total);
    ctp::GpuApi::Synchronize();
  }
  clio::run::u64 bad = 0;
  for (clio::run::u64 i = 0; i < total; ++i) {
    if (result[i] != static_cast<clio::run::u32>(i * 2u)) ++bad;
  }
  std::fprintf(stderr, "[PREFETCH] mismatches=%llu / %llu\n",
               (unsigned long long) bad, (unsigned long long) total);
  ctp::GpuApi::FreeHost(result);
  REQUIRE(bad == 0);
}

/** THE CASE UNDER TEST: cold cache holding exactly 1/2 the logical extent, so
 *  reads must fault AND evict continuously. */
TEST_CASE("gpu_vector: uncompressed on-device faulting at 2x oversubscription",
          "[gpu_vector][cte][fault][oversubscribe]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  const clio::run::u64 total = kLogicalPages * ElemsPerPage();

  PopulateViaPutBlob("gpu_vector_oversub", kLogicalPages);
  std::fprintf(stderr, "[OVERSUB] populated + flushed\n");
  ReadBackAndVerify("OVERSUB", "gpu_vector_oversub", kLogicalPages,
                    kCachePages, gpu_info, total);
}

/**
 * DEEP OVERSUBSCRIPTION. The cases above are a 2x bisect built to localise a
 * deadlock; these are the actual stress: read an entire dataset MANY times
 * larger than the cache, sequentially, in ONE long-running kernel, served
 * purely by the cold-miss fault path.
 *
 * NO PREFETCHING is involved, and ReadBackAndVerify asserts that rather than
 * assuming it: fault_get == cold_miss == logical_pages and hits == 0 means
 * every single page arrived through a device-side fault. (read_range does push
 * rescore/lookahead hints, but in kLegacy single-tier mode with
 * allow_cold_miss_fault the manager never populates a page from them -- the
 * assertion is what keeps that true if the manager ever changes.)
 *
 * Sizing: 4 KiB pages of u32. The cache is pinned at kDeepCachePages and the
 * extent grows, so each case forces (logical - cache) evictions, every one of
 * which recycles a per-slot PodGetBlobTask -- the exact path that was broken.
 */
constexpr clio::run::u32 kDeepCachePages = 16;

TEST_CASE("gpu_vector: sequential fault stress at 4x oversubscription",
          "[gpu_vector][cte][fault][oversubscribe][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  const clio::run::u32 logical = kDeepCachePages * 4;   // 64 pages
  const clio::run::u64 total = logical * ElemsPerPage();

  PopulateViaPutBlob("gpu_vector_deep4x", logical);
  ReadBackAndVerify("DEEP4X", "gpu_vector_deep4x", logical, kDeepCachePages,
                    gpu_info, total);
}

TEST_CASE("gpu_vector: sequential fault stress at 16x oversubscription",
          "[gpu_vector][cte][fault][oversubscribe][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  const clio::run::u32 logical = kDeepCachePages * 16;  // 256 pages
  const clio::run::u64 total = logical * ElemsPerPage();

  PopulateViaPutBlob("gpu_vector_deep16x", logical);
  ReadBackAndVerify("DEEP16X", "gpu_vector_deep16x", logical, kDeepCachePages,
                    gpu_info, total);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * CLIO_SHARED_PERSIST: block-uniform shared state that survives a park.
 *
 * A suspension in this design is a KERNEL EXIT, so __shared__ is garbage on
 * resume. The coroutine frame survives only because it was deliberately put
 * in global memory; nothing does that for shared, and the coroutine transform
 * never will -- shared has static storage duration in another address space,
 * so it is not frame state by definition.
 *
 * That gap cost a real bug: block-uniform tables staged in shared read back
 * stale pointers after a fault and took an MMU fault inside the coroutine's
 * resume path, found only from a GPU coredump. It was invisible for weeks
 * because RESIDENT RUNS NEVER PARK -- which is exactly what case A below
 * pins down, and why every parking case here asserts it really did park.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <cstdio>
#include <fstream>
#include <chrono>
#include <string>
#include <thread>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u64 kPageBytes = 4096;
constexpr u64 kElemsPerPage = kPageBytes / sizeof(u32);
constexpr u64 kPages = 32;
constexpr u64 kElems = kPages * kElemsPerPage;
/** Tiny on purpose: two frames against 32 pages, so a fetch of an unseen
 *  page cannot be a hit and the block is forced to park. */
constexpr u32 kSlots = 2;
constexpr u32 kMagic = 64;

/** BLOCK-DEPENDENT ON PURPOSE. A restore that indexes the backing store by
 *  blockIdx.x rather than the LOGICAL block hands back another block's bytes;
 *  a block-independent pattern would accept that silently. */
CTP_INLINE_CROSS_FUN u32 Pattern(u32 block, u32 i) {
  return block * 7919u + i * 31u + 1u;
}
CTP_INLINE_CROSS_FUN unsigned long long Cookie(u32 block) {
  return 0xC0FFEEULL ^ (static_cast<unsigned long long>(block) << 8);
}

/** The block-uniform state under test. */
struct Tables {
  u32 magic[kMagic];
  unsigned long long cookie;
};
}  // namespace

#if !CTP_IS_DEVICE_PASS
template <typename VecT, typename LaunchT>
static u32 RunYieldable(unsigned nblocks, VecT &vec, LaunchT &&launch) {
  (void)vec;
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, 8192);
  const u32 rounds = drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000, gv::ResumeWhenComplete);
  if (rounds >= 200000) {
    std::fprintf(stderr, "[gv] FATAL: hit the round cap\n");
    std::abort();
  }
  return rounds;
}
#endif  // !CTP_IS_DEVICE_PASS

/**
 * Stamp the arena BEFORE the first suspension, then re-check it after every
 * one. Nothing rewrites it, so a lost restore is a mismatch rather than a
 * value that happens to be refreshed.
 */
__device__ gy::YCoroMain PersistCoro(gv::DeviceVector<u32> v, u32 block,
                                     u32 nparks, unsigned long long *bad) {
  CLIO_SHARED_PERSIST(Tables, t);
  for (u32 i = threadIdx.x; i < kMagic; i += blockDim.x) {
    t.magic[i] = Pattern(block, i);
  }
  if (threadIdx.x == 0) t.cookie = Cookie(block);
  __syncthreads();

  for (u32 k = 0; k < nparks; ++k) {
    // A page this block has not touched, so the fetch must fault and park.
    const u64 pg = (static_cast<u64>(block) * 3ull + k + 1ull) % kPages;
    const u64 off = pg * kElemsPerPage;
    co_await v.Fetch(v.PageLo(off), v.PageSpan(off, 1));

    for (u32 i = threadIdx.x; i < kMagic; i += blockDim.x) {
      if (t.magic[i] != Pattern(block, i)) atomicAdd(bad, 1ull);
    }
    if (threadIdx.x == 0 && t.cookie != Cookie(block)) atomicAdd(bad, 1ull);
    __syncthreads();
  }
}

__global__ void PersistKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<u32> v, u32 nparks,
                              unsigned long long *bad, gy::YieldableView<> yv,
                              gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PersistCoro(v, yv.Block(), nparks, bad));
}

#if !CTP_IS_DEVICE_PASS
TEST_CASE("gpu_vector: shared state survives a park",
          "[gpu_vector][persist]") {
  {
    std::ofstream cfg("gpu_vector_persist.yaml");
    cfg << "networking:\n  port: 9431\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "gpu:\n  queue_depth: 4096\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_persist_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_persist.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  unsigned long long *d_bad =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));

  struct Case {
    const char *name;
    u32 nblocks;
    u32 nparks;
    bool expect_park;
  };
  // D runs many logical blocks so the driver compacts the grid between
  // rounds and a block resumes on different hardware than it started on.
  const Case cases[] = {
      {"A control: no park", 4, 0, false},
      {"B one park", 4, 1, true},
      {"C repeated parks", 4, 8, true},
      {"D grid compaction", 64, 4, true},
  };

  int rc = 0;
  for (const Case &c : cases) {
    gv::Vector<u32> vec("gv_persist", {0}, kPageBytes, c.nblocks, kSlots,
                        kElems);
    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    const u32 rounds = RunYieldable(
        c.nblocks, vec,
        [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
          PersistKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, vec.GetDevice(0), c.nparks, d_bad, vw, sv);
        });
    ctp::GpuApi::Synchronize();

    unsigned long long bad = 0;
    ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
    // THE ROUND ASSERTION IS NOT OPTIONAL. Without it a configuration that
    // never parks leaves the arena trivially intact and the case passes
    // having tested nothing -- which is how the original bug survived.
    const bool parked = rounds > 1;
    const bool ok = (bad == 0) && (parked == c.expect_park);
    std::printf("  %-22s blocks=%2u parks=%u rounds=%u stale=%llu -> %s\n",
                c.name, c.nblocks, c.nparks, rounds, bad,
                ok ? "PASS" : "FAIL");
    if (!ok) {
      if (bad != 0) {
        std::printf("    %llu stale reads: the arena did not survive a "
                    "park\n", bad);
      }
      if (parked != c.expect_park) {
        std::printf("    expected park=%d but rounds=%u -- the case did not "
                    "exercise what it claims\n",
                    static_cast<int>(c.expect_park), rounds);
      }
      rc = 1;
    }
  }
  REQUIRE(rc == 0);
}
#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

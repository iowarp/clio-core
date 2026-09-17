/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Vector::Copy -- the lazy checkpoint.
 *
 * Copy(new_name) creates a tag whose FAULT HANDLER is the checkpoint chimod
 * with this vector's tag as the source: no bytes move at Copy time, and each
 * page of the copy materialises from the source on first touch (host get,
 * host put, or an in-kernel page fault). Verifies:
 *
 *   - HOST READ of the copy returns the source's bytes;
 *   - INDEPENDENCE both ways: overwriting the source afterwards does not
 *     change the copy (it is a checkpoint), and writing the copy does not
 *     change the source;
 *   - DEVICE READ: a kernel faulting the COPY's pages through the ordinary
 *     paging path gets the source's bytes (the fault handler runs under a
 *     device-originated fault, raw-int32 blob names and all).
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
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

namespace {
constexpr clio::run::u64 kPageBytes = 4096;
constexpr clio::run::u64 kElems = 4096;  // 4 pages
constexpr clio::run::u32 kSeedA = 7;     // source pattern
constexpr clio::run::u32 kSeedB = 13;    // source overwrite pattern

clio::run::u32 Pat(clio::run::u64 i, clio::run::u32 seed) {
  return static_cast<clio::run::u32>(i * seed + 1);
}
}  // namespace

/** Read the whole vector and count elements that differ from Pat(i, seed). */
__device__ gy::YCoroMain CopyCheckCoro(gv::DeviceVector<clio::run::u32> v,
                                       clio::run::u64 n, clio::run::u32 seed,
                                       unsigned long long *bad) {
  for (clio::run::u64 i = 0; i < n;) {
    co_await v.Fetch(0, v.PageLo(i), v.PageSpan(i, 1));
    auto h = co_await v.HoldPage(i, n - i);
    for (clio::run::u64 k = threadIdx.x; k < h.run(); k += blockDim.x) {
      if (h[i + k] != static_cast<clio::run::u32>((i + k) * seed + 1)) {
        atomicAdd(bad, 1ull);
      }
    }
    const clio::run::u64 run = h.run();
    v.UnpinRange(v.PageLo(i), v.PageSpan(i, 1));
    i += run;
  }
}

__global__ void CopyCheckKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<clio::run::u32> v,
                                clio::run::u64 n, clio::run::u32 seed,
                                unsigned long long *bad,
                                gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(CopyCheckCoro(v, n, seed, bad));
}

#if !CTP_IS_DEVICE_PASS

/** Download `vec` and count mismatches against Pat(i, seed). */
static clio::run::u64 HostMismatches(gv::Vector<clio::run::u32> &vec,
                                     clio::run::u32 seed) {
  std::vector<clio::run::u32> host(kElems, 0);
  const clio::run::u64 got = vec.Download(host.data(), kElems);
  if (got != kElems) return kElems;
  clio::run::u64 bad = 0;
  for (clio::run::u64 i = 0; i < kElems; ++i) {
    bad += (host[i] != Pat(i, seed));
  }
  return bad;
}

TEST_CASE("gpu_vector: Copy is a lazy checkpoint", "[gpu_vector][copy]") {
  {
    std::ofstream cfg("gpu_vector_copy.yaml");
    REQUIRE(cfg.is_open());
    cfg << "networking:\n  port: 9437\n\n"
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
        << "      - path: \"ram::gv_copy_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_copy.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info{};
  gpu_info = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  gv::Vector<clio::run::u32> src("gv_copy_src", {0}, kPageBytes,
                                 /*nblocks=*/1, /*pages_per_block=*/4, kElems);

  // Seed the SOURCE from the host: 4 pages of Pat(i, kSeedA).
  {
    std::vector<clio::run::u32> host(kElems);
    for (clio::run::u64 i = 0; i < kElems; ++i) host[i] = Pat(i, kSeedA);
    src.Preload(host.data(), kElems);
  }

  // ---- the checkpoint ---------------------------------------------------
  auto ckpt = src.Copy("gv_copy_ckpt");
  REQUIRE(ckpt != nullptr);

  // 1. HOST READ through the copy: every page faults, materialises from the
  //    source, and serves the source's bytes.
  {
    const clio::run::u64 bad = HostMismatches(*ckpt, kSeedA);
    std::fprintf(stderr, "[copy-host-read] mismatches=%llu / %llu\n",
                 (unsigned long long)bad, (unsigned long long)kElems);
    REQUIRE(bad == 0);
  }

  // 2. INDEPENDENCE, source side: overwrite the SOURCE; the copy must keep
  //    the checkpointed bytes.
  {
    std::vector<clio::run::u32> host(kElems);
    for (clio::run::u64 i = 0; i < kElems; ++i) host[i] = Pat(i, kSeedB);
    src.Preload(host.data(), kElems);

    const clio::run::u64 bad_copy = HostMismatches(*ckpt, kSeedA);
    const clio::run::u64 bad_src = HostMismatches(src, kSeedB);
    std::fprintf(stderr,
                 "[independence] copy-vs-A=%llu src-vs-B=%llu / %llu\n",
                 (unsigned long long)bad_copy, (unsigned long long)bad_src,
                 (unsigned long long)kElems);
    REQUIRE(bad_copy == 0);
    REQUIRE(bad_src == 0);
  }

  // 3. INDEPENDENCE, copy side: write the COPY; the source is untouched.
  {
    std::vector<clio::run::u32> host(kElems);
    for (clio::run::u64 i = 0; i < kElems; ++i) host[i] = Pat(i, kSeedA + 2);
    ckpt->Preload(host.data(), kElems);

    const clio::run::u64 bad_src = HostMismatches(src, kSeedB);
    const clio::run::u64 bad_copy = HostMismatches(*ckpt, kSeedA + 2);
    std::fprintf(stderr,
                 "[copy-write] src-vs-B=%llu copy-vs-A2=%llu / %llu\n",
                 (unsigned long long)bad_src, (unsigned long long)bad_copy,
                 (unsigned long long)kElems);
    REQUIRE(bad_src == 0);
    REQUIRE(bad_copy == 0);
  }

  // 4. DEVICE READ: a FRESH copy of the (kSeedB) source, never touched from
  //    the host, read entirely by a kernel. Copy() returns a CTE-only
  //    handle, so the kernel binds ITS OWN Vector to the snapshot tag --
  //    the fault registration lives on the tag, not the handle. Every page
  //    fault goes device -> core -> checkpoint handler -> materialise ->
  //    serve.
  {
    auto dev_copy = src.Copy("gv_copy_ckpt_dev");
    REQUIRE(dev_copy != nullptr);
    gv::Vector<clio::run::u32> dev_view("gv_copy_ckpt_dev", {0}, kPageBytes,
                                        /*nblocks=*/1, /*pages_per_block=*/4,
                                        kElems);

    gy::Yieldable<> drv(1, 32);
    gy::YieldStack stack(1, 32, 8192);
    unsigned long long *d_bad = ctp::GpuApi::Malloc<unsigned long long>(
        sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));

    const clio::run::u32 rounds = drv.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> vw) {
          CopyCheckKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, dev_view.GetDevice(0), kElems, kSeedB, d_bad, vw,
              stack.View());
        },
        [] {}, /*max_rounds=*/200000, gv::ResumeWhenComplete);
    REQUIRE(rounds < 200000);
    ctp::GpuApi::Synchronize();

    unsigned long long bad = 1;
    ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
    ctp::GpuApi::Free(d_bad);
    std::fprintf(stderr, "[copy-device-read] mismatches=%llu / %llu\n",
                 (unsigned long long)bad, (unsigned long long)kElems);
    REQUIRE(bad == 0);
  }
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

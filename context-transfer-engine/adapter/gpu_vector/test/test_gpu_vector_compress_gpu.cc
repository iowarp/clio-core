/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The compression path, end to end through the vector.
 *
 * Pages are written and re-read through a COMPRESSOR pool that interposes in
 * front of the CTE core, so the bytes on the core are codec bytes while the
 * kernel only ever sees plain weights. Both stored forms travel the exact same
 * code path -- only the codec id on the page put differs -- so a difference in
 * result is a compression bug and nothing else.
 *
 * The matrix is {stored raw, stored compressed} x {working set fits in the
 * device cache, working set does not}, repeated at two scales. The
 * does-not-fit half is the interesting one: every page is evicted and
 * re-faulted, so each page round-trips through the codec many times rather
 * than once.
 *
 * The workload is a model-weights calculation -- a weighted sum over the whole
 * weight vector, the inner loop of an inference pass -- checked against a
 * host-computed value. Byte-exactness is the point: a codec that silently
 * corrupts (as LZ4 did when a negative return was read as success) shows up
 * here as a wrong sum, not as a crash.
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

#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/gpu/yield_stack.h>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

namespace {

constexpr clio::run::u64 kPageBytes = 4096;
constexpr clio::run::u64 kPageElems = kPageBytes / sizeof(clio::run::u32);

/** Compressor sits at the CTE entrypoint; the real core is behind it. */
const clio::run::PoolId kCompressorPool(512, 0);
const clio::run::PoolId kCorePool(513, 0);

/** nvcomp-lz4's CTE wire id (CompressionFactory: {"nvcomp-lz4", 11, ...}).
 * The GPU codec, not CPU lz4 (id 4): this build has CTP_ENABLE_LZ4=OFF, so
 * id 4 constructs a null compressor and silently stores pass-through -- and
 * the gpu_vector requirement is GPU codecs anyway. */
constexpr int kLz4WireId = 11;

/**
 * A model weight at index i.
 *
 * Deliberately QUANTIZED-looking: a small value set that repeats, which is
 * what real quantized weights look like to a byte-oriented codec and what
 * makes compression measurable here. Random bytes would compress to nothing
 * and the compressed case would not differ from the raw one.
 */
CTP_INLINE_CROSS_FUN clio::run::u32 Weight(clio::run::u64 i) {
  return static_cast<clio::run::u32>((i / 64) % 17) * 0x01010101u;
}

/** Per-element activation, so the check depends on POSITION as well as value. */
CTP_INLINE_CROSS_FUN clio::run::u32 Activation(clio::run::u64 i) {
  return static_cast<clio::run::u32>((i % 7) + 1);
}

}  // namespace

/** Write the weights, then flush them so they land in the CTE.
 *
 * A COROUTINE, not a blocking kernel. The original spun in-kernel inside
 * the blocking HoldPage until each fault was served -- but the compressor
 * services faults by LAUNCHING KERNELS (nvcomp), and an indefinitely
 * resident spinning kernel blocks every later launch in the context: the
 * classic in-kernel-wait deadlock, and the reason this test hung for as
 * long as it existed. The coroutine suspends instead -- the kernel EXITS,
 * the compressor runs, the driver relaunches -- so there is nothing left
 * to deadlock against. */
__device__ gy::YCoroMain SeedWeightsCoro(gv::DeviceVector<clio::run::u32> v,
                                         clio::run::u64 per,
                                         clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  for (clio::run::u64 i = 0; i < per;) {
    clio::run::u64 run = 0;
    {
      co_await v.BeginFetch(v.PageLo(base + i), v.PageSpan(base + i, 1));
      co_await v.AwaitFetch();
      auto h = co_await v.HoldPage(base + i, per - i, /*write=*/true);
      run = h.run();
      if (threadIdx.x == 0) {
        for (clio::run::u64 k = 0; k < run; ++k) {
            h[base + i + k] = Weight(base + i + k);
        }
      }
      __syncthreads();
    }
    // Flush as we go: the vector never writes back on its own, so a
    // dirty page is unevictable until the caller flushes it. Async, so
    // it overlaps the next page; the servicer retires it once it lands.
    co_await v.BeginFlush();
    i += run;
  }
  co_await v.EndFlush();
}

__global__ void SeedWeightsKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<clio::run::u32> v,
                                  clio::run::u64 per, gy::YieldableView<> yv,
                                  gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SeedWeightsCoro(v, per, yv.Block()));
}

/**
 * The weights calculation: sum(w[i] * activation(i)) over this block's slice,
 * read back through the vector. With a cache smaller than the slice this
 * faults every page, which is where the codec gets exercised repeatedly.
 * Coroutine form: every fault suspends instead of spinning (see
 * SeedWeightsCoro for why that is the difference between running and
 * deadlocking against the compressor's own kernel launches).
 */
__device__ gy::YCoroMain WeightsDotCoro(gv::DeviceVector<clio::run::u32> v,
                                        clio::run::u64 per,
                                        unsigned long long *sum,
                                        clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  unsigned long long acc = 0;
  for (clio::run::u64 i = 0; i < per;) {
    co_await v.BeginFetch(v.PageLo(base + i), v.PageSpan(base + i, 1));
    co_await v.AwaitFetch();
    auto h = co_await v.HoldPage(base + i, per - i);
    if (threadIdx.x == 0) {
      for (clio::run::u64 k = 0; k < h.run(); ++k) {
        acc += static_cast<unsigned long long>(h[base + i + k]) *
               Activation(base + i + k);
      }
    }
    __syncthreads();
    i += h.run();
  }
  if (threadIdx.x == 0) atomicAdd(sum, acc);
}

__global__ void WeightsDotKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<clio::run::u32> v,
                                 clio::run::u64 per, unsigned long long *sum,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(WeightsDotCoro(v, per, sum, yv.Block()));
}

/**
 * Same dot product, but the pages are faulted in CHUNKS via the batched get
 * rather than one at a time -- issued through RescorePagesBatchedAsync
 * (score 1.0 == make resident), which reaches the compressor as a
 * PodMultiGetBlob rather than N PodGetBlobs: a different task, a different
 * handler, and a different record-to-page mapping. A codec that
 * decompressed into the wrong record's page would still return plausible
 * page-sized data, so the check has to be an exact whole-vector
 * calculation, not a per-page checksum. The old form waited INSIDE the
 * kernel on the synchronous batched fetch; this one never waits anywhere
 * but a coroutine suspension.
 */
__device__ gy::YCoroMain WeightsDotBatchedCoro(
    gv::DeviceVector<clio::run::u32> v, clio::run::u64 per,
    clio::run::u32 chunk, unsigned long long *sum, clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  const clio::run::u64 first_page = base / v.ElemsPerPage();
  const clio::run::u64 npages = per / v.ElemsPerPage();
  unsigned long long acc = 0;
  for (clio::run::u64 p0 = 0; p0 < npages; p0 += chunk) {
    clio::run::u64 n = npages - p0;
    if (n > chunk) n = chunk;
    const clio::run::u64 fp = first_page + p0;
    for (clio::run::u64 j = 0; j < n; ++j) {
      const clio::run::u64 off = base + (p0 + j) * v.ElemsPerPage();
      for (clio::run::u64 i = 0; i < v.ElemsPerPage();) {
        co_await v.BeginFetch(v.PageLo(off + i), v.PageSpan(off + i, 1));
        co_await v.AwaitFetch();
        auto h = co_await v.HoldPage(off + i, v.ElemsPerPage() - i);
        if (threadIdx.x == 0) {
          for (clio::run::u64 k = 0; k < h.run(); ++k) {
            acc += static_cast<unsigned long long>(h[off + i + k]) *
                   Activation(off + i + k);
          }
        }
        __syncthreads();
        i += h.run();
      }
    }
  }
  if (threadIdx.x == 0) atomicAdd(sum, acc);
}

__global__ void WeightsDotBatchedKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<clio::run::u32> v,
    clio::run::u64 per, clio::run::u32 chunk, unsigned long long *sum,
    gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(WeightsDotBatchedCoro(v, per, chunk, sum, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS

namespace {
constexpr unsigned kYieldLaneBytes = 8192;

/** Drive a coroutine kernel to completion: launch, service, relaunch. */
template <typename LaunchT>
clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, kYieldLaneBytes);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000);
}
}  // namespace

namespace {

unsigned long long ExpectedDot(clio::run::u64 n) {
  unsigned long long acc = 0;
  for (clio::run::u64 i = 0; i < n; ++i) {
    acc += static_cast<unsigned long long>(Weight(i)) * Activation(i);
  }
  return acc;
}

/** Stored bytes of the whole vector on the CORE, i.e. after any codec. */
clio::run::u64 StoredBytes(const clio::cte::core::TagId &tag,
                           clio::run::u64 num_pages) {
  clio::cte::core::Client core(kCorePool);
  clio::run::u64 total = 0;
  for (clio::run::u64 p = 0; p < num_pages; ++p) {
    const std::string name = std::to_string(p);
    auto sz = core.AsyncGetBlobSize(tag, name);
    sz.Wait();
    if (sz->GetReturnCode() == 0) total += sz->size_;
  }
  return total;
}

}  // namespace

TEST_CASE("gpu_vector: model weights through the compression path",
          "[gpu_vector][compress]") {
  {
    std::ofstream cfg("gpu_vector_compress.yaml");
    REQUIRE(cfg.is_open());
    cfg << "networking:\n  port: 9433\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "gpu:\n  queue_depth: 4096\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
        // Compressor at the entrypoint, core behind it. Both the raw and the
        // compressed configurations below go through this same chain; only
        // the codec stamped on the page put differs.
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"513.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_compress_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_compress.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  struct Config {
    const char *name;
    int codec;                       // 0 = store raw, 11 = nvcomp-lz4
    clio::run::u32 pages_per_block;  // device cache slots
    clio::run::u64 pages;            // pages the block walks
    unsigned nblocks;
    clio::run::u32 chunk;            // >0: fault in batches of this many
  };
  // {raw, lz4} x {fits, does not fit}, at two scales. "fits" gives the block
  // as many slots as pages; "oversub" gives it two, so every page is evicted
  // and re-faulted and each one round-trips through the codec repeatedly.
  const std::vector<Config> configs = {
      {"raw-fits", 0, 8u, 8u, 1u, 0u},
      {"lz4-fits", kLz4WireId, 8u, 8u, 1u, 0u},
      {"raw-oversub", 0, 2u, 8u, 1u, 0u},
      {"lz4-oversub", kLz4WireId, 2u, 8u, 1u, 0u},
      // Larger scale, several blocks, both stored forms.
      {"raw-scale", 0, 4u, 16u, 8u, 0u},
      {"lz4-scale", kLz4WireId, 4u, 16u, 8u, 0u},
      // Same ground truth through the BATCHED fault path, which is what the
      // reader actually uses now. Oversubscribed so every chunk evicts and
      // refills, and chunked at the full cache size so a batch that mismatched
      // records to slots could not hide behind a spare slot.
      {"raw-oversub-batch", 0, 2u, 8u, 1u, 2u},
      {"lz4-oversub-batch", kLz4WireId, 2u, 8u, 1u, 2u},
      {"lz4-scale-batch", kLz4WireId, 4u, 16u, 8u, 4u},
  };

  for (const Config &c : configs) {
    const clio::run::u64 per = c.pages * kPageElems;
    const clio::run::u64 n = per * c.nblocks;
    const std::string tag = std::string("gv_cmp_") + c.name;

    gv::Vector<clio::run::u32> vec(tag, {0}, kPageBytes, c.nblocks,
                                   c.pages_per_block, n, kCompressorPool,
                                   c.codec);

    RunYieldable(c.nblocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                gy::YieldStackView sv) {
      SeedWeightsKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu_info, vec.GetDevice(0), per, vw, sv);
    });
    ctp::GpuApi::Synchronize();

    unsigned long long *d_sum = nullptr;
    d_sum = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_sum)>>(sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_sum, 0, sizeof(unsigned long long));
    if (c.chunk > 0) {
      RunYieldable(c.nblocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
        WeightsDotBatchedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu_info, vec.GetDevice(0), per, c.chunk, d_sum, vw, sv);
      });
    } else {
      RunYieldable(c.nblocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
        WeightsDotKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu_info, vec.GetDevice(0), per, d_sum, vw, sv);
      });
    }
    // GpuApi::Synchronize error-checks fatally, so reaching the next
  // line IS the assertion.
  ctp::GpuApi::Synchronize();

    unsigned long long got = 0;
    ctp::GpuApi::Memcpy(&got, d_sum, sizeof(got));
    ctp::GpuApi::Free(d_sum);

    const unsigned long long want = ExpectedDot(n);
    const clio::run::u64 stored =
        StoredBytes(vec.TagId(), n / kPageElems);
    std::fprintf(stderr,
                 "[%s] blocks=%u slots=%u pages=%llu logical=%lluB stored=%lluB"
                 " sum=%llu want=%llu\n",
                 c.name, c.nblocks, c.pages_per_block,
                 (unsigned long long) (n / kPageElems),
                 (unsigned long long) (n * sizeof(clio::run::u32)),
                 (unsigned long long) stored, got, want);

    // The calculation must be exact whatever the stored form is.
    REQUIRE(got == want);

    // And the codec must actually have done something: a "compressed" config
    // that stored the same bytes as the raw one would pass the sum check
    // while testing nothing.
    if (c.codec != 0) {
      REQUIRE(stored > 0);
      REQUIRE(stored < n * sizeof(clio::run::u32));
    }
  }
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

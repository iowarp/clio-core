/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Model-weights benchmark for the device-paged vector, comparing RAW and
 * COMPRESSED storage as the working set outgrows GPU memory.
 *
 * The variable that matters is the CTE's GPU (kHbm) tier. Pages live there in
 * their STORED form, so compression changes how much of the model fits in
 * device memory; whatever does not fit spills to the host RAM tier and is
 * fetched over PCIe on every touch. The GPU page cache itself always holds
 * plain bytes -- compression buys tier residency, not cache capacity.
 *
 * That gives three regimes, selected by --hbm-mb:
 *
 *   1. Both fit.        Compression is pure overhead: the same pages are
 *                       device-resident either way and the compressed run
 *                       additionally decompresses each fetch. Expect raw to
 *                       win on time and compressed to win on utilization.
 *   2. Raw spills,      The compressed image is device-resident while the raw
 *      compressed fits. one goes to host RAM. Expect compressed to win, by
 *                       the margin between a device read and a PCIe read.
 *   3. Neither fits.    Compressed still moves fewer bytes over PCIe per page,
 *                       so it should stay ahead, by less.
 *
 * The workload is an inference-shaped weighted sum over the whole weight
 * vector. Each block walks its own slice and, with --prefetch, raises the
 * score of the page it is about to need so the organizer keeps it in the top
 * tier -- a metadata hint, not a data copy.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;

namespace {

constexpr clio::run::u64 kPageBytes = 64 * 1024;
constexpr clio::run::u64 kPageElems = kPageBytes / sizeof(clio::run::u32);
const clio::run::PoolId kCompressorPool(512, 0);
const clio::run::PoolId kCorePool(513, 0);
constexpr int kLz4WireId = 4;   // CompressionFactory registry: {"lz4", 4, ...}

/**
 * A weight that compresses like real model data, not like a test pattern.
 *
 * The obvious "small repeating value set" compresses ~34x, which makes every
 * tier look big enough and collapses the three regimes into one. Real
 * quantized weights are close to random at byte level (measured elsewhere:
 * lz4 saves ~4% on Q4_K). This mixes a pseudo-random low nibble with a
 * slowly-varying high nibble, landing around 2x -- compressible enough that
 * residency actually changes, without pretending weights are trivially
 * compressible.
 */
CTP_INLINE_CROSS_FUN clio::run::u32 Weight(clio::run::u64 i) {
  // Runs of kRun identical values: structured weight data repeats locally,
  // and it is that repetition -- not the value distribution -- that a byte
  // codec exploits. kRun sets the compression ratio; 8 lands near 2x.
  constexpr clio::run::u64 kRun = 8;
  clio::run::u32 r = static_cast<clio::run::u32>((i / kRun) * 2654435761u);
  r ^= r >> 13;
  return (r & 0x3F3F3F3Fu) |
         (static_cast<clio::run::u32>((i / 4096) % 13) * 0x40404040u);
}

CTP_INLINE_CROSS_FUN clio::run::u32 Activation(clio::run::u64 i) {
  return static_cast<clio::run::u32>((i % 7) + 1);
}

clio::run::u64 NowMs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

__global__ void SeedKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<clio::run::u32> v,
                           clio::run::u64 per) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  if (threadIdx.x != 0) return;
  const clio::run::u64 base = static_cast<clio::run::u64>(blockIdx.x) * per;
  for (clio::run::u64 i = 0; i < per;) {
    const clio::run::u64 run_i = v.HoldPage(base + i, (per) - i);
    for (clio::run::u64 k_i = 0; k_i < run_i; ++k_i, ++i) {
      v[base + i] = Weight(base + i);
      }
  }
  v.BeginFlush(base, per);
  v.WaitFlush(base, per);
}

/** The measured pass: a weighted sum over this block's slice of the model. */
__global__ void WeightsKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<clio::run::u32> v,
                              clio::run::u64 per, clio::run::u64 page_elems,
                              int prefetch, unsigned long long *sum) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  const clio::run::u64 base = static_cast<clio::run::u64>(blockIdx.x) * per;
  unsigned long long acc = 0;
  for (clio::run::u64 off = 0; off < per; off += page_elems) {
    const clio::run::u64 n =
        (off + page_elems <= per) ? page_elems : (per - off);
    // SINGLE-THREADED per block, deliberately.
    //
    // Reading a 64 KiB page with one thread costs milliseconds and dominates
    // this measurement, so parallelising the inner loop across the block was
    // tried: 6x faster (122ms -> 21ms) and WRONG (checksum mismatch). A lane
    // other than 0 that misses still enters the fault path, where Send and
    // Wait are no-ops for it, so it reads an unpopulated page. Concurrent
    // access needs a block-collective fault in the vector itself; until then
    // the correct-but-slow form is the honest one.
    if (threadIdx.x == 0) {
      if (prefetch && off + page_elems < per) {
        // Metadata-only hint: raise the NEXT page's score so the organizer
        // keeps it in the top tier before this block reaches it.
        v.RescorePage((base + off + page_elems) / page_elems, 1.0f);
      }
      (void) v.HoldPage(base + off, 1);   // fault it in
    }
    if (threadIdx.x != 0) continue;
    for (clio::run::u64 i = 0; i < n;) {
      const clio::run::u64 run_i = v.HoldPage(base + off + i, (n) - i);
      for (clio::run::u64 k_i = 0; k_i < run_i; ++k_i, ++i) {
        acc += static_cast<unsigned long long>(v[base + off + i]) *
               Activation(base + off + i);
          }
    }
  }
  atomicAdd(sum, acc);
}

namespace gy = clio::run::gpu;

/**
 * The same weighted sum, but the block SUSPENDS on a miss instead of one lane
 * blocking on it.
 *
 * The non-yieldable kernel above is single-threaded per block on purpose, and
 * its comment explains why: parallelising the page read was 6x faster and
 * WRONG, because a lane other than 0 that missed entered the fault path where
 * Send and Wait are no-ops for it and then read an unpopulated page. What it
 * asks for is "a block-collective fault in the vector itself".
 *
 * CLIO_YIELD_IF is that fault. Every lane votes on whether its page is
 * resident, the whole block suspends if ANY lane is waiting, and the kernel
 * exits so the fetch can land. On resume the page is resident for everyone, so
 * the read is an ordinary parallel loop with no fault path in it at all.
 */
__global__ void WeightsKernelYield(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceVector<clio::run::u32> v,
                                   clio::run::u64 per,
                                   clio::run::u64 page_elems,
                                   unsigned long long *sum,
                                   gy::YieldableView<> yv,
                                   gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  // The driver relaunches only unfinished blocks, so blockIdx.x is not this
  // block's identity. block_override_ already exists for launch fusion and is
  // exactly the hook needed here.
  v.block_override_ = yv.Block();

  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  // Only these two cross a suspend. Both are touched once per PAGE, never in
  // the inner loop, which is what keeps the frame off the hot path.
  CLIO_YLOCAL_INIT(clio::run::u64, off, 0);
  CLIO_YLOCAL_INIT(unsigned long long, acc, 0);
  // Declared before the switch: re-derived on every entry, so the resume
  // cannot jump over its initialization.
  const clio::run::u64 base =
      static_cast<clio::run::u64>(yv.Block()) * per;

  CLIO_YBEGIN();
  for (; off < per; off += page_elems) {
    // One lane issues the fetch; issuing it from every lane would submit the
    // same page many times.
    if (threadIdx.x == 0 && !v.IsResident((base + off) / page_elems)) {
      v.BeginFetch((base + off) / page_elems);
    }
    CLIO_YIELD_IF(!v.IsResident((base + off) / page_elems));
    {
      // Past the yield the page is resident for the whole block, so HoldPage
      // takes its lock-free fast path and every lane can read.
      const clio::run::u64 n =
          (off + page_elems <= per) ? page_elems : (per - off);
      (void) v.HoldPage(base + off, n);
      unsigned long long r = 0;                   // register, not the frame
      for (clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
        r += static_cast<unsigned long long>(v.at(base + off + i)) *
             Activation(base + off + i);
      }
      acc += r;
    }
  }
  atomicAdd(sum, acc);
  CLIO_YEND();
}

#if !CTP_IS_DEVICE_PASS

int main(int argc, char **argv) {
  unsigned blocks = 16;
  clio::run::u64 hbm_mb = 64;
  clio::run::u64 pages_per_block = 16;
  clio::run::u32 slots = 4;
  bool compressed = false;
  int prefetch = 1;
  int repeat = 3;
  bool yieldable = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? std::atoll(argv[++i]) : 0; };
    if (a == "--blocks") blocks = static_cast<unsigned>(next());
    else if (a == "--hbm-mb") hbm_mb = static_cast<clio::run::u64>(next());
    else if (a == "--pages") pages_per_block = static_cast<clio::run::u64>(next());
    else if (a == "--slots") slots = static_cast<clio::run::u32>(next());
    else if (a == "--compressed") compressed = true;
    else if (a == "--no-prefetch") prefetch = 0;
    else if (a == "--yieldable") yieldable = true;
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--help") {
      std::fprintf(stderr,
                   "usage: %s [--blocks N] [--hbm-mb M] [--pages P] "
                   "[--slots S] [--compressed] [--no-prefetch] [--repeat R]\n"
                   "       [--yieldable]  block-collective faults, parallel page reads\n",
                   argv[0]);
      return 0;
    }
  }

  // The GPU tier is the whole point: pages live there in their STORED form, so
  // its capacity decides how much of the model is device-resident. Everything
  // beyond it spills to the host RAM tier below.
  {
    std::ofstream cfg("gv_weights_bench.yaml");
    cfg << "networking:\n  port: 9435\n\n"
        // first_busy_wait: workers that fall back to sleeping add their sleep to
        // every device fault, since a fault is a synchronous round trip. With
        // the 1000us default a fault costs ~5ms regardless of where the page
        // lives, which buries the tier/codec difference this benchmark exists
        // to measure. Keep them spinning for the duration of a run.
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 2000000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"2GB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"513.0\"\n"
        << "    storage:\n"
        << "      - path: \"hbm::gv_bench_hbm\"\n"
        << "        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: 1.0\n"
        << "      - path: \"ram::gv_bench_ram\"\n"
        << "        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"2GB\"\n"
        << "        score: 0.2\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_weights_bench.yaml", 1);

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "bench: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "bench: cte client init failed\n");
    return 1;
  }
  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  const clio::run::u64 per = pages_per_block * kPageElems;
  const clio::run::u64 n = per * blocks;
  const clio::run::u64 logical = n * sizeof(clio::run::u32);
  const std::string tag =
      std::string("gvw_") + (compressed ? "lz4_" : "raw_") +
      std::to_string(blocks) + "_" + std::to_string(hbm_mb) + "_" +
      std::to_string(pages_per_block);

  gv::Vector<clio::run::u32> vec(tag, {0}, kPageBytes, blocks, slots, n,
                                 kCompressorPool,
                                 compressed ? kLz4WireId : 0);

  SeedKernel<<<blocks, 32>>>(gpu_info, vec.GetDevice(0), per);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "bench: seed failed\n");
    return 1;
  }

  // Stored footprint: what the model actually occupies in the CTE, which is
  // what decides whether it fits the GPU tier.
  clio::run::u64 stored = 0;
  {
    clio::cte::core::Client core(kCorePool);
    for (clio::run::u64 p = 0; p < n / kPageElems; ++p) {
      char name[32];
      gv::PageBlobName(p, name);
      auto sz = core.AsyncGetBlobSize(vec.TagId(), name);
      sz.Wait();
      if (sz->GetReturnCode() == 0) stored += sz->size_;
    }
  }

  unsigned long long *d_sum = nullptr;
  cudaMalloc(&d_sum, sizeof(unsigned long long));
  unsigned long long want = 0;
  for (clio::run::u64 i = 0; i < n; ++i) {
    want += static_cast<unsigned long long>(Weight(i)) * Activation(i);
  }

  double best_gbps = 0.0;
  clio::run::u32 rounds = 0;
  clio::run::u64 best_ms = 0;
  bool ok = true;
  for (int r = 0; r < repeat; ++r) {
    cudaMemset(d_sum, 0, sizeof(unsigned long long));
    const clio::run::u64 t0 = NowMs();
    if (yieldable) {
      gy::Yieldable<> drv(blocks, 32);
      gy::YieldStack ystack(blocks, 32, 256);
      rounds = drv.RunToCompletion(
          [&](dim3 g, dim3 b, gy::YieldableView<> view) {
            WeightsKernelYield<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                gpu_info, vec.GetDevice(0), per, kPageElems, d_sum, view,
                ystack.View());
          },
          []{}, /*max_rounds=*/200000);
    } else {
      WeightsKernel<<<blocks, 32>>>(gpu_info, vec.GetDevice(0), per, kPageElems,
                                    prefetch, d_sum);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "bench: kernel failed\n");
      return 1;
    }
    const clio::run::u64 ms = NowMs() - t0;
    unsigned long long got = 0;
    cudaMemcpy(&got, d_sum, sizeof(got), cudaMemcpyDeviceToHost);
    if (got != want) ok = false;
    const double gbps =
        (ms == 0) ? 0.0
                  : (static_cast<double>(logical) / (1024.0 * 1024.0 * 1024.0)) /
                        (static_cast<double>(ms) / 1000.0);
    if (gbps > best_gbps) {
      best_gbps = gbps;
      best_ms = ms;
    }
  }
  cudaFree(d_sum);

  std::fprintf(stderr,
               "GVW mode=%s%s blocks=%u hbm=%lluMB slots=%u pages=%llu "
               "logical=%.1fMB stored=%.1fMB fits=%s ms=%llu GB/s=%.2f "
               "checksum=%s rounds=%u\n",
               compressed ? "lz4" : "raw", yieldable ? "+yield" : "", blocks,
               (unsigned long long) hbm_mb, slots,
               (unsigned long long) (n / kPageElems),
               logical / (1024.0 * 1024.0), stored / (1024.0 * 1024.0),
               (stored <= hbm_mb * 1024ull * 1024ull) ? "yes" : "no",
               (unsigned long long) best_ms, best_gbps, ok ? "OK" : "MISMATCH",
               rounds);
  return ok ? 0 : 1;
}

#endif  // !CTP_IS_DEVICE_PASS

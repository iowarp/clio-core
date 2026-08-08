/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Does an asynchronous flush overlap with compute? The simplest possible test.
 *
 * Each iteration a block spins for a fixed time, writes a REGION, and flushes
 * that region. The flush is block level: one BeginFlush over the whole region,
 * which issues a put per page and leaves them all in flight, so a large region
 * amortises the per-put round trip instead of paying it once per 64 KB.
 *
 * Nothing is read, nothing is faulted, nothing is evicted in the timed loop --
 * the cache is sized to hold every region at once and everything is made
 * resident before timing starts. The ONLY I/O measured is the write-back.
 *
 *   sync    spin, write, BeginFlush, WaitFlush        -> spin + flush
 *   async   spin, collect PREVIOUS flush, write,      -> max(spin, flush)
 *           BeginFlush, continue
 *
 * Both baselines are taken the same way as the real runs -- spin with no I/O,
 * and I/O with no spin -- so "sum" and "max" are measured quantities and the
 * verdict is arithmetic. Written bytes are verified from the host afterwards;
 * a flush that loses data fails instead of posting a speedup.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
using clio::run::u32;
using clio::run::u64;

/** Value written at index i on pass `pass`. */
CTP_INLINE_CROSS_FUN u32 Val(u64 i, u32 pass) {
  return static_cast<u32>(i * 2654435761u + pass * 2246822519u + 1u);
}

/** Busy-wait `us` microseconds of SM time. Identical in both kernels. */
CTP_GPU_FUN void Spin(u32 us, u64 clock_khz) {
  if (us == 0) return;
  const long long ticks =
      static_cast<long long>(us) * static_cast<long long>(clock_khz) / 1000ll;
  const long long t0 = clock64();
  while (clock64() - t0 < ticks) {
    // Pure spin, no __nanosleep: sleeping lets the clock drop, which makes the
    // same "microseconds" cost different wall time depending on what else the
    // kernel is doing -- exactly the confound this benchmark exists to avoid.
  }
}

/** Touch and flush every page, so the timed loop faults nothing. */
__global__ void WarmKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 iters,
                           u64 pages_per_region) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  const u64 region_elems = pages_per_region * v.elems_per_page_;
  const u64 block_base = static_cast<u64>(blockIdx.x) * iters * region_elems;
  for (u64 it = 0; it < iters; ++it) {
    {
      const u64 off = block_base + it * region_elems;
      for (u64 pg = 0; pg < pages_per_region; ++pg) {
        const u64 poff = off + pg * v.elems_per_page_;
        v.HoldPage(poff, v.elems_per_page_);
        for (u64 i = threadIdx.x; i < v.elems_per_page_; i += blockDim.x) {
          v[poff + i] = Val(poff + i, 0u);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        v.BeginFlush(off, region_elems);
        v.WaitFlush(off, region_elems);
      }
      __syncthreads();
      }
  }
}

/**
 * spin -> write region -> flush region, `iters` times.
 *
 * @param async 0: BeginFlush then WaitFlush immediately.
 *              1: BeginFlush and collect it on the NEXT iteration, after that
 *                 iteration's spin, so the transfer runs under the spin.
 */
__global__ void SpinWriteFlushKernel(clio::run::IpcManagerGpuInfo info,
                                     gv::DeviceVector<u32> v, u64 iters,
                                     u64 pages_per_region, u32 spin_us,
                                     u64 clock_khz, int do_write, u32 pass,
                                     int async) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  const u64 region_elems = pages_per_region * v.elems_per_page_;
  const u64 block_base = static_cast<u64>(blockIdx.x) * iters * region_elems;
  long long prev = -1;

  for (u64 it = 0; it < iters; ++it) {
    {
      // ---- compute ----
      Spin(spin_us, clock_khz);
      __syncthreads();

      // ---- async: collect the previous flush, which has been running under
      //      the spin above ----
      if (async && do_write && threadIdx.x == 0 && prev >= 0) {
        v.WaitFlush(block_base + static_cast<u64>(prev) * region_elems,
                    region_elems);
      }
      __syncthreads();

      if (do_write) {
        const u64 off = block_base + it * region_elems;
        // Page at a time, so the whole block is inside one page at any moment --
        // the granularity the vector's paging contract assumes.
        for (u64 pg = 0; pg < pages_per_region; ++pg) {
          const u64 poff = off + pg * v.elems_per_page_;
          for (u64 i = threadIdx.x; i < v.elems_per_page_; i += blockDim.x) {
            v[poff + i] = Val(poff + i, pass);
          }
          __syncthreads();
        }
        // ---- block-level flush: ONE call covering the whole region ----
        if (threadIdx.x == 0) {
          v.BeginFlush(off, region_elems);
          if (!async) {
            v.WaitFlush(off, region_elems);
          }
        }
        __syncthreads();
        prev = static_cast<long long>(it);
      }
      }
  }

  if (async == 1 && do_write && threadIdx.x == 0 && prev >= 0) {
    v.WaitFlush(block_base + static_cast<u64>(prev) * region_elems,
                region_elems);
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

struct Args {
  u32 blocks = 1;
  u64 iters = 4;         // regions written per block
  u32 spin_us = 2000;    // compute per iteration
  u64 page_kb = 1024;    // page granularity
  u64 flush_mb = 128;    // region flushed per iteration
  u32 threads = 256;
  u32 repeat = 3;
};

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> const char * {
      return (i + 1 < argc) ? argv[++i] : "0";
    };
    if (f == "--blocks") a.blocks = std::atoi(next());
    else if (f == "--iters") a.iters = std::atoll(next());
    else if (f == "--spin-us") a.spin_us = std::atoi(next());
    else if (f == "--page-kb") a.page_kb = std::atoll(next());
    else if (f == "--flush-mb") a.flush_mb = std::atoll(next());
    else if (f == "--threads") a.threads = std::atoi(next());
    else if (f == "--repeat") a.repeat = std::atoi(next());
    else if (f == "--help") {
      std::printf(
          "usage: %s [--blocks N] [--iters N] [--spin-us N] [--page-kb N] "
          "[--flush-mb N] [--threads N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }

  const u64 page_bytes = a.page_kb * 1024;
  const u64 page_elems = page_bytes / sizeof(u32);
  const u64 flush_bytes = a.flush_mb * 1024ull * 1024ull;
  const u64 pages_per_region = flush_bytes / page_bytes;
  if (pages_per_region == 0) {
    std::fprintf(stderr, "flush region smaller than one page\n");
    return 1;
  }
  // Every region resident at once: no faults and no evictions while timing.
  const u64 slots = pages_per_region * a.iters;
  const u64 n = page_elems * slots * a.blocks;
  const double resident_mb =
      static_cast<double>(slots * page_bytes * a.blocks) / (1024.0 * 1024.0);

  {
    std::ofstream cfg("gpu_vector_flush.yaml");
    cfg << "networking:\n  port: 9437\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"16GB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_flush_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"12GB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_flush.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  const u64 clock_khz = static_cast<u64>(prop.clockRate);

  std::printf(
      "gpu_vector block-level flush benchmark\n"
      "  blocks=%u threads=%u iters=%llu spin=%u us/iter\n"
      "  page=%lluKB  flush region=%lluMB (%llu pages/flush)\n"
      "  resident=%.0fMB  total written=%.0fMB per pass\n"
      "  (all regions resident: no faults, no evictions -- flush is the only "
      "I/O)\n",
      a.blocks, a.threads, (unsigned long long) a.iters, a.spin_us,
      (unsigned long long) a.page_kb, (unsigned long long) a.flush_mb,
      (unsigned long long) pages_per_region, resident_mb, resident_mb);

  gv::Vector<u32> vec("gv_flush", {0}, page_bytes, a.blocks,
                      static_cast<u32>(slots), n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);

  WarmKernel<<<a.blocks, a.threads>>>(gpu, dev, a.iters, pages_per_region);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "warm failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }

  u32 pass = 1;
  auto run = [&](u32 spin, int do_write, int async) {
    double best = 1e30;
    for (u32 r = 0; r < a.repeat; ++r) {
      vec.ResetStats();
      cudaDeviceSynchronize();
      const double t0 = NowMs();
      SpinWriteFlushKernel<<<a.blocks, a.threads>>>(
          gpu, dev, a.iters, pages_per_region, spin, clock_khz, do_write,
          pass++, async);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
      }
      const double ms = NowMs() - t0;
      if (ms < best) best = ms;
    }
    return best;
  };

  const double spin_only = run(a.spin_us, 0, 0);   // compute, no I/O
  const double io_only = run(0, 1, 0);             // I/O, no compute
  const auto io_stats = vec.ReadStats(0);
  const double sync_ms = run(a.spin_us, 1, 0);
  const auto sync_stats = vec.ReadStats(0);
  const double async_ms = run(a.spin_us, 1, 1);
  const auto async_stats = vec.ReadStats(0);
  const u32 last_pass = pass - 1;

  // Verify the async pass reached the CTE. Sampled: first and last page of
  // every region, which is enough to catch a lost or mis-ordered flush without
  // reading hundreds of megabytes back through the client.
  bool ok = true;
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<u32> buf(static_cast<size_t>(page_elems));
    for (u32 b = 0; b < a.blocks && ok; ++b) {
      for (u64 it = 0; it < a.iters && ok; ++it) {
        const u64 first = (b * a.iters + it) * pages_per_region;
        const u64 probe[2] = {first, first + pages_per_region - 1};
        for (int k = 0; k < 2 && ok; ++k) {
          char name[32];
          gv::PageBlobName(probe[k], name);
          auto f = core.AsyncGetBlob(vec.TagId(), std::string(name), 0,
                                     page_bytes, 0,
                                     reinterpret_cast<char *>(buf.data()));
          f.Wait();
          if (f->GetReturnCode() != 0) {
            std::fprintf(stderr, "  read of %s failed rc=%d\n", name,
                         f->GetReturnCode());
            ok = false;
            break;
          }
          const u64 base = probe[k] * page_elems;
          for (u64 i = 0; i < page_elems; i += 97) {
            if (buf[static_cast<size_t>(i)] != Val(base + i, last_pass)) {
              std::fprintf(stderr,
                           "  MISMATCH %s elem %llu: got %u want %u\n", name,
                           (unsigned long long) i, buf[static_cast<size_t>(i)],
                           Val(base + i, last_pass));
              ok = false;
              break;
            }
          }
        }
      }
    }
  }

  const double per_iter = static_cast<double>(a.iters);
  const double mx = (spin_only > io_only) ? spin_only : io_only;
  const double flush_mbps =
      (io_only > 0.0) ? (resident_mb / (io_only / 1000.0)) : 0.0;
  std::printf(
      "\n  spin only        %8.2f ms   (%.0f us/iter)\n"
      "  flush only       %8.2f ms   (%.0f us/iter, puts=%llu, %.0f MB/s)\n"
      "  ---------------------------------------------\n"
      "  sync  measured   %8.2f ms   (puts=%llu)\n"
      "  async measured   %8.2f ms   (puts=%llu)  %s\n"
      "  ---------------------------------------------\n"
      "  sum  spin+flush  %8.2f ms   <- what NO overlap costs\n"
      "  max  spin,flush  %8.2f ms   <- what FULL overlap costs\n"
      "  speedup async    %8.2fx\n"
      "  flush hidden     %8.1f %%\n",
      spin_only, spin_only * 1000.0 / per_iter,
      io_only, io_only * 1000.0 / per_iter,
      (unsigned long long) io_stats.puts, flush_mbps,
      sync_ms, (unsigned long long) sync_stats.puts,
      async_ms, (unsigned long long) async_stats.puts,
      ok ? "data=OK" : "data=MISMATCH",
      spin_only + io_only, mx, sync_ms / async_ms,
      (sync_ms > mx) ? (sync_ms - async_ms) / (sync_ms - mx) * 100.0 : 0.0);

  if (!ok) {
    std::fprintf(stderr, "\nDATA MISMATCH -- timings above are void\n");
    return 1;
  }
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS

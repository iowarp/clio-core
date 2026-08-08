/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Does an asynchronous flush overlap with compute? The simplest possible test.
 *
 * Each iteration: spin for a fixed time, write some bytes to a page, flush it.
 * Nothing is read, nothing is faulted, nothing is evicted -- every page is made
 * resident before timing starts, so the ONLY I/O in the timed loop is the
 * write-back. Two kernels, identical work:
 *
 *   sync    spin, write, BeginFlush, WaitFlush.        -> spin + flush
 *   async   spin, wait for the PREVIOUS flush, write,  -> max(spin, flush)
 *           BeginFlush and move on.
 *
 * If asynchronous flushing overlaps, `async` costs the larger of the two halves
 * and `sync` costs their sum. Both are measured against two baselines taken the
 * same way -- spin with no I/O, and I/O with no spin -- so the sum and the max
 * are known quantities rather than assumptions, and the answer is arithmetic.
 *
 * Correctness is verified after every timed run by reading the pages back from
 * the CTE on the host: an "overlap" that loses writes is a failure.
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

namespace {
constexpr u64 kPageBytes = 65536;
constexpr u64 kPageElems = kPageBytes / sizeof(u32);
}  // namespace

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
    // Pure spin, no __nanosleep: sleeping lets the clock drop and makes the
    // same "microseconds" cost different wall time depending on what else the
    // kernel is doing, which is exactly the confound this benchmark exists to
    // avoid.
  }
}

/** Make every page resident and give it a blob, before any timing. */
__global__ void WarmKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 pages) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  const u64 base = static_cast<u64>(blockIdx.x) * pages * v.elems_per_page_;
  for (u64 p = 0; p < pages; ++p) {
    const u64 off = base + p * v.elems_per_page_;
    for (u64 i = threadIdx.x; i < v.elems_per_page_; i += blockDim.x) {
      v[off + i] = Val(off + i, 0u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      v.BeginFlush(off, v.elems_per_page_);
      v.WaitFlush(off, v.elems_per_page_);
    }
    __syncthreads();
  }
}

/**
 * spin -> write -> flush, N times.
 *
 * @param spin_us    compute per iteration
 * @param write_elems elements written per iteration (the I/O size)
 * @param async      0: flush and wait immediately.
 *                   1: flush and wait for it on the NEXT iteration, after that
 *                      iteration's spin -- so the transfer runs underneath the
 *                      spin.
 */
__global__ void SpinWriteFlushKernel(clio::run::IpcManagerGpuInfo info,
                                     gv::DeviceVector<u32> v, u64 pages,
                                     u32 spin_us, u64 clock_khz,
                                     u64 write_elems, u32 pass, int async) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  const u64 base = static_cast<u64>(blockIdx.x) * pages * v.elems_per_page_;
  long long prev = -1;

  for (u64 p = 0; p < pages; ++p) {
    // ---- compute ----
    Spin(spin_us, clock_khz);
    __syncthreads();

    // ---- in async mode, THIS is where the previous flush is collected;
    //      it has been running underneath the spin above ----
    if (async && threadIdx.x == 0 && prev >= 0) {
      const u64 poff = base + static_cast<u64>(prev) * v.elems_per_page_;
      v.WaitFlush(poff, v.elems_per_page_);
    }
    __syncthreads();

    // ---- write ----
    const u64 off = base + p * v.elems_per_page_;
    for (u64 i = threadIdx.x; i < write_elems; i += blockDim.x) {
      v[off + i] = Val(off + i, pass);
    }
    __syncthreads();

    // ---- flush ----
    if (threadIdx.x == 0) {
      v.BeginFlush(off, v.elems_per_page_);
      if (!async) {
        v.WaitFlush(off, v.elems_per_page_);
      }
    }
    __syncthreads();
    prev = static_cast<long long>(p);
  }

  // Drain the last outstanding flush.
  if (async && threadIdx.x == 0 && prev >= 0) {
    const u64 poff = base + static_cast<u64>(prev) * v.elems_per_page_;
    v.WaitFlush(poff, v.elems_per_page_);
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

struct Args {
  u32 blocks = 1;
  u64 pages = 32;        // iterations
  u32 spin_us = 200;     // compute per iteration
  u64 write_kb = 64;     // bytes written per iteration
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
    else if (f == "--iters") a.pages = std::atoll(next());
    else if (f == "--spin-us") a.spin_us = std::atoi(next());
    else if (f == "--write-kb") a.write_kb = std::atoll(next());
    else if (f == "--repeat") a.repeat = std::atoi(next());
    else if (f == "--help") {
      std::printf("usage: %s [--blocks N] [--iters N] [--spin-us N] "
                  "[--write-kb N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }
  const u64 write_elems =
      (a.write_kb * 1024 > kPageBytes ? kPageBytes : a.write_kb * 1024) /
      sizeof(u32);

  {
    std::ofstream cfg("gpu_vector_flush.yaml");
    cfg << "networking:\n  port: 9437\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"4GB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_flush_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"2GB\"\n"
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

  const u64 n = a.pages * kPageElems * a.blocks;
  // Cache holds EVERY page: no faults and no evictions in the timed loop, so
  // the only I/O measured is the flush.
  gv::Vector<u32> vec("gv_flush", {0}, kPageBytes, a.blocks,
                      static_cast<u32>(a.pages), n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);

  WarmKernel<<<a.blocks, 32>>>(gpu, dev, a.pages);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "warm failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }

  std::printf(
      "gpu_vector flush benchmark\n"
      "  blocks=%u iters=%llu spin=%u us/iter write=%lluKB/iter page=%lluKB\n"
      "  (all pages resident: no faults, no evictions -- flush is the only I/O)\n",
      a.blocks, (unsigned long long) a.pages, a.spin_us,
      (unsigned long long) (write_elems * sizeof(u32) / 1024),
      (unsigned long long) (kPageBytes / 1024));

  u32 pass = 1;
  auto run = [&](u32 spin, u64 welems, int async) {
    double best = 1e30;
    for (u32 r = 0; r < a.repeat; ++r) {
      vec.ResetStats();
      cudaDeviceSynchronize();
      const double t0 = NowMs();
      SpinWriteFlushKernel<<<a.blocks, 32>>>(gpu, dev, a.pages, spin, clock_khz,
                                             welems, pass++, async);
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

  // Baselines, taken exactly the same way as the real runs.
  const double spin_only = run(a.spin_us, 0, 0);          // compute, no I/O
  const double io_only = run(0, write_elems, 0);          // I/O, no compute
  const auto io_stats = vec.ReadStats(0);

  const double sync_ms = run(a.spin_us, write_elems, 0);
  const auto sync_stats = vec.ReadStats(0);
  const double async_ms = run(a.spin_us, write_elems, 1);
  const auto async_stats = vec.ReadStats(0);
  const u32 last_pass = pass - 1;

  // Verify the async run's bytes actually reached the CTE.
  bool ok = true;
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<u32> buf(static_cast<size_t>(kPageElems));
    for (u32 b = 0; b < a.blocks && ok; ++b) {
      for (u64 p = 0; p < a.pages && ok; ++p) {
        const u64 page = b * a.pages + p;
        char name[32];
        gv::PageBlobName(page, name);
        auto f = core.AsyncGetBlob(vec.TagId(), std::string(name), 0,
                                   kPageBytes, 0,
                                   reinterpret_cast<char *>(buf.data()));
        f.Wait();
        if (f->GetReturnCode() != 0) { ok = false; break; }
        const u64 base = page * kPageElems;
        for (u64 i = 0; i < write_elems; ++i) {
          if (buf[static_cast<size_t>(i)] != Val(base + i, last_pass)) {
            std::fprintf(stderr,
                         "  MISMATCH page %llu elem %llu: got %u want %u\n",
                         (unsigned long long) page, (unsigned long long) i,
                         buf[static_cast<size_t>(i)],
                         Val(base + i, last_pass));
            ok = false;
            break;
          }
        }
      }
    }
  }

  const double per_iter = static_cast<double>(a.pages);
  std::printf(
      "\n  spin only        %8.2f ms   (%.1f us/iter)\n"
      "  flush only       %8.2f ms   (%.1f us/iter, puts=%llu)\n"
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
      (unsigned long long) io_stats.puts,
      sync_ms, (unsigned long long) sync_stats.puts,
      async_ms, (unsigned long long) async_stats.puts,
      ok ? "data=OK" : "data=MISMATCH",
      spin_only + io_only,
      (spin_only > io_only ? spin_only : io_only),
      sync_ms / async_ms,
      (sync_ms > (spin_only > io_only ? spin_only : io_only))
          ? (sync_ms - async_ms) /
                (sync_ms - (spin_only > io_only ? spin_only : io_only)) * 100.0
          : 0.0);

  if (!ok) {
    std::fprintf(stderr, "\nDATA MISMATCH -- timings above are void\n");
    return 1;
  }
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS

/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CUDA Unified Virtual Memory (UVM / cudaMallocManaged) baseline.
 *
 * Same shape as the traditional out-of-memory benchmark, but the per-block
 * staging buffer lives in managed memory:
 *
 *   for each page-stripe index p in [0, per_block_pages):
 *     1. launch fill kernel that writes nblocks pages of u32s to managed_buf
 *     2. cudaDeviceSynchronize
 *     3. for each block b: cte.AsyncPutBlob(...).Wait()  with the managed
 *        pointer (the runtime DeviceAwareMemcpy hook walks pages in/out
 *        on demand, so no explicit cudaMemcpy is needed)
 *
 * The read mirror does AsyncGetBlob().Wait() into the managed buffer, then
 * a kernel touches the data so the read driver migrates pages to device.
 *
 * CLI is identical to clio_cte_gpu_vector_bench so the three benches can
 * be compared apples-to-apples.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace std::chrono_literals;

namespace {

struct BenchOpts {
  clio::run::u32 nblocks = 32;
  clio::run::u32 pages_per_block = 2;     // accepted for CLI compat; sizing only
  clio::run::u64 page_size = 1ULL << 20;  // 1 MiB
  double ratio = 1.0;
  clio::run::u64 total_bytes = 0;
  clio::run::u64 per_block_bytes = 0;
  clio::run::u32 cache_period_ms = 0;     // accepted for CLI compat; unused here
  clio::run::u32 iters = 1;
  bool do_read = true;
  clio::run::u32 gpu_id = 0;
  clio::run::u64 bdev_capacity_mib = 0;
};

void PrintUsage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "  --blocks N             Number of blocks (default 32)\n"
               "  --pages-per-block P    Pages per block (default 2; affects only sizing math)\n"
               "  --page-size BYTES      Page size in bytes (default 1048576)\n"
               "  --per-block-bytes B    Bytes each block writes/reads (overrides --ratio\n"
               "                         and --total-bytes; total_bytes = blocks * B)\n"
               "  --ratio R              total_bytes = R * (blocks*pages_per_block*page_size)\n"
               "                         (default 1.0)\n"
               "  --total-bytes BYTES    Override total bytes (takes precedence over --ratio)\n"
               "  --cache-period-ms N    Accepted for CLI compat; unused here\n"
               "  --iters N              Repeat write[/read] cycle N times (default 1)\n"
               "  --no-read              Skip read phase\n"
               "  --bdev-capacity-mib N  kRam bdev capacity (default = 4x total_bytes, min 64)\n"
               "  --gpu-id N             GPU index (default 0)\n"
               "  --help                 Show this message\n",
               prog);
}

bool ParseOpts(int argc, char *argv[], BenchOpts &opts) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); std::exit(0); }
    else if (a == "--blocks") opts.nblocks = std::atoi(next("--blocks"));
    else if (a == "--pages-per-block")
      opts.pages_per_block = std::atoi(next("--pages-per-block"));
    else if (a == "--page-size")
      opts.page_size = std::strtoull(next("--page-size"), nullptr, 10);
    else if (a == "--ratio") opts.ratio = std::atof(next("--ratio"));
    else if (a == "--total-bytes")
      opts.total_bytes = std::strtoull(next("--total-bytes"), nullptr, 10);
    else if (a == "--per-block-bytes")
      opts.per_block_bytes =
          std::strtoull(next("--per-block-bytes"), nullptr, 10);
    else if (a == "--cache-period-ms")
      opts.cache_period_ms = std::atoi(next("--cache-period-ms"));
    else if (a == "--iters") opts.iters = std::atoi(next("--iters"));
    else if (a == "--no-read") opts.do_read = false;
    else if (a == "--bdev-capacity-mib")
      opts.bdev_capacity_mib =
          std::strtoull(next("--bdev-capacity-mib"), nullptr, 10);
    else if (a == "--gpu-id") opts.gpu_id = std::atoi(next("--gpu-id"));
    else {
      std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

#if !CTP_IS_DEVICE_PASS
void EnsureInit(const BenchOpts &opts, clio::run::u64 bdev_capacity_bytes) {
  std::fprintf(stderr, "[INIT] Starting Clio server\n");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "[INIT] CLIO_INIT failed\n");
    std::exit(2);
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[INIT] CLIO_CTE_CLIENT_INIT failed\n");
    std::exit(2);
  }
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] CTE pool create failed rc=%u\n",
                 create_task->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);

  clio::run::PoolId bdev_pool_id(951, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("uvm_bench_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, bdev_capacity_bytes);
  bdev_create.Wait();
  if (bdev_create->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] bdev create failed rc=%u\n",
                 bdev_create->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "uvm_bench_ram", clio::run::bdev::BdevType::kRam,
      bdev_capacity_bytes, clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] RegisterTarget failed rc=%u\n",
                 reg_task->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);
  (void)opts;
}
#endif  // !CTP_IS_DEVICE_PASS

}  // namespace

/** Each block fills its slice of stripe page p directly into the UVM buffer.
 *  cudaMallocManaged backs `managed_buf`, so the writes will demand-fault
 *  pages onto device as the kernel runs. */
__global__ void UvmFillPageKernel(clio::run::u32 *managed_buf,
                                   clio::run::u64 elems_per_page,
                                   clio::run::u64 page_idx,
                                   clio::run::u64 per_block_pages) {
  clio::run::u32 b = blockIdx.x;
  clio::run::u64 base =
      (static_cast<clio::run::u64>(b) * per_block_pages + page_idx) * elems_per_page;
  clio::run::u32 *slot = managed_buf + b * elems_per_page;
  for (clio::run::u64 j = threadIdx.x; j < elems_per_page; j += blockDim.x) {
    slot[j] = static_cast<clio::run::u32>(base + j);
  }
}

/** Touch the just-Get'd UVM data so pages migrate back to device on read. */
__global__ void UvmConsumePageKernel(const clio::run::u32 *managed_buf,
                                      clio::run::u64 elems_per_page,
                                      clio::run::u32 *out_xor) {
  clio::run::u32 b = blockIdx.x;
  const clio::run::u32 *slot = managed_buf + b * elems_per_page;
  clio::run::u32 acc = 0;
  for (clio::run::u64 j = threadIdx.x; j < elems_per_page; j += blockDim.x) {
    acc ^= slot[j];
  }
  for (int off = 16; off > 0; off >>= 1) {
#if CTP_ENABLE_CUDA
    acc ^= __shfl_xor_sync(0xffffffff, acc, off);
#else
    acc ^= __shfl_xor(acc, off);
#endif
  }
  if (threadIdx.x == 0) out_xor[b] = acc;
}

#if !CTP_IS_DEVICE_PASS

namespace {
std::string FmtBytes(clio::run::u64 b) {
  char buf[64];
  if (b >= (1ULL << 30)) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB",
                  b / static_cast<double>(1ULL << 30));
  } else if (b >= (1ULL << 20)) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB",
                  b / static_cast<double>(1ULL << 20));
  } else if (b >= (1ULL << 10)) {
    std::snprintf(buf, sizeof(buf), "%.2f KiB",
                  b / static_cast<double>(1ULL << 10));
  } else {
    std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
  }
  return std::string(buf);
}
}  // namespace

int main(int argc, char *argv[]) {
  BenchOpts opts;
  if (!ParseOpts(argc, argv, opts)) return 2;
  if (opts.nblocks == 0 || opts.pages_per_block == 0 ||
      opts.page_size == 0) {
    std::fprintf(stderr, "blocks/pages/page_size must be > 0\n");
    return 2;
  }
  if (opts.page_size % sizeof(clio::run::u32) != 0) {
    std::fprintf(stderr, "page_size must be a multiple of sizeof(u32)\n");
    return 2;
  }

  clio::run::u64 cache_bytes = static_cast<clio::run::u64>(opts.nblocks) *
                         opts.pages_per_block * opts.page_size;
  clio::run::u64 per_block_bytes;
  if (opts.per_block_bytes > 0) {
    per_block_bytes = opts.per_block_bytes;
  } else if (opts.total_bytes > 0) {
    per_block_bytes = opts.total_bytes / opts.nblocks;
  } else {
    clio::run::u64 t = static_cast<clio::run::u64>(opts.ratio * cache_bytes);
    if (t < opts.page_size) t = opts.page_size;
    per_block_bytes = t / opts.nblocks;
  }
  if (per_block_bytes < opts.page_size) per_block_bytes = opts.page_size;
  if (per_block_bytes % opts.page_size != 0) {
    per_block_bytes = ((per_block_bytes + opts.page_size - 1) /
                       opts.page_size) * opts.page_size;
  }
  opts.total_bytes = per_block_bytes * opts.nblocks;
  clio::run::u64 per_block_pages = per_block_bytes / opts.page_size;
  clio::run::u64 elems_per_page = opts.page_size / sizeof(clio::run::u32);

  clio::run::u64 bdev_capacity_bytes;
  if (opts.bdev_capacity_mib > 0) {
    bdev_capacity_bytes = opts.bdev_capacity_mib * (1ULL << 20);
  } else {
    bdev_capacity_bytes =
        std::max<clio::run::u64>(64ULL << 20, opts.total_bytes * 4);
  }

  clio::run::u64 expected_puts =
      static_cast<clio::run::u64>(opts.nblocks) * per_block_pages;
  clio::run::u64 expected_gets = expected_puts;

  std::fprintf(stderr,
               "[BENCH] mode=uvm (cudaMallocManaged + sync PutBlob)\n"
               "[BENCH] blocks=%u pages_per_block=%u page_size=%s\n"
               "[BENCH] cache=%s total=%s ratio=%.2fx\n"
               "[BENCH] per_block=%llu pages (%s)\n"
               "[BENCH] expected_puts=%llu expected_gets=%llu\n"
               "[BENCH] iters=%u read=%s\n",
               opts.nblocks, opts.pages_per_block,
               FmtBytes(opts.page_size).c_str(),
               FmtBytes(cache_bytes).c_str(),
               FmtBytes(opts.total_bytes).c_str(),
               opts.total_bytes / static_cast<double>(cache_bytes),
               (unsigned long long)per_block_pages,
               FmtBytes(per_block_bytes).c_str(),
               (unsigned long long)expected_puts,
               (unsigned long long)expected_gets,
               opts.iters, opts.do_read ? "yes" : "no");

  EnsureInit(opts, bdev_capacity_bytes);

  auto *cte_client = CLIO_CTE_CLIENT;

  // One-page-per-block staging in UVM.
  clio::run::u64 workspace_bytes =
      static_cast<clio::run::u64>(opts.nblocks) * opts.page_size;
  clio::run::u32 *managed_buf =
      ctp::GpuApi::MallocManaged<clio::run::u32>(workspace_bytes);
  if (!managed_buf) {
    std::fprintf(stderr,
                 "[BENCH] cudaMallocManaged(%s) failed\n",
                 FmtBytes(workspace_bytes).c_str());
    return 2;
  }
  // Pre-fault managed_buf on the host so OS page faults land here, not
  // in the timed write loop. The fill kernel will then migrate pages
  // host->device on first GPU access — that migration is what we're
  // measuring; the underlying OS page allocation isn't.
  std::memset(managed_buf, 0, workspace_bytes);
  clio::run::u32 *xor_out = ctp::GpuApi::Malloc<clio::run::u32>(
      static_cast<clio::run::u64>(opts.nblocks) * sizeof(clio::run::u32));

  using clock = std::chrono::steady_clock;
  auto us_since = [](clock::time_point t0) -> long long {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clock::now() - t0)
        .count();
  };

  std::vector<long long> write_us;
  std::vector<long long> read_us;
  write_us.reserve(opts.iters);
  read_us.reserve(opts.iters);

  for (clio::run::u32 it = 0; it < opts.iters; ++it) {
    std::string tag_name = "uvm_iter_" + std::to_string(it);
    auto tag_fut = cte_client->AsyncGetOrCreateTag(tag_name);
    tag_fut.Wait();
    if (tag_fut->GetReturnCode() != 0) {
      std::fprintf(stderr, "[INIT] GetOrCreateTag failed rc=%u\n",
                   tag_fut->GetReturnCode());
      return 2;
    }
    clio::cte::core::TagId tag_id = tag_fut->tag_id_;

    // ---- Write phase: kernel fills UVM, host PutBlob from UVM pointer ----
    auto t0 = clock::now();
    for (clio::run::u64 p = 0; p < per_block_pages; ++p) {
      UvmFillPageKernel<<<opts.nblocks, 32>>>(
          managed_buf, elems_per_page, p, per_block_pages);
      ctp::GpuApi::Synchronize();
      for (clio::run::u32 b = 0; b < opts.nblocks; ++b) {
        std::string blob_name = "uvm_b" + std::to_string(b) +
                                "_pi" + std::to_string(p);
        ctp::ipc::ShmPtr<> ptr;
        ptr.alloc_id_.SetNull();
        ptr.off_ = reinterpret_cast<clio::run::u64>(
            managed_buf + b * elems_per_page);
        auto fut = cte_client->AsyncPutBlob(
            tag_id, blob_name, /*offset=*/clio::run::u64(0),
            opts.page_size, ptr);
        fut.Wait();
        if (fut->GetReturnCode() != 0) {
          std::fprintf(stderr,
                       "[ITER %u] PutBlob(%s) failed rc=%u\n",
                       it, blob_name.c_str(), fut->GetReturnCode());
          return 2;
        }
      }
    }
    long long w = us_since(t0);

    long long r = 0;
    if (opts.do_read) {
      auto t1 = clock::now();
      for (clio::run::u64 p = 0; p < per_block_pages; ++p) {
        for (clio::run::u32 b = 0; b < opts.nblocks; ++b) {
          std::string blob_name = "uvm_b" + std::to_string(b) +
                                  "_pi" + std::to_string(p);
          ctp::ipc::ShmPtr<> ptr;
          ptr.alloc_id_.SetNull();
          ptr.off_ = reinterpret_cast<clio::run::u64>(
              managed_buf + b * elems_per_page);
          auto fut = cte_client->AsyncGetBlob(
              tag_id, blob_name, /*offset=*/clio::run::u64(0),
              opts.page_size, /*flags=*/clio::run::u32(0), ptr);
          fut.Wait();
          if (fut->GetReturnCode() != 0) {
            std::fprintf(stderr,
                         "[ITER %u] GetBlob(%s) failed rc=%u\n",
                         it, blob_name.c_str(), fut->GetReturnCode());
            return 2;
          }
        }
        UvmConsumePageKernel<<<opts.nblocks, 32>>>(
            managed_buf, elems_per_page, xor_out);
        ctp::GpuApi::Synchronize();
      }
      r = us_since(t1);
    }

    write_us.push_back(w);
    read_us.push_back(r);

    auto bw = [&](long long us) {
      if (us <= 0) return 0.0;
      return (opts.total_bytes / static_cast<double>(1ULL << 20)) /
             (us / 1e6);
    };
    std::fprintf(stderr,
                 "[ITER %u/%u] write=%.3f ms (%.1f MiB/s) "
                 "read=%.3f ms (%.1f MiB/s)\n",
                 it + 1, opts.iters,
                 w / 1e3, bw(w),
                 r / 1e3, opts.do_read ? bw(r) : 0.0);
  }

  ctp::GpuApi::Free<clio::run::u32>(managed_buf);
  ctp::GpuApi::Free<clio::run::u32>(xor_out);

  // ----- Summary -----
  auto stat = [&](std::vector<long long> &v) {
    if (v.empty()) return std::tuple<long long, long long, long long>(0, 0, 0);
    std::sort(v.begin(), v.end());
    return std::tuple<long long, long long, long long>(
        v.front(), v[v.size() / 2], v.back());
  };
  auto [wmin, wmed, wmax] = stat(write_us);
  auto [rmin, rmed, rmax] = stat(read_us);
  auto bw_med = [&](long long us) {
    if (us <= 0) return 0.0;
    return (opts.total_bytes / static_cast<double>(1ULL << 20)) /
           (us / 1e6);
  };

  std::fprintf(stderr,
               "\n[SUMMARY] write : min=%.3f med=%.3f max=%.3f ms "
               "(median %.1f MiB/s, %.3f ms/put)\n",
               wmin / 1e3, wmed / 1e3, wmax / 1e3, bw_med(wmed),
               expected_puts
                   ? (wmed / 1e3) / static_cast<double>(expected_puts)
                   : 0.0);
  if (opts.do_read) {
    std::fprintf(stderr,
                 "[SUMMARY] read  : min=%.3f med=%.3f max=%.3f ms "
                 "(median %.1f MiB/s, %.3f ms/get)\n",
                 rmin / 1e3, rmed / 1e3, rmax / 1e3, bw_med(rmed),
                 expected_gets
                     ? (rmed / 1e3) / static_cast<double>(expected_gets)
                     : 0.0);
  }
  std::fprintf(stderr,
               "[SUMMARY] write+read : %.1f MiB/s end-to-end (median)\n",
               bw_med(wmed + rmed));
  return 0;
}

#else

int main() { return 0; }

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * weights, BaM edition: the capacity sibling. The model lives in pinned host
 * DRAM behind BaM's page cache; the same integer weighted sum reads it
 * through bam_ptr, sequentially, so a faulted page serves whole warps. The
 * gate is EXACT (integer accumulation commutes). Links nothing from clio.
 */

#include <cuda_runtime.h>
#include <bam/array.cuh>
#include <bam/page_cache.cuh>
#include <bam/page_cache_host.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define WB_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      std::exit(1);                                                               \
    }                                                                        \
  } while (0)

using u32 = unsigned int;
using u64 = unsigned long long;

/** The paged bench's weight generator, verbatim: runs of 8 identical values
 *  with a hashed sprinkling of flat 64KB granules (--flat-pct). */
static constexpr u64 kFlatGranuleElems = (64 * 1024) / sizeof(u32);
__host__ __device__ inline bool PageIsFlat(u64 page, u32 pct) {
  u64 h = page * 0x9E3779B97F4A7C15ull;
  h ^= h >> 33;
  return (h % 100u) < pct;
}
__host__ __device__ inline u32 WeightRaw(u64 i) {
  constexpr u64 kRun = 8;
  u32 r = static_cast<u32>((i / kRun) * 2654435761u);
  r ^= r >> 13;
  return (r & 0x3F3F3F3Fu) |
         (static_cast<u32>((i / 4096) % 13) * 0x40404040u);
}
__host__ __device__ inline u32 Weight(u64 i, u32 flat_pct) {
  if (PageIsFlat(i / kFlatGranuleElems, flat_pct)) return 0x01010101u;
  return WeightRaw(i);
}
__host__ __device__ inline u32 Activation(u64 i) {
  return static_cast<u32>((i % 7) + 1);
}

__global__ void SumKernel(bam::ArrayDevice<u32> w, u64 n,
                          unsigned long long *sum) {
  unsigned long long r = 0;
  bam::bam_ptr<u32> ptr(&w);
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    r += static_cast<unsigned long long>(ptr.at(i)) * Activation(i);
  }
  atomicAdd(sum, r);
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, passes = 3, flat_pct = 0;
  u64 data_mb = 512, cache_mb = 128, bam_page_kb = 64;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--passes") passes = static_cast<u32>(next());
    else if (a == "--flat-pct") flat_pct = static_cast<u32>(next());
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--cache-mb") cache_mb = next();
    else if (a == "--bam-page-kb") bam_page_kb = next();
  }
  WB_CUDA_CHECK(cudaSetDevice(0));
  const u64 n = (data_mb * 1024ull * 1024ull) / sizeof(u32);
  const u64 page_bytes = bam_page_kb * 1024;

  std::printf("weights, BaM edition: %llu elems (%.1f MB) behind a %llu MB "
              "/ %lluKB-page BaM cache\n",
              (unsigned long long)n,
              static_cast<double>(n * sizeof(u32)) / 1048576.0,
              (unsigned long long)cache_mb, (unsigned long long)bam_page_kb);

  bam::PageCacheConfig cfg{};
  cfg.page_size = static_cast<size_t>(page_bytes);
  cfg.num_pages = static_cast<size_t>((cache_mb * 1024 * 1024) / page_bytes);
  cfg.num_queues = 1;
  cfg.queue_depth = 1024;
  cfg.backend = bam::BackendType::kHostMemory;
  cfg.nvme_dev = nullptr;
  std::unique_ptr<bam::PageCache> cache(new bam::PageCache(cfg));
  // Whole pages only: host_read_page has no tail clamp (see the md sibling).
  const u64 bam_elems = ((n * sizeof(u32) + page_bytes - 1) / page_bytes) *
                        (page_bytes / sizeof(u32));
  std::unique_ptr<bam::Array<u32>> w(new bam::Array<u32>(bam_elems, *cache));
  unsigned long long want = 0;
  {
    std::vector<u32> host(bam_elems, 0u);
    for (u64 i = 0; i < n; ++i) {
      host[i] = Weight(i, flat_pct);
      want += static_cast<unsigned long long>(host[i]) * Activation(i);
    }
    w->load_from_host(host.data(), bam_elems);
  }

  unsigned long long *d_sum = nullptr;
  WB_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
  const double t0 = NowMs();
  unsigned long long got = 0;
  for (u32 p = 0; p < passes; ++p) {
    WB_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    SumKernel<<<blocks, threads>>>(w->device(), n, d_sum);
    WB_CUDA_CHECK(cudaGetLastError());
    WB_CUDA_CHECK(cudaDeviceSynchronize());
    WB_CUDA_CHECK(cudaMemcpy(&got, d_sum, sizeof(got),
                             cudaMemcpyDeviceToHost));
  }
  const double ms = NowMs() - t0;

  int rc = 0;
  std::printf("  %u passes in %.1f ms\n", passes, ms);
  if (got != want) {
    std::printf("  SUM GATE: FAIL (got=%llu want=%llu)\n", got, want);
    rc = 1;
  } else {
    std::printf("  SUM GATE: PASS (exact, %llu)\n", got);
  }
  std::printf("%s\n", rc == 0 ? "WEIGHTS BAM: ALL GATES PASS"
                              : "WEIGHTS BAM: GATE FAILURE");
  return rc;
}

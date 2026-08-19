/* Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * BSD 3-Clause License. */
#ifndef CLIO_GPU_VECTOR_BENCH_MEMCPY_PROBE_H_
#define CLIO_GPU_VECTOR_BENCH_MEMCPY_PROBE_H_

/**
 * Achievable host-to-device bandwidth AT THE BENCHMARK'S OWN PAGE SIZE.
 *
 * A page fault on a host-tier page is, at bottom, one H2D copy of page_bytes.
 * So the ceiling for the paging path is not the link's headline number but
 * what a bare cudaMemcpy of exactly that size sustains, and the gap between
 * the two is the vector's overhead: task submission, slot claim, eviction,
 * completion polling.
 *
 * TRANSFER SIZE IS THE POINT. PCIe efficiency is strongly size-dependent --
 * small transfers are latency-bound and never approach the link rate -- so a
 * single headline figure cannot be compared against a 16 KB-page run and a
 * 4 MB-page run at once. Probing at the configured page size makes the
 * comparison honest at every point of a page-size sweep.
 *
 * BOTH pinned and pageable are measured, and the difference matters here:
 *   - pinned   is the device ceiling; DMA straight from page-locked memory.
 *   - pageable is what the CTE's shm-backed RAM tier actually is, so the
 *     driver stages through an internal bounce buffer. This is the number the
 *     paging path is really competing with.
 * Reporting only the pinned figure would overstate the overhead; reporting
 * only the pageable one would hide how much is lost to the allocation type.
 *
 * Runs before the runtime starts, so it measures an otherwise idle device.
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

struct MemcpyProbe {
  double pinned_gbps = 0.0;
  double pageable_gbps = 0.0;
};

/** Time `iters` H2D copies of `page_bytes` from `host`, in GB/s (1e9). */
static inline double ProbeOne(void *dev, void *host, size_t page_bytes,
                              int iters) {
  // Warm up: the first copies pay context and mapping costs that have nothing
  // to do with the steady-state rate.
  for (int i = 0; i < 8; ++i) {
    if (cudaMemcpy(dev, host, page_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      return 0.0;
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (cudaMemcpy(dev, host, page_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      return 0.0;
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return 0.0;
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  if (s <= 0.0) return 0.0;
  return static_cast<double>(page_bytes) * static_cast<double>(iters) / s / 1e9;
}

/**
 * Probe both paths. Returns zeros rather than failing the run: this is a
 * reference measurement, and losing it must not cost the workload's result.
 */
static inline MemcpyProbe ProbeMemcpyBandwidth(size_t page_bytes) {
  MemcpyProbe r;
  if (page_bytes == 0) return r;
  // Enough iterations that a 1 MB copy is timed over ~100 MB, but capped so a
  // 16 MB page does not spend seconds here.
  int iters = static_cast<int>((128ull << 20) / page_bytes);
  if (iters < 16) iters = 16;
  if (iters > 512) iters = 512;

  void *dev = nullptr;
  if (cudaMalloc(&dev, page_bytes) != cudaSuccess) return r;

  void *hp = nullptr;
  if (cudaHostAlloc(&hp, page_bytes, cudaHostAllocDefault) == cudaSuccess) {
    std::memset(hp, 1, page_bytes);
    r.pinned_gbps = ProbeOne(dev, hp, page_bytes, iters);
    cudaFreeHost(hp);
  }

  void *hq = std::malloc(page_bytes);
  if (hq != nullptr) {
    std::memset(hq, 1, page_bytes);
    r.pageable_gbps = ProbeOne(dev, hq, page_bytes, iters);
    std::free(hq);
  }

  cudaFree(dev);
  return r;
}

#endif  // CLIO_GPU_VECTOR_BENCH_MEMCPY_PROBE_H_

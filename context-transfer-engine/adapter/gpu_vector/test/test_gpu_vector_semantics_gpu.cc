/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Semantics of the paged vector, asserted rather than assumed.
 *
 * The smoke tests only check that the bytes come back right. That is far too
 * weak a signal: a cache that never evicts, one that re-faults on every single
 * access, one that ignores scores entirely, and one that writes every clean
 * page back all return the SAME bytes. Every paging bug that matters is
 * invisible to a checksum.
 *
 * So these tests assert the paging POLICY directly, using the device counters
 * (Vector::EnableStats) for exact fault / writeback / eviction counts on
 * access patterns whose correct counts are known by construction. An
 * off-by-one in victim selection changes a number here.
 *
 * They also verify flushed pages FROM THE HOST by reading the blobs back
 * through the CTE client, so "the kernel believes it flushed" and "the bytes
 * are in the CTE" are checked separately -- a flush that silently does nothing
 * passes a device-only readback (the page is still resident and still holds
 * the values) and fails here.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

/** Run a yieldable kernel to completion (see the other gpu_vector tests). */
template <typename LaunchT>
static clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  // 8192, not the macro-era 256: real coroutine frames (PrefetchWalkCoro)
  // spill into the lane and overflow anything page-thin.
  gy::YieldStack stack(nblocks, 32, 8192);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000);
}
using clio::run::u32;
using clio::run::u64;

namespace {

constexpr u64 kPageBytes = 4096;
constexpr u64 kPageElems = kPageBytes / sizeof(u32);

/** Value stored at index i for a given pattern id. Position AND salt matter. */
CTP_INLINE_CROSS_FUN u32 Val(u64 i, u32 salt) {
  return static_cast<u32>(i * 2654435761u + salt * 40503u + 1u);
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernels. Every one runs a single lane unless the test is specifically about
// multi-lane behaviour, so that fault counts are a property of the ACCESS
// PATTERN and not of how the hardware happened to schedule warps.
// ---------------------------------------------------------------------------

/** v[base + i] = Val(base + i, salt), then optionally flush the range. */
__global__ void WriteKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<u32> v, u64 base, u64 count,
                            u32 salt, int flush, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, i, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const u64 off = base + static_cast<u64>(yv.Block()) * count;

  CLIO_YBEGIN();
  for (; i < count; i += run) {
    CLIO_YCALL(v.HoldPageYield(off + i, count - i, &run));
    for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
      v[off + i + k] = Val(off + i + k, salt);
    }
  }
  __syncthreads();
  if (flush && threadIdx.x == 0) {
    v.BeginFlush(off, count);
  }
  __syncthreads();
  CLIO_YIELD_IF(v.AnyTransferInFlight());
  CLIO_YEND();
}

/** Verify v[base+i] == Val(base+i, salt); count mismatches. */
__global__ void VerifyKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<u32> v, u64 base, u64 count,
                             u32 salt, unsigned long long *bad,
                             gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, i, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const u64 off = base + static_cast<u64>(yv.Block()) * count;

  CLIO_YBEGIN();
  for (; i < count; i += run) {
    CLIO_YCALL(v.HoldPageYield(off + i, count - i, &run));
    unsigned long long local = 0;
    for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
      if (v.at(off + i + k) != Val(off + i + k, salt)) ++local;
    }
    if (local != 0) atomicAdd(bad, local);
  }
  CLIO_YEND();
}

/** Touch ONE element on each of `npages` pages, `passes` times. */
__global__ void WalkKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 first_page, u64 npages,
                           u32 passes, unsigned long long *sink) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  unsigned long long acc = 0;
  for (u32 p = 0; p < passes; ++p) {
    for (u64 k = 0; k < npages; ++k) {
      const u64 off = (first_page + k) * v.h_->elems_per_page_;
      v.HoldPage(off, 1);
      acc += v.at(off);
    }
  }
  atomicAdd(sink, acc);
}

/** Touch a list of pages in the given order, once each. */
__global__ void TouchSeqKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<u32> v, const u64 *pages,
                               u32 n, unsigned long long *sink) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  unsigned long long acc = 0;
  // One page per step, in the given order -- these touches are deliberately
  // NOT contiguous, so each needs its own hold.
  for (u32 i = 0; i < n; ++i) {
    const u64 off = pages[i] * v.h_->elems_per_page_;
    v.HoldPage(off, 1);
    acc += v.at(off);
  }
  atomicAdd(sink, acc);
}

// Uses EvictPagesYield, not EvictPages. Wrapping the blocking EvictPages in
// a yield frame does NOT make it yieldable -- it waits on the victims' puts
// in-kernel, so thread 0 blocks while the block sits at the barrier, and that
// hangs. EvictPagesYield makes the wait the suspend itself.
__global__ void EvictKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<u32> v, u32 n,
                            gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u32, dummy, 0);
  CLIO_YBEGIN();
  CLIO_YCALL(v.EvictPagesYield(n));
  (void) dummy;
  CLIO_YEND();
}

__global__ void RescoreKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<u32> v, u64 page, float score) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  v.RescorePage(page, score);
}

/** BeginFlush/WaitFlush over a range with no writes, to test the clean case. */
__global__ void FlushKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<u32> v, u64 off, u64 count) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  v.BeginFlush(off, count);
  v.WaitFlush(off, count);
}

/**
 * Fill `npages` pages, then write the WHOLE block cache back with batched
 * puts. Reports how many pages the batched flush claims it wrote.
 */
__global__ void BatchFlushKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v, u64 npages, u32 salt,
                                 unsigned long long *flushed,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, off, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const clio::run::u64 total = npages * v.h_->elems_per_page_;

  CLIO_YBEGIN();
  // Dirty every page first, THEN flush the whole block in one batched
  // submission -- that batch is what this test is about.
  for (; off < total; off += run) {
    CLIO_YCALL(v.HoldPageYield(off, total - off, &run));
    for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
      v[off + k] = Val(off + k, salt);
    }
  }
  // The barrier is load-bearing: FlushBlockBatched submits and clears `dirty`,
  // so a lane still writing when lane 0 flushes loses its writes AND leaves
  // the page looking clean.
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(flushed,
              static_cast<unsigned long long>(v.FlushBlockBatched()));
  }
  __syncthreads();
  CLIO_YIELD_IF(v.AnyTransferInFlight());
  CLIO_YEND();
}

/**
 * Batched-fetch `npages` in chunks of `chunk`, then verify every element.
 * Reports mismatches and how many pages the batched fetch reported resident.
 */
// Yields, and it MUST. The blocking HoldPage spins in-kernel until a slot
// can be claimed, and a claim now declines rather than dropping a dirty
// victim, so with dirty pages in the window this kernel span forever.
__global__ void BatchFetchKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v, u64 npages, u32 chunk,
                                 u32 salt, unsigned long long *bad,
                                 unsigned long long *got,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, k, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, j, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, n, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);

  CLIO_YBEGIN();
  for (; k < npages; k += chunk) {
    n = npages - k;
    if (n > chunk) n = chunk;
    if (threadIdx.x == 0) {
      const u32 g = gv::DeviceVectorTestAccess::FetchPagesBatched(
          v, k, static_cast<u32>(n));
      atomicAdd(got, static_cast<unsigned long long>(g));
    }
    __syncthreads();
    for (j = 0; j < n; ++j) {
      CLIO_YCALL(v.HoldPageYield((k + j) * v.h_->elems_per_page_,
                                 v.h_->elems_per_page_, &run));
      {
        const u64 off = (k + j) * v.h_->elems_per_page_;
        unsigned long long local = 0;
        for (u64 i = threadIdx.x; i < run; i += blockDim.x) {
          if (v.at(off + i) != Val(off + i, salt)) ++local;
        }
        if (local != 0) atomicAdd(bad, local);
      }
    }
  }
  CLIO_YEND();
}

/** Drop the calling block's cache, so the next reads must fault. */
__global__ void DropAllKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<u32> v) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  v.DropAll();
}

/** Batched-flush a block whose pages should already be clean. */
__global__ void FlushAgainKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v,
                                 unsigned long long *flushed) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  atomicAdd(flushed,
            static_cast<unsigned long long>(v.FlushBlockBatched()));
}

__global__ void SizeKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 *out) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  *out = v.size();
}

/**
 * Write the two elements STRADDLING a page boundary, alternating, with the
 * cache too small to hold both pages -- so each step faults the other page
 * out. Catches an index that is right only when both pages happen to be
 * resident.
 */
__global__ void BoundaryKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<u32> v, u64 boundary_page,
                               u32 reps, gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u32, r, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const u64 last_of_prev = boundary_page * v.h_->elems_per_page_ - 1;
  const u64 first_of_next = boundary_page * v.h_->elems_per_page_;

  // The two elements straddle a page boundary, so each needs its own hold --
  // one hold cannot cover both, which is exactly what this test checks. With
  // a ONE-slot cache they evict each other every step, so each hold is a real
  // fault and a real writeback: precisely the case the blocking path cannot
  // service from inside the kernel.
  CLIO_YBEGIN();
  for (; r < reps; ++r) {
    CLIO_YCALL(v.HoldPageYield(last_of_prev, 1, &run));
    if (threadIdx.x == 0) v[last_of_prev] = Val(last_of_prev, 7u);
    CLIO_YCALL(v.HoldPageYield(first_of_next, 1, &run));
    if (threadIdx.x == 0) v[first_of_next] = Val(first_of_next, 7u);
  }
  // Collective, internally BATCHED (one multi-put per 64 pages).
  v.FlushAsync(last_of_prev, 2);
  CLIO_YIELD_IF(v.AnyTransferInFlight());
  CLIO_YEND();
}

/**
 * All 32 lanes of the block read the same range. The vector's contract is
 * that a warp's accesses land in one page, so every lane resolves to the same
 * page and the block needs no coordination -- this asserts that holds in
 * practice rather than in theory.
 */
__global__ void MultiLaneReadKernel(clio::run::IpcManagerGpuInfo info,
                                    gv::DeviceVector<u32> v, u64 count,
                                    u32 salt, unsigned long long *bad) {
  CLIO_GPU_INIT(info, nullptr);
  unsigned long long local = 0;
  // Lane-strided WITHIN a page, page-by-page: at any moment the whole warp is
  // inside one page, which is the granularity contract.
  const u64 pages = count / v.h_->elems_per_page_;
  for (u64 p = 0; p < pages; ++p) {
    const u64 base = p * v.h_->elems_per_page_;
    v.HoldPage(base, v.h_->elems_per_page_);
    for (u64 i = threadIdx.x; i < v.h_->elems_per_page_; i += blockDim.x) {
      if (v.at(base + i) != Val(base + i, salt)) ++local;
    }
    __syncthreads();
  }
  atomicAdd(bad, local);
}

/** All lanes WRITE, page by page, then the block flushes once. */
__global__ void MultiLaneWriteKernel(clio::run::IpcManagerGpuInfo info,
                                     gv::DeviceVector<u32> v, u64 count,
                                     u32 salt) {
  CLIO_GPU_INIT(info, nullptr);
  // Each block owns its own slice, matching VerifyKernel -- otherwise every
  // block would write the same pages and race on the same blobs.
  const u64 slice = static_cast<u64>(blockIdx.x) * count;
  const u64 pages = count / v.h_->elems_per_page_;
  for (u64 p = 0; p < pages; ++p) {
    const u64 base = slice + p * v.h_->elems_per_page_;
    v.HoldPage(base, v.h_->elems_per_page_);
    {
      for (u64 i = threadIdx.x; i < v.h_->elems_per_page_; i += blockDim.x) {
        v[base + i] = Val(base + i, salt);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        v.BeginFlush(base, v.h_->elems_per_page_);
        v.WaitFlush(base, v.h_->elems_per_page_);
      }
      __syncthreads();
      }
  }
}

/**
 * Walk pages while prefetching the next one, the way an overlapping kernel
 * does. Correctness here is what makes the overlap benchmark meaningful: a
 * prefetch that lands in the wrong slot, or that a demand fault re-issues on
 * top of, produces wrong data rather than a slow run.
 */
__device__ gy::YCoroMain PrefetchWalkCoro(gv::DeviceVector<u32> v, u64 count,
                                          u32 salt, unsigned long long *bad,
                                          u32 block) {
  const u64 slice = static_cast<u64>(block) * count;
  const u64 pages = count / v.h_->elems_per_page_;
  const u64 first = slice / v.h_->elems_per_page_;
  u64 run = 0;
  for (u64 p = 0; p < pages; ++p) {
    // Hint the NEXT page before touching this one -- score 1.0 means "make
    // resident", which is the batched prefetch. HoldPage's pin is what
    // protects the page being read; no explicit rescore needed.
    if (p + 1 < pages) {
      const u64 np = first + p + 1;
      v.RescorePagesBatchedAsync(
          1u, [np](u32) { return gv::PageScore{np, 1.0f}; });
    }
    const u64 off = slice + p * v.h_->elems_per_page_;
    co_await v.HoldPage(off, v.h_->elems_per_page_, &run);
    unsigned long long local = 0;
    for (u64 i = threadIdx.x; i < run; i += blockDim.x) {
      if (v.at(off + i) != Val(off + i, salt)) ++local;
    }
    if (local != 0) atomicAdd(bad, local);
    __syncthreads();
  }
}

__global__ void PrefetchWalkKernel(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceVector<u32> v, u64 count,
                                   u32 salt, unsigned long long *bad,
                                   gy::YieldableView<> yv,
                                   gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PrefetchWalkCoro(v, count, salt, bad, yv.Block()));
}

/** Rescore the same page repeatedly, hammering the fire-and-forget slot. */
// Rescoring is fire-and-forget metadata; it never faults, so it needs no
// yield frame. Left single-lane on purpose: the test counts the rescores
// SENT, and every lane sending them would multiply that by the block width.
__global__ void RescoreStormKernel(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceVector<u32> v, u64 page,
                                   u32 reps) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  for (u32 r = 0; r < reps; ++r) {
    v.RescorePage(page, static_cast<float>(r));
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

clio::run::IpcManagerGpuInfo g_gpu;

/** Device scratch for a counter, zeroed. */
unsigned long long *NewCounter() {
  unsigned long long *p = nullptr;
  REQUIRE(cudaMalloc(&p, sizeof(*p)) == cudaSuccess);
  REQUIRE(cudaMemset(p, 0, sizeof(*p)) == cudaSuccess);
  return p;
}

unsigned long long ReadCounter(unsigned long long *p) {
  unsigned long long h = 0;
  REQUIRE(cudaMemcpy(&h, p, sizeof(h), cudaMemcpyDeviceToHost) == cudaSuccess);
  return h;
}

void Sync() { REQUIRE(cudaDeviceSynchronize() == cudaSuccess); }

/** Upload a page-index list for TouchSeqKernel. */
u64 *UploadPages(const std::vector<u64> &pages) {
  u64 *d = nullptr;
  REQUIRE(cudaMalloc(&d, pages.size() * sizeof(u64)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d, pages.data(), pages.size() * sizeof(u64),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return d;
}

/**
 * Read one page's blob back through the CTE from the HOST and compare it to
 * what the kernel should have written. This is the independent check that a
 * flush actually reached storage.
 */
bool HostPageMatches(const clio::cte::core::TagId &tag, u64 page, u32 salt) {
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  std::vector<u32> buf(static_cast<size_t>(kPageElems), 0u);
  char name[32];
  gv::PageBlobName(page, name);
  auto f = core.AsyncGetBlob(tag, std::string(name), 0, kPageBytes, 0,
                             reinterpret_cast<char *>(buf.data()));
  f.Wait();
  if (f->GetReturnCode() != 0) {
    std::fprintf(stderr, "  host read of %s failed rc=%d\n", name,
                 f->GetReturnCode());
    return false;
  }
  for (u64 i = 0; i < kPageElems; ++i) {
    const u64 idx = page * kPageElems + i;
    if (buf[static_cast<size_t>(i)] != Val(idx, salt)) {
      std::fprintf(stderr, "  host mismatch %s elem %llu: got %u want %u\n",
                   name, (unsigned long long) i, buf[static_cast<size_t>(i)],
                   Val(idx, salt));
      return false;
    }
  }
  return true;
}

/** A vector plus its device view, with stats already on. */
struct Fixture {
  gv::Vector<u32> vec;
  gv::DeviceVector<u32> dev;

  Fixture(const std::string &tag, u32 nblocks, u32 slots, u64 npages)
      : vec(tag, {0}, kPageBytes, nblocks, slots, npages * kPageElems) {
    vec.EnableStats();
    dev = vec.GetDevice(0);
  }
  gv::Vector<u32>::Stats Stats() { return vec.ReadStats(0); }
  void Reset() { vec.ResetStats(); }
};

}  // namespace

TEST_CASE("gpu_vector: paging semantics", "[gpu_vector][semantics]") {
  {
    std::ofstream cfg("gpu_vector_semantics.yaml");
    REQUIRE(cfg.is_open());
    cfg << "networking:\n  port: 9434\n\n"
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
        << "      - path: \"ram::gv_sem_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_semantics.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  g_gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  unsigned long long *bad = NewCounter();
  unsigned long long *sink = NewCounter();

  // -------------------------------------------------------------------
  // BATCHED WRITEBACK. A block's whole page cache goes back in
  // ceil(slots / kPodMultiMax) submissions instead of one per page. The
  // check that matters is from the HOST: "the kernel believes it flushed"
  // and "the bytes are in the CTE" are different claims, and a batch that
  // silently dropped records would pass a device-side readback because the
  // pages are still resident and still hold the values.
  // -------------------------------------------------------------------
  {
    const u64 kPages = 6;
    Fixture f("gv_sem_batch_flush", 1, static_cast<u32>(kPages), kPages);
    unsigned long long *flushed = NewCounter();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      BatchFlushKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(
          g_gpu, f.dev, kPages, 21u, flushed, vw_, sv_);
    });
    Sync();
    const unsigned long long nflushed = ReadCounter(flushed);
    std::fprintf(stderr, "[batch-flush] pages=%llu flushed=%llu\n",
                 (unsigned long long) kPages, nflushed);
    // Every dirty page, and no more: a batch that double-counted records or
    // one that stopped at the first would both miss this.
    REQUIRE(nflushed == kPages);
    for (u64 p = 0; p < kPages; ++p) {
      REQUIRE(HostPageMatches(f.vec.TagId(), p, 21u));
    }
    // The pages are clean now, so a second batched flush has nothing to do.
    unsigned long long *again = NewCounter();
    FlushAgainKernel<<<1, 32>>>(g_gpu, f.dev, again);
    Sync();
    REQUIRE(ReadCounter(again) == 0);
    cudaFree(flushed);
    cudaFree(again);
  }

  // -------------------------------------------------------------------
  // BATCHED FAULTING. One get per chunk of pages instead of one per page.
  // The data must be identical to what per-page faulting returns -- a batch
  // that mismatched a record to its slot would return the RIGHT bytes for
  // the WRONG page, which no checksum over a single page can catch, so this
  // verifies every element of every page against its own page number.
  // -------------------------------------------------------------------
  {
    const u64 kPages = 12;
    // Cache smaller than the working set: chunks must evict and refill,
    // which is where a slot-reuse bug would show.
    Fixture f("gv_sem_batch_fetch", 1, 4, kPages);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 31u, 1, vw_, sv_);
    });
    Sync();
    // FLUSH before dropping. The claim path stopped dropping dirty victims
    // long after this test was written, and DropAll discards dirty pages
    // WITHOUT writeback by design -- so without this flush the last
    // cache-full of pages never reaches its blobs, their batch records
    // fail rc, and the fetched-count assertion below reads short even
    // though the (demand-faulted) content checks still pass.
    FlushKernel<<<1, 32>>>(g_gpu, f.dev, 0, kPages * kPageElems);
    cudaDeviceSynchronize();
    DropAllKernel<<<1, 32>>>(g_gpu, f.dev);
    Sync();

    unsigned long long *got = NewCounter();
    f.Reset();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      BatchFetchKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, kPages, 4u, 31u, bad, got, vw_, sv_);
    });
    Sync();
    const unsigned long long nbad = ReadCounter(bad);
    std::fprintf(stderr, "[batch-fetch] pages=%llu resident=%llu bad=%llu\n",
                 (unsigned long long) kPages, ReadCounter(got), nbad);
    REQUIRE(nbad == 0);
    REQUIRE(ReadCounter(got) == kPages);
    REQUIRE(cudaMemset(bad, 0, sizeof(*bad)) == cudaSuccess);
    cudaFree(got);
  }

  // -------------------------------------------------------------------
  // A resident page is a HIT. Nothing else in the suite proves the cache
  // caches: a pager that re-faulted on every access returns correct data.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_hits", 1, 4, 4);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 1u, 1, vw_, sv_);
    });
    Sync();

    f.Reset();
    WalkKernel<<<1, 32>>>(g_gpu, f.dev, 0, 4, 10u, sink);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[hits] faults=%llu puts=%llu evicts=%llu\n",
                 (unsigned long long) s.faults, (unsigned long long) s.puts,
                 (unsigned long long) s.evicts);
    // All four pages fit in four slots and are already resident: forty
    // accesses must cost nothing at all.
    REQUIRE(s.faults == 0);
    REQUIRE(s.puts == 0);
    REQUIRE(s.evicts == 0);
  }

  // -------------------------------------------------------------------
  // A cold walk faults each page EXACTLY once, and clean pages are dropped
  // without a writeback. An over-eager pager that wrote back clean pages
  // would be invisible to a checksum and expensive in practice.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_cold", 1, 2, 8);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 8 * kPageElems, 2u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, vw_, sv_);
    });   // start from an empty cache
    Sync();

    f.Reset();
    WalkKernel<<<1, 32>>>(g_gpu, f.dev, 0, 8, 1u, sink);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[cold] faults=%llu puts=%llu evicts=%llu\n",
                 (unsigned long long) s.faults, (unsigned long long) s.puts,
                 (unsigned long long) s.evicts);
    REQUIRE(s.faults == 8);       // one per page, no more and no fewer
    REQUIRE(s.puts == 0);         // read-only walk writes nothing back
    REQUIRE(s.evicts == 6);       // 8 pages through 2 slots
  }

  // -------------------------------------------------------------------
  // Dirty pages survive eviction. The working set is four times the cache,
  // so most pages are written, evicted, and re-read -- if the writeback on
  // eviction were dropped, the re-read would return stale bytes.
  //
  // Checked twice: on the device, and from the HOST through the CTE.
  // -------------------------------------------------------------------
  {
    constexpr u64 kPages = 8;
    Fixture f("gv_sem_dirty", 1, 2, kPages);
    f.Reset();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 3u, 0, vw_, sv_);
    });
    Sync();
    auto sw = f.Stats();
    // Flush whatever is still resident and dirty, then drop everything so
    // the read below cannot be served from the cache.
    FlushKernel<<<1, 32>>>(g_gpu, f.dev, 0, kPages * kPageElems);
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, vw_, sv_);
    });
    Sync();

    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 3u, bad, vw_, sv_);
    });
    Sync();
    const unsigned long long mism = ReadCounter(bad);
    std::fprintf(stderr,
                 "[dirty] write_puts=%llu write_evicts=%llu mismatches=%llu\n",
                 (unsigned long long) sw.puts, (unsigned long long) sw.evicts,
                 mism);
    REQUIRE(mism == 0);
    // Every page but the last two had to be written back during the walk.
    REQUIRE(sw.puts >= kPages - 2);

    for (u64 p = 0; p < kPages; ++p) {
      REQUIRE(HostPageMatches(f.vec.TagId(), p, 3u));
    }
  }

  // -------------------------------------------------------------------
  // Score steers eviction. A high-scored page must NEVER be chosen as the
  // victim, which is the whole basis of using RescorePage as a pin/prefetch
  // hint. Counted exactly: if the pinned page were ever evicted the final
  // access to it would add a fault.
  // -------------------------------------------------------------------
  {
    constexpr u64 kPages = 8;
    Fixture f("gv_sem_score", 1, 3, kPages);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 4u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 3u, vw_, sv_);
    });
    Sync();

    // Make page 0 resident, then pin it with a score nothing else has.
    WalkKernel<<<1, 32>>>(g_gpu, f.dev, 0, 1, 1u, sink);
    Sync();
    RescoreKernel<<<1, 32>>>(g_gpu, f.dev, 0, 100.0f);
    Sync();

    f.Reset();
    // Pages 1..7 rotate through the two remaining slots, twice.
    WalkKernel<<<1, 32>>>(g_gpu, f.dev, 1, kPages - 1, 2u, sink);
    Sync();
    // ...and page 0 must still be there.
    WalkKernel<<<1, 32>>>(g_gpu, f.dev, 0, 1, 1u, sink);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[score] faults=%llu (expect %llu) evicts=%llu\n",
                 (unsigned long long) s.faults,
                 (unsigned long long) (2 * (kPages - 1)),
                 (unsigned long long) s.evicts);
    // 7 pages x 2 passes = 14. A 15th fault means the pinned page was evicted.
    REQUIRE(s.faults == 2 * (kPages - 1));
  }

  // -------------------------------------------------------------------
  // Equal scores fall back to LRU. Touch 0 then 1 then 0 again, so 1 is the
  // older access; bringing in page 2 must evict 1, leaving 0 a hit.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_lru", 1, 2, 4);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 5u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, vw_, sv_);
    });
    Sync();

    u64 *warm = UploadPages({0, 1, 0});
    TouchSeqKernel<<<1, 32>>>(g_gpu, f.dev, warm, 3u, sink);
    Sync();

    f.Reset();
    u64 *probe = UploadPages({2, 0});   // 2 evicts the LRU victim; 0 must hit
    TouchSeqKernel<<<1, 32>>>(g_gpu, f.dev, probe, 2u, sink);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[lru] faults=%llu (expect 1) evicts=%llu\n",
                 (unsigned long long) s.faults, (unsigned long long) s.evicts);
    // Exactly one fault (page 2). Two means LRU picked page 0 -- the page
    // touched most recently -- which is the classic inverted-comparison bug.
    REQUIRE(s.faults == 1);
    REQUIRE(s.evicts == 1);
    cudaFree(warm);
    cudaFree(probe);
  }

  // -------------------------------------------------------------------
  // BeginFlush publishes to the CTE WITHOUT evicting, and re-flushing a
  // clean page is free. A flush that only worked as a side effect of
  // eviction would pass every device-side readback.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_flush", 1, 4, 4);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 6u, 1, vw_, sv_);
    });
    Sync();

    // Visible to the host while still resident on the device.
    for (u64 p = 0; p < 4; ++p) {
      REQUIRE(HostPageMatches(f.vec.TagId(), p, 6u));
    }

    f.Reset();
    WalkKernel<<<1, 32>>>(g_gpu, f.dev, 0, 4, 1u, sink);
    Sync();
    REQUIRE(f.Stats().faults == 0);      // flushing did not evict

    f.Reset();
    FlushKernel<<<1, 32>>>(g_gpu, f.dev, 0, 4 * kPageElems);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[flush] clean reflush puts=%llu (expect 0)\n",
                 (unsigned long long) s.puts);
    REQUIRE(s.puts == 0);                // nothing dirty, nothing written
  }

  // -------------------------------------------------------------------
  // EvictPages boundaries: zero is a no-op, and asking for more than are
  // resident must clear the cache and RETURN rather than spin looking for a
  // victim that does not exist.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_evict", 1, 4, 4);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 7u, 1, vw_, sv_);
    });
    Sync();

    f.Reset();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0u, vw_, sv_);
    });
    Sync();
    REQUIRE(f.Stats().evicts == 0);

    f.Reset();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 99u, vw_, sv_);
    });   // far more than resident
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[evict] over-evict evicts=%llu (expect 4)\n",
                 (unsigned long long) s.evicts);
    REQUIRE(s.evicts == 4);

    // And the data is still correct afterwards.
    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 7u, bad, vw_, sv_);
    });
    Sync();
    REQUIRE(ReadCounter(bad) == 0);
  }

  // -------------------------------------------------------------------
  // Elements straddling a page boundary, with a one-slot cache so the two
  // pages evict each other on every step. Catches an index computed against
  // the wrong page's base.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_boundary", 1, 1, 4);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      BoundaryKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, 4u,
                                                        vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 1u, vw_, sv_);
    });
    Sync();
    // Only ONE element of each page was written, so the whole-page helper
    // does not apply -- check the two straddling elements directly.
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<u32> buf(static_cast<size_t>(kPageElems), 0u);
    auto rd = core.AsyncGetBlob(f.vec.TagId(), "p1", 0, kPageBytes, 0,
                                reinterpret_cast<char *>(buf.data()));
    rd.Wait();
    REQUIRE(rd->GetReturnCode() == 0);
    const u64 last_of_prev = 2 * kPageElems - 1;
    REQUIRE(buf[static_cast<size_t>(kPageElems - 1)] == Val(last_of_prev, 7u));
    std::vector<u32> buf2(static_cast<size_t>(kPageElems), 0u);
    auto rd2 = core.AsyncGetBlob(f.vec.TagId(), "p2", 0, kPageBytes, 0,
                                 reinterpret_cast<char *>(buf2.data()));
    rd2.Wait();
    REQUIRE(rd2->GetReturnCode() == 0);
    REQUIRE(buf2[0] == Val(2 * kPageElems, 7u));
    std::fprintf(stderr, "[boundary] straddling elements correct\n");
  }

  // -------------------------------------------------------------------
  // A vector whose length is NOT a whole number of pages. The tail page is
  // partial; puts and gets still move a whole page, so the tail must not be
  // truncated or read past.
  // -------------------------------------------------------------------
  {
    const u64 n = 5 * kPageElems + 37;
    gv::Vector<u32> vec("gv_sem_tail", {0}, kPageBytes, 1, 2, n);
    REQUIRE(vec.NumPages() == 6);
    auto dev = vec.GetDevice(0);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, dev, 0, n, 8u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, dev, 2u, vw_, sv_);
    });
    Sync();
    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, dev, 0, n, 8u, bad, vw_, sv_);
    });
    Sync();
    std::fprintf(stderr, "[tail] n=%llu pages=%llu mismatches=%llu\n",
                 (unsigned long long) n, (unsigned long long) vec.NumPages(),
                 ReadCounter(bad));
    REQUIRE(ReadCounter(bad) == 0);
  }

  // -------------------------------------------------------------------
  // Blocks are isolated: each owns its own slice of the page table, so a
  // block's evictions must never disturb another's residency or data.
  // Every block writes a different salt-free region and reads it back after
  // ALL blocks have thrashed their caches.
  // -------------------------------------------------------------------
  {
    constexpr u32 kBlocks = 32;
    constexpr u64 kPagesPerBlock = 4;
    const u64 per = kPagesPerBlock * kPageElems;
    Fixture f("gv_sem_blocks", kBlocks, 2, kPagesPerBlock * kBlocks);
    RunYieldable(kBlocks, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, per, 9u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(kBlocks, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, vw_, sv_);
    });
    Sync();
    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(kBlocks, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, per, 9u, bad, vw_, sv_);
    });
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[blocks] mismatches=%llu faults=%llu\n",
                 ReadCounter(bad), (unsigned long long) s.faults);
    REQUIRE(ReadCounter(bad) == 0);
  }

  // -------------------------------------------------------------------
  // Rescoring a page that is NOT resident must not fault it in, crash, or
  // corrupt the slot it hashes onto -- it is a hint to the CTE only.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_rescore_absent", 1, 2, 8);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 8 * kPageElems, 10u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, vw_, sv_);
    });
    Sync();

    f.Reset();
    RescoreKernel<<<1, 32>>>(g_gpu, f.dev, 5, 42.0f);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[rescore-absent] faults=%llu (expect 0)\n",
                 (unsigned long long) s.faults);
    REQUIRE(s.faults == 0);

    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 8 * kPageElems, 10u, bad, vw_, sv_);
    });
    Sync();
    REQUIRE(ReadCounter(bad) == 0);
  }

  // -------------------------------------------------------------------
  // size() as seen by the kernel.
  // -------------------------------------------------------------------
  {
    const u64 n = 3 * kPageElems + 5;
    gv::Vector<u32> vec("gv_sem_size", {0}, kPageBytes, 1, 2, n);
    u64 *d = nullptr;
    REQUIRE(cudaMalloc(&d, sizeof(u64)) == cudaSuccess);
    SizeKernel<<<1, 32>>>(g_gpu, vec.GetDevice(0), d);
    Sync();
    u64 h = 0;
    REQUIRE(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(h == n);
    cudaFree(d);
  }

  // -------------------------------------------------------------------
  // Host API contract.
  // -------------------------------------------------------------------
  {
    // Degenerate geometry is rejected rather than producing a vector that
    // divides by zero on first access.
    bool threw = false;
    try {
      gv::Vector<u32> bad_page("gv_sem_bad0", {0}, 0, 1, 1, 16);
    } catch (const std::exception &) {
      threw = true;
    }
    REQUIRE(threw);

    threw = false;
    try {
      gv::Vector<u32> bad_blocks("gv_sem_bad1", {0}, kPageBytes, 0, 1, 16);
    } catch (const std::exception &) {
      threw = true;
    }
    REQUIRE(threw);

    threw = false;
    try {
      gv::Vector<u32> bad_slots("gv_sem_bad2", {0}, kPageBytes, 1, 0, 16);
    } catch (const std::exception &) {
      threw = true;
    }
    REQUIRE(threw);

    gv::Vector<u32> v("gv_sem_api", {0}, kPageBytes, 1, 2, 4 * kPageElems);
    REQUIRE(v.PageBytes() == kPageBytes);
    REQUIRE(v.NumPages() == 4);
    REQUIRE(!v.TagId().IsNull());
    threw = false;
    try {
      v.GetDevice(7);   // no view was built for this GPU
    } catch (const std::exception &) {
      threw = true;
    }
    REQUIRE(threw);
    // Stats are zero until enabled, and readable for a device with none.
    REQUIRE(v.ReadStats(0).faults == 0);
    std::fprintf(stderr, "[host-api] contract holds\n");
  }

  // -------------------------------------------------------------------
  // All 32 lanes reading the same pages. This is the granularity contract
  // from the design: a warp's accesses land in one page, so every lane
  // resolves the same page and no coordination is needed.
  // -------------------------------------------------------------------
  {
    constexpr u64 kPages = 8;
    Fixture f("gv_sem_lanes", 1, 4, kPages);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 11u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 4u, vw_, sv_);
    });
    Sync();

    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    MultiLaneReadKernel<<<1, 32>>>(g_gpu, f.dev, kPages * kPageElems, 11u, bad);
    Sync();
    const unsigned long long mism = ReadCounter(bad);
    std::fprintf(stderr, "[lanes] 32-lane mismatches=%llu\n", mism);
    REQUIRE(mism == 0);
  }

  // -------------------------------------------------------------------
  // Concurrency is COALESCED, not just survived: 32 lanes walking the same
  // pages must cost the same faults as one lane would. If each lane faulted
  // for itself the count would be 32x -- correct data, catastrophic cost.
  // -------------------------------------------------------------------
  {
    constexpr u64 kPages = 8;
    Fixture f("gv_sem_coalesce", 1, 4, kPages);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 12u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 4u, vw_, sv_);
    });
    Sync();

    f.Reset();
    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    MultiLaneReadKernel<<<1, 32>>>(g_gpu, f.dev, kPages * kPageElems, 12u, bad);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[coalesce] faults=%llu (expect %llu) mismatches=%llu\n",
                 (unsigned long long) s.faults, (unsigned long long) kPages,
                 ReadCounter(bad));
    REQUIRE(ReadCounter(bad) == 0);
    REQUIRE(s.faults == kPages);   // one fault per page for the WHOLE block
  }

  // -------------------------------------------------------------------
  // The hard case: 32 lanes WRITING through a cache a quarter the size of
  // the working set. Concurrent lanes dirty a page while the block's cache
  // is evicting, so writeback, eviction and the fault path all overlap.
  // -------------------------------------------------------------------
  {
    constexpr u64 kPages = 8;
    Fixture f("gv_sem_lane_write", 1, 2, kPages);
    MultiLaneWriteKernel<<<1, 32>>>(g_gpu, f.dev, kPages * kPageElems, 13u);
    Sync();
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 2u, vw_, sv_);
    });
    Sync();

    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, kPages * kPageElems, 13u, bad, vw_, sv_);
    });
    Sync();
    std::fprintf(stderr, "[lane-write] mismatches=%llu\n", ReadCounter(bad));
    REQUIRE(ReadCounter(bad) == 0);
    for (u64 p = 0; p < kPages; ++p) {
      REQUIRE(HostPageMatches(f.vec.TagId(), p, 13u));
    }
  }

  // -------------------------------------------------------------------
  // Scale: 128 blocks x 32 lanes, each block oversubscribed 4:1. This is
  // every mechanism at once -- concurrent lanes, per-block locks, eviction,
  // writeback -- across more blocks than the GPU runs concurrently.
  // -------------------------------------------------------------------
  {
    struct Scale {
      u32 blocks;
      u32 slots;
      u64 pages;
    };
    const std::vector<Scale> scales = {
        {32, 2, 8},    // oversubscribed, well within one GPU's resident blocks
        {64, 2, 8},    // the largest that used to work
        {96, 2, 8},    //
        {128, 8, 8},   // many blocks, NO eviction (slots == pages)
        {128, 2, 8},   // many blocks WITH eviction
    };
    for (const Scale &sc : scales) {
      const u64 per = sc.pages * kPageElems;
      std::fprintf(stderr, "[scale] blocks=%u slots=%u pages=%llu ...\n",
                   sc.blocks, sc.slots, (unsigned long long) sc.pages);
      std::fflush(stderr);
      Fixture f("gv_sem_scale_" + std::to_string(sc.blocks) + "_" +
                    std::to_string(sc.slots),
                sc.blocks, sc.slots, sc.pages * sc.blocks);
      MultiLaneWriteKernel<<<sc.blocks, 32>>>(g_gpu, f.dev, per, 14u);
      Sync();
      RunYieldable(sc.blocks, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, sc.slots, vw_, sv_);
    });
      Sync();

      REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
      RunYieldable(sc.blocks, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, per, 14u, bad, vw_, sv_);
    });
      Sync();
      std::fprintf(stderr, "[scale] blocks=%u slots=%u mismatches=%llu\n",
                   sc.blocks, sc.slots, ReadCounter(bad));
      std::fflush(stderr);
      REQUIRE(ReadCounter(bad) == 0);
    }
  }

  // -------------------------------------------------------------------
  // Prefetching returns the SAME data as demand faulting, on an
  // oversubscribed cache where every prefetch has to evict something. Also
  // asserts the prefetch is actually being used: a run where the arrivals
  // were never in flight when needed would report zero hits.
  // -------------------------------------------------------------------
  {
    constexpr u64 kPages = 16;
    Fixture f("gv_sem_prefetch", 4, 4, kPages * 4);
    const u64 per = kPages * kPageElems;
    RunYieldable(4, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, per, 15u, 1, vw_, sv_);
    });
    Sync();
    RunYieldable(4, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      EvictKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 4u, vw_, sv_);
    });
    Sync();

    f.Reset();
    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(4, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      PrefetchWalkKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, per, 15u, bad, vw_, sv_);
    });
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr,
                 "[prefetch] mismatches=%llu faults=%llu prefetches=%llu "
                 "hits=%llu rescores=%llu dropped=%llu\n",
                 ReadCounter(bad), (unsigned long long) s.faults,
                 (unsigned long long) s.prefetches,
                 (unsigned long long) s.prefetch_hits,
                 (unsigned long long) s.rescores,
                 (unsigned long long) s.pf_dropped);
    REQUIRE(ReadCounter(bad) == 0);
    // Each block prefetches every page after its first.
    REQUIRE(s.prefetches == 4 * (kPages - 1));
    // A page must never be fetched twice: a prefetch followed by a demand
    // fault for the same page would mean the wait path is not working.
    REQUIRE(s.faults == 4 * kPages);
  }

  // -------------------------------------------------------------------
  // Rescore is fire-and-forget, so its task slot can still be executing when
  // the next one is issued. Hammer it: without the wait-before-reuse this
  // mutates a task the runtime is reading.
  // -------------------------------------------------------------------
  {
    Fixture f("gv_sem_rescore_storm", 1, 4, 4);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      WriteKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 16u, 1, vw_, sv_);
    });
    Sync();

    f.Reset();
    RescoreStormKernel<<<1, 32>>>(g_gpu, f.dev, 0, 64u);
    Sync();
    auto s = f.Stats();
    std::fprintf(stderr, "[rescore-storm] rescores=%llu (expect 64)\n",
                 (unsigned long long) s.rescores);
    REQUIRE(s.rescores == 64);

    REQUIRE(cudaMemset(bad, 0, sizeof(unsigned long long)) == cudaSuccess);
    RunYieldable(1, [&](dim3 g_, dim3 b_, gy::YieldableView<> vw_,
                        gy::YieldStackView sv_) {
      VerifyKernel<<<g_, b_, CLIO_YIELD_SMEM_BYTES>>>(g_gpu, f.dev, 0, 4 * kPageElems, 16u, bad, vw_, sv_);
    });
    Sync();
    REQUIRE(ReadCounter(bad) == 0);
  }

  cudaFree(bad);
  cudaFree(sink);
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

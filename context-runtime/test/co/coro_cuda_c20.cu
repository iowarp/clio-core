/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * THE THIRD EDITION: the same workload as a C++20 device coroutine.
 *
 * coro_cuda.cu measures the transpiled state machine against the synchronous
 * source. That answers "what does the mechanism cost", but not the question
 * that decides whether to refactor anything, which is "does it cost less than
 * what we have". This file is the incumbent, on the identical workload:
 * StreamTile > HoldPage > Fetch / Flush, the same three-level chain with the
 * same four suspend points, written with `co_await`.
 *
 * WHY THE CHAIN AND NOT A MICROBENCHMARK. The delta pipeline's finding is that
 * a coroutine kernel's register count is near-INDEPENDENT of its body, because
 * NVPTX has no tail calls: CoroSplit merges every resume segment into a single
 * function and the allocator then takes the liveness UNION across all suspend
 * points. If that is the mechanism, this six-suspend-point toy should land
 * near the 192 registers the real paged benches measure, and the comparison
 * generalises. If it lands near 22, the mechanism is something else and the
 * comparison does not. Either outcome is worth knowing before refactoring
 * 39,000 lines of benchmark.
 *
 * THE COROUTINE SHIM. nvcc's <coroutine> is host-only -- coroutine_handle's
 * members are not __device__ -- so the handle is declared here against the
 * compiler builtins, which is the same technique test/unit/gpu/spike uses.
 * Promise frames come from a fixed per-lane arena rather than the heap, again
 * as the existing spikes do, because a device `operator new` would price the
 * allocator rather than the lowering.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "coro_cuda_report.h"

/* ---------------------------------------------------------------------------
 * The device-side coroutine_handle. Declared in namespace std because that is
 * where the language looks for it; nothing here is a reimplementation of the
 * library, only the three builtins the lowering emits calls to.
 * ------------------------------------------------------------------------- */
namespace std {
template <typename Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  void *ptr_ = nullptr;
  __host__ __device__ constexpr coroutine_handle() noexcept {}
  __host__ __device__ constexpr coroutine_handle(decltype(nullptr)) noexcept {}
  __host__ __device__ static coroutine_handle from_address(void *a) noexcept {
    coroutine_handle h;
    h.ptr_ = a;
    return h;
  }
  __host__ __device__ void *address() const noexcept { return ptr_; }
  __host__ __device__ void resume() const { __builtin_coro_resume(ptr_); }
  __host__ __device__ void destroy() const { __builtin_coro_destroy(ptr_); }
  __host__ __device__ bool done() const { return __builtin_coro_done(ptr_); }
  __host__ __device__ explicit operator bool() const { return ptr_ != nullptr; }
};

template <typename Promise>
struct coroutine_handle : coroutine_handle<void> {
  __host__ __device__ static coroutine_handle from_promise(Promise &p) {
    coroutine_handle h;
    h.ptr_ = __builtin_coro_promise(reinterpret_cast<char *>(&p),
                                    alignof(Promise), true);
    return h;
  }
  __host__ __device__ static coroutine_handle from_address(void *a) noexcept {
    coroutine_handle h;
    h.ptr_ = a;
    return h;
  }
  __host__ __device__ Promise &promise() const {
    return *reinterpret_cast<Promise *>(
        __builtin_coro_promise(ptr_, alignof(Promise), false));
  }
};

template <typename R, typename...>
struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

struct suspend_always {
  __host__ __device__ bool await_ready() const noexcept { return false; }
  __host__ __device__ void await_suspend(coroutine_handle<>) const noexcept {}
  __host__ __device__ void await_resume() const noexcept {}
};

struct suspend_never {
  __host__ __device__ bool await_ready() const noexcept { return true; }
  __host__ __device__ void await_suspend(coroutine_handle<>) const noexcept {}
  __host__ __device__ void await_resume() const noexcept {}
};
}  // namespace std

namespace {

using u32 = unsigned;
using u64 = unsigned long long;

/* The geometry coro_cuda.cu uses, so the two runs are directly comparable. */
constexpr u64 kPages = 6;
constexpr u64 kPer = 32;
constexpr u32 kGroups = 3;
constexpr u32 kLanes = 8;

/* Per-THREAD frame arena, as test/unit/gpu/spike does it: a coroutine frame
 * holds that thread's locals, so it cannot be shared across a block. Sized
 * for far more threads than these tests launch, because the numbers being
 * collected are static kernel attributes and an arena overrun would corrupt a
 * neighbour rather than report anything. */
constexpr u32 kMaxLanes = 4096;
constexpr u32 kLaneBytes = 512;

__device__ char g_arena[(std::size_t)kMaxLanes * kLaneBytes];
__device__ u32 g_off[kMaxLanes];
/** Bump cursor as it stood before the frame now being constructed.
 *
 * The chain is a stack, so a frame must be released when its call returns --
 * StreamTile's loop makes six HoldPage and six Flush calls, and without a
 * release the cursor walks out of the thread's slice and into its
 * neighbour's. `operator new` runs before the promise exists, so it parks the
 * value here and the promise constructor takes it. */
__device__ u32 g_pending[kMaxLanes];
/** Innermost live coroutine per thread: what the driver resumes next. */
__device__ void *g_cur[kMaxLanes];
/** Set once a thread's whole chain has run off its end. */
__device__ int g_done[kMaxLanes];

/** This thread's slot index. */
__device__ __forceinline__ u32 Tid() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

/* ---------------------------------------------------------------------------
 * The coroutine type.
 *
 * NO SYMMETRIC TRANSFER. Returning a handle from await_suspend lowers to a
 * tail call, and the delta pipeline's register finding turns on NVPTX not
 * having those. So the chain is an explicit stack instead: await_suspend
 * records the caller, publishes the callee as "resume this next", and returns
 * true. The kernel below pops the stack. That is also what makes a park a
 * plain kernel exit, exactly as in the transpiled edition.
 * ------------------------------------------------------------------------- */
struct Task {
  struct promise_type {
    void *caller_ = nullptr;
    u32 saved_off_ = 0;

    __device__ promise_type() : saved_off_(g_pending[Tid()]) {}

    __device__ Task get_return_object() {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    __device__ std::suspend_always final_suspend() noexcept { return {}; }
    __device__ void return_void() {}
    __device__ void unhandled_exception() {}

    __device__ void *operator new(std::size_t n) {
      const u32 l = Tid();
      const u32 o = g_off[l];
      g_pending[l] = o;
      g_off[l] = o + static_cast<u32>((n + 15) & ~std::size_t{15});
      return g_arena + static_cast<std::size_t>(l) * kLaneBytes + o;
    }
    __device__ void operator delete(void *, std::size_t) {}
  };

  std::coroutine_handle<promise_type> h_;

  __device__ bool await_ready() const noexcept { return false; }
  __device__ bool await_suspend(std::coroutine_handle<> caller) noexcept {
    h_.promise().caller_ = caller.address();
    g_cur[Tid()] = h_.address();
    return true;
  }
  __device__ void await_resume() const noexcept {}
};

/* ---------------------------------------------------------------------------
 * The workload, structurally identical to coro_workload.h: the same
 * three-level chain, the same four suspend points, the same barrier between
 * two of them.
 * ------------------------------------------------------------------------- */
struct PageCache {
  float *data;
  unsigned *resident;
  unsigned *flushed;
  u64 npages;
  u64 per;
  __device__ float *Find(u64 page) const {
    return resident[page] ? data + page * per : nullptr;
  }
};

/** Where a parked block records what it is waiting for. */
struct Park {
  int *want;
  u64 *tag;
};

/** Block-collective park decision, the analogue of Ctx::Any.
 *  @param p     this block's park slot
 *  @param need  whether this thread must wait
 *  @param tag   what to wait for; written by thread 0
 *  @return true if the whole block must suspend */
__device__ __forceinline__ bool ParkVote(Park p, bool need, u64 tag) {
  const bool any = __syncthreads_or(need ? 1 : 0) != 0;
  if (any && threadIdx.x == 0) {
    *p.want = 1;
    *p.tag = tag;
  }
  __syncthreads();
  return any;
}

__device__ Task Fetch(PageCache c, u64 page, Park p) {
  while (ParkVote(p, c.resident[page] == 0, page)) {
    co_await std::suspend_always{};
  }
}

__device__ Task Flush(PageCache c, u64 page, Park p) {
  while (ParkVote(p, c.flushed[page] == 0, page | (1ull << 32))) {
    co_await std::suspend_always{};
  }
}

__device__ Task HoldPage(PageCache c, u64 page, Park p, float **out) {
  float *q = c.Find(page);
  if (q == nullptr) {
    co_await Fetch(c, page, p);
    q = c.Find(page);
  }
  *out = q;
}

__device__ Task StreamTile(PageCache c, float *out, u64 npages, u32 lane,
                           u32 nlanes, Park p) {
  for (u64 i = 0; i < npages; ++i) {
    {
      float *q = nullptr;
      co_await HoldPage(c, i, p, &q);
      for (u64 k = lane; k < c.per; k += nlanes) {
        out[i * c.per + k] = q[k] * 2.0f + static_cast<float>(i);
      }
    }
    __syncthreads();
    co_await Flush(c, i, p);
  }
}

/* ------------------------------------------------------------------------- */
#if defined(CORO_CUDA_LB_THREADS) && defined(CORO_CUDA_LB_BLOCKS)
#define CORO_LB_APPLY(t, b) __launch_bounds__(t, b)
#define CORO_CUDA_LB CORO_LB_APPLY(CORO_CUDA_LB_THREADS, CORO_CUDA_LB_BLOCKS)
#else
#define CORO_CUDA_LB
#endif

/**
 * The kernel: build the chain on the first entry, then pop the explicit stack
 * until the block parks or the chain runs out.
 *
 * Every thread walks its own chain, but the park decision inside ParkVote is
 * block-collective, so all threads leave the loop on the same iteration and
 * the __syncthreads inside the workload is always reached uniformly.
 */
__global__ CORO_CUDA_LB void StreamKernelC20(PageCache cache, float *out,
                                             u64 npages, int *want, u64 *tag) {
  const u32 l = Tid();
  if (g_done[l]) return;
  Park p{want + blockIdx.x, tag + blockIdx.x};
  if (threadIdx.x == 0) *p.want = 0;
  __syncthreads();

  if (g_cur[l] == nullptr) {
    g_off[l] = 0;
    Task t = StreamTile(cache, out + blockIdx.x * npages * cache.per, npages,
                        threadIdx.x, blockDim.x, p);
    g_cur[l] = t.h_.address();
  }

  for (;;) {
    auto h = std::coroutine_handle<Task::promise_type>::from_address(g_cur[l]);
    h.resume();
    if (*p.want != 0) return;  // parked: resume this same handle next launch
    if (!h.done()) continue;   // pushed a callee; g_cur now points at it
    // Returned normally: pop the frame, releasing its arena bytes, and
    // resume whoever awaited it.
    void *caller = h.promise().caller_;
    g_off[l] = h.promise().saved_off_;
    h.destroy();
    g_cur[l] = caller;
    if (caller == nullptr) {
      g_done[l] = 1;
      return;
    }
  }
}

/** Measure this edition's kernel with the shared reporter. */
void ReportStatic() {
  clio::co::report::ReportStatic("cuda-c20", StreamKernelC20);
}

/** Occupancy sweep for this edition's kernel, with the shared reporter. */
void ReportOccupancy() {
  clio::co::report::ReportOccupancy("cuda-c20", StreamKernelC20);
}

}  // namespace

int main() {
  float *data = nullptr;
  unsigned *resident = nullptr;
  unsigned *flushed = nullptr;
  float *out = nullptr;
  int *want = nullptr;
  u64 *tag = nullptr;

  CUDA_CHECK(cudaMallocManaged(&data, kPages * kPer * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&resident, kPages * sizeof(unsigned)));
  CUDA_CHECK(cudaMallocManaged(&flushed, kPages * sizeof(unsigned)));
  CUDA_CHECK(cudaMallocManaged(&out, kGroups * kPages * kPer * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&want, kGroups * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&tag, kGroups * sizeof(u64)));

  for (u64 i = 0; i < kPages * kPer; ++i) data[i] = static_cast<float>(i) * 0.5f;
  for (u64 i = 0; i < kPages; ++i) {
    resident[i] = 0u;
    flushed[i] = 0u;
  }
  for (u64 i = 0; i < kGroups * kPages * kPer; ++i) out[i] = -1.0f;
  for (u32 g = 0; g < kGroups; ++g) {
    want[g] = 0;
    tag[g] = 0;
  }

  u64 launches = 0;
  u64 fetches = 0;
  u64 flushes = 0;
  int rc = 0;
  for (;;) {
    StreamKernelC20<<<kGroups, kLanes>>>(
        PageCache{data, resident, flushed, kPages, kPer}, out, kPages, want,
        tag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ++launches;

    bool pending = false;
    for (u32 g = 0; g < kGroups; ++g) {
      if (want[g] == 0) continue;
      pending = true;
      const u64 t = tag[g];
      if ((t & (1ull << 32)) != 0) {
        flushed[t & 0xffffffffu] = 1u;
        ++flushes;
      } else {
        resident[t] = 1u;
        ++fetches;
      }
    }
    if (!pending) break;
    if (launches > 1000) {
      std::fprintf(stderr, "cuda-c20: runaway relaunch loop\n");
      rc = 1;
      break;
    }
  }

  for (u64 i = 0; i < kGroups * kPages * kPer; ++i) {
    std::printf("%.9g\n", static_cast<double>(out[i]));
  }
  std::fprintf(stderr, "cuda-c20: launches=%llu fetches=%llu flushes=%llu\n",
               launches, fetches, flushes);

  ReportStatic();
  ReportOccupancy();

  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaFree(resident));
  CUDA_CHECK(cudaFree(flushed));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(want));
  CUDA_CHECK(cudaFree(tag));
  return rc;
}

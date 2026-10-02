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
 * The transpiler's output, on CUDA -- and the occupancy it costs.
 *
 * This is the leg the branch called out as unexercised: the CUDA/HIP arm of
 * `Item` was written from the SYCL one and never compiled. It answers two
 * separate questions, and keeping them separate matters:
 *
 *   SEMANTICS   the transpiled state machine on a real NVIDIA GPU must print
 *               exactly what the synchronous source prints, from a 0xA5
 *               poisoned stack, with the same launch/park statistics as the
 *               host and SYCL runs.
 *   OCCUPANCY   what the mechanism costs in registers, and therefore in
 *               resident warps. The design claims locals stay in registers and
 *               the device-memory stack is touched only at an actual suspend;
 *               the register count of the transpiled kernel against the same
 *               kernel with CO_AWAIT as the identity is that claim, measured.
 *
 * Built twice from this one file, which is why the only `#if` is on which
 * edition is being built and never on the backend:
 *
 *   -DCORO_CUDA_REF=1  + -I test/co        the synchronous source. Everything
 *                                          resident, one launch, no parking.
 *   -DCORO_CUDA_REF=0  + -I <mirror> first the TRANSPILED header.
 *
 * The two stdout streams must be identical. Compare them with diff, exactly as
 * coro_ref and coro_gen are compared on the host.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <clio_runtime/co/coro.h>
#include <clio_runtime/co/driver.h>

#include "coro_cuda_report.h"
#include "coro_types.h"
#include <coro_workload.h>  // ORIGINAL or TRANSPILED, by include order alone

namespace co = clio::co;
namespace d = clio::co::demo;

/* Launch bounds, off unless asked for.
 *
 * The delta pipeline established that `__launch_bounds__(256, 4)` is what cuts
 * the C++20 coroutine paged kernels from 192 registers to 64, with no local
 * spill, so the same annotation is the only fair way to ask whether the
 * transpiled state machine reaches the same occupancy target. Turn it on with
 * -DCORO_CUDA_LB_THREADS=256 -DCORO_CUDA_LB_BLOCKS=4. */
#if defined(CORO_CUDA_LB_THREADS) && defined(CORO_CUDA_LB_BLOCKS)
#define CORO_LB_APPLY(t, b) __launch_bounds__(t, b)
#define CORO_CUDA_LB CORO_LB_APPLY(CORO_CUDA_LB_THREADS, CORO_CUDA_LB_BLOCKS)
#else
#define CORO_CUDA_LB
#endif

namespace {

/* The geometry is the one the host and SYCL runs use, so the statistics line
 * is directly comparable across all three backends. Occupancy is reported
 * separately over realistic block sizes, because eight threads per block says
 * nothing about a GPU. */
constexpr co::u64 kPages = 6;
constexpr co::u64 kPer = 32;
constexpr co::u32 kGroups = 3;
constexpr co::u32 kLanes = 8;
constexpr co::u32 kStackBytes = 512;

/**
 * The kernel under test: one work-group per block, one work-item per thread.
 *
 * Scope retires a group that already finished, so a relaunched grid costs a
 * block launch and nothing else for the groups that are done. Note there is no
 * backend token below the Scope: the call into the workload is the same call
 * the SYCL and host drivers make.
 */
__global__ CORO_CUDA_LB void StreamKernel(co::StackView view,
                                          d::PageCache cache, float *out,
                                          co::u64 npages) {
#if CORO_CUDA_REF
  (void)view;
  d::StreamTile(cache, out + blockIdx.x * npages * cache.per, npages,
                threadIdx.x, blockDim.x, co::Item{});
#else
  co::Scope scope(view);
  if (scope) {
    d::StreamTile(cache, out + blockIdx.x * npages * cache.per, npages,
                  threadIdx.x, blockDim.x, co::Item{}, scope.Context());
  }
#endif
}

/** Measure this edition's kernel with the shared reporter. */
void ReportStatic() {
  clio::co::report::ReportStatic(CORO_CUDA_TAG, StreamKernel);
}

/** Occupancy sweep for this edition's kernel, with the shared reporter. */
void ReportOccupancy() {
  clio::co::report::ReportOccupancy(CORO_CUDA_TAG, StreamKernel);
}

}  // namespace

int main(int argc, char **argv) {
  // --sync makes every page resident up front. Under the transpiled header it
  // must still complete in a single launch, which is the self-check that the
  // fast path through a CO_AWAIT is a predicated branch and not a park.
  const bool sync_mode = argc > 1 && std::string(argv[1]) == "--sync";

  // Managed memory, as driver.h documents: Stack lays out and reads headers
  // from the host, the kernel writes frames from the device. Every launch is
  // followed by a full sync, so the WDDM restriction on touching managed
  // memory while a kernel is resident is never approached.
  float *data = nullptr;
  unsigned *resident = nullptr;
  unsigned *flushed = nullptr;
  float *out = nullptr;
  char *stack_mem = nullptr;
  const std::size_t stack_bytes =
      co::Stack::BytesNeeded(kGroups, kLanes, kStackBytes);

  CUDA_CHECK(cudaMallocManaged(&data, kPages * kPer * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&resident, kPages * sizeof(unsigned)));
  CUDA_CHECK(cudaMallocManaged(&flushed, kPages * sizeof(unsigned)));
  CUDA_CHECK(
      cudaMallocManaged(&out, kGroups * kPages * kPer * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&stack_mem, stack_bytes));

  for (co::u64 i = 0; i < kPages * kPer; ++i) {
    data[i] = static_cast<float>(i) * 0.5f;
  }
  for (co::u64 i = 0; i < kPages; ++i) {
    // The reference edition cannot park, so it needs everything resident.
    resident[i] = (sync_mode || CORO_CUDA_REF) ? 1u : 0u;
    flushed[i] = (sync_mode || CORO_CUDA_REF) ? 1u : 0u;
  }
  for (co::u64 i = 0; i < kGroups * kPages * kPer; ++i) out[i] = -1.0f;
  std::memset(stack_mem, 0xA5, stack_bytes);

  co::Stack stack(stack_mem, kGroups, kLanes, kStackBytes);
  co::Driver drv(stack);
  const d::PageCache cache{data, resident, flushed, kPages, kPer};

  co::u64 fetches = 0;
  co::u64 flushes = 0;
  co::u32 max_depth = 0;
  co::u32 hwm = 0;
  int rc = 0;
  for (;;) {
    StreamKernel<<<kGroups, kLanes>>>(stack.View(), cache, out, kPages);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (co::u32 g = 0; g < kGroups; ++g) {
      const co::GroupHeader *h = stack.Header(g);
      if (h->overflow != 0) {
        std::fprintf(stderr, "%s: STACK OVERFLOW in group %u\n", CORO_CUDA_TAG,
                     g);
        rc = 1;
      }
      if (h->hwm > hwm) hwm = h->hwm;
      if (h->status == co::kParked && h->park_depth > max_depth) {
        max_depth = h->park_depth;
      }
    }

#if CORO_CUDA_REF
    // The synchronous edition constructs no Scope, so it never writes a
    // header and every group stays kFresh. There is nothing to relaunch for:
    // one launch is the whole run, and Step is called only to keep the
    // launch counter comparable with the transpiled edition.
    (void)drv.Step([](co::u32, co::u64) {});
    break;
#else
    const co::Progress p = drv.Step([&](co::u32, co::u64 tag) {
      if ((tag & (1ull << 32)) != 0) {
        flushed[tag & 0xffffffffu] = 1u;
        ++flushes;
      } else {
        resident[tag] = 1u;
        ++fetches;
      }
    });
    if (!p.Pending()) break;
    if (drv.Launches() > 1000) {
      std::fprintf(stderr, "%s: runaway relaunch loop\n", CORO_CUDA_TAG);
      rc = 1;
      break;
    }
#endif
  }

  for (co::u64 i = 0; i < kGroups * kPages * kPer; ++i) {
    std::printf("%.9g\n", static_cast<double>(out[i]));
  }

  std::fprintf(stderr,
               "%s: launches=%llu fetches=%llu flushes=%llu max_park_depth=%u "
               "hwm=%u/%u bytes\n",
               CORO_CUDA_TAG,
               static_cast<unsigned long long>(drv.Launches()),
               static_cast<unsigned long long>(fetches),
               static_cast<unsigned long long>(flushes), max_depth, hwm,
               kStackBytes);

#if !CORO_CUDA_REF
  const co::u64 want_launches = sync_mode ? 1 : kPages * 2 + 1;
  if (drv.Launches() != want_launches) {
    std::fprintf(stderr, "%s: expected %llu launches, got %llu\n",
                 CORO_CUDA_TAG,
                 static_cast<unsigned long long>(want_launches),
                 static_cast<unsigned long long>(drv.Launches()));
    rc = 1;
  }
  if (!sync_mode && max_depth != 2) {
    std::fprintf(stderr, "%s: expected to park at depth 2, got %u\n",
                 CORO_CUDA_TAG, max_depth);
    rc = 1;
  }
#else
  (void)sync_mode;
#endif

  ReportStatic();
  ReportOccupancy();

  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaFree(resident));
  CUDA_CHECK(cudaFree(flushed));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(stack_mem));
  return rc;
}

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
 * The transpiler's output, on SYCL.
 *
 * Compiled from the SAME generated header the host test uses -- not a port, not
 * a variant, the identical file. That is the property the whole design exists
 * to buy: the generated code contains no backend token, so one transpiled
 * source serves SYCL, CUDA and ROCm.
 *
 * Built two ways:
 *   - JIT, and run on whatever device is present: tests SEMANTICS.
 *   - AOT for spir64_gen: runs IGC and tests the GPU CODEGEN PATH, which is
 *     where a `switch` with case labels inside a loop could have failed and did
 *     not, without needing a GPU on the machine doing the build.
 *
 * Prints its result vector in the same format as coro_ref, so the same diff
 * that checks the host build checks this one.
 */
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

#include <clio_runtime/co/coro.h>
#include <clio_runtime/co/driver.h>

#include "coro_types.h"
#include <coro_workload.h>  // the TRANSPILED copy

namespace co = clio::co;
namespace d = clio::co::demo;

namespace {

constexpr co::u64 kPages = 6;
constexpr co::u64 kPer = 32;
constexpr co::u32 kGroups = 3;
constexpr co::u32 kLanes = 8;
constexpr co::u32 kStackBytes = 512;

sycl::nd_range<1> Range() {
  return sycl::nd_range<1>{
      sycl::range<1>(static_cast<std::size_t>(kGroups) * kLanes),
      sycl::range<1>(kLanes)};
}

}  // namespace

int main(int argc, char **argv) {
  // --sync runs the workload with everything resident, which under the
  // TRANSPILED header must still never park. Useful as a self-check that the
  // fast path through a CO_AWAIT really is just a predicated branch.
  const bool sync_mode = argc > 1 && std::string(argv[1]) == "--sync";

  sycl::queue q;
  std::fprintf(stderr, "sycl device: %s\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

  float *data = sycl::malloc_shared<float>(kPages * kPer, q);
  unsigned *resident = sycl::malloc_shared<unsigned>(kPages, q);
  unsigned *flushed = sycl::malloc_shared<unsigned>(kPages, q);
  float *out = sycl::malloc_shared<float>(kGroups * kPages * kPer, q);
  const std::size_t stack_bytes =
      co::Stack::BytesNeeded(kGroups, kLanes, kStackBytes);
  char *stack_mem = sycl::malloc_shared<char>(stack_bytes, q);

  for (co::u64 i = 0; i < kPages * kPer; ++i) {
    data[i] = static_cast<float>(i) * 0.5f;
  }
  for (co::u64 i = 0; i < kPages; ++i) {
    resident[i] = sync_mode ? 1u : 0u;
    flushed[i] = sync_mode ? 1u : 0u;
  }
  for (co::u64 i = 0; i < kGroups * kPages * kPer; ++i) out[i] = -1.0f;
  // Poisoned rather than zeroed, so a forgotten save-list entry comes back as
  // garbage instead of a plausible zero.
  for (std::size_t i = 0; i < stack_bytes; ++i) {
    stack_mem[i] = static_cast<char>(0xA5);
  }

  co::Stack stack(stack_mem, kGroups, kLanes, kStackBytes);
  co::Driver drv(stack);
  const d::PageCache cache{data, resident, flushed, kPages, kPer};

  co::u64 fetches = 0;
  co::u64 flushes = 0;
  co::u32 max_depth = 0;
  co::u32 hwm = 0;
  int rc = 0;
  for (;;) {
    const co::StackView view = stack.View();
    q.parallel_for(Range(), [=](sycl::nd_item<1> it) {
       co::Scope scope(view, co::Item(it));
       if (scope) {
         const co::u32 g = static_cast<co::u32>(it.get_group(0));
         d::StreamTile(cache, out + g * kPages * kPer, kPages,
                       static_cast<co::u32>(it.get_local_id(0)),
                       static_cast<co::u32>(it.get_local_range(0)),
                       co::Item(it), scope.Context());
       }
     }).wait();

    for (co::u32 g = 0; g < kGroups; ++g) {
      const co::GroupHeader *h = stack.Header(g);
      if (h->overflow != 0) {
        std::fprintf(stderr, "sycl: STACK OVERFLOW in group %u\n", g);
        rc = 1;
      }
      if (h->hwm > hwm) hwm = h->hwm;
      if (h->status == co::kParked && h->park_depth > max_depth) {
        max_depth = h->park_depth;
      }
    }

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
      std::fprintf(stderr, "sycl: runaway relaunch loop\n");
      rc = 1;
      break;
    }
  }

  for (co::u64 i = 0; i < kGroups * kPages * kPer; ++i) {
    std::printf("%.9g\n", static_cast<double>(out[i]));
  }
  std::fprintf(stderr,
               "sycl: launches=%llu fetches=%llu flushes=%llu "
               "max_park_depth=%u hwm=%u/%u bytes\n",
               static_cast<unsigned long long>(drv.Launches()),
               static_cast<unsigned long long>(fetches),
               static_cast<unsigned long long>(flushes), max_depth, hwm,
               kStackBytes);

  const co::u64 want_launches = sync_mode ? 1 : kPages * 2 + 1;
  if (drv.Launches() != want_launches) {
    std::fprintf(stderr, "sycl: expected %llu launches\n",
                 static_cast<unsigned long long>(want_launches));
    rc = 1;
  }
  if (!sync_mode && max_depth != 2) {
    std::fprintf(stderr, "sycl: expected to park at depth 2\n");
    rc = 1;
  }

  sycl::free(data, q);
  sycl::free(resident, q);
  sycl::free(flushed, q);
  sycl::free(out, q);
  sycl::free(stack_mem, q);
  return rc;
}

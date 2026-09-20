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
 * P0 spike S1 on SYCL: does the generated shape survive a real device compiler?
 *
 * This is the gate the design doc (section 9) puts everything else behind,
 * because the one assumption it could not settle by argument is that a `switch`
 * whose case labels sit inside a loop -- an irreducible CFG -- comes out of a
 * device backend correct. Nothing in the mechanism can rescue that if it fails;
 * the fallback is a different emitter (`--emit=flat`), decided here on evidence.
 *
 * Two things are being tested and they are separable:
 *   - JIT/run on whatever device is present, which tests SEMANTICS.
 *   - AOT to spir64_gen, which runs IGC and tests the GPU CODEGEN PATH without
 *     needing a GPU on the machine doing the build.
 *
 * The same spike_workload.h that the host backend uses is compiled here without
 * a single conditional: `gen::StreamTile` is one source for all backends, which
 * is the property the whole design exists to buy.
 */
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

#include <clio_runtime/co/coro.h>
#include <clio_runtime/co/driver.h>

#include "spike_workload.h"

namespace co = clio::co;
namespace sp = clio::co::spike;

namespace {

int g_failures = 0;

void Check(bool ok, const char *what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

constexpr co::u64 kPages = 6;
constexpr co::u64 kPer = 32;
constexpr co::u32 kGroups = 3;
constexpr co::u32 kLanes = 8;

/** Everything the kernel touches, in USM shared memory so the host servicer can
 *  read the group headers and flip the residency flags between launches. A
 *  production driver would use a device allocation plus a copy-back of just the
 *  GroupHeader array; shared memory keeps the spike to the point. */
struct World {
  sycl::queue &q;
  float *data;
  unsigned *resident;
  unsigned *flushed;
  float *out;
  char *stack_mem;

  explicit World(sycl::queue &queue) : q(queue) {
    data = sycl::malloc_shared<float>(kPages * kPer, q);
    resident = sycl::malloc_shared<unsigned>(kPages, q);
    flushed = sycl::malloc_shared<unsigned>(kPages, q);
    out = sycl::malloc_shared<float>(kGroups * kPages * kPer, q);
    stack_mem = sycl::malloc_shared<char>(
        co::Stack::BytesNeeded(kGroups, kLanes,
                               sp::gen::kStackBytes_StreamTile),
        q);
    for (co::u64 i = 0; i < kPages * kPer; ++i) {
      data[i] = static_cast<float>(i) * 0.5f;
    }
  }
  ~World() {
    sycl::free(data, q);
    sycl::free(resident, q);
    sycl::free(flushed, q);
    sycl::free(out, q);
    sycl::free(stack_mem, q);
  }

  sp::PageCache Cache() const {
    return sp::PageCache{data, resident, flushed, kPages, kPer};
  }
  void SetResidency(unsigned v) {
    for (co::u64 i = 0; i < kPages; ++i) resident[i] = v;
  }
  void SetFlushed(unsigned v) {
    for (co::u64 i = 0; i < kPages; ++i) flushed[i] = v;
  }
  void ClearOut() {
    for (co::u64 i = 0; i < kGroups * kPages * kPer; ++i) out[i] = -1.0f;
  }
  /** Poisoned rather than zeroed: a forgotten save-list entry must come back as
   *  garbage, not as a plausible zero, or the differential would not see it. */
  void PoisonStack() {
    const std::size_t n = co::Stack::BytesNeeded(
        kGroups, kLanes, sp::gen::kStackBytes_StreamTile);
    for (std::size_t i = 0; i < n; ++i) stack_mem[i] = static_cast<char>(0xA5);
  }
};

sycl::nd_range<1> Range() {
  return sycl::nd_range<1>{sycl::range<1>(static_cast<std::size_t>(kGroups) *
                                          kLanes),
                           sycl::range<1>(kLanes)};
}

/** The synchronous form, everything resident: the S5 oracle, on device. */
void RunReference(World &w) {
  w.SetResidency(1);
  w.SetFlushed(1);
  w.ClearOut();
  const sp::PageCache c = w.Cache();
  float *out = w.out;
  w.q.parallel_for(Range(), [=](sycl::nd_item<1> it) {
     const co::u32 g = static_cast<co::u32>(it.get_group(0));
     sp::src::StreamTile(c, out + g * kPages * kPer, kPages,
                         static_cast<co::u32>(it.get_local_id(0)),
                         static_cast<co::u32>(it.get_local_range(0)),
                         co::Item(it));
   }).wait();
}

struct RunStats {
  co::u64 launches = 0;
  co::u64 parks = 0;
  co::u32 max_park_depth = 0;
  co::u64 fetches = 0;
  co::u64 flushes = 0;
};

/** The generated state machine, driven through the park/relaunch loop. */
void RunParking(World &w, RunStats *stats) {
  w.SetResidency(0);
  w.SetFlushed(0);
  w.ClearOut();
  w.PoisonStack();

  co::Stack stack(w.stack_mem, kGroups, kLanes,
                  sp::gen::kStackBytes_StreamTile);
  co::Driver drv(stack);
  const sp::PageCache c = w.Cache();
  float *out = w.out;

  for (;;) {
    const co::StackView view = stack.View();
    w.q.parallel_for(Range(), [=](sycl::nd_item<1> it) {
       co::Scope scope(view, co::Item(it));
       if (scope) {
         const co::u32 g = static_cast<co::u32>(it.get_group(0));
         sp::gen::StreamTile(c, out + g * kPages * kPer, kPages,
                             static_cast<co::u32>(it.get_local_id(0)),
                             static_cast<co::u32>(it.get_local_range(0)),
                             scope.Context());
       }
     }).wait();

    for (co::u32 g = 0; g < kGroups; ++g) {
      const co::GroupHeader *h = stack.Header(g);
      if (h->status == co::kParked && h->park_depth > stats->max_park_depth) {
        stats->max_park_depth = h->park_depth;
      }
    }

    const co::Progress p = drv.Step([&](co::u32, co::u64 tag) {
      if (tag & (1ull << 32)) {
        w.flushed[tag & 0xffffffffu] = 1u;
        ++stats->flushes;
      } else {
        w.resident[tag] = 1u;
        ++stats->fetches;
      }
    });
    stats->launches = drv.Launches();
    stats->parks += p.parked;
    if (!p.Pending()) break;
    if (stats->launches > 1000) {
      std::printf("  runaway: %llu launches\n",
                  static_cast<unsigned long long>(stats->launches));
      break;
    }
  }
}

}  // namespace

int main() {
  sycl::queue q;
  std::printf("P0 spike S1, SYCL backend\n");
  std::printf("  device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("  stack bytes per work-item: %u\n",
              sp::gen::kStackBytes_StreamTile);

  World w(q);

  RunReference(w);
  std::vector<float> ref(w.out, w.out + kGroups * kPages * kPer);
  bool ref_written = true;
  for (float v : ref) ref_written = ref_written && (v != -1.0f);
  Check(ref_written, "S1  synchronous reference wrote every cell");

  RunStats st;
  RunParking(w, &st);
  std::printf("  launches=%llu parks=%llu fetches=%llu flushes=%llu "
              "max_park_depth=%u\n",
              static_cast<unsigned long long>(st.launches),
              static_cast<unsigned long long>(st.parks),
              static_cast<unsigned long long>(st.fetches),
              static_cast<unsigned long long>(st.flushes), st.max_park_depth);

  bool identical = true;
  std::size_t first_bad = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != w.out[i]) {
      identical = false;
      first_bad = i;
      break;
    }
  }
  if (!identical) {
    std::printf("  first mismatch at %zu: ref=%f got=%f\n", first_bad,
                ref[first_bad], w.out[first_bad]);
  }
  Check(identical, "S1  parking state machine == synchronous source, on device");
  Check(st.max_park_depth == 2, "S1  parked at depth 2 through the chain");
  Check(st.fetches == kPages * kGroups, "S1  each group fetched each page once");
  Check(st.flushes == kPages * kGroups, "S1  each group flushed each page once");
  Check(st.launches == kPages * 2 + 1, "S1  launch count is npages*2 + 1");

  std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}

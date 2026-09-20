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
 * The TRANSPILED half of the differential test.
 *
 * Includes the GENERATED coro_workload.h -- same file name, resolved by
 * include order alone -- where the workload is a group-scoped state machine.
 * Nothing is resident at the start, so every page faults twice (a fetch and a
 * flush) and the whole park/relaunch path is exercised. It must print exactly
 * what coro_ref prints.
 */
#include "coro_test_common.h"
#include <coro_workload.h>  // the TRANSPILED copy, from the mirror directory

int main() {
  namespace co = clio::co;
  namespace d = clio::co::demo;
  d::World w;
  const d::PageCache c = w.Cache();

  const std::size_t bytes =
      co::Stack::BytesNeeded(d::kGroups, d::kLanes, d::kStackBytes);
  // Poisoned, not zeroed: a save list that forgets a value must come back as
  // garbage rather than a plausible zero, or the comparison would not see it.
  std::vector<char> mem(bytes, static_cast<char>(0xA5));
  co::Stack stack(mem.data(), d::kGroups, d::kLanes, d::kStackBytes);
  co::Driver drv(stack);

  co::u64 fetches = 0;
  co::u64 flushes = 0;
  co::u32 max_depth = 0;
  co::u32 hwm = 0;
  for (;;) {
    const co::StackView view = stack.View();
    d::Launch(d::kGroups, d::kLanes, [&](co::Item it) {
      co::Scope scope(view, it);
      if (scope) {
        d::StreamTile(c, w.OutFor(it.Group()), d::kPages, it.Local(),
                      it.Size(), it, scope.Context());
      }
    });

    for (co::u32 g = 0; g < d::kGroups; ++g) {
      const co::GroupHeader *h = stack.Header(g);
      if (h->overflow != 0) {
        std::fprintf(stderr, "gen: STACK OVERFLOW in group %u\n", g);
        return 1;
      }
      if (h->hwm > hwm) hwm = h->hwm;
      if (h->status == co::kParked && h->park_depth > max_depth) {
        max_depth = h->park_depth;
      }
    }

    // The servicer runs strictly BETWEEN launches, which is why the park path
    // needs no fence and no atomic anywhere (invariant I8).
    const co::Progress p = drv.Step([&](co::u32, co::u64 tag) {
      if ((tag & (1ull << 32)) != 0) {
        w.flushed[tag & 0xffffffffu] = 1u;
        ++flushes;
      } else {
        w.resident[tag] = 1u;
        ++fetches;
      }
    });
    if (!p.Pending()) break;
    if (drv.Launches() > 1000) {
      std::fprintf(stderr, "gen: runaway relaunch loop\n");
      return 1;
    }
  }

  d::PrintResult(w.out);
  std::fprintf(stderr,
               "gen: launches=%llu fetches=%llu flushes=%llu max_park_depth=%u "
               "hwm=%u/%u bytes\n",
               static_cast<unsigned long long>(drv.Launches()),
               static_cast<unsigned long long>(fetches),
               static_cast<unsigned long long>(flushes), max_depth, hwm,
               d::kStackBytes);

  const bool ok = drv.Launches() == d::kPages * 2 + 1 &&
                  fetches == d::kPages * d::kGroups &&
                  flushes == d::kPages * d::kGroups && max_depth == 2;
  if (!ok) {
    std::fprintf(stderr, "gen: park/relaunch statistics are wrong\n");
    return 1;
  }
  return 0;
}

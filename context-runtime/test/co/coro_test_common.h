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
 * Shared scaffolding for the transpiler's differential test.
 *
 * The two halves are SEPARATE PROGRAMS on purpose. Both include a header named
 * coro_workload.h -- one the original, one the transpiled copy, resolved by
 * include order alone -- and both define the same inline functions. Linking
 * them together would be an ODR violation with the linker silently picking a
 * winner, which is precisely the bug a differential test must not have. So each
 * prints its result vector and a script compares the two streams.
 */
#ifndef CLIO_RUNTIME_TEST_CO_CORO_TEST_COMMON_H_
#define CLIO_RUNTIME_TEST_CO_CORO_TEST_COMMON_H_

#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include <clio_runtime/co/coro.h>
#include <clio_runtime/co/driver.h>

#include "coro_types.h"

namespace clio::co::demo {

inline constexpr u64 kPages = 6;
inline constexpr u64 kPer = 32;
inline constexpr u32 kGroups = 3;
inline constexpr u32 kLanes = 8;

/** Per-item bytes for the deepest chain, StreamTile > HoldPage > Fetch. The
 *  runtime reports the high-water mark, so a value that is too small shows up
 *  as GroupHeader::overflow rather than as corruption. */
inline constexpr u32 kStackBytes = 512;

/** The host "launch": one thread per work-item, joined at the end. Real
 *  threads and a real barrier, so group collectives are genuinely exercised. */
template <class BodyFn>
void Launch(u32 n_groups, u32 group_size, BodyFn body) {
  std::vector<std::unique_ptr<HostGroup>> groups;
  groups.reserve(n_groups);
  for (u32 g = 0; g < n_groups; ++g) {
    groups.push_back(std::make_unique<HostGroup>(group_size, g));
  }
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_groups) * group_size);
  for (u32 g = 0; g < n_groups; ++g) {
    for (u32 lane = 0; lane < group_size; ++lane) {
      threads.emplace_back([&, g, lane] { body(Item(groups[g].get(), lane)); });
    }
  }
  for (auto &t : threads) t.join();
}

/** The workload's buffers. */
struct World {
  std::vector<float> data;
  std::vector<unsigned> resident;
  std::vector<unsigned> flushed;
  std::vector<float> out;

  World()
      : data(kPages * kPer),
        resident(kPages, 0),
        flushed(kPages, 0),
        out(kGroups * kPages * kPer, -1.0f) {
    for (u64 i = 0; i < kPages * kPer; ++i) {
      data[i] = static_cast<float>(i) * 0.5f;
    }
  }

  PageCache Cache() {
    return PageCache{data.data(), resident.data(), flushed.data(), kPages,
                     kPer};
  }
  /** Each group writes its own slice, so groups cannot mask each other's bugs
   *  by overwriting the same cells. */
  float *OutFor(u32 g) { return out.data() + g * kPages * kPer; }
};

/** Print the result vector for the comparison script. Exact decimal, because
 *  the two builds run the same arithmetic in the same order: any difference at
 *  all is a lost value, not a rounding artefact. */
inline void PrintResult(const std::vector<float> &v) {
  for (float x : v) std::printf("%.9g\n", static_cast<double>(x));
}

}  // namespace clio::co::demo

#endif  // CLIO_RUNTIME_TEST_CO_CORO_TEST_COMMON_H_

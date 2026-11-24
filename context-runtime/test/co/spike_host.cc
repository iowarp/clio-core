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
 * P0 spikes S2-S5 on the host backend.
 *
 * The host backend is not a mock. A work-group is emulated with real threads
 * and a real std::barrier, so barrier convergence, the collective park vote and
 * the replay descent are all genuinely exercised -- the only thing it cannot
 * prove is backend codegen, which is what spike S1 (spike_sycl.cc) is for.
 *
 *   S2  a barrier sits between two suspend points, and the group crosses it
 *       after resuming. Legal only if the whole group resumes at the same
 *       point (invariant I2). A violation deadlocks or trips the barrier.
 *   S3  a three-level suspending chain parks at depth 2. Frame offsets must
 *       reproduce on the replay descent (invariant I1); if they do not, Pop
 *       reads another frame's bytes and the output is wrong.
 *   S4  state survives the "kernel exit". On the host that is the thread join;
 *       every launch builds fresh Ctx and Frame objects from scratch.
 *   S5  the differential oracle: the same workload, run synchronously from the
 *       `src` namespace with everything resident, must produce bit-identical
 *       output to the parking `gen` state machine.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

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

/* ------------------------------------------------------------------ */
/* The host "launch": one thread per work-item, joined at the end.     */
/* ------------------------------------------------------------------ */

template <class BodyFn>
void Launch(co::u32 n_groups, co::u32 group_size, BodyFn body) {
  std::vector<std::unique_ptr<co::HostGroup>> groups;
  groups.reserve(n_groups);
  for (co::u32 g = 0; g < n_groups; ++g) {
    groups.push_back(std::make_unique<co::HostGroup>(group_size, g));
  }
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_groups) * group_size);
  for (co::u32 g = 0; g < n_groups; ++g) {
    for (co::u32 lane = 0; lane < group_size; ++lane) {
      threads.emplace_back(
          [&, g, lane] { body(co::Item(groups[g].get(), lane)); });
    }
  }
  for (auto &t : threads) t.join();
}

/* ------------------------------------------------------------------ */
/* The workload's buffers.                                            */
/* ------------------------------------------------------------------ */

struct World {
  static constexpr co::u64 kPages = 6;
  static constexpr co::u64 kPer = 32;
  static constexpr co::u32 kGroups = 3;
  static constexpr co::u32 kLanes = 8;

  std::vector<float> data;
  std::vector<unsigned> resident;
  std::vector<unsigned> flushed;
  std::vector<float> out;

  World()
      : data(kPages * kPer),
        resident(kPages, 0),
        flushed(kPages, 0),
        out(kGroups * kPages * kPer, -1.0f) {
    for (co::u64 i = 0; i < kPages * kPer; ++i) {
      data[i] = static_cast<float>(i) * 0.5f;
    }
  }

  sp::PageCache Cache() {
    return sp::PageCache{data.data(), resident.data(), flushed.data(), kPages,
                         kPer};
  }
  /** Each group writes its own slice of `out`, so groups cannot mask each
   *  other's bugs by overwriting the same cells. */
  float *OutFor(co::u32 g) { return out.data() + g * kPages * kPer; }
};

/* ------------------------------------------------------------------ */
/* S5 reference: the synchronous form, everything already resident.   */
/* ------------------------------------------------------------------ */

std::vector<float> Reference(World &w) {
  std::fill(w.resident.begin(), w.resident.end(), 1u);
  std::fill(w.flushed.begin(), w.flushed.end(), 1u);
  std::fill(w.out.begin(), w.out.end(), -1.0f);
  sp::PageCache c = w.Cache();
  Launch(World::kGroups, World::kLanes, [&](co::Item it) {
    sp::src::StreamTile(c, w.OutFor(it.Group()), World::kPages, it.Local(),
                        it.Size(), it);
  });
  return w.out;
}

/* ------------------------------------------------------------------ */
/* The parking run.                                                   */
/* ------------------------------------------------------------------ */

struct RunStats {
  co::u64 launches = 0;
  co::u64 parks = 0;
  co::u32 max_park_depth = 0;
  co::u64 fetches = 0;
  co::u64 flushes = 0;
};

std::vector<float> ParkingRun(World &w, RunStats *stats) {
  std::fill(w.resident.begin(), w.resident.end(), 0u);
  std::fill(w.flushed.begin(), w.flushed.end(), 0u);
  std::fill(w.out.begin(), w.out.end(), -1.0f);

  const std::size_t bytes = co::Stack::BytesNeeded(
      World::kGroups, World::kLanes, sp::gen::kStackBytes_StreamTile);
  // Poisoned, not zeroed: a save list that forgets a value must produce garbage
  // rather than a plausible zero, or S5 would not catch it.
  std::vector<char> mem(bytes, static_cast<char>(0xA5));
  co::Stack stack(mem.data(), World::kGroups, World::kLanes,
                  sp::gen::kStackBytes_StreamTile);
  co::Driver drv(stack);

  sp::PageCache c = w.Cache();
  for (;;) {
    co::StackView view = stack.View();
    Launch(World::kGroups, World::kLanes, [&](co::Item it) {
      co::Scope scope(view, it);
      if (scope) {
        sp::gen::StreamTile(c, w.OutFor(it.Group()), World::kPages, it.Local(),
                            it.Size(), scope.Context());
      }
    });

    for (co::u32 g = 0; g < World::kGroups; ++g) {
      const co::GroupHeader *h = stack.Header(g);
      if (h->status == co::kParked && h->park_depth > stats->max_park_depth) {
        stats->max_park_depth = h->park_depth;
      }
    }

    // The servicer. Runs strictly between launches, which is the whole reason
    // suspension is a kernel exit: no concurrency with the kernel means no
    // system-scope atomics and no fences anywhere (invariant I8).
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
  return w.out;
}

}  // namespace

int main() {
  std::printf("P0 spikes, host backend\n");
  std::printf("  frame bytes: StreamTile=%u HoldPage=%u Fetch=%u Flush=%u\n",
              sp::gen::kFrameBytes_StreamTile, sp::gen::kFrameBytes_HoldPage,
              sp::gen::kFrameBytes_Fetch, sp::gen::kFrameBytes_Flush);
  std::printf("  stack bytes per work-item (deepest chain): %u\n",
              sp::gen::kStackBytes_StreamTile);

  World w;
  const std::vector<float> ref = Reference(w);

  RunStats st;
  const std::vector<float> got = ParkingRun(w, &st);

  std::printf("  launches=%llu parks=%llu fetches=%llu flushes=%llu "
              "max_park_depth=%u\n",
              static_cast<unsigned long long>(st.launches),
              static_cast<unsigned long long>(st.parks),
              static_cast<unsigned long long>(st.fetches),
              static_cast<unsigned long long>(st.flushes), st.max_park_depth);

  // S5: the two forms must agree exactly. Not approximately -- these are the
  // same arithmetic in the same order, so any difference is a lost value.
  bool identical = ref.size() == got.size();
  std::size_t first_bad = 0;
  for (std::size_t i = 0; identical && i < ref.size(); ++i) {
    if (ref[i] != got[i]) {
      identical = false;
      first_bad = i;
    }
  }
  if (!identical && ref.size() == got.size()) {
    std::printf("  first mismatch at %zu: ref=%f got=%f\n", first_bad,
                ref[first_bad], got[first_bad]);
  }
  Check(identical, "S5  parking state machine == synchronous source");

  // The reference itself must have done something, or S5 compares two blanks.
  bool ref_written = true;
  for (float v : ref) ref_written = ref_written && (v != -1.0f);
  Check(ref_written, "S5  reference actually wrote every cell");

  // S3: the deepest chain really did park at its leaf. If this is 0 or 1 the
  // three-level replay descent was never exercised and S3 proved nothing.
  Check(st.max_park_depth == 2, "S3  parked at depth 2 (StreamTile>HoldPage>Fetch)");

  // S2/S4: every page was fetched and flushed exactly once, and the run
  // terminated. A repeated fetch would mean a resume re-ran work it should have
  // skipped; a missing one would mean a resume skipped work it should have run.
  Check(st.fetches == World::kPages * World::kGroups,
        "S4  each group fetched each page exactly once");
  Check(st.flushes == World::kPages * World::kGroups,
        "S4  each group flushed each page exactly once");

  // Deterministic servicer: one park per fetch and one per flush, plus the
  // final launch that finds everything ready.
  Check(st.launches == World::kPages * 2 + 1,
        "S2  launch count is exactly npages*2 + 1");

  std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}

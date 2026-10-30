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
 * Per-callback profile of the FUSE adapter (CLIO_FUSE_PROFILE=<seconds>).
 *
 * Every entry of the fuse_operations table is wrapped (CLIO_FUSE_TIMED), so
 * each callback's calls and wall time are counted without touching the
 * handlers. The time is what the adapter spends serving the call, runtime
 * round trips included; what an application sees on top of it is the FUSE
 * layer's own cost (WinFsp's kernel <-> user transitions on Windows).
 *
 * With the variable unset the wrapper costs one relaxed atomic load. Set, a
 * reporter thread logs the calls made in each interval, costliest first.
 */

#ifndef CLIO_CTE_ADAPTER_LIBFUSE_FUSE_CTE_PROFILE_H_
#define CLIO_CTE_ADAPTER_LIBFUSE_FUSE_CTE_PROFILE_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "clio_runtime/clio_runtime.h"

namespace clio::cte::fuse_profile {

/** Counters of one FUSE callback. */
struct OpStat {
  const char *name;                      ///< callback name (getattr, ...)
  std::atomic<uint64_t> calls{0};        ///< calls since start
  std::atomic<uint64_t> ns{0};           ///< total wall time (ns)
  std::atomic<uint64_t> max_ns{0};       ///< slowest call (ns)
  OpStat *next = nullptr;                ///< registry link
  explicit OpStat(const char *n);
};

/** Report labels of the wrapped callbacks (template arguments need linkage). */
#define CLIO_FUSE_PROFILE_NAME(n) inline constexpr char kName_##n[] = #n;
CLIO_FUSE_PROFILE_NAME(getattr)
CLIO_FUSE_PROFILE_NAME(readlink)
CLIO_FUSE_PROFILE_NAME(mkdir)
CLIO_FUSE_PROFILE_NAME(unlink)
CLIO_FUSE_PROFILE_NAME(rmdir)
CLIO_FUSE_PROFILE_NAME(symlink)
CLIO_FUSE_PROFILE_NAME(rename)
CLIO_FUSE_PROFILE_NAME(link)
CLIO_FUSE_PROFILE_NAME(chmod)
CLIO_FUSE_PROFILE_NAME(chown)
CLIO_FUSE_PROFILE_NAME(truncate)
CLIO_FUSE_PROFILE_NAME(open)
CLIO_FUSE_PROFILE_NAME(read)
CLIO_FUSE_PROFILE_NAME(write)
CLIO_FUSE_PROFILE_NAME(statfs)
CLIO_FUSE_PROFILE_NAME(flush)
CLIO_FUSE_PROFILE_NAME(release)
CLIO_FUSE_PROFILE_NAME(fsync)
CLIO_FUSE_PROFILE_NAME(setxattr)
CLIO_FUSE_PROFILE_NAME(getxattr)
CLIO_FUSE_PROFILE_NAME(listxattr)
CLIO_FUSE_PROFILE_NAME(removexattr)
CLIO_FUSE_PROFILE_NAME(readdir)
CLIO_FUSE_PROFILE_NAME(fsyncdir)
CLIO_FUSE_PROFILE_NAME(create)
CLIO_FUSE_PROFILE_NAME(utimens)
CLIO_FUSE_PROFILE_NAME(fallocate)
#undef CLIO_FUSE_PROFILE_NAME

/** True when CLIO_FUSE_PROFILE is set (decided once, at mount). */
inline std::atomic<bool> g_enabled{false};

/** Time one callback into its OpStat when profiling is on. */
class Scope {
 public:
  explicit Scope(OpStat &s)
      : stat_(g_enabled.load(std::memory_order_relaxed) ? &s : nullptr) {
    if (stat_ != nullptr) t0_ = std::chrono::steady_clock::now();
  }
  ~Scope() {
    if (stat_ == nullptr) return;
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0_)
            .count());
    stat_->calls.fetch_add(1, std::memory_order_relaxed);
    stat_->ns.fetch_add(ns, std::memory_order_relaxed);
    uint64_t m = stat_->max_ns.load(std::memory_order_relaxed);
    while (ns > m && !stat_->max_ns.compare_exchange_weak(
                         m, ns, std::memory_order_relaxed)) {
    }
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

 private:
  OpStat *stat_;
  std::chrono::steady_clock::time_point t0_;
};

/** Head of the registry of every OpStat (one per wrapped callback). */
inline std::atomic<OpStat *> g_head{nullptr};

/** One callback's numbers over a report interval. */
struct Row {
  const char *name;
  uint64_t calls;
  uint64_t ns;
  uint64_t max_ns;
};

/**
 * Log the calls made since the previous report, costliest first.
 * @param last per-OpStat totals at the previous report (updated)
 * @param secs length of the interval
 */
inline void Report(std::vector<std::pair<OpStat *, Row>> *last, double secs) {
  std::vector<Row> rows;
  for (OpStat *s = g_head.load(); s != nullptr; s = s->next) {
    auto it = std::find_if(last->begin(), last->end(),
                           [s](const auto &p) { return p.first == s; });
    if (it == last->end()) {
      last->push_back({s, Row{s->name, 0, 0, 0}});
      it = last->end() - 1;
    }
    const Row now{s->name, s->calls.load(), s->ns.load(),
                  s->max_ns.exchange(0)};
    if (now.calls > it->second.calls) {
      rows.push_back(Row{s->name, now.calls - it->second.calls,
                         now.ns - it->second.ns, now.max_ns});
    }
    it->second = now;
  }
  if (rows.empty()) return;
  std::sort(rows.begin(), rows.end(),
            [](const Row &a, const Row &b) { return a.ns > b.ns; });
  std::string out;
  char line[160];
  for (const Row &r : rows) {
    std::snprintf(line, sizeof(line),
                  "\n  %-12s %9llu calls %9.1f ms total %9.1f us mean "
                  "%9.1f us max",
                  r.name, static_cast<unsigned long long>(r.calls),
                  r.ns / 1e6, r.ns / 1e3 / r.calls, r.max_ns / 1e3);
    out += line;
  }
  const double unix_s =
      std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  HLOG(kInfo, "clio_cte_fuse profile at {:.3f} (unix s), last {} s:{}",
       unix_s, secs, out);
}
inline OpStat::OpStat(const char *n) : name(n) {
  OpStat *head = g_head.load();
  do {
    next = head;
  } while (!g_head.compare_exchange_weak(head, this));
}

/**
 * Read CLIO_FUSE_PROFILE and, if set, start the reporter thread.
 * Called once from the adapter's init callback.
 */
inline void Start() {
  const char *e = std::getenv("CLIO_FUSE_PROFILE");
  if (e == nullptr || *e == '\0') return;
  const double secs = std::max(1.0, std::atof(e));
  g_enabled.store(true);
  HLOG(kInfo, "clio_cte_fuse: profiling callbacks, reporting every {} s",
       secs);
  std::thread([secs]() {
    std::vector<std::pair<OpStat *, Row>> last;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::duration<double>(secs));
      Report(&last, secs);
    }
  }).detach();
}

/** Wrapper with the exact signature of the callback it times. */
template <typename F, F Fn>
struct Timed;

template <typename R, typename... A, R (*Fn)(A...)>
struct Timed<R (*)(A...), Fn> {
  /**
   * Call the wrapped callback, timing it under `name`.
   * @param a the callback's arguments
   * @return the callback's result
   */
  template <const char *Name>
  static R Call(A... a) {
    static OpStat stat(Name);
    Scope s(stat);
    return Fn(a...);
  }
};

}  // namespace clio::cte::fuse_profile

/**
 * Wrap a fuse_operations entry: CLIO_FUSE_TIMED(getattr, cte_fuse_getattr).
 * The name is the callback's, used as the report label.
 */
#define CLIO_FUSE_TIMED(name, fn)                                      \
  ::clio::cte::fuse_profile::Timed<decltype(&fn), &fn>::template Call< \
      ::clio::cte::fuse_profile::kName_##name>

#endif  // CLIO_CTE_ADAPTER_LIBFUSE_FUSE_CTE_PROFILE_H_

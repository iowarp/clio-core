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
 * Lock Workload Benchmark
 *
 * A configurable read/write workload generator for the
 * context-transport-primitives locking implementations (ctp::Mutex,
 * ctp::RwLock, ctp::CvRwLock). It spawns reader and writer threads that follow
 * a duty cycle and reports the acquisition-latency percentiles per role.
 *
 * Per-thread workload:
 *   repeat until runtime elapses:
 *     - if shift > 0: stay idle for `shift` seconds (not holding the lock)
 *     - acquire the lock, timing how long acquisition takes (the latency sample)
 *     - if burst > 0: hold the lock for `burst` seconds
 *     - release the lock
 *   shift == 0 => no idle gap; burst == 0 => hold ~0s. Both 0 => hammer the
 *   lock continuously.
 *
 * Readers take the shared/read lock; writers take the exclusive/write lock.
 * ctp::Mutex has no shared mode, so its "readers" also take the exclusive lock.
 *
 * Each config is run against all three lock types. Latency samples are gathered
 * with per-thread reservoir sampling so an uncontended hammer (hundreds of
 * millions of ops) stays within bounded memory. Reported latencies include the
 * fixed cost of two steady_clock reads around each acquire.
 *
 * Usage:
 *   lock_latency_benchmark [readers writers read_shift read_burst \
 *                           write_shift write_burst [runtime_s]]
 *   (no args => run the built-in scenario suite)
 *
 * All shift/burst/runtime values are in seconds (floating point).
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/thread/lock/cvrwlock.h"
#include "clio_ctp/thread/lock/mutex.h"
#include "clio_ctp/thread/lock/rwlock.h"

namespace {

using Clock = std::chrono::steady_clock;

/** One benchmark configuration. */
struct Config {
  std::string name;
  int readers = 1;
  int writers = 0;
  double read_shift = 0.0;
  double read_burst = 0.0;
  double write_shift = 0.0;
  double write_burst = 0.0;
  double runtime = 5.0;
};

/** Latency percentiles (nanoseconds) plus the true acquisition count. */
struct Stats {
  uint64_t count = 0;  // total acquisitions (not the sampled subset)
  double p50 = 0, p75 = 0, p90 = 0, p99 = 0;
};

/**
 * Bounded, unbiased latency sampler (Algorithm R reservoir sampling). Keeps at
 * most kCap samples regardless of how many are offered, so hammering a lock for
 * seconds does not blow up memory while still yielding representative
 * percentiles.
 */
class Reservoir {
 public:
  static constexpr size_t kCap = 1u << 19;  // 524288 samples (~4 MB)

  explicit Reservoir(uint64_t seed) : rng_(seed) { samples_.reserve(kCap); }

  void Add(double v) {
    if (samples_.size() < kCap) {
      samples_.push_back(v);
    } else {
      const uint64_t j = rng_() % (count_ + 1);  // uniform in [0, count_]
      if (j < kCap) samples_[j] = v;
    }
    ++count_;
  }

  uint64_t count() const { return count_; }
  const std::vector<double> &samples() const { return samples_; }

 private:
  std::vector<double> samples_;
  uint64_t count_ = 0;
  std::mt19937_64 rng_;
};

void SleepSeconds(double s) {
  std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

/** Aggregate one role's per-thread reservoirs into percentile stats. */
Stats Aggregate(std::vector<Reservoir> &reservoirs) {
  Stats s;
  std::vector<double> all;
  for (auto &r : reservoirs) {
    s.count += r.count();
    all.insert(all.end(), r.samples().begin(), r.samples().end());
  }
  if (all.empty()) return s;
  std::sort(all.begin(), all.end());
  auto pct = [&](double p) {
    const size_t i = static_cast<size_t>((p / 100.0) *
                                         static_cast<double>(all.size() - 1));
    return all[i];
  };
  s.p50 = pct(50);
  s.p75 = pct(75);
  s.p90 = pct(90);
  s.p99 = pct(99);
  return s;
}

// Compile-time adapters so the hot loop dispatches with no per-op branch.
struct MutexOps {
  using Lock = ctp::Mutex;
  static const char *Name() { return "ctp::Mutex"; }
  static void RLock(Lock &l) { l.Lock(0); }
  static void RUnlock(Lock &l) { l.Unlock(); }
  static void WLock(Lock &l) { l.Lock(0); }
  static void WUnlock(Lock &l) { l.Unlock(); }
};
struct RwLockOps {
  using Lock = ctp::RwLock;
  static const char *Name() { return "ctp::RwLock"; }
  static void RLock(Lock &l) { l.ReadLock(0); }
  static void RUnlock(Lock &l) { l.ReadUnlock(); }
  static void WLock(Lock &l) { l.WriteLock(0); }
  static void WUnlock(Lock &l) { l.WriteUnlock(); }
};
struct CvRwLockOps {
  using Lock = ctp::CvRwLock;
  static const char *Name() { return "ctp::CvRwLock"; }
  static void RLock(Lock &l) { l.ReadLock(); }
  static void RUnlock(Lock &l) { l.ReadUnlock(); }
  static void WLock(Lock &l) { l.WriteLock(); }
  static void WUnlock(Lock &l) { l.WriteUnlock(); }
};

/** The per-thread duty-cycle loop. Records acquisition latency into `out`. */
template <typename Ops>
void Worker(typename Ops::Lock *lock, bool is_writer, double shift,
            double burst, Clock::time_point deadline, Reservoir *out) {
  while (Clock::now() < deadline) {
    if (shift > 0.0) {
      SleepSeconds(shift);
      if (Clock::now() >= deadline) break;
    }
    const auto t0 = Clock::now();
    if (is_writer) {
      Ops::WLock(*lock);
    } else {
      Ops::RLock(*lock);
    }
    const auto t1 = Clock::now();
    out->Add(std::chrono::duration<double, std::nano>(t1 - t0).count());

    if (burst > 0.0) SleepSeconds(burst);

    if (is_writer) {
      Ops::WUnlock(*lock);
    } else {
      Ops::RUnlock(*lock);
    }
  }
}

/** Run one config against a single lock type; print a reader and/or writer row. */
template <typename Ops>
void RunLock(const Config &cfg) {
  typename Ops::Lock lock;

  std::vector<Reservoir> readers;
  readers.reserve(cfg.readers);
  for (int i = 0; i < cfg.readers; ++i) readers.emplace_back(0x9E3779B97F4A7C15ULL + i);
  std::vector<Reservoir> writers;
  writers.reserve(cfg.writers);
  for (int i = 0; i < cfg.writers; ++i) writers.emplace_back(0xD1B54A32D192ED03ULL + i);

  std::vector<std::thread> threads;
  const auto deadline =
      Clock::now() + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(cfg.runtime));
  for (int i = 0; i < cfg.readers; ++i) {
    threads.emplace_back([&, i]() {
      Worker<Ops>(&lock, /*is_writer=*/false, cfg.read_shift, cfg.read_burst,
                  deadline, &readers[i]);
    });
  }
  for (int i = 0; i < cfg.writers; ++i) {
    threads.emplace_back([&, i]() {
      Worker<Ops>(&lock, /*is_writer=*/true, cfg.write_shift, cfg.write_burst,
                  deadline, &writers[i]);
    });
  }
  for (auto &t : threads) t.join();

  auto row = [&](const char *role, const Stats &s) {
    std::cout << std::left << std::setw(16) << Ops::Name() << std::setw(8)
              << role << std::right << std::setw(14) << s.count << std::setw(13)
              << std::fixed << std::setprecision(1) << s.p50 << std::setw(13)
              << s.p75 << std::setw(13) << s.p90 << std::setw(13) << s.p99
              << "\n";
  };
  if (cfg.readers > 0) row("reader", Aggregate(readers));
  if (cfg.writers > 0) row("writer", Aggregate(writers));
}

void RunConfig(const Config &cfg) {
  std::cout << "\n=== " << cfg.name << " ===\n";
  std::cout << "readers=" << cfg.readers << " writers=" << cfg.writers
            << " read_shift=" << cfg.read_shift
            << " read_burst=" << cfg.read_burst
            << " write_shift=" << cfg.write_shift
            << " write_burst=" << cfg.write_burst << " runtime=" << cfg.runtime
            << "s\n";
  std::cout << std::left << std::setw(16) << "lock" << std::setw(8) << "role"
            << std::right << std::setw(14) << "acquires" << std::setw(13)
            << "p50(ns)" << std::setw(13) << "p75(ns)" << std::setw(13)
            << "p90(ns)" << std::setw(13) << "p99(ns)" << "\n";
  std::cout << std::string(90, '-') << "\n";
  RunLock<MutexOps>(cfg);
  RunLock<RwLockOps>(cfg);
  RunLock<CvRwLockOps>(cfg);
}

std::vector<Config> BuiltinScenarios() {
  return {
      {"Uncontested Reads", 1, 0, 0, 0, 0, 0, 2.0},
      {"Uncontested Writes", 0, 1, 0, 0, 0, 0, 2.0},
      {"Mostly Reads, Few Writes", 3, 1, 0, 0, 0.5, 0.1, 5.0},
      {"Mixed reads + writes", 2, 2, 0, 0, 0, 0, 5.0},
      {"Mostly Writes, Few Reads", 1, 3, 0.5, 0.1, 0, 0, 5.0},
  };
}

}  // namespace

int main(int argc, char **argv) {
  std::cout << "Lock workload benchmark\n";
  std::cout << "sizeof(ctp::Mutex)    = " << sizeof(ctp::Mutex) << " bytes\n";
  std::cout << "sizeof(ctp::RwLock)   = " << sizeof(ctp::RwLock) << " bytes\n";
  std::cout << "sizeof(ctp::CvRwLock) = " << sizeof(ctp::CvRwLock) << " bytes\n";

  if (argc > 1) {
    Config cfg;
    cfg.name = "Custom";
    cfg.readers = std::atoi(argv[1]);
    if (argc > 2) cfg.writers = std::atoi(argv[2]);
    if (argc > 3) cfg.read_shift = std::atof(argv[3]);
    if (argc > 4) cfg.read_burst = std::atof(argv[4]);
    if (argc > 5) cfg.write_shift = std::atof(argv[5]);
    if (argc > 6) cfg.write_burst = std::atof(argv[6]);
    if (argc > 7) cfg.runtime = std::atof(argv[7]);
    if (cfg.runtime <= 0.0) cfg.runtime = 5.0;
    RunConfig(cfg);
  } else {
    for (const auto &cfg : BuiltinScenarios()) RunConfig(cfg);
  }
  return 0;
}

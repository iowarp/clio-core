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

// Zero-IPC read path under sustained write load (issue #787).
//
// One hot blob is overwritten in place by a paced writer while R reader
// threads read it through the public API: Tag::GetBlobSize (metadata) or
// Tag::GetBlob (payload). The per-thread ShmReadStats counters tell each
// call's outcome apart: served from shared memory, or fell back to the
// runtime round trip, and why. One CSV row per point on stdout.
//
// Attaches to an external runtime, so readers and writer are clients in
// their own process and a fallback is a real cross-process round trip. The
// blob must land on a RAM tier; cte_config_shm_read.yaml is that setup:
//
//   CLIO_SERVER_CONF=cte_config_shm_read.yaml clio_run runtime start &
//   CLIO_SERVER_CONF=cte_config_shm_read.yaml clio_cte_shm_read_bench
//
// Env knobs (comma lists):
//   BENCH_SIZES    (4096)              blob size in bytes
//   BENCH_RATES    (none,100,1000,max) target puts/s; none = no writer,
//                                      max = back-to-back puts
//   BENCH_READERS  (1,4,8)
//   BENCH_MODES    (meta,payload)
//   BENCH_SECS     (2)                 seconds per point

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/shm_read_stats.h>
#include <clio_runtime/clio_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using clio::cte::core::ShmReadCounters;
using clio::cte::core::ShmReadStats;
using clio::run::u64;
using Clock = std::chrono::steady_clock;

std::vector<std::string> EnvList(const char *name, const char *dflt) {
  const char *e = std::getenv(name);
  std::stringstream ss((e && *e) ? e : dflt);
  std::vector<std::string> out;
  for (std::string item; std::getline(ss, item, ',');) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

/** Log-linear latency histogram: 8 sub-buckets per power of two (~9%). */
struct Hist {
  std::array<u64, 64 * 8> b{};
  u64 n = 0;
  double sum_ns = 0;
  static size_t Bucket(u64 ns) {
    if (ns < 8) return ns;
    int lg = 0;
    while ((ns >> (lg + 1)) != 0) ++lg;
    return static_cast<size_t>(lg) * 8 + ((ns >> (lg - 3)) & 7);
  }
  static double Value(size_t i) {
    if (i < 8) return static_cast<double>(i);
    size_t lg = i / 8, sub = i % 8;
    return std::ldexp(1.0 + (sub + 0.5) / 8.0, static_cast<int>(lg));
  }
  void Add(u64 ns) {
    ++b[Bucket(ns)];
    ++n;
    sum_ns += static_cast<double>(ns);
  }
  void Merge(const Hist &o) {
    for (size_t i = 0; i < b.size(); ++i) b[i] += o.b[i];
    n += o.n;
    sum_ns += o.sum_ns;
  }
  double MeanUs() const { return n ? sum_ns / n / 1000.0 : 0.0; }
  double PctUs(double p) const {
    if (n == 0) return 0.0;
    u64 want = static_cast<u64>(std::ceil(p * n)), acc = 0;
    for (size_t i = 0; i < b.size(); ++i) {
      acc += b[i];
      if (acc >= want) return Value(i) / 1000.0;
    }
    return 0.0;
  }
};

struct ReaderResult {
  Hist all, fast, slow;
  ShmReadStats stats;
  u64 wrong = 0;  // size mismatch or torn page
};

#define CLIO_SHM_READ_STAT_FIELDS(F)                                      \
  F(record_reads) F(record_retries) F(record_fallbacks) F(payload_reads) \
  F(payload_hits) F(payload_write_in_flight) F(payload_raced_writer)     \
  F(payload_not_servable)

ShmReadStats Diff(const ShmReadStats &a, const ShmReadStats &b) {
  ShmReadStats d;
#define F(x) d.x = a.x - b.x;
  CLIO_SHM_READ_STAT_FIELDS(F)
#undef F
  return d;
}

void Accumulate(ShmReadStats *a, const ShmReadStats &b) {
#define F(x) a->x += b.x;
  CLIO_SHM_READ_STAT_FIELDS(F)
#undef F
}

u64 Load(const std::atomic<u64> &c) {
  return c.load(std::memory_order_relaxed);
}

/** Fill a page with its generation in every word, so a reader can tell a
 *  whole page (all words equal) from a torn one. */
void FillGen(char *p, u64 size, u64 gen) {
  u64 *w = reinterpret_cast<u64 *>(p);
  for (u64 i = 0; i < size / 8; ++i) w[i] = gen;
}

bool PageWhole(const char *p, u64 size) {
  const u64 *w = reinterpret_cast<const u64 *>(p);
  return w[0] == w[size / 8 - 1] && w[0] == w[size / 16];
}

struct Point {
  u64 size;
  std::string mode;
  std::string rate;
  int readers;
};

void RunPoint(clio::cte::core::Tag &tag, const std::string &blob,
              const Point &pt, double secs) {
  std::atomic<bool> stop{false};
  std::atomic<bool> go{false};
  std::atomic<u64> writes{0};
  std::atomic<u64> write_ns{0};
  std::atomic<u64> write_fail{0};

  const bool has_writer = pt.rate != "none";
  const double period_ns =
      (has_writer && pt.rate != "max") ? 1e9 / std::atof(pt.rate.c_str()) : 0;

  std::thread writer;
  if (has_writer) {
    writer = std::thread([&] {
      auto *ipc = CLIO_IPC;
      ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(pt.size);
      if (wb.IsNull()) {
        write_fail.store(1);
        return;
      }
      while (!go.load()) std::this_thread::yield();
      u64 gen = 1;
      auto next = Clock::now();
      while (!stop.load(std::memory_order_relaxed)) {
        FillGen(wb.ptr_, pt.size, ++gen);
        auto t0 = Clock::now();
        auto put = tag.AsyncPutBlob(blob, wb.shm_.template Cast<void>(),
                                    pt.size, 0, 1.0f);
        put.Wait();
        auto t1 = Clock::now();
        if (put->GetReturnCode() != 0) write_fail.fetch_add(1);
        writes.fetch_add(1, std::memory_order_relaxed);
        write_ns.fetch_add(static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count()));
        if (period_ns > 0) {
          next += std::chrono::nanoseconds(static_cast<long long>(period_ns));
          if (next < t1 - std::chrono::milliseconds(10)) next = t1;
          std::this_thread::sleep_until(next);
        }
      }
      ipc->FreeBuffer(wb);
    });
  }

  std::vector<ReaderResult> results(pt.readers);
  std::vector<std::thread> readers;
  for (int r = 0; r < pt.readers; ++r) {
    readers.emplace_back([&, r] {
      ReaderResult &res = results[r];
      std::vector<char> buf(pt.size);
      const ShmReadCounters &st = clio::cte::core::ThreadShmReadCounters();
      while (!go.load()) std::this_thread::yield();
      const ShmReadStats start = st.Snapshot();
      while (!stop.load(std::memory_order_relaxed)) {
        bool fast;
        auto t0 = Clock::now();
        if (pt.mode == "meta") {
          const u64 fb0 = Load(st.record_fallbacks);
          u64 sz = tag.GetBlobSize(blob);
          fast = Load(st.record_fallbacks) == fb0;
          if (sz != pt.size) ++res.wrong;
        } else {
          const u64 hits0 = Load(st.payload_hits);
          try {
            tag.GetBlob(blob, buf.data(), pt.size, 0);
          } catch (...) {
            ++res.wrong;
          }
          fast = Load(st.payload_hits) != hits0;
        }
        auto ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - t0)
                .count());
        if (pt.mode == "payload" && !PageWhole(buf.data(), pt.size)) {
          ++res.wrong;
        }
        res.all.Add(ns);
        (fast ? res.fast : res.slow).Add(ns);
      }
      res.stats = Diff(st.Snapshot(), start);
    });
  }

  auto t_start = Clock::now();
  go.store(true);
  std::this_thread::sleep_for(std::chrono::duration<double>(secs));
  stop.store(true);
  for (auto &t : readers) t.join();
  if (writer.joinable()) writer.join();
  double elapsed =
      std::chrono::duration<double>(Clock::now() - t_start).count();

  ReaderResult tot;
  for (auto &r : results) {
    tot.all.Merge(r.all);
    tot.fast.Merge(r.fast);
    tot.slow.Merge(r.slow);
    Accumulate(&tot.stats, r.stats);
    tot.wrong += r.wrong;
  }
  const auto &s = tot.stats;
  const double calls = static_cast<double>(tot.all.n);
  const u64 w = writes.load();
  std::printf(
      "%llu,%s,%s,%.0f,%.2f,%d,%.0f,%.4f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,"
      "%.3f,%.3f,%llu,%llu\n",
      static_cast<unsigned long long>(pt.size), pt.mode.c_str(),
      pt.rate.c_str(), w / elapsed,
      w ? write_ns.load() / 1000.0 / w : 0.0, pt.readers, calls / elapsed,
      s.record_reads ? static_cast<double>(s.record_retries) / s.record_reads
                     : 0.0,
      s.record_reads
          ? static_cast<double>(s.record_fallbacks) / s.record_reads
          : 0.0,
      calls ? tot.slow.n / calls : 0.0,
      s.payload_reads
          ? static_cast<double>(s.payload_write_in_flight) / s.payload_reads
          : 0.0,
      s.payload_reads
          ? static_cast<double>(s.payload_raced_writer) / s.payload_reads
          : 0.0,
      tot.all.MeanUs(), tot.all.PctUs(0.5), tot.all.PctUs(0.99),
      tot.fast.MeanUs(), tot.slow.MeanUs(),
      static_cast<unsigned long long>(tot.wrong),
      static_cast<unsigned long long>(write_fail.load()));
  std::fflush(stdout);
}

}  // namespace

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient)) {
    std::fprintf(stderr, "CLIO_INIT failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "CLIO_CTE_CLIENT_INIT failed\n");
    return 1;
  }
  auto *cte = CLIO_CTE_CLIENT;
  if (!cte->HasShmCache() && !cte->AttachShmCache()) {
    std::fprintf(stderr, "SHM metadata cache unavailable; nothing to measure\n");
    return 1;
  }

  const double secs = std::atof(EnvList("BENCH_SECS", "2")[0].c_str());
  clio::cte::core::Tag tag("shm787_tag");

  std::printf(
      "size,mode,rate,writes_per_s,put_us,readers,reads_per_s,"
      "record_retries_per_read,record_fallback_frac,fallback_frac,"
      "write_in_flight_frac,raced_writer_frac,mean_us,p50_us,p99_us,"
      "fast_mean_us,slow_mean_us,wrong,write_fail\n");
  for (const auto &size_s : EnvList("BENCH_SIZES", "4096")) {
    const u64 size = std::strtoull(size_s.c_str(), nullptr, 10);
    const std::string blob = "hot_" + size_s;
    {
      std::vector<char> seed(size);
      FillGen(seed.data(), size, 1);
      auto put = tag.AsyncPutBlob(blob, seed.data(), size, 0, 1.0f);
      put.Wait();
      clio::cte::core::ShmBlobRecord rec;
      if (put->GetReturnCode() != 0 ||
          !cte->TryGetBlobRecordShm(tag.GetTagId(), blob, &rec) ||
          !rec.IsDirectReadable()) {
        std::fprintf(stderr,
                     "blob %s not cached as direct-readable; check that the "
                     "only tier is RAM (cte_config_shm_read.yaml)\n",
                     blob.c_str());
        return 1;
      }
    }
    for (const auto &mode : EnvList("BENCH_MODES", "meta,payload")) {
      for (const auto &rate : EnvList("BENCH_RATES", "none,100,1000,max")) {
        for (const auto &rd : EnvList("BENCH_READERS", "1,4,8")) {
          RunPoint(tag, blob, Point{size, mode, rate, std::atoi(rd.c_str())},
                   secs);
        }
      }
    }
  }
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

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
 * clio_cte_vector_stress: the CTE driven the way the gpu_vector adapter
 * drives it, with no GPU anywhere.
 *
 * A "vector" is one tag holding P pages per node, blob name = decimal global
 * page number, so pages are hash-distributed over the nodes exactly as the
 * adapter's are (an owner is whichever node the name hashes to, not the
 * writer). Each step, T worker threads per node (the adapter's work-groups)
 * do what a stencil or a streaming kernel does on the host side:
 *
 *   halo:      read the left and right neighbours of each own page at the
 *              generation the previous step published (generational get);
 *   stream:    read R random pages of the whole deck at that generation;
 *   writeback: publish each own page's new contents as the next generation;
 *   barrier:   the CTE-blob barrier the benchmarks use (bench_dist.h);
 *   checkpoint (every --ckpt-every steps): the vector's Copy: a tag in the
 *              checkpoint pool that resolves through the live tag, then
 *              every own page materialised server-side.
 *
 * Reads and writes go through the same batched PodMultiGet/PutBlob tasks the
 * device flush uses (--batch records per task, remote records forwarded by
 * the owner-resolving runtime); --batch 1 issues scalar puts and gets.
 *
 * Every byte read is verified against the deterministic contents the writer
 * must have published, and every return code is counted, so a stale
 * generation, a lost put, a refused put (rc 11) or a network-timeout
 * completion (rc -1000) is a named failure, not a hang. If this passes at a
 * node count where the vector fails, the defect is on the device side; if it
 * fails, it is clio-core's.
 *
 * One process per node; each embeds its runtime (CLIO_INIT(kClient, true))
 * unless CLIO_STRESS_ATTACH=1. Runs under run_colocated.sh unchanged.
 */

#include <clio_cte/checkpoint/checkpoint_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bench_dist.h"

namespace {

using clio::run::u32;
using clio::run::u64;
using Clock = std::chrono::steady_clock;

struct Args {
  u32 nodes = 1, node = 0;
  u64 pages_per_node = 1024;  // deck per node, in pages
  u64 page_kb = 1024;
  u32 threads = 64;
  u32 steps = 4;
  u32 halo = 1;               // neighbours read on each side per own page
  u32 stream = 0;             // random whole-deck reads per own page per step
  bool generational = true;   // demand/publish generations (the vector does)
  u32 batch = 16;             // records per PodMulti task; 1 = scalar ops
  u32 ckpt_every = 0;         // checkpoint after every k steps; 0 = never
  int barrier_timeout_s = 300;
  u32 barriers = 0;           // extra barrier-only rounds after the steps
  u32 max_error_lines = 8;
};

/** @brief Parse the command line; returns false and prints usage on error. */
bool ParseArgs(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
    if (k == "--nodes") a.nodes = static_cast<u32>(std::atoi(next()));
    else if (k == "--node") a.node = static_cast<u32>(std::atoi(next()));
    else if (k == "--pages-per-node") a.pages_per_node = std::strtoull(next(), nullptr, 10);
    else if (k == "--page-kb") a.page_kb = std::strtoull(next(), nullptr, 10);
    else if (k == "--threads") a.threads = static_cast<u32>(std::atoi(next()));
    else if (k == "--steps") a.steps = static_cast<u32>(std::atoi(next()));
    else if (k == "--halo") a.halo = static_cast<u32>(std::atoi(next()));
    else if (k == "--stream") a.stream = static_cast<u32>(std::atoi(next()));
    else if (k == "--no-gen") a.generational = false;
    else if (k == "--batch") a.batch = static_cast<u32>(std::atoi(next()));
    else if (k == "--ckpt-every") a.ckpt_every = static_cast<u32>(std::atoi(next()));
    else if (k == "--barrier-timeout") a.barrier_timeout_s = std::atoi(next());
    else if (k == "--barriers") a.barriers = static_cast<u32>(std::atoi(next()));
    else {
      std::fprintf(stderr,
                   "usage: %s [--nodes N --node r] [--pages-per-node P] "
                   "[--page-kb K] [--threads T] [--steps S] [--halo H] "
                   "[--stream R] [--no-gen] [--batch B] [--ckpt-every k] "
                   "[--barrier-timeout s] [--barriers N]\n",
                   argv[0]);
      return false;
    }
  }
  if (a.nodes == 0 || a.node >= a.nodes || a.threads == 0 || a.page_kb == 0) {
    std::fprintf(stderr, "bad --nodes/--node/--threads/--page-kb\n");
    return false;
  }
  if (a.batch == 0) a.batch = 1;
  if (a.batch > clio::cte::core::kPodMultiMax) a.batch = clio::cte::core::kPodMultiMax;
  return true;
}

/** @brief The 64-bit word at index i of page p at generation g. */
inline u64 Word(u64 p, u64 g, u64 i) {
  u64 x = (p + 1) * 0x9E3779B97F4A7C15ull ^ (g + 1) * 0xC2B2AE3D27D4EB4Full ^ i;
  x ^= x >> 31; x *= 0x7FB5D329728EA185ull; x ^= x >> 27;
  x *= 0x81DADEF4BC2DD44Dull; x ^= x >> 33;
  return x;
}

/** @brief Fill a page buffer with the contents of page p at generation g. */
void FillPage(char *buf, u64 bytes, u64 p, u64 g) {
  u64 *w = reinterpret_cast<u64 *>(buf);
  const u64 n = bytes / sizeof(u64);
  w[0] = p; w[1] = g;
  for (u64 i = 2; i < n; ++i) w[i] = Word(p, g, i);
}

/**
 * @brief Check a page buffer against page p read at generation g.
 *
 * Readers and writers of one step overlap on purpose, as the vector's do:
 * a page demanded at gen g may already carry gen g+1 (its owner published
 * the next step), never anything else, because every step ends in a
 * barrier. The content must match whichever generation the header claims.
 * @param seen  receives the generation the page carried
 * @return the index of the first wrong word, or -1 when the page is right
 */
long long VerifyPage(const char *buf, u64 bytes, u64 p, u64 g, u64 *seen) {
  const u64 *w = reinterpret_cast<const u64 *>(buf);
  const u64 n = bytes / sizeof(u64);
  *seen = w[1];
  if (w[0] != p) return 0;
  if (w[1] != g && w[1] != g + 1) return 1;
  for (u64 i = 2; i < n; ++i) if (w[i] != Word(p, w[1], i)) return static_cast<long long>(i);
  return -1;
}

/** Counters shared by the worker threads of one node. */
struct Stats {
  std::atomic<u64> gets{0}, puts{0}, get_errors{0}, put_errors{0}, mismatches{0};
  std::atomic<u64> ahead{0};  /**< reads that returned the next generation */
  std::atomic<u64> rc_no_targets{0}, rc_net_timeout{0}, rc_gen{0}, rc_other{0};
  std::atomic<u64> ckpt_pages{0}, ckpt_errors{0};
  std::atomic<u32> error_lines{0};
  std::mutex log_mu;
};

/** @brief Attribute a non-zero return code to a named bucket. */
void CountRc(Stats &st, u32 rc) {
  if (rc == 11) st.rc_no_targets++;
  else if (rc == static_cast<u32>(-1000)) st.rc_net_timeout++;
  else if (rc == 1) st.rc_gen++;
  else st.rc_other++;
}

/** @brief Log the first few failures in full; count the rest. */
void LogFailure(Stats &st, const Args &a, const char *what, u64 page, u64 gen,
                u32 rc, long long word) {
  if (st.error_lines.fetch_add(1) >= a.max_error_lines) return;
  std::lock_guard<std::mutex> lk(st.log_mu);
  std::fprintf(stderr, "STRESS ERROR node %u: %s page %llu gen %llu rc=%u (%d) word=%lld\n",
               a.node, what, (unsigned long long)page, (unsigned long long)gen, rc,
               static_cast<int>(rc), word);
}

class VectorStress {
 public:
  VectorStress(const Args &a) : a_(a), page_bytes_(a.page_kb * 1024) {}

  /** @brief Create the tags; returns false when the CTE is unusable. */
  bool Init() {
    auto *cte = CLIO_CTE_CLIENT;
    auto t = cte->AsyncGetOrCreateTag("cte_stress_vector");
    t.Wait();
    if (t->return_code_.load() != 0) {
      HLOG(kError, "GetOrCreateTag(cte_stress_vector) rc={}", t->return_code_.load());
      return false;
    }
    tag_ = t->tag_id_;
    auto b = cte->AsyncGetOrCreateTag("cte_stress_barrier");
    b.Wait();
    if (b->return_code_.load() != 0) {
      HLOG(kError, "GetOrCreateTag(cte_stress_barrier) rc={}", b->return_code_.load());
      return false;
    }
    bar_tag_ = b->tag_id_;
    return true;
  }

  /** @brief Seed, then run the steps; returns false on any failure. */
  bool Run() {
    const u64 first = a_.node * a_.pages_per_node;
    std::printf("cte vector stress: nodes=%u node=%u pages/node=%llu page_kb=%llu "
                "threads=%u steps=%u halo=%u stream=%u gen=%d\n",
                a_.nodes, a_.node, (unsigned long long)a_.pages_per_node,
                (unsigned long long)a_.page_kb, a_.threads, a_.steps, a_.halo,
                a_.stream, a_.generational ? 1 : 0);
    const auto t0 = Clock::now();
    RunPhase("seed", first, /*gen=*/1, /*read=*/false);
    if (!Sync(0)) return false;
    const double seed_ms = Ms(t0);
    std::printf("  seed: %.1f ms, %llu puts, %llu errors\n", seed_ms,
                (unsigned long long)st_.puts.load(), (unsigned long long)Errors());
    double total_ms = 0.0;
    for (u32 s = 1; s <= a_.steps && Errors() == 0; ++s) {
      const auto ts = Clock::now();
      RunPhase("step", first, s, /*read=*/true);
      if (!Sync(s)) return false;
      if (a_.ckpt_every != 0 && s % a_.ckpt_every == 0) {
        const auto tc = Clock::now();
        Checkpoint(first, s);
        std::printf("  checkpoint after step %u: %.1f ms, %llu pages, %llu errors\n", s,
                    Ms(tc), (unsigned long long)st_.ckpt_pages.load(),
                    (unsigned long long)st_.ckpt_errors.load());
        if (!Sync(1000 + s)) return false;
      }
      const double ms = Ms(ts);
      total_ms += ms;
      std::printf("  step %u: %.1f ms gets=%llu puts=%llu errors=%llu\n", s, ms,
                  (unsigned long long)st_.gets.load(),
                  (unsigned long long)st_.puts.load(),
                  (unsigned long long)Errors());
      std::fflush(stdout);
    }
    // Barrier-only rounds: the put-then-poll exchange every gate uses, with
    // nothing else in flight, to see how often a round is missed and how
    // long a clean one takes.
    if (a_.barriers != 0 && Errors() == 0) {
      const auto tb = Clock::now();
      double worst_ms = 0.0;
      u32 done = 0;
      for (; done < a_.barriers; ++done) {
        const auto t1 = Clock::now();
        if (!Sync(2000 + done)) break;
        worst_ms = std::max(worst_ms, Ms(t1));
      }
      std::printf("  barriers: %u/%u rounds in %.1f ms, slowest %.1f ms\n", done,
                  a_.barriers, Ms(tb), worst_ms);
      if (done != a_.barriers) return false;
    }
    std::fprintf(stderr,
                 "STRESS nodes=%u node=%u pages=%llu page_kb=%llu threads=%u steps=%u "
                 "halo=%u stream=%u ms=%.1f seed_ms=%.1f gets=%llu puts=%llu "
                 "get_errors=%llu put_errors=%llu mismatches=%llu ahead=%llu rc11=%llu "
                 "rc_net_timeout=%llu rc_gen=%llu rc_other=%llu batch=%u "
                 "ckpt_pages=%llu ckpt_errors=%llu\n",
                 a_.nodes, a_.node, (unsigned long long)a_.pages_per_node,
                 (unsigned long long)a_.page_kb, a_.threads, a_.steps, a_.halo,
                 a_.stream, total_ms, seed_ms, (unsigned long long)st_.gets.load(),
                 (unsigned long long)st_.puts.load(),
                 (unsigned long long)st_.get_errors.load(),
                 (unsigned long long)st_.put_errors.load(),
                 (unsigned long long)st_.mismatches.load(),
                 (unsigned long long)st_.ahead.load(),
                 (unsigned long long)st_.rc_no_targets.load(),
                 (unsigned long long)st_.rc_net_timeout.load(),
                 (unsigned long long)st_.rc_gen.load(),
                 (unsigned long long)st_.rc_other.load(), a_.batch,
                 (unsigned long long)st_.ckpt_pages.load(),
                 (unsigned long long)st_.ckpt_errors.load());
    return Errors() == 0;
  }

 private:
  u64 Errors() const {
    return st_.get_errors.load() + st_.put_errors.load() + st_.mismatches.load() +
           st_.ckpt_errors.load();
  }
  static double Ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  }
  bool Sync(u64 round) {
    if (a_.nodes <= 1) return true;
    if (clio_bench_dist::Barrier(*CLIO_CTE_CLIENT, bar_tag_, a_.node, a_.nodes,
                                 round, "stressbar", a_.barrier_timeout_s)) {
      return true;
    }
    std::fprintf(stderr, "STRESS ERROR node %u: barrier round %llu failed\n", a_.node,
                 (unsigned long long)round);
    return false;
  }
  clio::cte::core::Context Ctx(u64 gen, bool get = false) const {
    clio::cte::core::Context c;
    if (a_.generational) {
      c.op_flags_ |= clio::cte::core::Context::kGenerational;
      c.generation_ = gen;
    }
    if (get) c.create_on_get_ = true;  // the device fetch sets this
    return c;
  }
  static std::string Name(u64 page) { return std::to_string(page); }

  /**
   * One phase over this node's pages, split across the worker threads.
   * @param first  this node's first global page
   * @param gen    the generation being published (reads demand gen; the
   *               seed publishes gen 1 and reads nothing)
   * @param read   whether to do the halo and stream reads before writing
   */
  void RunPhase(const char *what, u64 first, u64 gen, bool read) {
    std::vector<std::thread> ts;
    ts.reserve(a_.threads);
    const u64 per = (a_.pages_per_node + a_.threads - 1) / a_.threads;
    for (u32 t = 0; t < a_.threads; ++t) {
      const u64 lo = first + t * per;
      const u64 hi = std::min(first + a_.pages_per_node, lo + per);
      if (lo >= hi) break;
      ts.emplace_back([=] { Worker(what, lo, hi, gen, read, t); });
    }
    for (auto &th : ts) th.join();
  }

  /** A page buffer in shared memory plus what it currently holds. */
  struct Slot {
    ctp::ipc::FullPtr<char> buf;
    ctp::ipc::ShmPtr<> ptr;
    u64 page = 0, gen = 0;
  };

  /** @brief Allocate n page slots; frees them in FreeSlots. */
  std::vector<Slot> AllocSlots(u32 n) {
    std::vector<Slot> v(n);
    for (auto &sl : v) {
      sl.buf = CLIO_IPC->AllocateBuffer(page_bytes_);
      sl.ptr = sl.buf.shm_.template Cast<void>();
    }
    return v;
  }
  void FreeSlots(std::vector<Slot> &v) {
    for (auto &sl : v) CLIO_IPC->FreeBuffer(sl.buf);
  }

  /** @brief One thread's share of a phase: reads at gen, then publishes gen+1 (seed: gen). */
  void Worker(const char *what, u64 lo, u64 hi, u64 gen, bool read, u32 tid) {
    auto *cte = CLIO_CTE_CLIENT;
    std::vector<Slot> gets = AllocSlots(a_.batch), puts = AllocSlots(a_.batch);
    u32 ng = 0, np = 0;
    const u64 total = a_.nodes * a_.pages_per_node;
    u64 rng = (a_.node + 1) * 0x9E3779B97F4A7C15ull ^ (tid + 1) * 0xD1B54A32D192ED03ull ^ gen;
    auto queue_get = [&](u64 page) {
      gets[ng].page = page; gets[ng].gen = gen;
      if (++ng == a_.batch) { IssueGets(cte, gets, ng); ng = 0; }
    };
    const u64 publish = read ? gen + 1 : gen;
    for (u64 p = lo; p < hi; ++p) {
      if (read) {
        for (u32 h = 1; h <= a_.halo; ++h) {
          if (p >= h) queue_get(p - h);
          if (p + h < total) queue_get(p + h);
        }
        for (u32 r = 0; r < a_.stream; ++r) {
          rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
          queue_get(rng % total);
        }
      }
      puts[np].page = p; puts[np].gen = publish;
      FillPage(puts[np].buf.ptr_, page_bytes_, p, publish);
      if (++np == a_.batch) { IssuePuts(cte, what, puts, np); np = 0; }
    }
    if (ng) IssueGets(cte, gets, ng);
    if (np) IssuePuts(cte, what, puts, np);
    FreeSlots(gets);
    FreeSlots(puts);
  }

  /** @brief Publish n queued pages: one PodMultiPutBlob, or scalar puts when batch is 1. */
  void IssuePuts(clio::cte::core::Client *cte, const char *what, std::vector<Slot> &sl, u32 n) {
    st_.puts += n;
    if (a_.batch == 1) {
      auto t = cte->AsyncPutBlob(tag_, Name(sl[0].page), 0, page_bytes_, sl[0].ptr, 0.5f,
                                 Ctx(sl[0].gen));
      t.Wait();
      const u32 rc = t->return_code_.load();
      if (rc != 0) { st_.put_errors++; CountRc(st_, rc); LogFailure(st_, a_, what, sl[0].page, sl[0].gen, rc, -1); }
      return;
    }
    auto b = cte->NewPodBatch<clio::cte::core::PodMultiPutBlobTask>(tag_, Ctx(sl[0].gen));
    for (u32 i = 0; i < n; ++i) {
      b.get()->Add(Name(sl[i].page).c_str(), 0, page_bytes_, sl[i].ptr, 0.5f);
    }
    auto f = cte->AsyncPodBatch(b);
    f.Wait();
    for (u32 i = 0; i < n; ++i) {
      const u32 rc = f->reqs_[i].rc_;
      if (rc != 0) { st_.put_errors++; CountRc(st_, rc); LogFailure(st_, a_, what, sl[i].page, sl[i].gen, rc, -1); }
    }
    if (f->return_code_.load() != 0 && f->num_ok_ == n) {
      st_.put_errors++; CountRc(st_, f->return_code_.load());
      LogFailure(st_, a_, "multiput", sl[0].page, sl[0].gen, f->return_code_.load(), -1);
    }
  }

  /** @brief Fetch n queued pages at their generations and verify every byte. */
  void IssueGets(clio::cte::core::Client *cte, std::vector<Slot> &sl, u32 n) {
    st_.gets += n;
    for (u32 i = 0; i < n; ++i) std::memset(sl[i].buf.ptr_, 0, page_bytes_);
    if (a_.batch == 1) {
      auto t = cte->AsyncGetBlob(tag_, Name(sl[0].page), 0, page_bytes_, 0u, sl[0].ptr,
                                 clio::run::PoolQuery::Dynamic(), Ctx(sl[0].gen, true));
      t.Wait();
      const u32 rc = t->return_code_.load();
      if (rc != 0) { st_.get_errors++; CountRc(st_, rc); LogFailure(st_, a_, "get", sl[0].page, sl[0].gen, rc, -1); return; }
      Verify(sl[0]);
      return;
    }
    auto b = cte->NewPodBatch<clio::cte::core::PodMultiGetBlobTask>(tag_, Ctx(sl[0].gen, true));
    for (u32 i = 0; i < n; ++i) {
      b.get()->Add(Name(sl[i].page).c_str(), 0, page_bytes_, sl[i].ptr, 0.5f);
    }
    auto f = cte->AsyncPodBatch(b);
    f.Wait();
    for (u32 i = 0; i < n; ++i) {
      const u32 rc = f->reqs_[i].rc_;
      if (rc != 0) { st_.get_errors++; CountRc(st_, rc); LogFailure(st_, a_, "multiget", sl[i].page, sl[i].gen, rc, -1); continue; }
      Verify(sl[i]);
    }
  }

  void Verify(const Slot &sl) {
    u64 seen = 0;
    const long long bad = VerifyPage(sl.buf.ptr_, page_bytes_, sl.page, sl.gen, &seen);
    if (bad >= 0) {
      st_.mismatches++;
      LogFailure(st_, a_, "verify", sl.page, sl.gen, static_cast<u32>(seen), bad);
    } else if (seen != sl.gen) {
      st_.ahead++;
    }
  }

  /**
   * The vector's Copy: a tag in the checkpoint pool that resolves through the
   * live tag, then this node's pages materialised server-side (a size-0 get
   * to the checkpoint pool per page, 32 in flight), exactly as
   * gpu_vector.h's Copy + MaterializeAll do.
   */
  void Checkpoint(u64 first, u32 step) {
    namespace ck = clio::cte::checkpoint;
    auto *ipc = CLIO_CPU_IPC;
    if (!ckpt_ready_) {
      ck::Client ckpt(ck::kCheckpointPoolId, clio::cte::core::kCtePoolId);
      ck::CheckpointConfig params;
      params.next_pool_id_ = clio::cte::core::kCtePoolId;
      auto cf = ckpt.AsyncCreateCheckpoint(clio::run::PoolQuery::Local(), ck::kCheckpointPoolName,
                                           ck::kCheckpointPoolId, params);
      cf.Wait();
      if (cf->GetReturnCode() != 0) {
        st_.ckpt_errors++;
        LogFailure(st_, a_, "checkpoint-pool", 0, step, cf->GetReturnCode(), -1);
        return;
      }
      ckpt_ready_ = true;
    }
    const std::string name = "cte_stress_ckpt_" + std::to_string(step);
    auto tf = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name, ck::kCheckpointPoolId, ck::kCheckpointPoolName,
                                                    std::string("cte_stress_vector"));
    tf.Wait();
    if (tf->GetReturnCode() != 0) {
      st_.ckpt_errors++;
      LogFailure(st_, a_, "checkpoint-tag", 0, step, tf->GetReturnCode(), -1);
      return;
    }
    const clio::cte::core::TagId ctag = tf->tag_id_;
    std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futs;
    auto reap = [&]() {
      for (auto &f : futs) {
        f.Wait();
        st_.ckpt_pages++;
        if (f->GetReturnCode() != 0) {
          st_.ckpt_errors++;
          LogFailure(st_, a_, "materialize", 0, step, f->GetReturnCode(), -1);
        }
      }
      futs.clear();
    };
    for (u64 p = first; p < first + a_.pages_per_node; ++p) {
      clio::cte::core::Context ctx;
      std::snprintf(ctx.fault_params_, clio::cte::core::Context::kFaultParamsSize, "%s",
                    "cte_stress_vector");
      auto sub = ipc->NewTask<clio::cte::core::GetBlobTask>(
          clio::run::CreateTaskId(), ck::kCheckpointPoolId, clio::run::PoolQuery::Local(), ctag,
          Name(p), 0, 0, 0, ctp::ipc::ShmPtr<>::GetNull(), ctx);
      futs.push_back(ipc->Send(sub));
      if (futs.size() >= 32) reap();
    }
    reap();
  }

  Args a_;
  u64 page_bytes_;
  clio::cte::core::TagId tag_{}, bar_tag_{};
  bool ckpt_ready_ = false;
  Stats st_;
};

}  // namespace

int main(int argc, char **argv) {
  Args a;
  if (!ParseArgs(argc, argv, a)) return 2;
  const char *attach = std::getenv("CLIO_STRESS_ATTACH");
  const bool embed = !(attach != nullptr && attach[0] == '1');
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, embed)) {
    HLOG(kError, "CLIO_INIT failed");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "CLIO_CTE_CLIENT_INIT failed");
    return 1;
  }
  VectorStress bench(a);
  if (!bench.Init()) return 1;
  const bool ok = bench.Run();
  clio::run::CLIO_RUNTIME_FINALIZE();
  return ok ? 0 : 1;
}

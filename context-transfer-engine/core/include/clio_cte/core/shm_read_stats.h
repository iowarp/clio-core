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

#ifndef CLIO_CTE_CORE_SHM_READ_STATS_H_
#define CLIO_CTE_CORE_SHM_READ_STATS_H_

#include <clio_cte/api.h>
#include <clio_runtime/types.h>

#include <atomic>

namespace clio::cte::core {

/**
 * How this process's zero-IPC reads of the shared-memory metadata cache
 * (issue #783) have fared (issue #787).
 *
 * The fast path degrades to RPC instead of returning wrong data, and before
 * these counters that degradation was invisible: a workload could lose the
 * whole speedup and nothing would say so. Every field is a running total
 * since process start.
 *
 * A record read is one TryGetBlobRecordShm (GetBlobSize, and both ends of
 * every payload read). A payload read is one TryReadBlobShm (GetBlob served
 * from the RAM tier). The record counters include the payload path's own
 * record reads.
 */
struct ShmReadStats {
  /** Blob record lookups. */
  clio::run::u64 record_reads = 0;
  /** Probe passes repeated because a writer was mid-update. */
  clio::run::u64 record_retries = 0;
  /** Lookups that ran out of retries; the caller took the RPC path. */
  clio::run::u64 record_fallbacks = 0;
  /** Payload reads attempted. */
  clio::run::u64 payload_reads = 0;
  /** Payload reads served from shared memory. */
  clio::run::u64 payload_hits = 0;
  /** Refused because an in-place put of the blob was in flight. */
  clio::run::u64 payload_write_in_flight = 0;
  /** Discarded because the blob moved or was rewritten during the copy. */
  clio::run::u64 payload_raced_writer = 0;
  /** Not servable here: uncached, not RAM-resident, transformed, out of
   *  range, or its record read fell back. */
  clio::run::u64 payload_not_servable = 0;
};

/**
 * One thread's counters. Only the owning thread writes them, with plain
 * relaxed load+store (no read-modify-write), so a hit costs no shared
 * cache-line traffic; other threads read them only to sum.
 */
struct ShmReadCounters {
  std::atomic<clio::run::u64> record_reads{0};
  std::atomic<clio::run::u64> record_retries{0};
  std::atomic<clio::run::u64> record_fallbacks{0};
  std::atomic<clio::run::u64> payload_reads{0};
  std::atomic<clio::run::u64> payload_hits{0};
  std::atomic<clio::run::u64> payload_write_in_flight{0};
  std::atomic<clio::run::u64> payload_raced_writer{0};
  std::atomic<clio::run::u64> payload_not_servable{0};

  /** Owner-thread increment. */
  static void Bump(std::atomic<clio::run::u64> &c, clio::run::u64 n = 1) {
    c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  ShmReadStats Snapshot() const;
};

/** The calling thread's registered counters. Out of line so the client
 *  library and the executable share one instance per thread. */
CLIO_CTE_API ShmReadCounters &RegisteredThreadShmReadCounters();

/** The calling thread's counters, for the read hot path. Caches the
 *  registered instance in a plain thread_local pointer: no initialization
 *  guard and no call after the first use (about 10 ns per read otherwise,
 *  against a 0.08 us metadata hit). */
inline ShmReadCounters &ThreadShmReadCounters() {
  thread_local ShmReadCounters *counters = nullptr;
  if (counters == nullptr) {
    counters = &RegisteredThreadShmReadCounters();
  }
  return *counters;
}

/** Totals for the whole process, including threads that have exited. */
CLIO_CTE_API ShmReadStats GetShmReadStats();

/** Totals for the calling thread only. */
CLIO_CTE_API ShmReadStats GetThreadShmReadStats();

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_SHM_READ_STATS_H_

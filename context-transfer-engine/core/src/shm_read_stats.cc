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

#include <clio_cte/core/shm_read_stats.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace clio::cte::core {

namespace {

void Add(ShmReadStats *to, const ShmReadStats &s) {
  to->record_reads += s.record_reads;
  to->record_retries += s.record_retries;
  to->record_fallbacks += s.record_fallbacks;
  to->payload_reads += s.payload_reads;
  to->payload_hits += s.payload_hits;
  to->payload_write_in_flight += s.payload_write_in_flight;
  to->payload_raced_writer += s.payload_raced_writer;
  to->payload_not_servable += s.payload_not_servable;
}

/** Every live thread's counters, plus what exited threads left behind.
 *  Leaked on purpose: threads can exit after static destructors run. */
struct Registry {
  std::mutex mu;
  std::vector<const ShmReadCounters *> live;
  ShmReadStats retired;
};

Registry &GetRegistry() {
  static Registry *r = new Registry();
  return *r;
}

struct ThreadSlot {
  ShmReadCounters counters;
  ThreadSlot() {
    Registry &r = GetRegistry();
    std::lock_guard<std::mutex> lock(r.mu);
    r.live.push_back(&counters);
  }
  ~ThreadSlot() {
    Registry &r = GetRegistry();
    std::lock_guard<std::mutex> lock(r.mu);
    Add(&r.retired, counters.Snapshot());
    r.live.erase(std::remove(r.live.begin(), r.live.end(), &counters),
                 r.live.end());
  }
};

}  // namespace

ShmReadStats ShmReadCounters::Snapshot() const {
  constexpr auto kRelaxed = std::memory_order_relaxed;
  ShmReadStats s;
  s.record_reads = record_reads.load(kRelaxed);
  s.record_retries = record_retries.load(kRelaxed);
  s.record_fallbacks = record_fallbacks.load(kRelaxed);
  s.payload_reads = payload_reads.load(kRelaxed);
  s.payload_hits = payload_hits.load(kRelaxed);
  s.payload_write_in_flight = payload_write_in_flight.load(kRelaxed);
  s.payload_raced_writer = payload_raced_writer.load(kRelaxed);
  s.payload_not_servable = payload_not_servable.load(kRelaxed);
  return s;
}

ShmReadCounters &RegisteredThreadShmReadCounters() {
  thread_local ThreadSlot slot;
  return slot.counters;
}

ShmReadStats GetShmReadStats() {
  Registry &r = GetRegistry();
  std::lock_guard<std::mutex> lock(r.mu);
  ShmReadStats total = r.retired;
  for (const ShmReadCounters *c : r.live) {
    Add(&total, c->Snapshot());
  }
  return total;
}

ShmReadStats GetThreadShmReadStats() {
  return ThreadShmReadCounters().Snapshot();
}

}  // namespace clio::cte::core

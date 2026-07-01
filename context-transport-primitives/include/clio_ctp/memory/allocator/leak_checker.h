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

#ifndef CLIO_CTP_MEMORY_ALLOCATOR_LEAK_CHECKER_H_
#define CLIO_CTP_MEMORY_ALLOCATOR_LEAK_CHECKER_H_

#include "clio_ctp/constants/macros.h"

// Host-only facility: the registry stores std::string/std::vector and logs via
// HLOG, none of which belong in device code. Device translation units get an
// empty header; the allocator destructors that call it are themselves guarded
// by `#if CTP_IS_HOST`, so there are no dangling references on the device pass.
#if CTP_IS_HOST

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

#include "clio_ctp/util/logging.h"

namespace ctp::ipc {

/**
 * Process-global registry that an allocator's *destructor* reports to when it
 * is torn down with a non-empty heap (outstanding allocated-but-not-freed
 * bytes). This is one of two complementary leak facilities:
 *
 *   * This checker (destructor driven) catches every allocator whose C++
 *     destructor actually runs — the process-lifetime singletons such as
 *     CTP_MALLOC, and any stack/heap allocator. It does NOT observe
 *     placement-constructed shared-memory allocators, whose destructors are
 *     never invoked; those are covered by IpcManager's own shutdown scan
 *     (IpcManager::ReportRuntimeLeaks) instead.
 *   * IpcManager::ReportRuntimeLeaks (context-runtime) scans the runtime-owned
 *     allocators (CTP_MALLOC + the per-process SHM segments) at ServerFinalize.
 *
 * Reports only carry meaning in leak-tracking builds (CTP_ALLOC_TRACK_SIZE, via
 * CLIO_CORE_ENABLE_LEAK_CHECK): allocators only call Report() under that macro,
 * and GetCurrentlyAllocatedSize() is a constant 0 otherwise. The class itself
 * is always compiled on host so tests can query it unconditionally.
 *
 * The singleton is intentionally leaked (heap-allocated, never freed) so that
 * allocator destructors firing during static teardown can always reach it,
 * regardless of static destruction order.
 */
class AllocatorLeakChecker {
 public:
  /** One recorded leak: which allocator, and how many bytes it still held. */
  struct Entry {
    std::string label;
    size_t bytes;
  };

  /** Access the process-global (leaked) instance. */
  static AllocatorLeakChecker &Get() {
    static AllocatorLeakChecker *inst = new AllocatorLeakChecker();
    return *inst;
  }

  /**
   * Record that the allocator identified by \a label was destroyed while still
   * holding \a bytes. Zero-byte reports are ignored (a clean allocator is not a
   * leak). Every non-zero report is also emitted immediately to the log.
   */
  void Report(const char *label, size_t bytes) {
    if (bytes == 0) {
      return;
    }
    const char *name = (label != nullptr) ? label : "<unknown>";
    std::lock_guard<std::mutex> lk(mu_);
    entries_.push_back(Entry{std::string(name), bytes});
    total_bytes_ += bytes;
    HLOG(kError,
         "[leak][allocator] '{}' destroyed with {} bytes still allocated",
         name, bytes);
  }

  /** Total outstanding bytes across all recorded leaks. */
  size_t TotalLeakedBytes() {
    std::lock_guard<std::mutex> lk(mu_);
    return total_bytes_;
  }

  /** Number of allocators that reported a leak. */
  size_t LeakCount() {
    std::lock_guard<std::mutex> lk(mu_);
    return entries_.size();
  }

  /** Snapshot of the recorded leaks (copied under lock). */
  std::vector<Entry> Entries() {
    std::lock_guard<std::mutex> lk(mu_);
    return entries_;
  }

  /** Clear all recorded leaks (mainly for tests that check specific phases). */
  void Reset() {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.clear();
    total_bytes_ = 0;
  }

 private:
  AllocatorLeakChecker() = default;

  std::mutex mu_;
  std::vector<Entry> entries_;
  size_t total_bytes_ = 0;
};

}  // namespace ctp::ipc

#endif  // CTP_IS_HOST
#endif  // CLIO_CTP_MEMORY_ALLOCATOR_LEAK_CHECKER_H_

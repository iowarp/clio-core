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

#ifndef CTP_MEMORY_SMART_PTR_LEASE_H_
#define CTP_MEMORY_SMART_PTR_LEASE_H_

#include "clio_ctp/thread/lock/timed_mutex.h"
#include "clio_ctp/types/atomic.h"

namespace ctp::ipc {

/**
 * Mix-in for a shared-memory object that cross-process readers may access
 * (issue #783).
 *
 * Carries the TWO mechanisms the design keeps deliberately separate:
 *
 *   gen_          -- CORRECTNESS. A seqlock counter, odd while a writer is
 *                    mid-update. Readers validate it before and after copying
 *                    and retry on mismatch. Readers store NOTHING, so a reader
 *                    that dies mid-read leaves no state to reclaim. This is
 *                    the default read path.
 *
 *   lease_mutex_  -- PROGRESS and LIFETIME. Taken only when a reader needs a
 *                    stable view for longer than a retry loop tolerates, or
 *                    when the runtime must not free the object underneath a
 *                    reader. Reclaimable, so a client that dies holding one is
 *                    eventually serviced.
 *
 * The asymmetry is the whole point: a lease is the only client-side construct
 * whose abandonment needs servicing, so the design minimizes how often one is
 * held. Prefer SeqRead; reach for Lease only when you must.
 *
 * SHM-SAFE: no virtuals, no pointers.
 */
struct Leaseable {
  ipc::atomic<min_u64> gen_;
  TimedMutex lease_mutex_;

  CTP_INLINE_CROSS_FUN
  Leaseable() : gen_(0) { lease_mutex_.Init(); }

  CTP_INLINE_CROSS_FUN
  void InitLeaseable() {
    gen_ = 0;
    lease_mutex_.Init();
  }

  /** Writer: open an update window (generation becomes odd). */
  CTP_INLINE_CROSS_FUN
  void BeginUpdate() { gen_.fetch_add(1); }

  /** Writer: close the update window (generation becomes even). */
  CTP_INLINE_CROSS_FUN
  void EndUpdate() { gen_.fetch_add(1); }

  /** True if a writer currently has this object open. */
  CTP_INLINE_CROSS_FUN
  bool IsUpdating() const { return (gen_.load() & 1) != 0; }
};

/**
 * Optimistic read of a Leaseable object.
 *
 * Runs `copy_out` between two reads of the generation counter and accepts the
 * result only if the counter did not move and was not odd. `copy_out` MUST be
 * side-effect-free and must tolerate reading torn bytes -- it may run against
 * a half-written object and have its result discarded. In particular it must
 * not dereference a pointer read out of the object without re-validating.
 *
 * @return true if a consistent snapshot was taken.
 */
template <typename T, typename Fn>
CTP_INLINE_CROSS_FUN bool SeqRead(const T &obj, Fn &&copy_out,
                                  min_u32 max_retries = 16) {
  for (min_u32 attempt = 0; attempt < max_retries; ++attempt) {
    min_u64 g0 = obj.gen_.load();
    if ((g0 & 1) != 0) {
      continue;  // writer mid-update
    }
    copy_out();
    if (obj.gen_.load() == g0) {
      return true;
    }
  }
  // Bounded rather than infinite: a writer that died mid-update leaves the
  // generation odd forever, and a reader must degrade to the RPC path instead
  // of spinning. (Runtime death is already fatal, but a bounded loop keeps a
  // client from hanging on the way down.)
  return false;
}

/**
 * RAII lease over a Leaseable object.
 *
 * Acquiring is a STORE, which is why clients map the metadata segment
 * read-write even though metadata itself is runtime-owned (see design §5.3b).
 *
 * Holds the TimedMutex acquisition id so release is the ABA-safe CAS form: if
 * this lease was broken because the holder was presumed dead, releasing it must
 * not unlock whoever legitimately acquired it afterwards.
 *
 * HOLD IT BRIEFLY. Copy what you need and let it go. A lease held across a
 * syscall, an I/O, or any unbounded operation can delay runtime progress --
 * the reclamation timeout is a backstop, not a budget.
 */
template <typename T>
class Lease {
 public:
  /** Blocking acquire. */
  CTP_INLINE_CROSS_FUN
  explicit Lease(T *obj, min_u64 timeout_ms = TimedMutex::kDefaultTimeoutMs)
      : obj_(obj), id_(0) {
    if (obj_ != nullptr) {
      id_ = obj_->lease_mutex_.Lock(timeout_ms);
    }
  }

  /** Non-blocking acquire. Check Owns() before touching the object. */
  CTP_INLINE_CROSS_FUN
  static Lease TryAcquire(T *obj) {
    Lease l;
    l.obj_ = obj;
    if (obj != nullptr) {
      l.id_ = obj->lease_mutex_.TryLock();
    }
    return l;
  }

  CTP_INLINE_CROSS_FUN
  Lease() : obj_(nullptr), id_(0) {}

  CTP_INLINE_CROSS_FUN
  ~Lease() { Release(); }

  Lease(const Lease &) = delete;
  Lease &operator=(const Lease &) = delete;

  CTP_INLINE_CROSS_FUN
  Lease(Lease &&other) noexcept : obj_(other.obj_), id_(other.id_) {
    other.obj_ = nullptr;
    other.id_ = 0;
  }

  CTP_INLINE_CROSS_FUN
  Lease &operator=(Lease &&other) noexcept {
    if (this != &other) {
      Release();
      obj_ = other.obj_;
      id_ = other.id_;
      other.obj_ = nullptr;
      other.id_ = 0;
    }
    return *this;
  }

  /** True if this lease actually holds the object. */
  CTP_INLINE_CROSS_FUN
  bool Owns() const { return id_ != 0; }

  /** Guarded object, or nullptr if the lease was not acquired. Returning
   *  nullptr rather than the raw pointer makes a missing Owns() check fail
   *  loudly instead of silently reading unprotected memory. */
  CTP_INLINE_CROSS_FUN
  T *get() const { return (id_ != 0) ? obj_ : nullptr; }

  CTP_INLINE_CROSS_FUN
  T *operator->() const { return get(); }

  CTP_INLINE_CROSS_FUN
  T &operator*() const { return *obj_; }

  CTP_INLINE_CROSS_FUN
  explicit operator bool() const { return Owns(); }

  /** Release early. Idempotent. */
  CTP_INLINE_CROSS_FUN
  void Release() {
    if (obj_ != nullptr && id_ != 0) {
      obj_->lease_mutex_.Unlock(id_);
    }
    obj_ = nullptr;
    id_ = 0;
  }

 private:
  T *obj_;
  min_u64 id_;
};

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_SMART_PTR_LEASE_H_

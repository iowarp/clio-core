/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_GPU_SUBMIT_PROBE_H_
#define CLIO_RUNTIME_INCLUDE_GPU_SUBMIT_PROBE_H_

// Per-submit latency probe for the GPU->CPU task path (measurement only).
//
// Splits one GPU-initiated Put/Get into the hops the design claims are cheap:
//
//   device  d_enter    -> d_fields     task-field stores into the slot
//   device  d_fields   -> d_pushed     __threadfence_system() + ring push
//   cross   d_pushed   -> t_pop        CPU poll latency until Pop() succeeds
//   host    t_pop      -> t_pre_route  RecvIn prologue (wrap + BeginRunContext)
//   host    t_pre_route-> t_post_route the forced dispatch hop (RouteTask)
//   host    t_post_route->t_exec_start worker-queue wait before the coroutine runs
//   host    t_exec_start->t_sendout_b  execution in the transfer engine
//   host    t_sendout_b->t_sendout_e   completion writeback (SendOut)
//   cross   t_sendout_e-> d_wait_end   completion-flag visibility on the device
//   device  d_wait_begin->d_wait_end   the device-side spin (end-to-end, device view)
//
// Two design constraints, both load-bearing for the measurement's validity:
//
//   1. The DEVICE half writes only to device global memory (cudaMalloc), never
//      to pinned/mapped host memory. A probe store must not add PCIe traffic to
//      the very path whose PCIe cost we are measuring. Records are copied back
//      once, after the kernel retires.
//
//   2. The HOST half never hashes, locks, or allocates on the task path. RecvIn
//      opens the record at Pop and stashes its address in the task's RunContext
//      (`probe_rec_`); every later hop dereferences that pointer directly. For a
//      non-GPU task probe_rec_ is 0, so the cost on the ordinary CPU task path is
//      one load and one not-taken branch.
//
// The join key between the two halves is the task POD's address. kvhdf5 forces
// the task backend to kPinnedHost, so the device and the host name that slot by
// the same address and the join is exact.
//
// Device timestamps come from %globaltimer (nanoseconds, invariant to SM clock
// boost); clock64() cycles are captured alongside purely as a cross-check.
// %globaltimer and CLOCK_MONOTONIC have different epochs, so the two cross-domain
// hops require the offset measured by the caller's clock handshake.

#include <cstdint>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CLIO_PROBE_HD __host__ __device__
#else
#define CLIO_PROBE_HD
#endif

namespace clio::run::gpu {

/**
 * One GPU submit.
 *
 * TWO device clocks, because neither one alone can do the job:
 *
 *   %globaltimer (`d_*`) is a wall clock in ns and is the ONLY device clock
 *   comparable to the host's — but on Ada it advances in ~1024 ns steps, which is
 *   coarser than the entire task-field-store hop. It is therefore used only where
 *   a cross-domain join needs it (the push, and the wait), never for a fine hop.
 *
 *   clock64() (`c_*`) is an SM cycle counter with single-cycle resolution — the
 *   only clock that can resolve the fence and the push. It is valid ONLY within
 *   one kernel launch on one block (a different block may land on a different SM
 *   with a different counter base), and its cycles are not nanoseconds until
 *   divided by the SM clock, which boosts. Both caveats are handled offline: the
 *   cycles->ns ratio is derived per-record from the long enter->wait span, which
 *   is bracketed by both clocks on the same block.
 */
struct SubmitProbeRec {
  unsigned long long task_ptr;      /**< task POD address — the join key */
  unsigned long long seq;           /**< global submit order */
  /* Cross-domain anchors (globaltimer ns; ~1024 ns resolution). */
  unsigned long long d_enter;       /**< SendIn entry */
  unsigned long long d_pushed;      /**< after ring Push returns */
  unsigned long long d_wait_begin;  /**< before the Wait() spin */
  unsigned long long d_wait_end;    /**< completion observed */
  /* The fine hops (clock64 cycles; single-cycle resolution). */
  unsigned long long c_enter;       /**< SendIn entry */
  unsigned long long c_fields;      /**< after fut_ field stores  -> hop 1 */
  unsigned long long c_prefence;    /**< after queue-entry setup */
  unsigned long long c_postfence;   /**< after __threadfence_system() -> hop 2a = the fence */
  unsigned long long c_pushed;      /**< after Push                   -> hop 2b = the ring push */
  unsigned long long c_wait_begin;  /**< before the Wait() spin */
  unsigned long long c_wait_end;    /**< completion observed */
};

/**
 * Kernel-facing probe handle. Passed BY VALUE inside IpcManagerGpuInfo, so it
 * costs the kernel-arg budget three words and needs no extra indirection.
 * `recs == nullptr` means the probe is off and every device-side stamp compiles
 * down to a null check.
 */
struct SubmitProbeDev {
  SubmitProbeRec *recs = nullptr;  /**< device global memory, `cap` entries */
  unsigned *counter = nullptr;     /**< device global memory, slot allocator */
  unsigned cap = 0;

  CLIO_PROBE_HD bool On() const { return recs != nullptr; }
};

/** "no record" — probe off, or capacity exhausted. */
static constexpr unsigned kProbeNoSlot = 0xffffffffu;

#if defined(__CUDACC__) || defined(__HIPCC__)

/** Wall-clock nanoseconds on the device. Invariant to SM clock boost. */
__device__ inline unsigned long long ProbeNowNs() {
#if defined(__CUDA_ARCH__)
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
#else
  return static_cast<unsigned long long>(clock64());
#endif
}

/** SM cycles. Boost-dependent — a cross-check on ProbeNowNs, not a substitute. */
__device__ inline unsigned long long ProbeCycles() {
  return static_cast<unsigned long long>(clock64());
}

/** Claim a record. Returns kProbeNoSlot once `cap` is exhausted. */
__device__ inline unsigned ProbeClaim(const SubmitProbeDev &p) {
  if (!p.recs) return kProbeNoSlot;
  unsigned i = atomicAdd(p.counter, 1u);
  return (i < p.cap) ? i : kProbeNoSlot;
}

__device__ inline SubmitProbeRec *ProbeRec(const SubmitProbeDev &p, unsigned slot) {
  if (!p.recs || slot >= p.cap) return nullptr;
  return &p.recs[slot];
}

#endif  // __CUDACC__ || __HIPCC__

// ---------------------------------------------------------------------------
// Host half
// ---------------------------------------------------------------------------

/** The CPU-side hops. All timestamps CLOCK_MONOTONIC ns; 0 = never reached. */
struct SubmitProbeHostRec {
  unsigned long long task_ptr;
  unsigned long long t_pop;
  unsigned long long t_pre_route;
  unsigned long long t_post_route;
  unsigned long long t_exec_start;
  unsigned long long t_sendout_begin;
  unsigned long long t_sendout_end;
  /** Worker state at the moment of the Pop — the hot-spin / idle-wakeup split.
   *  A worker that was spinning has idle_iters_at_pop == 0 and sleep_us == 0;
   *  one that had backed off into SuspendMe() shows both nonzero. */
  unsigned long long idle_iters_at_pop;
  unsigned sleep_us_at_pop;
  unsigned worker_pop;   /**< worker that popped the gpu2cpu lane */
  unsigned worker_exec;  /**< worker that ran the coroutine (the funnel: see S3) */
};

/**
 * Process-wide registry. Off unless Enable() is called, and when off every entry
 * point is a null return — nothing is allocated and no task-path branch is taken
 * beyond the RunContext's probe_rec_ null check.
 */
class SubmitProbe {
 public:
  static SubmitProbe &Get();

  /** Reserve `cap` records and arm the probe. Not thread-safe; call at setup. */
  void Enable(unsigned cap);
  bool On() const { return on_; }

  /** Open the record for a just-popped task. `t_pop` is passed in rather than
   *  read here: the caller samples the clock the instant Pop() returns, before
   *  it has even decoded the task address this record is keyed by. Returns the
   *  record to stash in RunContext::probe_rec_, or nullptr when the probe is off
   *  / out of capacity. */
  SubmitProbeHostRec *Open(unsigned long long task_ptr, unsigned long long t_pop,
                           unsigned worker_id, unsigned long long idle_iters,
                           unsigned sleep_us);

  /** Append every opened record to `path` as CSV (header included). */
  void Dump(const char *path) const;

  unsigned Count() const { return count_; }

  static unsigned long long NowNs();

 private:
  SubmitProbe() = default;
  bool on_ = false;
  unsigned cap_ = 0;
  unsigned count_ = 0;
  SubmitProbeHostRec *recs_ = nullptr;
};

}  // namespace clio::run::gpu

#undef CLIO_PROBE_HD

#endif  // CLIO_RUNTIME_INCLUDE_GPU_SUBMIT_PROBE_H_

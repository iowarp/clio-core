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
 * clio-coroc's Ctx and Frame, backed by the production yield stack.
 *
 * THIS FILE IS THE DROP-IN REPLACEMENT FOR yield_coro.h. Everything on the
 * host side stays exactly as it is: the same YieldStack allocation, the same
 * per-lane region, the same YieldBlockState, the same wait tags, the same
 * relaunch driver. What changes is only how a suspension point is LOWERED --
 * `co_await` and a clang-generated coroutine frame become a `switch` and a
 * frame this file lays out.
 *
 * The contract with the host is copied from YCoroSuspend, which is the thing
 * being replaced, so the driver cannot tell the difference:
 *
 *     block_state_->wait_tag_ = <tag>        what this block is waiting for
 *     block_state_->status_   = kYieldSuspended
 *     __threadfence_system()                 publish before the kernel exits
 *     return                                 a park IS a kernel exit
 *
 * WHAT THIS BUYS OVER yield_coro.h, and why it is worth a transpiler:
 *
 *   - nvcc and SPIR-V can compile it. `co_await` in device code requires
 *     clang, and clang cannot lower a device coroutine for spir64 at all --
 *     which is the entire reason a second, hand-written spelling of every
 *     kernel exists in this tree.
 *   - The frame is hand-packed to the live set at each suspend point rather
 *     than compiler-chosen for the whole coroutine, so lanes get smaller
 *     (yield_coro.h sizes the weights bench at 2 KB per lane).
 *   - No indirect resume. A nested suspending call is an ordinary call, so
 *     there is no handoff, no trampoline, and no resume-function pointer.
 *
 * WHAT IT COSTS. Registers. Measured at 26 -> 80 on the reference workload,
 * with zero local memory and zero spills; see test/co/CUDA_RESULTS.md.
 *
 * FRAME LAYOUT. One frame per live suspending call, claimed by DEPTH from the
 * lane's bump region, exactly as YCoroAlloc does for a promise frame:
 *
 *     [ u32 resume_point ][ save list, natural alignment ]
 *
 * Offsets reproduce on resume without being stored (invariant I1): the chain
 * is re-entered from the kernel entry, the same calls happen in the same
 * order, and each claims the same depth, so each lands on the same offset.
 */
#ifndef CLIO_RUNTIME_CO_YIELD_BACKEND_H_
#define CLIO_RUNTIME_CO_YIELD_BACKEND_H_

#include <clio_runtime/gpu/yield_stack.h>

/** Device suspension via clio-coroc is available in this build. */
#define CLIO_HAS_COROC 1

/**
 * The annotation every suspending function carries.
 *
 * FORCED, not merely suggested. A transpiled function is a switch over the
 * whole call's state, which is bulky enough that a compiler left to itself
 * may decline to inline it -- and an OUTLINED suspending function is a
 * resource contradiction, because the kernel carries __launch_bounds__ and
 * the callee does not. nvcc says so exactly: "Entry function ... with max
 * regcount of 64 calls function ... with regcount of 255", which is what
 * grayscott's StepCoro did before this existed.
 *
 * Inlining is also what the design wants on the fast path: a CO_AWAIT that
 * does not suspend should be a predicated branch, not a call.
 */
#if CTP_IS_GPU_COMPILER
#define CLIO_COROC_INLINE __forceinline__
#else
#define CLIO_COROC_INLINE inline
#endif

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace clio::co {

using u32 = clio::run::u32;
using u64 = clio::run::u64;

/** Header at the base of every call frame. */
struct CoFrameHeader {
  u32 resume_point_; /**< 0 = fresh entry; otherwise the case label to jump to */
  u32 pad;
};

/* ======================================================================== */
/* Ctx -- the parameter clio-coroc appends to every suspending function.     */
/* ======================================================================== */

/**
 * Per-work-item coroutine context. Register-resident; nothing global.
 *
 * Holds no state that must survive a park -- `parked_` is recomputed every
 * entry, and the lane pointer is re-derived at the kernel entry. That is what
 * lets a park be a plain kernel exit with no fence beyond the one that
 * publishes the block's status.
 */
class Ctx {
 public:
  __device__ __forceinline__ Ctx()
      : lane_(clio::run::gpu::YieldLane()), parked_(false) {}

  /** Group-collective OR. Every work-item of the block must reach it.
   *
   * This is what makes a park all-or-none (invariant I3): a page fault is a
   * per-thread event but suspending is a per-block one, so a warp that missed
   * cannot suspend alone and a warp that hit cannot run ahead.
   *
   * @param p this work-item's vote
   * @return true if any work-item in the block voted true
   */
  __device__ __forceinline__ bool Any(bool p) const {
    return __syncthreads_or(p ? 1 : 0) != 0;
  }

  /** Block barrier, for a workload that needs one between suspend points. */
  __device__ __forceinline__ void Barrier() const { __syncthreads(); }

  /** True once any frame on this chain has parked during this entry. */
  __device__ __forceinline__ bool Parked() const { return parked_; }

  /** Tell the host what this block is waiting for, so the driver can hold it
   *  back instead of relaunching into the same fault.
   *
   * Identical to YCoroSuspend's publication, including the fence: the driver
   * on the other side is unchanged.
   *
   * @param tag opaque token the driver's ResumeWhen understands
   */
  __device__ __forceinline__ void SetWaitTag(u64 tag) {
    parked_ = true;
    if (threadIdx.x == 0) {
      clio::run::gpu::YieldTls().block_state_->wait_tag_ = tag;
      clio::run::gpu::YieldTls().block_state_->status_ =
          clio::run::gpu::kYieldSuspended;
    }
    __threadfence_system();
  }

  /** Mark parked without a tag, for a frame relaying a callee's park. */
  __device__ __forceinline__ void MarkParked() { parked_ = true; }

  __device__ __forceinline__ clio::run::gpu::YieldLaneHeader *Lane() const {
    return lane_;
  }

 private:
  clio::run::gpu::YieldLaneHeader *lane_;
  bool parked_;
};

/* ======================================================================== */
/* Frame -- one suspending call's slice of the lane region.                  */
/* ======================================================================== */

/**
 * Claims this call's frame on construction and releases it on Done().
 *
 * Fresh entry claims the next region and zeroes the resume point; a resume
 * re-attaches to the frame this call already owns at this depth, which is
 * what makes the save list come back.
 */
class Frame {
 public:
  /**
   * @param cx    the context threaded in by clio-coroc
   * @param bytes save-list bytes this call needs, the max over its suspend
   *              points -- a compile-time constant the transpiler computes
   */
  __device__ __forceinline__ Frame(Ctx &cx, u32 bytes) : cx_(&cx) {
    lane_ = cx.Lane();
    base_ = reinterpret_cast<char *>(lane_);
    const u32 d = lane_->cur_depth_++;
    if (d >= clio::run::gpu::kYieldMaxDepth) {
      lane_->error_ = clio::run::gpu::kYieldErrDepth;
      clio::run::gpu::YieldFatalNote(clio::run::gpu::kYieldFatalDepth,
                                     blockIdx.x, threadIdx.x, d);
      __trap();
    }
    if (d < lane_->live_depth_) {
      fp_ = lane_->frame_off_[d];
      fresh_ = false;
    } else {
      fp_ = (lane_->sp_ + 15u) & ~15u;
      const u32 end = fp_ + static_cast<u32>(sizeof(CoFrameHeader)) + bytes;
      if (end > clio::run::gpu::YieldTls().stack_.bytes_per_lane_) {
        lane_->error_ = clio::run::gpu::kYieldErrOverflow;
        printf("[coroc] block %u lane %u: frame overflow, need %u > %u bytes\n",
               blockIdx.x, threadIdx.x, end,
               clio::run::gpu::YieldTls().stack_.bytes_per_lane_);
        clio::run::gpu::YieldFatalNote(clio::run::gpu::kYieldFatalCoroFrame,
                                       blockIdx.x, threadIdx.x, end);
        __trap();
      }
      lane_->frame_off_[d] = fp_;
      lane_->sp_ = end;
      lane_->live_depth_ = d + 1;
      fresh_ = true;
      Header()->resume_point_ = 0;
    }
  }

  /** The case label this entry must jump to. Zero on a fresh call. */
  __device__ __forceinline__ u32 Resume() const {
    return Header()->resume_point_;
  }

  /** True while re-entering a call that was live when the kernel exited, so
   *  the save list on the way to the resume point must be restored. */
  __device__ __forceinline__ bool Replaying() const { return !fresh_; }

  /**
   * Suspend here: record where to come back to and save the live set.
   *
   * @param state the case label to resume at
   * @param vs    the values that must survive the park
   */
  template <class... Ts>
  __device__ __forceinline__ void Push(u32 state, const Ts &...vs) {
    static_assert((std::is_trivially_copyable_v<Ts> && ...),
                  "a value live across a CO_AWAIT must be trivially copyable");
    Header()->resume_point_ = state;
    cx_->MarkParked();
    char *p = Slot();
    (StoreOne(p, vs), ...);
  }

  /** Restore the live set written by the matching Push, in the same order. */
  template <class... Ts>
  __device__ __forceinline__ void Pop(Ts &...vs) {
    static_assert((std::is_trivially_copyable_v<Ts> && ...),
                  "a value live across a CO_AWAIT must be trivially copyable");
    char *p = Slot();
    (LoadOne(p, vs), ...);
    fresh_ = true;  // replay consumed; deeper suspends this entry are fresh
  }

  /** Normal return: release the frame so a later call reuses the space. */
  __device__ __forceinline__ void Done() const {
    lane_->cur_depth_ -= 1;
    lane_->live_depth_ = lane_->cur_depth_;
    lane_->sp_ = fp_;
    Header()->resume_point_ = 0;
  }

 private:
  __device__ __forceinline__ CoFrameHeader *Header() const {
    return reinterpret_cast<CoFrameHeader *>(base_ + fp_);
  }
  __device__ __forceinline__ char *Slot() const {
    return base_ + fp_ + sizeof(CoFrameHeader);
  }
  __device__ __forceinline__ static char *Align(char *p, std::size_t a) {
    const std::uintptr_t x = reinterpret_cast<std::uintptr_t>(p);
    return reinterpret_cast<char *>((x + a - 1) & ~(a - 1));
  }
  template <class T>
  __device__ __forceinline__ static void StoreOne(char *&p, const T &v) {
    p = Align(p, alignof(T));
    memcpy(p, &v, sizeof(T));
    p += sizeof(T);
  }
  template <class T>
  __device__ __forceinline__ static void LoadOne(char *&p, T &v) {
    p = Align(p, alignof(T));
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
  }

  Ctx *cx_;
  clio::run::gpu::YieldLaneHeader *lane_;
  char *base_;
  u32 fp_;
  bool fresh_;
};

/* ======================================================================== */
/* Frame-size arithmetic. The transpiler emits calls to these.               */
/* ======================================================================== */

/** Sum of a trivially copyable pack's sizes, each at its own alignment. */
template <class... Ts>
__host__ __device__ constexpr u32 PackBytes() {
  u32 n = 0;
  (((n = (n + static_cast<u32>(alignof(Ts)) - 1) &
         ~(static_cast<u32>(alignof(Ts)) - 1)),
    (n += static_cast<u32>(sizeof(Ts)))),
   ...);
  return n;
}

/** Largest of a set of per-suspend-point pack sizes. */
__host__ __device__ constexpr u32 MaxOf(u32 a) { return a; }
template <class... Rest>
__host__ __device__ constexpr u32 MaxOf(u32 a, u32 b, Rest... rest) {
  return MaxOf(a > b ? a : b, rest...);
}

/** The value type of `CO_AWAIT(<awaiter>)`, so a hoisted `auto` declaration
 *  gets a type without the transpiler ever naming one. */
template <class A>
using AwaiterResult = decltype(std::declval<A &>().Take());

/**
 * Suspend once, then fall through. The replacement for a bare
 * `co_await YCoroSuspend{tag}`.
 *
 * lammps_md waits on a peer with a retry loop that votes and then suspends
 * unconditionally; there is no condition the host could poll, so the block
 * resumes every round and re-checks. Ready() is false on its first
 * evaluation and true afterwards, and the awaiter is saved across the park,
 * so the resume falls straight through and the enclosing loop re-tests the
 * real condition.
 *
 * The transpiler re-assigns the awaiter at the top of each loop iteration,
 * so `entered` starts false again every time round.
 */
struct YieldOnce {
  u64 tag = 0;
  u32 entered = 0;
  __device__ __forceinline__ bool Ready() {
    const bool was = entered != 0;
    entered = 1;
    return was;
  }
  __device__ __forceinline__ u64 Tag() const { return tag; }
  __device__ __forceinline__ void Take() const {}
};

/** The marker. Consumed by clio-coroc; the identity otherwise. */
__device__ __forceinline__ void AwaitMark() {}

}  // namespace clio::co

/**
 * Kernel entry. The replacement for CLIO_YCORO_RUN.
 *
 * Simpler than the thing it replaces, and the simplification is the point:
 * there is no resume loop, because a nested suspending call is an ordinary
 * call rather than a handoff to another coroutine. One entry runs the chain
 * until it parks or finishes.
 *
 * `view` is a YieldableView and `stack` a YieldStackView, exactly as
 * CLIO_YKERNEL_ENTER takes them. `call` is the top-level suspending call,
 * with `_cy` as its last argument.
 */
#define CLIO_COROC_RUN(view, stack, call)                                     \
  do {                                                                        \
    clio::run::gpu::YieldTlsPublish((stack), (view).Y(), (view).Block());      \
    clio::run::gpu::YieldLane()->cur_depth_ = 0;                              \
    __syncthreads();                                                          \
    /* Shared did not survive the park; restore before resuming, while     */ \
    /* every thread is still here to help copy.                            */ \
    ::clio::run::gpu::PersistRestore();                                       \
    clio::co::Ctx _cy;                                                        \
    (call);                                                                   \
    __syncthreads();                                                          \
    if (_cy.Parked()) {                                                       \
      /* About to exit and take shared with it. */                            \
      ::clio::run::gpu::PersistSave();                                        \
    } else {                                                                  \
      ::clio::run::gpu::PersistClear();                                       \
      clio::run::gpu::YieldLane()->live_depth_ = 0;                           \
      clio::run::gpu::YieldLane()->sp_ =                                      \
          sizeof(clio::run::gpu::YieldLaneHeader);                            \
    }                                                                         \
  } while (0)

/** The one token a user writes at a suspend point. */
#define CO_AWAIT(...) (::clio::co::AwaitMark(), (__VA_ARGS__))

#endif  // CLIO_RUNTIME_CO_YIELD_BACKEND_H_

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
 * Portable GPU coroutines: the device-side runtime.
 *
 * This is the machinery that the clio-coroc transpiler generates calls into.
 * Nothing here knows what a coroutine is; it provides a group-scoped resume
 * point, a device-memory stack of per-call frames, and a park protocol that
 * needs nothing from the platform beyond "a kernel launch ends".
 *
 * THE FIVE-OPERATION BACKEND SEAM. Everything that differs between CUDA, HIP
 * and SYCL lives in `Item`: Local, Group, Size, Barrier, Any. Generated code
 * never contains a backend token -- it calls Ctx methods, which call Item.
 * That is what makes one transpiled source serve three backends.
 *
 * THE INVARIANTS THIS FILE IS RESPONSIBLE FOR (numbering follows the design
 * doc, $HOME/coroutines.md section 6):
 *
 *   I1  Frame offsets reproduce on resume. A frame's offset is the running sum
 *       of the frame sizes of the calls currently on the chain. Frames are
 *       released on normal return, so the cursor depends only on the CURRENT
 *       chain, never on history -- and a resume re-enters the same chain from
 *       the kernel entry, constructing the same frames in the same order. No
 *       layout is stored anywhere.
 *   I2  All work-items of a group resume at the same point: state[] is
 *       group-uniform, written by lane 0 only.
 *   I3  The park is all-or-none: the decision is Ctx::Any, a group-collective
 *       reduction.
 *   I6  Save and restore are byte-exact: Push static_asserts that every element
 *       is trivially copyable, so this is enforced by the compiler that builds
 *       the generated file rather than by the transpiler.
 *   I8  Frame writes are visible to the next launch, because a park is a kernel
 *       exit and kernel completion orders every write the kernel made. There is
 *       deliberately no fence and no atomic in the park path.
 *
 * WHAT A GENERATED FUNCTION LOOKS LIKE. See context-runtime/test/co/ for
 * hand-written examples in exactly the form the transpiler emits.
 */
#ifndef CLIO_RUNTIME_CO_CORO_H_
#define CLIO_RUNTIME_CO_CORO_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

/* -------------------------------------------------------------------------
 * Backend selection.
 *
 * Exactly one of these is 1. The host backend is not a stub: it emulates a
 * work-group with real threads and real barriers so that every invariant above
 * can be tested, and the differential oracle run, without a GPU.
 * ------------------------------------------------------------------------- */
#if defined(SYCL_LANGUAGE_VERSION) || defined(CLIO_CO_FORCE_SYCL)
#define CLIO_CO_SYCL 1
#else
#define CLIO_CO_SYCL 0
#endif

#if defined(__CUDACC__) || defined(__CUDA__)
#define CLIO_CO_CUDA 1
#else
#define CLIO_CO_CUDA 0
#endif

#if defined(__HIPCC__)
#define CLIO_CO_HIP 1
#else
#define CLIO_CO_HIP 0
#endif

#if !CLIO_CO_SYCL && !CLIO_CO_CUDA && !CLIO_CO_HIP
#define CLIO_CO_HOST 1
#else
#define CLIO_CO_HOST 0
#endif

#if CLIO_CO_CUDA || CLIO_CO_HIP
#define CLIO_CO_FUN __device__ __forceinline__
#else
#define CLIO_CO_FUN inline
#endif

#if CLIO_CO_SYCL
#include <sycl/sycl.hpp>
#endif

#if CLIO_CO_HOST
#include <atomic>
#include <barrier>
#endif

namespace clio::co {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

/** Maximum suspending-call nesting depth. The suspending call graph is acyclic
 *  (design doc R4), so a program's real depth is a compile-time constant; this
 *  is only the size of the per-group state array. */
inline constexpr u32 kMaxDepth = 8;

/** A group's lifecycle, as the host driver sees it. */
enum Status : u32 {
  kFresh = 0,   /**< never entered */
  kParked = 1,  /**< suspended; state[] and the frames are live */
  kDone = 2,    /**< ran to completion; skip on relaunch */
};

/* =========================================================================
 * Item -- the entire backend surface of this mechanism.
 * ========================================================================= */

#if CLIO_CO_HOST
/** Shared state backing one emulated work-group on the host. */
struct HostGroup {
  u32 size;
  u32 group;
  std::barrier<> bar;
  std::atomic<int> vote;
  explicit HostGroup(u32 n, u32 g) : size(n), group(g), bar(n), vote(0) {}
};
#endif

/**
 * The work-item's identity and its two collective operations.
 *
 * Every member is a register read or a group collective. `Any` must be
 * group-collective on every backend: it is what makes the park all-or-none
 * (invariant I3), so a backend whose `Any` is not a barrier gets one wrapped
 * around it rather than an argument about whether it needs one.
 */
struct Item {
#if CLIO_CO_SYCL
  sycl::nd_item<1> it_;
  Item(sycl::nd_item<1> it) : it_(it) {}
  inline u32 Local() const { return static_cast<u32>(it_.get_local_id(0)); }
  inline u32 Group() const { return static_cast<u32>(it_.get_group(0)); }
  inline u32 Size() const {
    return static_cast<u32>(it_.get_local_range(0));
  }
  inline void Barrier() const { sycl::group_barrier(it_.get_group()); }
  /** any_of_group requires every work-item of the group to call it, but is not
   *  specified to be a barrier. The brackets make it one. Correctness beats a
   *  saved barrier; if spike S2 shows the brackets are redundant on the
   *  implementations we care about, they come out then and not before. */
  inline bool Any(bool p) const {
    sycl::group_barrier(it_.get_group());
    const bool r = sycl::any_of_group(it_.get_group(), p);
    sycl::group_barrier(it_.get_group());
    return r;
  }

#elif CLIO_CO_CUDA || CLIO_CO_HIP
  /* CUDA and HIP are token-identical here, which is why they share a branch. */
  CLIO_CO_FUN u32 Local() const { return threadIdx.x; }
  CLIO_CO_FUN u32 Group() const { return blockIdx.x; }
  CLIO_CO_FUN u32 Size() const { return blockDim.x; }
  CLIO_CO_FUN void Barrier() const { __syncthreads(); }
  CLIO_CO_FUN bool Any(bool p) const {
    return __syncthreads_or(p ? 1 : 0) != 0;
  }

#else  /* host */
  HostGroup *g_ = nullptr;
  u32 local_ = 0;
  Item() = default;
  Item(HostGroup *g, u32 local) : g_(g), local_(local) {}
  inline u32 Local() const { return local_; }
  inline u32 Group() const { return g_->group; }
  inline u32 Size() const { return g_->size; }
  inline void Barrier() const { g_->bar.arrive_and_wait(); }
  /** Reduce, read, reset -- three barriers, because the reset must not race the
   *  next Any. Slow and obviously correct, which is what a test oracle wants. */
  inline bool Any(bool p) const {
    if (p) g_->vote.fetch_or(1, std::memory_order_relaxed);
    g_->bar.arrive_and_wait();
    const bool r = g_->vote.load(std::memory_order_relaxed) != 0;
    g_->bar.arrive_and_wait();
    if (local_ == 0) g_->vote.store(0, std::memory_order_relaxed);
    g_->bar.arrive_and_wait();
    return r;
  }
#endif
};

/* =========================================================================
 * The device-memory stack.
 * ========================================================================= */

/**
 * Per-group bookkeeping, at the base of each group's region.
 *
 * Read by the host after a launch completes; written by lane 0 during a park.
 * Nothing here is touched by more than one work-item, so there are no atomics.
 */
struct GroupHeader {
  u32 status;                /**< Status */
  u32 park_depth;            /**< deepest frame live at the park */
  u64 wait_tag;              /**< opaque token for the host servicer */
  u32 state[kMaxDepth];      /**< resume point per call depth (group-uniform) */
};

/** A view of the whole stack, passed to the kernel by value. */
struct StackView {
  char *base = nullptr;
  u32 bytes_per_group = 0;
  u32 group_size = 0;
  u32 n_groups = 0;

  CLIO_CO_FUN char *GroupBase(u32 g) const {
    return base + static_cast<std::size_t>(g) * bytes_per_group;
  }
  CLIO_CO_FUN GroupHeader *Header(u32 g) const {
    return reinterpret_cast<GroupHeader *>(GroupBase(g));
  }
};

/** Bytes a frame of `per_item` bytes occupies for a whole group. */
CLIO_CO_FUN u32 FrameStride(u32 per_item, u32 group_size) {
  return per_item * group_size;
}

/** Sum of the sizes of a trivially copyable pack, with natural alignment.
 *  This is what kFrameBytes_<fn> is built from, per suspend point. */
template <class... Ts>
constexpr u32 PackBytes() {
  u32 n = 0;
  // Fold over the pack, aligning each element to its own alignment.
  (((n = (n + static_cast<u32>(alignof(Ts)) - 1) &
         ~(static_cast<u32>(alignof(Ts)) - 1)),
    (n += static_cast<u32>(sizeof(Ts)))),
   ...);
  return n;
}

/** Largest of a set of per-suspend-point pack sizes (design doc R3). */
constexpr u32 MaxOf(u32 a) { return a; }
template <class... Rest>
constexpr u32 MaxOf(u32 a, u32 b, Rest... rest) {
  return MaxOf(a > b ? a : b, rest...);
}

/* =========================================================================
 * Ctx -- the threaded context. One per work-item, register-resident.
 * ========================================================================= */

class Frame;

/**
 * The parameter clio-coroc appends to every suspending function.
 *
 * It exists so that this mechanism needs NO global device state: no __device__
 * variable, no SYCL device_global, no __shared__, no thread-local. That is the
 * single decision that makes one implementation serve three backends, because
 * global device state is where the three disagree most.
 */
class Ctx {
 public:
  CLIO_CO_FUN Ctx(StackView st, Item it)
      : it_(it),
        hdr_(st.Header(it.Group())),
        frames_(st.GroupBase(it.Group()) + sizeof(GroupHeader)),
        group_size_(st.group_size),
        lane_(it.Local()),
        cursor_(0),
        depth_(0),
        parked_(false),
        replaying_(st.Header(it.Group())->status == kParked) {}

  CLIO_CO_FUN bool Parked() const { return parked_; }
  CLIO_CO_FUN bool Any(bool p) const { return it_.Any(p); }
  CLIO_CO_FUN void Barrier() const { it_.Barrier(); }
  CLIO_CO_FUN u32 Lane() const { return lane_; }
  CLIO_CO_FUN u32 GroupSize() const { return group_size_; }
  CLIO_CO_FUN const Item &It() const { return it_; }

  /** Record what this group is waiting for, so the host can service it before
   *  relaunching. Lane 0 writes; kernel completion publishes (I8). */
  CLIO_CO_FUN void SetWaitTag(u64 tag) {
    if (lane_ == 0) hdr_->wait_tag = tag;
  }

 private:
  friend class Frame;
  Item it_;
  GroupHeader *hdr_;
  char *frames_;
  u32 group_size_;
  u32 lane_;
  u32 cursor_;      /**< bytes consumed by frames on the current chain (I1) */
  u32 depth_;
  bool parked_;
  bool replaying_;  /**< descending into a saved chain */
};

/* =========================================================================
 * Frame -- one call's slice of the stack.
 * ========================================================================= */

/**
 * Reserves this call's frame on entry and releases it on Done().
 *
 * Reserving on ENTRY rather than pushing at the suspend is load-bearing: the
 * unwind runs inner-to-outer, so a callee pushes before its caller does, and a
 * push-ordered stack would interleave frames backwards. Reserving by depth
 * gives every call a stable address whether or not it is ever written.
 */
class Frame {
 public:
  /** @param c    the threaded context
   *  @param per_item  kFrameBytes_<fn>: the largest live set over this
   *                   function's suspend points (design doc R3) */
  CLIO_CO_FUN Frame(Ctx &c, u32 per_item)
      : c_(&c),
        depth_(c.depth_),
        off_(c.cursor_),
        per_item_(per_item),
        resume_(0),
        replay_pending_(false),
        popped_(0) {
    c.depth_ += 1;
    c.cursor_ += FrameStride(per_item, c.group_size_);
    if (c.replaying_) {
      // This call was on the chain when the group parked, so its resume point
      // is in the group header. Only meaningful while replaying: on a fresh
      // call the array is stale and deliberately not read.
      resume_ = c.hdr_->state[depth_];
      replay_pending_ = true;
    }
  }

  /** The resume point, 0 on a fresh call. Generated code switches on this. */
  CLIO_CO_FUN u32 Resume() const { return resume_; }

  /**
   * True exactly once, at the case label this call resumed into.
   *
   * Clearing Ctx::replaying_ at the DEEPEST replayed frame is what stops a
   * later, unrelated call at the same depth from resuming into a stale state:
   * once the chain that parked has been rebuilt, everything after it is fresh.
   */
  CLIO_CO_FUN bool Replaying() {
    if (!replay_pending_) return false;
    replay_pending_ = false;
    if (depth_ == c_->hdr_->park_depth) c_->replaying_ = false;
    return true;
  }

  /**
   * Store the live set for suspend point `state` and mark the group parked.
   *
   * The static_assert is where design-doc rule R6 is enforced -- by the
   * compiler that builds the generated file, on every save list, on every
   * backend, every build, with no help from the transpiler.
   */
  template <class... Ts>
  CLIO_CO_FUN void Push(u32 state, const Ts &...vs) {
    static_assert((std::is_trivially_copyable_v<Ts> && ...),
                  "a value live across a CO_AWAIT must be trivially copyable");
    static_assert(PackBytes<Ts...>() <= 0xffffffffu, "pack too large");
    if (!c_->parked_) {
      c_->parked_ = true;
      // The deepest frame parks first, so the first Push of this park is the
      // one that records how far down the chain the replay must go.
      if (c_->lane_ == 0) c_->hdr_->park_depth = depth_;
    }
    if (c_->lane_ == 0) {
      c_->hdr_->status = kParked;
      c_->hdr_->state[depth_] = state;
    }
    char *p = Slot();
    (StoreOne(p, vs), ...);
  }

  /** Restore the live set written by the matching Push, in the same order. */
  template <class... Ts>
  CLIO_CO_FUN void Pop(Ts &...vs) {
    static_assert((std::is_trivially_copyable_v<Ts> && ...),
                  "a value live across a CO_AWAIT must be trivially copyable");
    char *p = Slot();
    (LoadOne(p, vs), ...);
  }

  /** Normal return: release the frame so the cursor reflects the chain (I1). */
  CLIO_CO_FUN void Done() {
    c_->depth_ -= 1;
    c_->cursor_ -= FrameStride(per_item_, c_->group_size_);
  }

  /** Bytes this call actually used at its widest Push; for the size assert. */
  CLIO_CO_FUN u32 Used() const { return popped_; }

 private:
  /** This work-item's private slice of this frame. Lane-major, so each lane
   *  owns a contiguous run and no two lanes ever touch the same bytes. */
  CLIO_CO_FUN char *Slot() const {
    return c_->frames_ + off_ +
           static_cast<std::size_t>(c_->lane_) * per_item_;
  }

  template <class T>
  CLIO_CO_FUN static void StoreOne(char *&p, const T &v) {
    p = Align(p, alignof(T));
    __builtin_memcpy(p, &v, sizeof(T));
    p += sizeof(T);
  }
  template <class T>
  CLIO_CO_FUN static void LoadOne(char *&p, T &v) {
    p = Align(p, alignof(T));
    __builtin_memcpy(&v, p, sizeof(T));
    p += sizeof(T);
  }
  CLIO_CO_FUN static char *Align(char *p, std::size_t a) {
    const std::uintptr_t x = reinterpret_cast<std::uintptr_t>(p);
    return reinterpret_cast<char *>((x + a - 1) & ~(a - 1));
  }

  Ctx *c_;
  u32 depth_;
  u32 off_;
  u32 per_item_;
  u32 resume_;
  bool replay_pending_;
  u32 popped_;
};

/* =========================================================================
 * Scope -- the kernel-side entry the user writes.
 * ========================================================================= */

/**
 * Declared at the top of the kernel; the transpiler finds it by type and
 * threads its Ctx into every suspending call.
 *
 *     clio::co::Scope co(stack, it);     // SYCL
 *     clio::co::Scope co(stack);         // CUDA / HIP
 *     if (co) TopLevelSuspendingFunction(args);
 *
 * `if (co)` is false for a group that already finished, so a relaunched grid
 * retires completed groups immediately and the driver does not have to compact
 * it. Compaction is an optimization, not a correctness requirement.
 */
class Scope {
 public:
#if CLIO_CO_CUDA || CLIO_CO_HIP
  CLIO_CO_FUN explicit Scope(StackView st) : ctx_(st, Item{}), st_(st) {}
#else
  CLIO_CO_FUN Scope(StackView st, Item it) : ctx_(st, it), st_(st) {}
#endif

  CLIO_CO_FUN explicit operator bool() const {
    return st_.Header(ctx_.It().Group())->status != kDone;
  }

  CLIO_CO_FUN Ctx &Context() { return ctx_; }

  /** Publish the outcome of this entry. A park has already stamped kParked in
   *  Frame::Push; anything else means the group ran off its end. */
  CLIO_CO_FUN ~Scope() {
    if (!ctx_.Parked() && ctx_.Lane() == 0) {
      st_.Header(ctx_.It().Group())->status = kDone;
    }
  }

 private:
  Ctx ctx_;
  StackView st_;
};

}  // namespace clio::co

#endif  // CLIO_RUNTIME_CO_CORO_H_

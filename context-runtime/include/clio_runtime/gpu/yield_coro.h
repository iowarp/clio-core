/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * C++20 coroutines as the yieldable-kernel mechanism (clang-CUDA only).
 *
 * The macro system in yield_stack.h is a hand-rolled stackless coroutine: a
 * switch on a saved resume point, with every local that crosses a yield
 * hoisted by hand into a bump-allocated frame — and nothing diagnoses a
 * hoist that was missed. The compiler already solves exactly this problem
 * for coroutines: it splits the function into a state machine AND computes
 * frame liveness itself. nvcc refuses coroutines in device code, but clang
 * compiles them (proved by spike_device_coroutine.cu), so a clang-CUDA build
 * can use the real thing. This header is that mechanism, speaking the SAME
 * host protocol as the macros: the same YieldStack lane regions hold the
 * frames, the same YieldBlockState status/wait_tag drives the host's
 * relaunch/park loop, so the host driver cannot tell which mechanism a
 * kernel used.
 *
 * SHAPE
 *
 *   - One coroutine chain PER LANE (coroutines are per-thread), suspending
 *     block-collectively: every suspend site is guarded by a __syncthreads_or
 *     vote, so all lanes suspend together or none do — the same rule, and
 *     the same reason, as the macros' CLIO_YIELD_IF.
 *   - The kernel itself is NOT a coroutine (a __global__ cannot be). It is a
 *     small wrapper (CLIO_YCORO_RUN) that creates the lane's coroutine on
 *     first entry and resumes the DEEPEST suspended frame on later entries.
 *   - Nested yieldable calls are child coroutines, but control is handed
 *     over by TRAMPOLINE, not by symmetric transfer: every transfer records
 *     "who runs next" in the lane header and RETURNS to the wrapper, which
 *     loops. NVPTX does not tail-call coroutine transfers, so the textbook
 *     symmetric transfer nests a real device stack frame per handoff -- on a
 *     fully-resident run (no suspends for a thousand pages) that grew ~2
 *     frames per page until compute-sanitizer reported a device stack
 *     overflow. The trampoline caps the depth at ONE resume regardless of
 *     chain length, at the cost of an indirect call per handoff.
 *
 * WHY THE INTRINSICS AND NOT std::coroutine_handle METHODS
 *
 * This TU also contains the HOST runtime, which includes the real
 * <coroutine>; libstdc++'s resume()/done()/from_promise() are host-only
 * there (not constexpr, so clang does not make them implicitly
 * host+device). The constexpr members (from_address, address) ARE
 * device-usable and are used where the coroutine protocol demands the std
 * type; everything else goes through clang's __builtin_coro_* intrinsics,
 * which the spikes proved work in device code.
 *
 * FRAMES
 *
 * promise_type::operator new bump-allocates from the SAME per-lane region
 * the macro frames use (after YieldLaneHeader), so frames live in global
 * memory and survive the kernel exiting. Child frames free LIFO when the
 * parent's co_await completes. Frame sizes are compiler-chosen and bigger
 * than the hand-packed macro frames (the frame carries the coroutine's
 * whole live state plus resume/destroy pointers): size lanes accordingly
 * (the weights bench uses 2 KB per lane in coro mode vs 256 B for macros).
 */
#ifndef CLIO_RUNTIME_GPU_YIELD_CORO_H_
#define CLIO_RUNTIME_GPU_YIELD_CORO_H_

/* SYCL reaches this file too. The gate is "clang, compiling for a device":
 * clang-CUDA (__CUDA__) or a SYCL compiler. Both lower coroutines the same
 * way and both provide __builtin_coro_*; what differs is only that SYCL has
 * no host/device attributes, so <coroutine> works there as written and the
 * __device__ spellings below vanish (CTP_GPU_FUN). nvcc still cannot come
 * anywhere near this file. */
#include <clio_ctp/constants/macros.h>

/* CTP_ENABLE_SYCL rather than CTP_IS_SYCL_COMPILER: a SYCL program's HOST
 * translation units are compiled without -fsycl, yet they include
 * device_vector.h (through gpu_vector.h) and so must PARSE the YCoroTask
 * return types on its verbs. They never instantiate a coroutine -- these are
 * only names to them. Under CUDA nothing changes: those TUs are compiled by
 * clang-CUDA, so __CUDA__ already covers them.
 *
 * Requires clang either way. That is not a new constraint for SYCL, where
 * the device compiler is always clang (DPC++) or AdaptiveCpp -- but it does
 * mean the HOST compiler has to be the SYCL one too, not a stray g++. */
#if defined(CLIO_YIELD_CORO) && defined(__clang__) && \
    (defined(__CUDA__) || CTP_ENABLE_SYCL)

/* ------------------------------------------------------------------ */
/* MEASUREMENT MODE: -DCLIO_YIELD_SYNC                                  */
/*                                                                      */
/* Compiles the SAME kernels with the coroutine-ness removed: co_await  */
/* is elided, co_return becomes a plain return, the task types collapse */
/* to their value types, and a yield becomes a SPIN on the same         */
/* condition. The paging machinery is untouched -- probe, fetch, evict  */
/* and flush all still compile in. The only thing that goes away is the */
/* coroutine.                                                           */
/*                                                                      */
/* WHY: it answers "what would these kernels cost if co_await were      */
/* free?" -- the register count of the real workload, with real paging, */
/* and no coroutine ABI. That number is the target the coroutine build  */
/* should be judged against, and it cannot be obtained any other way.   */
/*                                                                      */
/* NEVER RUN A BINARY BUILT THIS WAY. A spin cannot service a fault:    */
/* the fetch needs the kernel to EXIT so the servicer can run, so this  */
/* build deadlocks by construction on any miss. It is compiled and read */
/* with cuobjdump, never launched.                                      */
/* ------------------------------------------------------------------ */
#if defined(CLIO_YIELD_SYNC)

namespace clio::run::gpu {
template <class V> using YCoroTaskT = V;
using YCoroTask = void;
using YCoroMain = void;
struct YCoroSuspend {   // constructed and discarded; never suspends
  clio::run::u64 tag;
};
}  // namespace clio::run::gpu

#define co_await
#define co_return return

// The driver becomes a plain call: no frame, no resume loop, no indirection.
#define CLIO_YCORO_RUN(create_expr) \
  do {                              \
    (create_expr);                  \
  } while (0)

// A yield is DELETED. Not replaced by a spin -- a spin has its own live
// state (the resumed flag, the vote, the barrier) and would be measured as
// coroutine cost that is not coroutine cost. Nothing is substituted.
#define CLIO_CO_YIELD_WHEN(reap, cond, tag) do { (void)(tag); } while (0)

#else  // !CLIO_YIELD_SYNC -- the real coroutine implementation follows



#include <coroutine>

#include <clio_runtime/gpu/yield_stack.h>

namespace clio::run::gpu {

/** The lane's frame arena: bump within [sizeof(YieldLaneHeader),
 *  bytes_per_lane). sp_ is reused as the cursor. Traps on overflow for the
 *  same reason YieldFrame::Alloc does: the next lane's frames are directly
 *  after this one, and an overflow silently corrupts a NEIGHBOUR. */
__device__ __forceinline__ void *YCoroAlloc(size_t n) {
  YieldLaneHeader *lane = YieldLane();
  const clio::run::u32 off = (lane->sp_ + 15u) & ~15u;
  const clio::run::u32 end = off + static_cast<clio::run::u32>((n + 15) & ~15);
  if (end > YieldTls().stack_.bytes_per_lane_) {
    lane->error_ = kYieldErrOverflow;
    printf("[ycoro] block %u lane %u: frame overflow, need %u > %u bytes\n",
           blockIdx.x, threadIdx.x, end, YieldTls().stack_.bytes_per_lane_);
    YieldFatalNote(kYieldFatalCoroFrame, blockIdx.x, threadIdx.x, end);
    __trap();
  }
  lane->sp_ = end;
  return reinterpret_cast<char *>(lane) + off;
}

/** LIFO pop: `p` is the frame's own base, so the cursor rewinds to it. */
__device__ __forceinline__ void YCoroFree(void *p) {
  YieldLaneHeader *lane = YieldLane();
  lane->sp_ = static_cast<clio::run::u32>(reinterpret_cast<char *>(p) -
                                          reinterpret_cast<char *>(lane));
}

/**
 * Suspend the lane's chain here; block-collective use only (see
 * CLIO_CO_YIELD_WHEN). Records the deepest frame for the wrapper to resume,
 * and — from lane 0 — publishes the suspend and its wait tag to the host,
 * exactly as the macro yield does.
 */
struct YCoroSuspend {
  clio::run::u64 tag_;
  __device__ bool await_ready() const noexcept { return false; }
  template <typename P>
  __device__ void await_suspend(std::coroutine_handle<P> h) noexcept {
    YieldLaneHeader *lane = YieldLane();
    lane->coro_resume_ = reinterpret_cast<clio::run::u64>(h.address());
    lane->coro_park_ = 1u;   // for the HOST: the wrapper's loop must stop
    if (threadIdx.x == 0) {
      YieldSmem &s = YieldTls();
      s.block_state_->wait_tag_ = tag_;
      s.block_state_->status_ = kYieldSuspended;
    }
    __threadfence_system();
  }
  __device__ void await_resume() const noexcept {
    if (threadIdx.x == 0) {
      YieldTls().block_state_->wait_tag_ = 0;
    }
  }
};

/**
 * A nested yieldable call: `co_await SomeChildCoro(...)`.
 *
 * The child is created suspended (initial_suspend). Awaiting it stores our
 * handle in the child for the trip back, queues the child on the lane's
 * trampoline, and suspends; the wrapper's loop then runs the child. When the
 * child runs off its end, its final awaiter queues the parent the same way,
 * and await_resume (running in the parent) frees the child's frame (LIFO).
 * A suspend INSIDE the child records the child's own frame as the lane's
 * resume point, so the next kernel entry resumes the child directly — the
 * parent just stays suspended at the co_await until the child finishes.
 */
struct YCoroTask {
  struct promise_type {
    void *parent_ = nullptr;
    __device__ void *operator new(size_t n) { return YCoroAlloc(n); }
    __device__ void operator delete(void *p) { YCoroFree(p); }
    __device__ void operator delete(void *p, size_t) { YCoroFree(p); }
    __device__ YCoroTask get_return_object() {
      return YCoroTask{
          __builtin_coro_promise(reinterpret_cast<char *>(this),
                                 alignof(promise_type), /*from-promise=*/true)};
    }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    struct FinalXfer {
      __device__ bool await_ready() const noexcept { return false; }
      __device__ void await_suspend(
          std::coroutine_handle<promise_type> h) noexcept {
        // h.promise() is host-only in libstdc++; recover the promise (and its
        // stored parent) through the intrinsic instead. Trampoline: hand the
        // parent to the wrapper's loop rather than resuming it from here.
        auto *p = reinterpret_cast<promise_type *>(__builtin_coro_promise(
            h.address(), alignof(promise_type), /*from-promise=*/false));
        YieldLane()->coro_resume_ =
            reinterpret_cast<clio::run::u64>(p->parent_);
      }
      __device__ void await_resume() const noexcept {}
    };
    __device__ FinalXfer final_suspend() noexcept { return {}; }
    __device__ void return_void() noexcept {}
    __device__ void unhandled_exception() {}
  };

  void *frame_;

  __device__ bool await_ready() const noexcept { return false; }
  template <typename P>
  __device__ void await_suspend(std::coroutine_handle<P> parent) noexcept {
    auto *p = reinterpret_cast<promise_type *>(__builtin_coro_promise(
        frame_, alignof(promise_type), /*from-promise=*/false));
    p->parent_ = parent.address();
    // Trampoline: queue the child for the wrapper's loop and return.
    YieldLane()->coro_resume_ = reinterpret_cast<clio::run::u64>(frame_);
  }
  __device__ void await_resume() noexcept {
    // The child sits at its final suspend; destroying it runs no user code
    // (everything already ran) and releases the frame through operator
    // delete, rewinding the lane's bump cursor.
    __builtin_coro_destroy(frame_);
  }
};

/**
 * YCoroTask that RETURNS A VALUE to its awaiter: `V v = co_await Child(...)`.
 *
 * Identical trampoline to YCoroTask; the only additions are the value slot in
 * the promise and the move-out in await_resume, which runs in the parent
 * right before the child's frame is freed.
 */
template <typename V>
struct YCoroTaskT {
  struct promise_type {
    void *parent_ = nullptr;
    V value_{};
    __device__ void *operator new(size_t n) { return YCoroAlloc(n); }
    __device__ void operator delete(void *p) { YCoroFree(p); }
    __device__ void operator delete(void *p, size_t) { YCoroFree(p); }
    __device__ YCoroTaskT get_return_object() {
      return YCoroTaskT{
          __builtin_coro_promise(reinterpret_cast<char *>(this),
                                 alignof(promise_type), /*from-promise=*/true)};
    }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    struct FinalXfer {
      __device__ bool await_ready() const noexcept { return false; }
      __device__ void await_suspend(
          std::coroutine_handle<promise_type> h) noexcept {
        auto *p = reinterpret_cast<promise_type *>(__builtin_coro_promise(
            h.address(), alignof(promise_type), /*from-promise=*/false));
        YieldLane()->coro_resume_ =
            reinterpret_cast<clio::run::u64>(p->parent_);
      }
      __device__ void await_resume() const noexcept {}
    };
    __device__ FinalXfer final_suspend() noexcept { return {}; }
    // BY RVALUE REFERENCE, not by value: clang's device lowering
    // materializes `co_return <prvalue>` in the coroutine frame and then
    // BITWISE-copies it into a by-value parameter -- no move constructor
    // runs, so an RAII operand ends up with two live owners and its
    // destructor fires twice (observed as page pins underflowing by one
    // blockDim per hold). Binding the reference to the materialized
    // temporary lets the move-assign steal from -- and null -- the one and
    // only object.
    __device__ void return_value(V &&v) noexcept {
      value_ = static_cast<V &&>(v);
    }
    __device__ void unhandled_exception() {}
  };

  void *frame_;

  __device__ bool await_ready() const noexcept { return false; }
  template <typename P>
  __device__ void await_suspend(std::coroutine_handle<P> parent) noexcept {
    auto *p = reinterpret_cast<promise_type *>(__builtin_coro_promise(
        frame_, alignof(promise_type), /*from-promise=*/false));
    p->parent_ = parent.address();
    YieldLane()->coro_resume_ = reinterpret_cast<clio::run::u64>(frame_);
  }
  __device__ V await_resume() noexcept {
    auto *p = reinterpret_cast<promise_type *>(__builtin_coro_promise(
        frame_, alignof(promise_type), /*from-promise=*/false));
    V v = static_cast<V &&>(p->value_);
    __builtin_coro_destroy(frame_);
    return v;
  }
};

/** The lane's TOP-LEVEL coroutine. Created by the wrapper, resumed by the
 *  wrapper, parks at final_suspend so done() answers "lane finished?". */
struct YCoroMain {
  struct promise_type {
    __device__ void *operator new(size_t n) { return YCoroAlloc(n); }
    __device__ void operator delete(void *p) { YCoroFree(p); }
    __device__ void operator delete(void *p, size_t) { YCoroFree(p); }
    __device__ YCoroMain get_return_object() {
      return YCoroMain{
          __builtin_coro_promise(reinterpret_cast<char *>(this),
                                 alignof(promise_type), /*from-promise=*/true)};
    }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    __device__ std::suspend_always final_suspend() noexcept { return {}; }
    __device__ void return_void() noexcept {}
    __device__ void unhandled_exception() {}
  };
  void *frame_;
};

}  // namespace clio::run::gpu

/**
 * Block-collective conditional yield with a host wait tag — the coroutine
 * spelling of CLIO_YIELD_IF_RESUME_WHEN, usable only INSIDE a YCoroTask /
 * YCoroMain coroutine. Same semantics: every thread votes, the block
 * suspends if ANY thread is waiting, and a resume RE-EVALUATES the
 * condition, making it a retry loop with the host in the middle.
 *
 * `reap` runs (on thread 0) before EVERY evaluation of the condition,
 * including after each resume. The macro system got this for free — a
 * resume re-enters the whole function, so the pre-switch prologue re-ran —
 * but a coroutine resumes INSIDE the loop, so what must re-run on resume
 * has to live in the loop. Pass `;` when nothing needs reaping.
 */
#define CLIO_CO_YIELD_WHEN(reap, cond, tag)                                   \
  for (bool _cy_resumed = false;;) {                                          \
    /* Reap only AFTER a resume -- the macro system's semantics exactly.   */ \
    /* Its prologue reap covers the first evaluation; reaping before the   */ \
    /* first vote here doubled the locked slot scans per page hold, which  */ \
    /* measured as most of the coroutine path's nvcomp-regime overhead.    */ \
    if (_cy_resumed && threadIdx.x == 0) {                                    \
      reap;                                                                   \
    }                                                                         \
    __syncthreads();                                                          \
    if (!__syncthreads_or((cond) ? 1 : 0)) {                                  \
      break;                                                                  \
    }                                                                         \
    co_await ::clio::run::gpu::YCoroSuspend{                                  \
        static_cast<clio::run::u64>(tag)};                                    \
    _cy_resumed = true;                                                       \
  }

/**
 * Kernel-side driver: create this lane's coroutine on first entry, resume
 * the deepest suspended frame on later entries, and reset the lane when the
 * chain finishes. YieldTlsPublish must have run first (it stamps kYieldDone,
 * which stands unless something suspends during this entry — identical to
 * the macro kernels). `create_expr` must evaluate to a YCoroMain.
 */
#define CLIO_YCORO_RUN(create_expr)                                           \
  do {                                                                        \
    clio::run::gpu::YieldLaneHeader *_yc_lane = clio::run::gpu::YieldLane();  \
    if (_yc_lane->coro_top_ == 0) {                                           \
      _yc_lane->sp_ = sizeof(clio::run::gpu::YieldLaneHeader);                \
      clio::run::gpu::YCoroMain _yc_t = (create_expr);                        \
      _yc_lane->coro_top_ = reinterpret_cast<clio::run::u64>(_yc_t.frame_);   \
      _yc_lane->coro_resume_ = _yc_lane->coro_top_;                           \
    }                                                                         \
    __syncthreads();                                                          \
    /* Shared did not survive the park; bring the persistent arena back  */   \
    /* BEFORE resuming, while every thread is still here to help copy.   */   \
    ::clio::run::gpu::PersistRestore();                                       \
    for (;;) {                                                                \
      __builtin_coro_resume(                                                  \
          reinterpret_cast<void *>(_yc_lane->coro_resume_));                  \
      if (_yc_lane->coro_park_ != 0u ||                                       \
          __builtin_coro_done(                                                \
              reinterpret_cast<void *>(_yc_lane->coro_top_))) {               \
        break;                                                                \
      }                                                                       \
    }                                                                         \
    _yc_lane->coro_park_ = 0u;                                                \
    __syncthreads();                                                          \
    if (__builtin_coro_done(                                                  \
            reinterpret_cast<void *>(_yc_lane->coro_top_))) {                 \
      ::clio::run::gpu::PersistClear();                                       \
      __builtin_coro_destroy(                                                 \
          reinterpret_cast<void *>(_yc_lane->coro_top_));                     \
      _yc_lane->coro_top_ = 0;                                                \
      _yc_lane->coro_resume_ = 0;                                             \
      _yc_lane->sp_ = sizeof(clio::run::gpu::YieldLaneHeader);                \
    } else {                                                                  \
      /* Parked: the kernel is about to exit and take shared with it. */      \
      ::clio::run::gpu::PersistSave();                                        \
    }                                                                         \
  } while (0)

#endif  // CLIO_YIELD_SYNC
#endif  // CLIO_YIELD_CORO && __clang__ && __CUDA__
#endif  // CLIO_RUNTIME_GPU_YIELD_CORO_H_

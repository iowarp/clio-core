/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Lane-divided continuation stack for yieldable kernels.
 *
 * The base yieldable.h keeps ONE state struct per block, which is wrong for
 * anything but the simplest kernel: locals are per THREAD. Each lane has its
 * own `i`, its own accumulator, its own everything. A yield suspends the whole
 * block, so every lane's locals have to come back.
 *
 * So the stack is divided per (logical block, lane), and inside a lane it is a
 * real stack of FRAMES -- one per live yieldable call -- which is what lets a
 * device function yield without its caller knowing or caring.
 *
 *   stack ── block 0 ── lane 0 [hdr][frame][frame]...
 *         │          ── lane 1 [hdr][frame]...
 *         │          ── ...
 *         └─ block 1 ── ...
 *
 * WHY LOCALS STOP BEING A MANUAL CHORE
 *
 * Declare them at the top of the function and they are references INTO the
 * frame:
 *
 *     CLIO_YFRAME();                       // attach this call's frame
 *     CLIO_YLOCAL_INIT(int, i, 0);         // lives in the frame, survives
 *     CLIO_YBEGIN();
 *     for (; i < 10; ++i) { CLIO_YIELD(); }
 *     CLIO_YEND();
 *
 * Nothing is saved or restored by hand, and no state struct is declared: the
 * frame layout is computed by bump-allocating in declaration order, which is
 * stable because the declarations sit BEFORE the switch and therefore re-run
 * on every entry, including a resume. That ordering is the whole trick -- it
 * is also why the switch cannot jump over them, so the construct that would be
 * ill-formed is structurally impossible here.
 *
 * What is still manual is the DECLARATION SITE: locals live across a yield
 * only if declared with CLIO_YLOCAL at the top. Nothing diagnoses a miss --
 * nvcc's device pass accepts a plain initialized local held across a yield
 * without a word, so it survives compilation and then reads back garbage after
 * a resume. Host g++ rejects the same shape, so a clang-CUDA lint build is the
 * cheapest place to catch it until the AST pass exists. Deciding that automatically
 * means a liveness analysis over the CFG and hoisting the declarations, i.e. a
 * Clang pass over the AST. That pass would emit exactly these macros, so it is
 * additive rather than a redesign.
 *
 * FINDING THE STACK WITHOUT PASSING IT
 *
 * The block's stack base is published once at kernel entry into DYNAMIC shared
 * memory, so any __device__ function reaches it through `extern __shared__`
 * without taking a parameter. The first sizeof(YieldSmem) bytes of the dynamic
 * shared allocation are reserved for this; a kernel that also wants dynamic
 * shared memory must offset past it (CLIO_YIELD_SMEM_BYTES).
 *
 * WHY YIELDS STAY BLOCK-COLLECTIVE (decided, not defaulted)
 *
 * The per-lane frame and per-lane resume point would ALSO support lanes
 * suspending at different places; the only thing forbidding it is the
 * __syncthreads inside the yield macros. That barrier is kept on purpose.
 *
 * Consumers here need a divergent PREDICATE, not a divergent POSITION: with a
 * page cache whose contract is that a warp's accesses land in one page, the
 * only per-thread question is "must I wait?", which CLIO_YIELD_IF already
 * votes on with __syncthreads_or. Nobody needs lane 3 suspended at one line
 * while lane 7 is suspended at another.
 *
 * Keeping suspension uniform is also what allows __syncthreads ANYWHERE ELSE
 * in the body. If lanes could exit at arbitrary points, every later
 * cooperative barrier in that kernel could deadlock -- and that hazard is not
 * specific to this mechanism, it applies equally to a coroutine-based design
 * whose lanes suspend independently.
 *
 * COST OF A YLOCAL, AND THE ONE RULE THAT FOLLOWS
 *
 * A YLOCAL is a reference into device memory, and nvcc does NOT promote it to
 * a register across a loop. Measured, 64 blocks x 256 threads summing a 4 MB
 * array (times relative to the same loop with an ordinary local):
 *
 *     ordinary local                                    1.00x
 *     YLOCAL, register copy for the loop                0.90x   <-- free
 *     YLOCAL used directly, yield OUTSIDE the loop     28.8x
 *     YLOCAL used directly, yield INSIDE the loop      49.6x
 *
 * So: a YLOCAL is for values that must SURVIVE a yield, not for values a loop
 * touches. Pull a register copy for the yield-free region and write it back:
 *
 *     { u64 r = acc;                          // register
 *       for (...) r += data[e];               // hot loop, no frame traffic
 *       acc = r; }                            // back to the frame
 *
 * That single move is the difference between 29x and free, and it is why the
 * realistic paged benchmark (spike 4) came out at 1.19x rather than 10x.
 *
 * A suspend point INSIDE the loop costs another ~1.7x on top, because the live
 * values must be in the frame at that point on every iteration. Yield at the
 * coarsest granularity the algorithm allows -- per page, not per element.
 *
 * LAYOUT NOTE (deliberate, and a known cost)
 *
 * A lane's frames are CONTIGUOUS, so `T &` references into them are possible
 * and locals behave like ordinary variables. The cost is that neighbouring
 * lanes touch addresses one lane-region apart, so a warp reading the same
 * local is not coalesced. The alternative -- interleaving word w of every lane
 * together -- coalesces perfectly but cannot hand out references for anything
 * wider than a word, which would turn every local back into a get/set chore.
 * Ergonomics won; if a kernel is bound by frame traffic, that is the knob.
 */
#ifndef CLIO_RUNTIME_GPU_YIELD_STACK_H_
#define CLIO_RUNTIME_GPU_YIELD_STACK_H_

#include <clio_ctp/util/gpu_api.h>
#include <clio_ctp/util/sycl_cuda_compat.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::run::gpu {

/** Deepest nest of yieldable calls a single lane may have live at once. */
GLOBAL_CROSS_CONST clio::run::u32 kYieldMaxDepth = 8;

/**
 * Per-lane bookkeeping, at the front of each lane's region.
 *
 * `frame_off_` exists so that a resume can re-attach to the SAME frames rather
 * than guessing: on re-entry the call chain re-executes in the same order, and
 * each call finds its old frame by depth instead of re-deriving an offset.
 */
struct YieldLaneHeader {
  /** Byte offset of the next free frame within this lane's region. */
  clio::run::u32 sp_;
  /** Frames currently live (suspended call chain depth). */
  clio::run::u32 live_depth_;
  /** Depth reached so far during THIS entry. Reset at kernel entry. */
  clio::run::u32 cur_depth_;
  /** Nonzero if this lane blew a limit; see kYieldErr* below. */
  clio::run::u32 error_;
  /** Offset of each live frame, indexed by depth. */
  clio::run::u32 frame_off_[kYieldMaxDepth];
  /**
   * COROUTINE MODE ONLY (yield_coro.h, clang-CUDA builds): the deepest
   * suspended coroutine frame (what the next entry resumes) and the lane's
   * top-level frame (whose done() ends the lane). Zero = no coroutine live.
   * The macro machinery never reads these; they ride here so both mechanisms
   * share one stack allocation and one init kernel.
   */
  clio::run::u64 coro_resume_;
  clio::run::u64 coro_top_;
  /** 1 = the chain parked for the HOST (vs. an in-device handoff). */
  clio::run::u32 coro_park_;
  clio::run::u32 coro_pad_;
};

/** Limit violations. These corrupt silently if unchecked, so they trap. */
enum YieldError : clio::run::u32 {
  kYieldErrNone = 0,
  /** More nested yieldable calls live at once than kYieldMaxDepth. */
  kYieldErrDepth = 1,
  /** This lane's frames outgrew bytes_per_lane. */
  kYieldErrOverflow = 2,
};

/** Reasons the yield driver traps, latched in the fatal channel below. */
enum YieldFatal : unsigned long long {
  kYieldFatalNone = 0,
  kYieldFatalDepth = 101,
  kYieldFatalFrame = 102,
  kYieldFatalCoroFrame = 103,
};

/**
 * THE FATAL CHANNEL: four slots of PINNED HOST memory the driver writes just
 * before it traps.
 *
 * A trap kills the CUDA context, which takes the printf FIFO and all device
 * memory with it -- so a kernel that traps reports nothing but "CUDA Error
 * 715: an illegal instruction", and the message explaining why never
 * arrives. Host memory survives, so a poller can read the reason. The pointer
 * is installed by the host (YieldSetFatalSlots) and is null until then, so
 * this costs nothing when unused.
 */
#if CTP_ENABLE_SYCL
/**
 * SYCL has no __device__ variables: a mutable namespace-scope global is not
 * addressable from a kernel. device_global is the sanctioned replacement --
 * program-scope device storage the host writes with queue::copy. Same role,
 * same "null until the host installs it" contract.
 *
 * The same mechanism carries the per-block YieldSmem base, because
 * `extern __shared__` has no SYCL equivalent either; see YieldTls below.
 *
 * ONLY THE OWNING TRANSLATION UNIT MAY DEFINE THESE, and it says so by
 * defining CLIO_SYCL_KERNEL_TU before including this header. A device_global
 * is registered with the SYCL runtime once per DEVICE IMAGE, and every TU
 * compiled with -fsycl produces an image of its own -- so leaving these
 * inline in a widely-included header made the second image re-register them
 * and DPC++ aborted at startup:
 *
 *   Assertion `!MDeviceGlobalPtr && "Device global pointer has already been
 *   initialized."' failed.
 *
 * A TU with no yieldable kernels needs the accessors only to PARSE (host
 * code reaches them through device_vector.h's verbs), so it gets a null
 * base rather than a definition. Any kernel that actually ran there would
 * dereference null immediately, which is the loudest failure available.
 */
#if defined(CLIO_SYCL_KERNEL_TU)
/* device_image_scope, not the default USM-backed form.
 *
 * A default device_global is backed by a USM allocation the SYCL runtime
 * frees from a static destructor -- after the CUDA context has already gone
 * away at process exit, which aborted every run at:
 *
 *   exception in ~DeviceGlobalUSMMem cuda backend failed with error ...
 *   DeviceGlobalUSMMem::~DeviceGlobalUSMMem(): Assertion `false' failed.
 *
 * device_image_scope puts the variable in the device image instead, so
 * there is no USM allocation to outlive anything. It also states the
 * constraint that is already true here: only the owning image reads it. */
using SyclImageScope = decltype(::sycl::ext::oneapi::experimental::properties(
    ::sycl::ext::oneapi::experimental::device_image_scope));
inline ::sycl::ext::oneapi::experimental::device_global<unsigned long long *,
                                                        SyclImageScope>
    g_yield_fatal_dg;
inline ::sycl::ext::oneapi::experimental::device_global<char *, SyclImageScope>
    g_yield_smem_dg;
inline unsigned long long *YieldFatalBase() { return g_yield_fatal_dg.get(); }
inline char *YieldSmemBase() { return g_yield_smem_dg.get(); }
#else
inline unsigned long long *YieldFatalBase() { return nullptr; }
inline char *YieldSmemBase() { return nullptr; }
#endif
#define CLIO_YIELD_FATAL_PTR (::clio::run::gpu::YieldFatalBase())
#else
__device__ inline unsigned long long *g_yield_fatal = nullptr;
#define CLIO_YIELD_FATAL_PTR (::clio::run::gpu::g_yield_fatal)
#endif

CTP_GPU_FUN inline void YieldFatalNote(unsigned long long code,
                                       unsigned long long a1,
                                       unsigned long long a2,
                                       unsigned long long a3) {
#if defined(__CUDA_ARCH__) || CTP_IS_SYCL_COMPILER
  unsigned long long *g_yield_fatal = CLIO_YIELD_FATAL_PTR;
  if (g_yield_fatal == nullptr) return;
  if (atomicCAS(g_yield_fatal, kYieldFatalNone, code) != kYieldFatalNone) {
    return;   // first writer wins: it is the one that explains the rest
  }
  g_yield_fatal[1] = a1;
  g_yield_fatal[2] = a2;
  g_yield_fatal[3] = a3;
  __threadfence_system();
#else
  (void) code; (void) a1; (void) a2; (void) a3;
#endif
}

/** Bytes of block-uniform shared state that survive a park. */
#define CLIO_PERSIST_BYTES 1024
/** Per-block backing-store stride: the arena plus its header. */
#define CLIO_PERSIST_STRIDE (CLIO_PERSIST_BYTES + 16)

/** Header of one block's persist backing store. */
struct PersistHeader {
  /** Live bytes of the arena, or 0 when nothing is saved. */
  clio::run::u32 bytes_;
  clio::run::u32 pad_[3];
};

/** Header of one call frame: everything before the locals. */
struct YieldFrameHeader {
  /** 0 = this call has not yielded yet; else the __LINE__ to resume at. */
  clio::run::u32 resume_point_;
  clio::run::u32 pad_;
};

/**
 * Device-side view of the whole stack. Small and passed by value; the base
 * pointer for the running block is what gets published to shared memory.
 */
struct YieldStackView {
  char *base_ = nullptr;
  clio::run::u32 bytes_per_lane_ = 0;
  clio::run::u32 lanes_per_block_ = 0;
  /** Backing store for the persistent shared arena, one region per LOGICAL
   *  block. Shared memory does not survive a park -- the driver exits the
   *  kernel -- so the arena is copied out here on suspend and back on
   *  resume. Indexed by LOGICAL block, never blockIdx.x: the driver compacts
   *  the grid between rounds, so a block resumes on different hardware. */
  char *persist_base_ = nullptr;

#if defined(__CUDACC__) || CTP_ENABLE_SYCL
  /** Region for one lane of one LOGICAL block. */
  CTP_GPU_FUN char *Lane(clio::run::u32 logical_block,
                        clio::run::u32 lane) const {
    const clio::run::u64 idx =
        static_cast<clio::run::u64>(logical_block) * lanes_per_block_ + lane;
    return base_ + idx * bytes_per_lane_;
  }
  /** This logical block's persistent-arena backing store. */
  CTP_GPU_FUN char *PersistLane(clio::run::u32 logical_block) const {
    return persist_base_ +
           static_cast<clio::run::u64>(logical_block) * CLIO_PERSIST_STRIDE;
  }
#endif
};

/** What lives at offset 0 of a yieldable kernel's dynamic shared memory. */
struct YieldSmem {
  YieldStackView stack_;
  /** This block's driver-visible status, so a yield deep inside a nested
   *  device function can mark the block suspended without the kernel having
   *  passed anything down to it. */
  YieldBlockState *block_state_;
  /** This block's logical id, so device functions need not be told it. */
  clio::run::u32 logical_block_;
  /** Live bytes of persist_, set by CLIO_SHARED_PERSIST. */
  clio::run::u32 persist_bytes_;
  /** THE PERSISTENT SHARED ARENA. Lives inside YieldSmem on purpose:
   *  CLIO_YIELD_SMEM_BYTES is sizeof(YieldSmem), so every kernel that already
   *  offsets its own shared past it keeps working with no change. */
  __align__(16) char persist_[CLIO_PERSIST_BYTES];
};

}  // namespace clio::run::gpu

/** Dynamic shared memory a yieldable kernel must reserve, in bytes. */
#define CLIO_YIELD_SMEM_BYTES (sizeof(clio::run::gpu::YieldSmem))

/**
 * Block-uniform shared state that SURVIVES A PARK.
 *
 *     struct Tables { const float *sp[9]; clio::run::u64 run[9]; };
 *     CLIO_SHARED_PERSIST(Tables, t);      // t.sp[q], t.run[q]
 *
 * Declares `Type &name` over the block's persistent arena and tells the
 * driver how much of it to carry across suspensions. Plain __shared__ is
 * GARBAGE after a co_await, because the driver exits the kernel to suspend;
 * anything block-uniform that is read after a suspension belongs here.
 *
 * One arena per block, so one such declaration per coroutine. Declare it
 * before the first co_await -- the size has to be known before anything can
 * suspend.
 */
#define CLIO_SHARED_PERSIST(Type, name)                                       \
  static_assert(sizeof(Type) <= CLIO_PERSIST_BYTES,                           \
                "CLIO_SHARED_PERSIST: type is larger than the arena; raise "  \
                "CLIO_PERSIST_BYTES or stage less");                          \
  Type &name = *reinterpret_cast<Type *>(                                     \
      ::clio::run::gpu::SharedPersistBase());                                 \
  ::clio::run::gpu::PersistDeclare(sizeof(Type))

/**
 * Limit checks. On by default: the failures they catch (nesting too deep, a
 * lane's frames overrunning into its neighbour's) are silent corruption that
 * surfaces later as wrong data somewhere unrelated. Each is a compare against
 * a value already in registers, against a path that is already touching
 * global memory.
 */
#ifndef CLIO_YIELD_CHECKS
#define CLIO_YIELD_CHECKS 1
#endif

/* CTP_ENABLE_SYCL, not CTP_IS_SYCL_COMPILER: the host TUs of a SYCL program
 * are compiled without -fsycl but still include device_vector.h, whose verbs
 * name YieldLane/YieldTls. They only need the declarations. See the same
 * widening in yield_coro.h and sycl_cuda_compat.h. */
#if defined(__CUDACC__) || CTP_ENABLE_SYCL

namespace clio::run::gpu {

#if CTP_ENABLE_SYCL
/**
 * SYCL: the block's YieldSmem lives in GLOBAL memory, one per PHYSICAL
 * block, at a base the host installs in g_yield_smem_dg.
 *
 * WHY NOT SHARED. `extern __shared__` is what makes the CUDA version
 * reachable from any function without a parameter, and SYCL has no
 * equivalent: local memory arrives through a local_accessor the kernel was
 * given, or through group_local_memory, which allocates a DISTINCT block
 * per call site -- so twenty callers would get twenty arenas, not one view
 * of the block's state. Global memory has one address per block and no such
 * ambiguity.
 *
 * PHYSICAL block, not logical, and that is not a bug: YieldSmem is
 * per-LAUNCH scratch that YieldTlsPublish rewrites at every kernel entry.
 * The only thing in it that must outlive the launch is the persist arena,
 * and that is already copied to and from persist_base_[LOGICAL block] by
 * PersistSave/PersistRestore -- which is exactly why those two functions
 * need no SYCL branch at all.
 *
 * The cost is that the persist arena is global rather than shared. It is
 * block-uniform and L2-resident, and it buys the same guarantee CUDA gets
 * from copying it out on every park.
 */
inline YieldSmem &YieldTls() {
  char *base = YieldSmemBase();
  return *reinterpret_cast<YieldSmem *>(
      base + static_cast<size_t>(blockIdx.x) * sizeof(YieldSmem));
}
#else
/**
 * The published handle. Every yieldable device function reaches the stack
 * through this, which is why none of them take it as a parameter.
 *
 * `extern __shared__` names the single dynamic shared allocation, so every
 * translation unit and every function sees the same bytes.
 */
__device__ __forceinline__ YieldSmem &YieldTls() {
  extern __shared__ char clio_yield_smem[];
  return *reinterpret_cast<YieldSmem *>(clio_yield_smem);
}
#endif

/**
 * PERSISTENT SHARED STATE ACROSS A PARK.
 *
 * A suspension here is a KERNEL EXIT, which destroys shared memory. The
 * coroutine frame survives only because it was deliberately placed in global
 * memory; the language preserves nothing else, and __shared__ has static
 * storage duration in another address space, so the coroutine transform will
 * never promote it into the frame. These three calls close that gap for one
 * block-uniform arena: the driver copies it out on suspend and back on
 * resume.
 *
 * THIS PRESERVES BITS, NOT VALIDITY. Restoring a pointer gives back the same
 * address; whether it still points at what you meant is the caller's problem.
 * It is safe for pointers into page frames only because the guards that pin
 * those frames live in the coroutine frame and so survive the park too.
 */
__device__ __forceinline__ char *SharedPersistBase() {
  return YieldTls().persist_;
}

/** Declare how much of the arena is live. Block-collective. */
__device__ __forceinline__ void PersistDeclare(clio::run::u32 bytes) {
  if (threadIdx.x == 0) {
    YieldTls().persist_bytes_ = (bytes + 3u) & ~3u;
  }
  __syncthreads();
}

/** Copy the arena back from global. Block-collective; call before resuming. */
__device__ __forceinline__ void PersistRestore() {
  YieldSmem &s = YieldTls();
  if (s.stack_.persist_base_ != nullptr) {
    PersistHeader *h =
        reinterpret_cast<PersistHeader *>(s.stack_.PersistLane(s.logical_block_));
    const clio::run::u32 n =
        (h->bytes_ <= CLIO_PERSIST_BYTES) ? h->bytes_ : 0u;
    if (n != 0) {
      const clio::run::u32 *src =
          reinterpret_cast<const clio::run::u32 *>(h + 1);
      clio::run::u32 *dst = reinterpret_cast<clio::run::u32 *>(s.persist_);
      for (clio::run::u32 i = threadIdx.x; i < (n >> 2); i += blockDim.x) {
        dst[i] = src[i];
      }
      // Shared was wiped, so the live-byte count has to come back too.
      if (threadIdx.x == 0) s.persist_bytes_ = n;
    }
  }
  __syncthreads();
}

/** Copy the arena out to global. Block-collective; call after a suspend. */
__device__ __forceinline__ void PersistSave() {
  __syncthreads();   // every thread's writes to the arena must be visible
  YieldSmem &s = YieldTls();
  if (s.stack_.persist_base_ != nullptr) {
    const clio::run::u32 n =
        (s.persist_bytes_ <= CLIO_PERSIST_BYTES) ? s.persist_bytes_ : 0u;
    PersistHeader *h =
        reinterpret_cast<PersistHeader *>(s.stack_.PersistLane(s.logical_block_));
    if (n != 0) {
      const clio::run::u32 *src =
          reinterpret_cast<const clio::run::u32 *>(s.persist_);
      clio::run::u32 *dst = reinterpret_cast<clio::run::u32 *>(h + 1);
      for (clio::run::u32 i = threadIdx.x; i < (n >> 2); i += blockDim.x) {
        dst[i] = src[i];
      }
    }
    if (threadIdx.x == 0) h->bytes_ = n;
  }
  __syncthreads();
}

/** Forget the saved arena; the coroutine that owned it has finished. */
__device__ __forceinline__ void PersistClear() {
  YieldSmem &s = YieldTls();
  if (s.stack_.persist_base_ != nullptr && threadIdx.x == 0) {
    reinterpret_cast<PersistHeader *>(
        s.stack_.PersistLane(s.logical_block_))->bytes_ = 0;
  }
}

/** Publish the stack for this block. Called once, by the kernel prologue. */
__device__ __forceinline__ void YieldTlsPublish(const YieldStackView &stack,
                                                YieldBlockState *block_state,
                                                clio::run::u32 logical_block) {
  if (threadIdx.x == 0) {
    YieldSmem &s = YieldTls();
    s.stack_ = stack;
    s.block_state_ = block_state;
    s.logical_block_ = logical_block;
    // ZERO THE ARENA'S LIVE-BYTE COUNT. Shared memory starts uninitialized,
    // so a kernel that never declares CLIO_SHARED_PERSIST would otherwise
    // hand PersistSave a garbage length and walk off the end of shared.
    s.persist_bytes_ = 0;
    // Stamp "finished" up front: any return that is not a yield -- including
    // paths nobody thought about -- then ends the block instead of leaving the
    // driver relaunching it forever.
    block_state->status_ = kYieldDone;
  }
  __syncthreads();
}

/** True when some frame in this block has suspended during THIS entry. */
__device__ __forceinline__ bool YieldSuspended() {
  return YieldTls().block_state_->status_ == kYieldSuspended;
}

/** This lane's header, found without any parameter threading. */
__device__ __forceinline__ YieldLaneHeader *YieldLane() {
  YieldSmem &s = YieldTls();
  return reinterpret_cast<YieldLaneHeader *>(
      s.stack_.Lane(s.logical_block_, threadIdx.x));
}

/**
 * One call's frame, plus the bump allocator that lays out its locals.
 *
 * Constructed at the top of every yieldable function. On a fresh call it
 * claims the next frame; on a resume it re-attaches to the frame this call
 * already owns at this depth, which is what makes the locals come back.
 */
struct YieldFrame {
  YieldLaneHeader *lane_;
  char *lane_base_;
  clio::run::u32 fp_;    // this frame's offset within the lane region
  clio::run::u32 lcur_;  // bump cursor for locals
  bool fresh_;           // first entry into this call?

  __device__ __forceinline__ YieldFrame() {
    lane_ = YieldLane();
    lane_base_ = reinterpret_cast<char *>(lane_);
    const clio::run::u32 d = lane_->cur_depth_++;
#if CLIO_YIELD_CHECKS
    if (d >= kYieldMaxDepth) {
      lane_->error_ = kYieldErrDepth;
      printf("[yield] block %u lane %u: nesting past kYieldMaxDepth=%u\n",
             blockIdx.x, threadIdx.x, (unsigned)kYieldMaxDepth);
      YieldFatalNote(kYieldFatalDepth, blockIdx.x, threadIdx.x, d);
      __trap();
    }
#endif
    if (d < lane_->live_depth_) {
      // Resuming: this call already has a frame, with its locals in it.
      fp_ = lane_->frame_off_[d];
      fresh_ = false;
    } else {
      fp_ = lane_->sp_;
      lane_->frame_off_[d] = fp_;
      lane_->sp_ = fp_ + sizeof(YieldFrameHeader);
      lane_->live_depth_ = d + 1;
      fresh_ = true;
      Header()->resume_point_ = 0;
    }
    lcur_ = fp_ + sizeof(YieldFrameHeader);
  }

  __device__ __forceinline__ YieldFrameHeader *Header() const {
    return reinterpret_cast<YieldFrameHeader *>(lane_base_ + fp_);
  }

  /**
   * Hand out storage for one local. Offsets are assigned in DECLARATION order
   * and that order re-runs identically on every entry, so a resume recovers
   * the same addresses without storing a layout anywhere.
   */
  __device__ __forceinline__ void *Alloc(clio::run::u32 size,
                                         clio::run::u32 align) {
    clio::run::u32 off = (lcur_ + (align - 1)) & ~(align - 1);
    lcur_ = off + size;
#if CLIO_YIELD_CHECKS
    // Silent otherwise: the next lane's frames are directly after this one, so
    // an overflow writes a NEIGHBOUR's locals and both lanes go wrong later,
    // far from here.
    if (lcur_ > YieldTls().stack_.bytes_per_lane_) {
      lane_->error_ = kYieldErrOverflow;
      printf("[yield] block %u lane %u: frame overflow, need %u > %u bytes\n",
             blockIdx.x, threadIdx.x, lcur_,
             YieldTls().stack_.bytes_per_lane_);
      YieldFatalNote(kYieldFatalFrame, blockIdx.x, threadIdx.x, lcur_);
      __trap();
    }
#endif
    if (fresh_ && lcur_ > lane_->sp_) {
      lane_->sp_ = lcur_;
    }
    return lane_base_ + off;
  }

  /** Pop on normal return, so a later call reuses the space. */
  __device__ __forceinline__ void Pop() const {
    lane_->cur_depth_ -= 1;
    lane_->live_depth_ = lane_->cur_depth_;
    lane_->sp_ = fp_;
    Header()->resume_point_ = 0;
  }

  /** Leave the frame live; the kernel is unwinding to be resumed later. */
  __device__ __forceinline__ void Suspend(clio::run::u32 point) const {
    Header()->resume_point_ = point;
  }
};

}  // namespace clio::run::gpu

// ---------------------------------------------------------------------------
// Kernel / function macros
// ---------------------------------------------------------------------------

/**
 * Kernel prologue: publish the stack and reset this lane's chain depth.
 * `view` is a YieldableView; `stack` is a YieldStackView.
 */
#define CLIO_YKERNEL_ENTER(view, stack)                                       \
  clio::run::gpu::YieldTlsPublish((stack), (view).Y(), (view).Block());       \
  clio::run::gpu::YieldLane()->cur_depth_ = 0;                                \
  __syncthreads();

/** Attach this call's frame. Must precede every CLIO_YLOCAL. */
#define CLIO_YFRAME() clio::run::gpu::YieldFrame _cy_f

/**
 * Declare a local that survives yields. Must appear BEFORE CLIO_YBEGIN, which
 * is what keeps the switch from ever jumping over it.
 */
#define CLIO_YLOCAL(T, name)                                                  \
  T &name = *reinterpret_cast<T *>(_cy_f.Alloc(sizeof(T), alignof(T)))

/** CLIO_YLOCAL plus a value applied only on the first entry to this call. */
#define CLIO_YLOCAL_INIT(T, name, init)                                       \
  CLIO_YLOCAL(T, name);                                                       \
  if (_cy_f.fresh_) {                                                         \
    name = (init);                                                            \
  }

/** Open the resumable body. */
#define CLIO_YBEGIN()                                                         \
  switch (_cy_f.Header()->resume_point_) {                                    \
    case 0:

/**
 * Suspend the block here. Block-collective: every thread must reach it.
 * One yield per source line (the resume label is __LINE__).
 */
#define CLIO_YIELD_STACK()                                                    \
  do {                                                                        \
    __syncthreads();                                                          \
    _cy_f.Suspend(__LINE__);                                                  \
    if (threadIdx.x == 0) {                                                   \
      clio::run::gpu::YieldTls().block_state_->status_ =                      \
          clio::run::gpu::kYieldSuspended;                                    \
    }                                                                         \
    __threadfence_system();                                                   \
    __syncthreads();                                                          \
    return;                                                                   \
  } while (0);                                                                \
  case __LINE__:                                                              \
    __syncthreads();

/**
 * Yield if ANY thread in the block asks to, otherwise fall straight through.
 *
 * This is the shape a page-fault wait actually has: a fault is a per-thread
 * event, but suspending is a per-block one, so a warp that missed cannot
 * suspend alone and a warp that hit cannot run ahead. Every thread votes with
 * `cond` and __syncthreads_or makes the decision uniform, which is what keeps
 * the yield -- and the __syncthreads inside it -- legal.
 *
 * The resume label sits BEFORE the vote, so coming back RE-EVALUATES `cond`.
 * That turns the construct into a retry loop with the host in the middle of
 * it: suspend, host services the faults, resume, re-check, and fall through
 * once nobody is still waiting. A thread that never faulted simply votes 0
 * every round and rides along.
 *
 * `cond` therefore runs once per entry and must be a pure test -- it is a
 * question about state, not the thing that changes it.
 */
#define CLIO_YIELD_IF(cond)                                                   \
  case __LINE__:                                                              \
    if (__syncthreads_or((cond) ? 1 : 0)) {                                   \
      _cy_f.Suspend(__LINE__);                                                \
      if (threadIdx.x == 0) {                                                 \
        clio::run::gpu::YieldTls().block_state_->status_ =                    \
            clio::run::gpu::kYieldSuspended;                                  \
      }                                                                       \
      __threadfence_system();                                                 \
      __syncthreads();                                                        \
      return;                                                                 \
    }

/**
 * CLIO_YIELD_IF, plus a token telling the HOST what this block is waiting for.
 *
 * Identical suspend semantics; the difference is on the other side. A plain
 * yield leaves the driver with nothing to go on, so it relaunches immediately
 * and the block looks again -- which is fine when the wait is a copy and
 * ruinous when it is a GPU decompress, measured at ~50 relaunches per fault
 * against ~2. `tag` is handed to the driver's ResumeWhen so the host can test
 * readiness cheaply and hold the block back until it is worth resuming.
 *
 * The token is opaque to the machinery: a device address to poll, a page id,
 * a sequence number -- whatever the ResumeWhen on the host understands. It is
 * cleared on resume so a later plain yield is not mistaken for this one.
 */
#define CLIO_YIELD_IF_RESUME_WHEN(cond, tag)                                  \
  case __LINE__:                                                              \
    if (__syncthreads_or((cond) ? 1 : 0)) {                                   \
      _cy_f.Suspend(__LINE__);                                                \
      if (threadIdx.x == 0) {                                                 \
        clio::run::gpu::YieldTls().block_state_->wait_tag_ =                  \
            static_cast<clio::run::u64>(tag);                                 \
        clio::run::gpu::YieldTls().block_state_->status_ =                    \
            clio::run::gpu::kYieldSuspended;                                  \
      }                                                                       \
      __threadfence_system();                                                 \
      __syncthreads();                                                        \
      return;                                                                 \
    }                                                                         \
    if (threadIdx.x == 0) {                                                   \
      clio::run::gpu::YieldTls().block_state_->wait_tag_ = 0;                 \
    }

/**
 * Call another yieldable function, propagating a yield out through this one.
 *
 * The resume label sits BEFORE the call on purpose: coming back means
 * re-entering the callee, which re-attaches to its own still-live frame and
 * jumps to its own resume point. Landing after the call would strand the
 * callee mid-way with nobody to finish it.
 */
#define CLIO_YCALL(call)                                                      \
  case __LINE__:                                                              \
    (call);                                                                   \
    if (clio::run::gpu::YieldSuspended()) {                                   \
      _cy_f.Suspend(__LINE__);                                                \
      return;                                                                 \
    }

/** Close the resumable body and pop the frame (this call is finished). */
#define CLIO_YEND()                                                           \
  }                                                                           \
  __syncthreads();                                                            \
  _cy_f.Pop();


namespace clio::run::gpu {

/** Put every lane header into its start-of-call state. */
__global__ inline void YieldStackInitKernel(YieldStackView v,
                                            clio::run::u32 nlanes) {
  const clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nlanes) {
    return;
  }
  auto *h = reinterpret_cast<YieldLaneHeader *>(
      v.base_ + static_cast<clio::run::u64>(idx) * v.bytes_per_lane_);
  // sp_ starts AFTER the header: the header is not frame space.
  h->sp_ = sizeof(YieldLaneHeader);
  h->live_depth_ = 0;
  h->cur_depth_ = 0;
  h->error_ = kYieldErrNone;
  h->coro_resume_ = 0;
  h->coro_top_ = 0;
  h->coro_park_ = 0;
}

}  // namespace clio::run::gpu

#endif  // __CUDACC__

// NOT guarded out of the device pass. clang compiles a CUDA TU twice and
// PARSES host code in both passes; hiding these host-only classes behind
// !defined(__CUDA_ARCH__) made every test and bench that defines a host
// launch helper fail to compile for the device target ("no member named
// 'Yieldable'"), because the helper's non-dependent names are looked up at
// parse time. The classes remain host-only in USE -- their members call
// host CUDA APIs -- but they must be visible to both passes.

namespace clio::run::gpu {

#if CTP_ENABLE_SYCL
/**
 * Reset a stack's lane headers and publish its YieldSmem base.
 *
 * DEFINED BY THE TU THAT OWNS THE YIELDABLE KERNELS (the one that defines
 * CLIO_SYCL_KERNEL_TU), because both halves need -fsycl: the lane-header
 * initialization is a kernel, and the base pointer goes into a
 * device_global that only that TU's device image contains. A YieldStack
 * itself is constructed in ordinary host code, which can do neither.
 *
 * It is NOT a memset: sp_ starts at sizeof(YieldLaneHeader), because the
 * header is not frame space.
 */
void SyclYieldStackReset(const YieldStackView &view, clio::run::u32 nlanes,
                         char *smem_base);
#endif

/**
 * Host owner of the lane-divided stack.
 *
 * Sized per lane rather than per block because that is the unit that actually
 * holds locals; `bytes_per_lane` bounds the deepest call chain a lane can have
 * live, and running past it is the one failure this design cannot detect
 * cheaply on the device, so pick it with headroom.
 */
class YieldStack {
 public:
  YieldStack(clio::run::u32 nblocks, clio::run::u32 lanes_per_block,
             clio::run::u32 bytes_per_lane)
      : nblocks_(nblocks),
        lanes_per_block_(lanes_per_block),
        bytes_per_lane_(bytes_per_lane) {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    // Through GpuApi, not cuda* directly: this header is part of the yielding
    // machinery every paged kernel depends on, and hard-coding CUDA confines
    // the whole design to one vendor. GpuApi dispatches to hip/sycl on the
    // same call.
    const size_t total =
        static_cast<size_t>(nblocks_) * lanes_per_block_ * bytes_per_lane_;
    d_base_ = ctp::GpuApi::Malloc<char>(total);
    ctp::GpuApi::Memset(d_base_, 0, total);
    // Backing store for the persistent shared arena, one region per LOGICAL
    // block. Zeroed, so bytes_ == 0 and the first entry restores nothing.
    const size_t ptotal =
        static_cast<size_t>(nblocks_) * CLIO_PERSIST_STRIDE;
    d_persist_ = ctp::GpuApi::Malloc<char>(ptotal);
    ctp::GpuApi::Memset(d_persist_, 0, ptotal);
#endif
#if CTP_ENABLE_SYCL
    // CTP_ENABLE_SYCL, not CTP_IS_SYCL_COMPILER: a YieldStack is constructed
    // in ORDINARY host code, which is not compiled with -fsycl. Under the
    // narrower guard d_smem_ stayed null, SyclYieldStackReset published a
    // null base, and the first kernel's YieldTlsPublish wrote through it --
    // memcheck: "Invalid __global__ write of size 8 bytes ... Access to 0x10".
    //
    // SYCL only: the per-block YieldSmem the CUDA build gets from dynamic
    // shared memory. One per PHYSICAL block, so it is sized by the widest
    // grid this stack will ever back -- which is nblocks_, since the driver
    // only ever COMPACTS the grid between rounds.
    d_smem_ = ctp::GpuApi::Malloc<char>(static_cast<size_t>(nblocks_) *
                                        sizeof(YieldSmem));
    ctp::GpuApi::Memset(d_smem_, 0,
                        static_cast<size_t>(nblocks_) * sizeof(YieldSmem));
#endif
    Reset();
  }

  ~YieldStack() {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    if (d_base_ != nullptr) {
      ctp::GpuApi::Free(d_base_);
    }
    if (d_persist_ != nullptr) {
      ctp::GpuApi::Free(d_persist_);
    }
    if (d_smem_ != nullptr) {
      ctp::GpuApi::Free(d_smem_);
    }
#endif
    d_base_ = nullptr;
    d_persist_ = nullptr;
    d_smem_ = nullptr;
  }

  YieldStack(const YieldStack &) = delete;
  YieldStack &operator=(const YieldStack &) = delete;

  YieldStackView View() const {
    YieldStackView v;
    v.base_ = d_base_;
    v.bytes_per_lane_ = bytes_per_lane_;
    v.lanes_per_block_ = lanes_per_block_;
    v.persist_base_ = d_persist_;
    return v;
  }

  void Reset() {
#if CTP_ENABLE_CUDA && defined(__CUDACC__)
    const clio::run::u32 nlanes = nblocks_ * lanes_per_block_;
    const clio::run::u32 threads = 128;
    const clio::run::u32 blocks = (nlanes + threads - 1) / threads;
    YieldStackInitKernel<<<blocks, threads>>>(View(), nlanes);
    ctp::GpuApi::Synchronize();
#elif CTP_ENABLE_SYCL
    // Out of line, into the kernel-owning TU. Both halves of this -- the
    // lane-header init kernel and the device_global publish -- are things
    // only a -fsycl TU can do, and a YieldStack is constructed in ORDINARY
    // host code. See SyclYieldStackReset's declaration above.
    SyclYieldStackReset(View(), nblocks_ * lanes_per_block_, d_smem_);
#endif
  }

 private:
  clio::run::u32 nblocks_;
  clio::run::u32 lanes_per_block_;
  clio::run::u32 bytes_per_lane_;
  char *d_base_ = nullptr;
  char *d_persist_ = nullptr;
  char *d_smem_ = nullptr;
};

#if !CTP_IS_DEVICE_PASS

/**
 * Host runner for a coroutine kernel: the per-lane frame stack plus the
 * relaunch driver, reset together and run to completion.
 *
 *     CoroRunner<> runner(nblocks, nthreads, lane_bytes);
 *     runner.Run([&](dim3 g, dim3 b, YieldableView<> v, YieldStackView s) {
 *       MyKernel<<<g, b>>>(v, s, other_args...);
 *     });
 *
 * The service callback is deliberately empty, and it was worth checking: a
 * suspended block publishes wait_tag_ so the host could wait on what it is
 * blocked on instead of relaunching, but spinning in the service step until
 * parked tokens could have landed cuts rounds without making the kernel any
 * faster -- the relaunches are not the cost.
 */
template <typename StateT = YieldNoState>
class CoroRunner {
 public:
  CoroRunner(clio::run::u32 nblocks, clio::run::u32 nthreads,
             clio::run::u32 lane_bytes)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, lane_bytes) {}

  /** `launch(grid, block, yieldable_view, stack_view)` issues the kernel.
   *  @return rounds taken. */
  template <typename LaunchFn>
  clio::run::u32 Run(LaunchFn &&launch,
                     clio::run::u32 max_rounds = 2000000) {
    drv_.ResetTimers();
    drv_.Reset();
    stack_.Reset();
    const clio::run::u32 rounds = drv_.RunToCompletion(
        [&](dim3 g, dim3 b, YieldableView<StateT> v) {
          launch(g, b, v, stack_.View());
        },
        [] {}, max_rounds,
        [](clio::run::u32, clio::run::u64 tag) -> bool {
          // Honor the wait tag: a parked block relaunches only once the
          // 32-bit completion flag it published (bit 0 = complete) has
          // fired. Relaunching unconditionally costs each parked block
          // its full coroutine-resume price (~125us of GPU time measured)
          // per round just to re-park -- most of a fault-heavy step's
          // kernel time. Tag 0 means "no specific wait"; the driver
          // relaunches those unconditionally.
          unsigned int v = 0;
          ctp::GpuApi::Memcpy(&v, reinterpret_cast<const unsigned int *>(tag),
                              sizeof(v));
          return (v & 1u) != 0u;
        });
    // SAY SO. The cap returning quietly is how a livelocked kernel gets
    // mistaken for a finished one by every caller that only looks at the
    // data it produced.
    if (drv_.HitRoundCap() && !warned_cap_) {
      warned_cap_ = true;
      printf("[yield] driver GAVE UP after %u rounds: blocks are still "
             "parked and their remaining work did NOT run -- results are "
             "incomplete (see CoroRunner::HitRoundCap)\n",
             (unsigned)rounds);
    }
    return rounds;
  }

  /** See Yieldable::HitRoundCap(). A kernel that hit the cap was ABANDONED
   *  with blocks still parked; anything it produced is incomplete. */
  bool HitRoundCap() const { return drv_.HitRoundCap(); }
  double KernelMs() const { return drv_.KernelMs(); }
  double CopyMs() const { return drv_.CopyMs(); }
  double UploadMs() const { return drv_.UploadMs(); }
  const std::vector<std::pair<double, clio::run::u32>> &RoundLog() const {
    return drv_.RoundLog();
  }
  Yieldable<StateT> &Driver() { return drv_; }
  YieldStack &Stack() { return stack_; }

 private:
  Yieldable<StateT> drv_;
  YieldStack stack_;
  bool warned_cap_ = false;
};

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::run::gpu



#endif  // CLIO_RUNTIME_GPU_YIELD_STACK_H_

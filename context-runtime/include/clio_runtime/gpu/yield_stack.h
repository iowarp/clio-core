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
 * only if declared with CLIO_YLOCAL at the top. Deciding that automatically
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
  clio::run::u32 pad_;
  /** Offset of each live frame, indexed by depth. */
  clio::run::u32 frame_off_[kYieldMaxDepth];
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

#if defined(__CUDACC__)
  /** Region for one lane of one LOGICAL block. */
  __device__ char *Lane(clio::run::u32 logical_block,
                        clio::run::u32 lane) const {
    const clio::run::u64 idx =
        static_cast<clio::run::u64>(logical_block) * lanes_per_block_ + lane;
    return base_ + idx * bytes_per_lane_;
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
  clio::run::u32 pad_;
};

}  // namespace clio::run::gpu

/** Dynamic shared memory a yieldable kernel must reserve, in bytes. */
#define CLIO_YIELD_SMEM_BYTES (sizeof(clio::run::gpu::YieldSmem))

#if defined(__CUDACC__)

namespace clio::run::gpu {

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

/** Publish the stack for this block. Called once, by the kernel prologue. */
__device__ __forceinline__ void YieldTlsPublish(const YieldStackView &stack,
                                                YieldBlockState *block_state,
                                                clio::run::u32 logical_block) {
  if (threadIdx.x == 0) {
    YieldSmem &s = YieldTls();
    s.stack_ = stack;
    s.block_state_ = block_state;
    s.logical_block_ = logical_block;
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
}

}  // namespace clio::run::gpu

#endif  // __CUDACC__

#if !defined(__CUDA_ARCH__)

namespace clio::run::gpu {

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
#if CTP_ENABLE_CUDA
    const size_t total =
        static_cast<size_t>(nblocks_) * lanes_per_block_ * bytes_per_lane_;
    cudaMalloc(reinterpret_cast<void **>(&d_base_), total);
    cudaMemset(d_base_, 0, total);
#endif
    Reset();
  }

  ~YieldStack() {
#if CTP_ENABLE_CUDA
    if (d_base_ != nullptr) {
      cudaFree(d_base_);
    }
#endif
    d_base_ = nullptr;
  }

  YieldStack(const YieldStack &) = delete;
  YieldStack &operator=(const YieldStack &) = delete;

  YieldStackView View() const {
    YieldStackView v;
    v.base_ = d_base_;
    v.bytes_per_lane_ = bytes_per_lane_;
    v.lanes_per_block_ = lanes_per_block_;
    return v;
  }

  void Reset() {
#if CTP_ENABLE_CUDA && defined(__CUDACC__)
    const clio::run::u32 nlanes = nblocks_ * lanes_per_block_;
    const clio::run::u32 threads = 128;
    const clio::run::u32 blocks = (nlanes + threads - 1) / threads;
    YieldStackInitKernel<<<blocks, threads>>>(View(), nlanes);
    cudaDeviceSynchronize();
#endif
  }

 private:
  clio::run::u32 nblocks_;
  clio::run::u32 lanes_per_block_;
  clio::run::u32 bytes_per_lane_;
  char *d_base_ = nullptr;
};

}  // namespace clio::run::gpu

#endif  // !__CUDA_ARCH__


#endif  // CLIO_RUNTIME_GPU_YIELD_STACK_H_

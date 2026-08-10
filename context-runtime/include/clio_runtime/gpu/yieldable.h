/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Cooperative (yieldable) CUDA kernels: a block can suspend itself, let the
 * kernel EXIT, and be resumed later from where it stopped.
 *
 * WHY THIS EXISTS
 *
 * A kernel that waits in-place for the host blocks the whole context: an
 * indefinitely-resident kernel prevents every LATER kernel launch from running
 * on that device, even though copy engines keep working. That is why a
 * gpu_vector page fault can be serviced by a memcpy but not by anything that
 * needs a kernel -- a GPU codec entered on the fault path waits for a device
 * that is waiting for it. Spinning is also expensive even when it works: the
 * block holds its SM slot for the entire round trip.
 *
 * Yielding inverts that. The block records where it was, returns, and the
 * kernel ends. The host is then free to do ANYTHING, including launching
 * kernels, and relaunches only the blocks that had not finished.
 *
 * HOW IT WORKS
 *
 * The classic stackless-coroutine trick (protothreads / Duff's device): the
 * kernel body is wrapped in a `switch` on a saved resume point, and every
 * yield expands to "record __LINE__, return, and a `case __LINE__:` label to
 * come back to". No compiler support is needed, which matters because CUDA
 * does NOT allow C++20 coroutines in device code -- co_await / co_yield /
 * co_return in a device function are compile errors -- so this is the only
 * mechanism available.
 *
 * THE ONE RULE: LOCALS DO NOT SURVIVE A YIELD
 *
 * The `switch` jumps back into the middle of the function; automatic variables
 * are NOT restored, and C++ forbids jumping over an initialization, so
 * anything live across a yield must live in the block's state struct instead:
 *
 *     struct MyState { clio::run::u64 i; };            // survives
 *     ...
 *     for (st->i = 0; st->i < 10; ++st->i) {           // NOT `u64 i = 0`
 *       CLIO_YIELD(y);
 *     }
 *
 * Getting this wrong is a COMPILE error ("jump to case label crosses
 * initialization"), not silent corruption, which is the main reason to prefer
 * this shape over anything cleverer. Automatic capture of the whole live stack
 * needs a source-to-source pass over the AST (what FLEP, ASPLOS'17, built);
 * such a pass could target this same runtime later.
 *
 * YIELDS ARE BLOCK-COLLECTIVE
 *
 * A yield suspends the BLOCK, so every thread in the block must reach the same
 * yield: it contains __syncthreads(), and the resume point is per block, not
 * per thread. Never put a yield under `if (threadIdx.x == 0)`.
 */
#ifndef CLIO_RUNTIME_GPU_YIELDABLE_H_
#define CLIO_RUNTIME_GPU_YIELDABLE_H_

#include <clio_runtime/types.h>

#include <vector>

namespace clio::run::gpu {

/** What a block was doing when its kernel returned. */
enum YieldStatus : clio::run::u32 {
  /** Ran to CLIO_YIELDABLE_END, or returned without yielding. Not relaunched.
   *  This is the value stamped at entry, so any early `return` counts as done
   *  and a kernel cannot strand the driver in an infinite resume loop. */
  kYieldDone = 0,
  /** Suspended at a yield point; the driver will relaunch this block. */
  kYieldSuspended = 1,
};

/**
 * Per-block continuation. Deliberately tiny and separate from user state: the
 * driver copies THIS array to the host every round to decide what to relaunch,
 * and user state (which can be large) never has to leave the device.
 */
struct YieldBlockState {
  /** 0 = start from the top; otherwise the __LINE__ of the yield to resume. */
  clio::run::u32 resume_point_;
  clio::run::u32 status_;
};

/** State type for kernels that need no state of their own. */
struct YieldNoState {};

/**
 * Device-side handle passed to a yieldable kernel by value.
 *
 * `Block()` is the LOGICAL block id, which is not blockIdx.x: the driver
 * relaunches only unfinished blocks, so on the second round blockIdx.x 0 may
 * be logical block 7. A yieldable kernel must use Block() everywhere it would
 * have used blockIdx.x -- including for anything that partitions state by
 * block, such as a gpu_vector page table.
 */
template <typename StateT = YieldNoState>
struct YieldableView {
  YieldBlockState *yield_ = nullptr;
  StateT *user_ = nullptr;
  /** pending_[blockIdx.x] -> logical block id. */
  const clio::run::u32 *pending_ = nullptr;
  clio::run::u32 num_pending_ = 0;
  clio::run::u32 num_blocks_ = 0;

#if defined(__CUDACC__)
  __device__ clio::run::u32 Block() const { return pending_[blockIdx.x]; }
  __device__ YieldBlockState *Y() const { return &yield_[Block()]; }
  __device__ StateT *State() const { return &user_[Block()]; }
#endif
};

}  // namespace clio::run::gpu

// ---------------------------------------------------------------------------
// Device macros
// ---------------------------------------------------------------------------

/**
 * Open a yieldable kernel body. Must be the outermost statement of the kernel,
 * and must be paired with CLIO_YIELDABLE_END.
 *
 * Stamps kYieldDone up front so that returning early -- including any path the
 * author did not think about -- ends the block rather than leaving the driver
 * relaunching it forever.
 */
#define CLIO_YIELDABLE_BEGIN(view)                                            \
  auto *_cy_y = (view).Y();                                                   \
  if (threadIdx.x == 0) {                                                     \
    _cy_y->status_ = clio::run::gpu::kYieldDone;                              \
  }                                                                           \
  __syncthreads();                                                            \
  switch (_cy_y->resume_point_) {                                             \
    case 0:

/**
 * Suspend the block here and end the kernel; the next Resume() re-enters at
 * this point.
 *
 * BLOCK-COLLECTIVE: every thread must reach this. One yield per source line
 * (the resume label is __LINE__).
 */
#define CLIO_YIELD(view)                                                      \
  do {                                                                        \
    __syncthreads();                                                          \
    if (threadIdx.x == 0) {                                                   \
      _cy_y->resume_point_ = __LINE__;                                        \
      _cy_y->status_ = clio::run::gpu::kYieldSuspended;                       \
    }                                                                         \
    /* The host reads this state after the launch completes, so the store is  \
     * ordered by kernel completion rather than by this fence; the fence is   \
     * for a host polling the state of a STILL-RUNNING grid. */               \
    __threadfence_system();                                                   \
    __syncthreads();                                                          \
    return;                                                                   \
  } while (0);                                                                \
  case __LINE__:                                                              \
    __syncthreads();

/** Close a yieldable kernel body. Falling out of the switch means done. */
#define CLIO_YIELDABLE_END()                                                  \
  }                                                                           \
  __syncthreads();                                                            \
  if (threadIdx.x == 0) {                                                     \
    _cy_y->status_ = clio::run::gpu::kYieldDone;                              \
  }

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
#if !defined(__CUDA_ARCH__)

namespace clio::run::gpu {

/**
 * Host driver for a set of yieldable blocks.
 *
 * Owns the per-block continuation and user state, and relaunches only the
 * blocks that suspended. Each round launches ONE grid sized to the number of
 * pending blocks, with a pending-id table mapping blockIdx.x to a logical
 * block -- rather than one single-block launch per pending block, which costs
 * a launch per block per round and buys nothing.
 *
 * The launch itself is a caller-supplied callable so that kernel arguments
 * stay ordinary:
 *
 *     Yieldable<MyState> y(nblocks, nthreads);
 *     y.RunToCompletion(
 *         [&](dim3 grid, dim3 block, YieldableView<MyState> v) {
 *           MyKernel<<<grid, block>>>(v, other_args...);
 *         },
 *         [&]() { ServiceWhateverTheBlocksAreWaitingFor(); });
 */
template <typename StateT = YieldNoState>
class Yieldable {
 public:
  Yieldable(clio::run::u32 nblocks, clio::run::u32 nthreads)
      : nblocks_(nblocks), nthreads_(nthreads) {
    host_yield_.resize(nblocks_);
    host_pending_.resize(nblocks_);
    Alloc();
    Reset();
  }

  ~Yieldable() { Free(); }

  Yieldable(const Yieldable &) = delete;
  Yieldable &operator=(const Yieldable &) = delete;

  clio::run::u32 NumBlocks() const { return nblocks_; }
  clio::run::u32 NumThreads() const { return nthreads_; }
  /** Blocks that have not finished. */
  clio::run::u32 NumPending() const { return num_pending_; }
  /** Device user state, for host-side setup before the first round. */
  StateT *DeviceState() const { return d_user_; }

  /** Rewind every block to its entry point and mark them all pending. */
  void Reset() {
    for (clio::run::u32 i = 0; i < nblocks_; ++i) {
      host_yield_[i].resume_point_ = 0;
      host_yield_[i].status_ = kYieldSuspended;
      host_pending_[i] = i;
    }
    num_pending_ = nblocks_;
    Upload();
  }

  /**
   * Run one round: launch the pending blocks, wait, and recompute the pending
   * set. Returns true if any block is still unfinished.
   */
  template <typename LaunchFn>
  bool Round(LaunchFn &&launch) {
    if (num_pending_ == 0) {
      return false;
    }
    launch(dim3(num_pending_), dim3(nthreads_), View());
#if CTP_ENABLE_CUDA
    if (cudaDeviceSynchronize() != cudaSuccess) {
      return false;
    }
    if (cudaMemcpy(host_yield_.data(), d_yield_,
                   nblocks_ * sizeof(YieldBlockState),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
#endif
    // Compact the still-suspended blocks. Order is preserved so that a block's
    // work stays as sequential as the caller wrote it.
    clio::run::u32 n = 0;
    for (clio::run::u32 i = 0; i < nblocks_; ++i) {
      if (host_yield_[i].status_ == kYieldSuspended) {
        host_pending_[n++] = i;
      }
    }
    num_pending_ = n;
    UploadPending();
    return num_pending_ > 0;
  }

  /**
   * Round() in a loop, calling `service` between rounds -- that gap is the
   * whole point: it is host time with NO kernel resident, so it may launch
   * kernels of its own (a GPU codec, a copy, another vector's work).
   *
   * `max_rounds` bounds a kernel that yields forever; 0 means unbounded.
   * Returns the number of rounds executed.
   */
  template <typename LaunchFn, typename ServiceFn>
  clio::run::u32 RunToCompletion(LaunchFn &&launch, ServiceFn &&service,
                                 clio::run::u32 max_rounds = 0) {
    clio::run::u32 rounds = 0;
    while (Round(launch)) {
      ++rounds;
      if (max_rounds != 0 && rounds >= max_rounds) {
        break;
      }
      service();
    }
    if (num_pending_ == 0) {
      ++rounds;  // count the round that finished the last block
    }
    return rounds;
  }

  YieldableView<StateT> View() const {
    YieldableView<StateT> v;
    v.yield_ = d_yield_;
    v.user_ = d_user_;
    v.pending_ = d_pending_;
    v.num_pending_ = num_pending_;
    v.num_blocks_ = nblocks_;
    return v;
  }

 private:
  void Alloc() {
#if CTP_ENABLE_CUDA
    cudaMalloc(reinterpret_cast<void **>(&d_yield_),
               nblocks_ * sizeof(YieldBlockState));
    cudaMalloc(reinterpret_cast<void **>(&d_user_), nblocks_ * sizeof(StateT));
    cudaMemset(d_user_, 0, nblocks_ * sizeof(StateT));
    cudaMalloc(reinterpret_cast<void **>(&d_pending_),
               nblocks_ * sizeof(clio::run::u32));
#endif
  }

  void Free() {
#if CTP_ENABLE_CUDA
    if (d_yield_ != nullptr) cudaFree(d_yield_);
    if (d_user_ != nullptr) cudaFree(d_user_);
    if (d_pending_ != nullptr) cudaFree(d_pending_);
#endif
    d_yield_ = nullptr;
    d_user_ = nullptr;
    d_pending_ = nullptr;
  }

  void Upload() {
#if CTP_ENABLE_CUDA
    cudaMemcpy(d_yield_, host_yield_.data(),
               nblocks_ * sizeof(YieldBlockState), cudaMemcpyHostToDevice);
#endif
    UploadPending();
  }

  void UploadPending() {
#if CTP_ENABLE_CUDA
    if (num_pending_ > 0) {
      cudaMemcpy(d_pending_, host_pending_.data(),
                 num_pending_ * sizeof(clio::run::u32), cudaMemcpyHostToDevice);
    }
#endif
  }

  clio::run::u32 nblocks_ = 0;
  clio::run::u32 nthreads_ = 0;
  clio::run::u32 num_pending_ = 0;
  YieldBlockState *d_yield_ = nullptr;
  StateT *d_user_ = nullptr;
  clio::run::u32 *d_pending_ = nullptr;
  std::vector<YieldBlockState> host_yield_;
  std::vector<clio::run::u32> host_pending_;
};

}  // namespace clio::run::gpu

#endif  // !__CUDA_ARCH__

#endif  // CLIO_RUNTIME_GPU_YIELDABLE_H_

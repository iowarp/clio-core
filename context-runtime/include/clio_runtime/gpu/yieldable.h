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
 * DO NOT COUNT ON THE COMPILER TO CATCH THIS. Host g++ does reject the shape
 * ("jump to case label crosses initialization"), but nvcc's DEVICE pass was
 * measured accepting an un-hoisted, initialized local across a yield with no
 * error and no warning. The variable is then simply not re-initialized on
 * resume and holds whatever the slot happens to contain -- silent, and exactly
 * the failure this comment used to claim was impossible.
 *
 * Automatic capture of the whole live stack needs a source-to-source pass over
 * the AST (what FLEP, ASPLOS'17, built); such a pass could target this same
 * runtime later, and would also be the natural place to make the un-hoisted
 * case diagnosable.
 *
 * YIELDS ARE BLOCK-COLLECTIVE
 *
 * A yield suspends the BLOCK, so every thread in the block must reach the same
 * yield: it contains __syncthreads(), and the resume point is per block, not
 * per thread. Never put a yield under `if (threadIdx.x == 0)`.
 */
#ifndef CLIO_RUNTIME_GPU_YIELDABLE_H_
#define CLIO_RUNTIME_GPU_YIELDABLE_H_

#include <cstdio>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/types.h>

#include <type_traits>
#include <chrono>
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
  /**
   * What this block is waiting FOR, for the host to interpret.
   *
   * A plain yield tells the driver only "not done", so the driver can do
   * nothing but relaunch and let the block look again. When the thing being
   * waited on takes much longer than a launch -- a GPU decompress, say -- that
   * turns into polling by kernel launch: measured at ~50 relaunches per page
   * fault against ~2 for a plain copy. This field carries a token the HOST can
   * test cheaply, so it can wait instead of relaunching.
   *
   * Opaque here on purpose: the driver never interprets it, it just hands it
   * back to the caller's ResumeWhen. 0 means "no token, relaunch freely".
   */
  clio::run::u64 wait_tag_;
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
// NOT guarded out of the device pass. clang compiles a CUDA TU twice and
// PARSES host code in both passes; hiding these host-only classes behind
// !defined(__CUDA_ARCH__) made every test and bench that defines a host
// launch helper fail to compile for the device target ("no member named
// 'Yieldable'"), because the helper's non-dependent names are looked up at
// parse time. The classes remain host-only in USE -- their members call
// host CUDA APIs -- but they must be visible to both passes.

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
  /** Host copy of one block's state (as of the last completed round). */
  const YieldBlockState &BlockState(clio::run::u32 b) const {
    return host_yield_[b];
  }
  clio::run::u32 PendingBlock(clio::run::u32 i) const {
    return host_pending_[i];
  }
  /** Device user state, for host-side setup before the first round. */
  StateT *DeviceState() const { return d_user_; }

  /** Rewind every block to its entry point and mark them all pending. */
  void Reset() {
    for (clio::run::u32 i = 0; i < nblocks_; ++i) {
      host_yield_[i].resume_point_ = 0;
      host_yield_[i].status_ = kYieldSuspended;
      host_yield_[i].wait_tag_ = 0;
      host_pending_[i] = i;
    }
    num_pending_ = nblocks_;
    // Restore the per-block user state too. The constructor zeroes d_user_ and
    // Reset did not, so a REUSED driver started its next run with whatever the
    // previous one left there -- Reset has to leave the object in the state
    // construction leaves it in, or it is not a reset.
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    if (d_user_ != nullptr) {
      ctp::GpuApi::Memset(d_user_, 0, nblocks_ * sizeof(StateT));
    }
#endif
    Upload();
  }

  /**
   * Run one round: launch the pending blocks, wait, and recompute the pending
   * set. Returns true if any block is still unfinished.
   */
  /**
   * Round() with a host-side readiness test.
   *
   * `resume_when(logical_block, wait_tag)` returns whether that block is worth
   * relaunching. Blocks that are not ready stay pending and cost nothing; if
   * NO block is ready the grid is not launched at all and the caller's service
   * step runs again. That is the difference between waiting for a decompress
   * and polling for it with kernel launches.
   */
  template <typename LaunchFn, typename ResumeWhenFn>
  bool Round(LaunchFn &&launch, ResumeWhenFn &&resume_when) {
    if (num_pending_ == 0) {
      return false;
    }
    // Keep only the blocks whose wait has actually been satisfied.
    clio::run::u32 ready = 0;
    for (clio::run::u32 i = 0; i < num_pending_; ++i) {
      const clio::run::u32 b = host_pending_[i];
      if (host_yield_[b].wait_tag_ == 0 ||
          resume_when(b, host_yield_[b].wait_tag_)) {
        host_pending_[ready++] = b;
      }
    }
    if (ready == 0) {
      return true;              // still pending, just nothing worth launching
    }
    const clio::run::u32 saved = num_pending_;
    num_pending_ = ready;
    UploadPending();
    const bool more = Round(std::forward<LaunchFn>(launch));
    // Blocks held back this round are still pending.
    if (ready < saved) {
      RestorePendingFromState();
      return true;
    }
    return more;
  }

  /** Where a round's time goes. Split because "the driver is slow" was not
   *  actionable: launch+sync, the D2H state copy, and the H2D pending upload
   *  are three different problems with three different fixes. */
  double t_kernel_ms_ = 0.0;
  double t_copy_ms_ = 0.0;
  double t_upload_ms_ = 0.0;
  /** Per-round (gpu_ms, blocks launched). Aggregate round counts told us
   *  kernel time tracks rounds at a near-constant cost each, which is only
   *  actionable once we know whether a round is expensive because it does
   *  work or expensive regardless of how few blocks are left in it. */
  std::vector<std::pair<double, clio::run::u32>> round_log_;

 public:
  double KernelMs() const { return t_kernel_ms_; }
  double CopyMs() const { return t_copy_ms_; }
  double UploadMs() const { return t_upload_ms_; }
  void ResetTimers() {
    t_kernel_ms_ = t_copy_ms_ = t_upload_ms_ = 0.0;
    round_log_.clear();
  }
  const std::vector<std::pair<double, clio::run::u32>> &RoundLog() const {
    return round_log_;
  }

  template <typename LaunchFn>
  bool Round(LaunchFn &&launch) {
    if (num_pending_ == 0) {
      return false;
    }
    const auto _t0 = std::chrono::steady_clock::now();
    const clio::run::u32 launched_this_round = num_pending_;
    launch(dim3(num_pending_), dim3(nthreads_), View());
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    // GpuApi checks its own errors and aborts on failure, so the previous
    // "return false on a bad status" branches no longer exist. That is a
    // behaviour change worth naming: a launch failure used to end the round
    // quietly and look like completion.
    // Wait for THIS KERNEL, not for the whole device. cudaDeviceSynchronize
    // blocks until every stream drains, and the Clio runtime is concurrently
    // servicing page faults on its own (non-blocking) streams -- so every
    // round waited for unrelated page transfers to finish before the host
    // could even look at which blocks had suspended.
    //
    // That is the difference between a round costing tens of microseconds and
    // costing the better part of a millisecond, and rounds are the whole cost
    // of this driver: measured 30 rounds per step at 13,500 atoms, 0.77 ms
    // each, for a step whose arithmetic a GPU finishes in well under 1 ms.
    //
    // The kernels launch on the legacy default stream, so that is what has to
    // be waited on. Clio's streams are created cudaStreamNonBlocking, so they
    // do not implicitly join it and are correctly excluded here.
    ctp::GpuApi::Synchronize(/*stream=*/nullptr);
    const auto _t1 = std::chrono::steady_clock::now();
    ctp::GpuApi::Memcpy(host_yield_.data(), d_yield_,
                        nblocks_ * sizeof(YieldBlockState));
    const auto _t2 = std::chrono::steady_clock::now();
    const double _rms =
        std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    round_log_.emplace_back(_rms, launched_this_round);
    t_kernel_ms_ += _rms;
    t_copy_ms_ += std::chrono::duration<double, std::milli>(_t2 - _t1).count();
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
    const auto _t3 = std::chrono::steady_clock::now();
    UploadPending();
    t_upload_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - _t3).count();
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
  /**
   * `service` may return void, or bool to ABORT: returning false stops the
   * loop immediately.
   *
   * Without that, the only way out of a kernel that can never finish is
   * max_rounds, which is a guess -- and a guess that has to be huge to avoid
   * killing a slow-but-healthy run, so a genuine dead end burns minutes
   * before it reports anything. The caller usually knows the difference
   * (gpu_vector: a writeback that FAILED means no slot can ever be cleaned,
   * so the fault path will spin forever), and this lets it say so between
   * rounds, which is exactly when it can read device state safely.
   */
  template <typename LaunchFn, typename ServiceFn>
  clio::run::u32 RunToCompletion(LaunchFn &&launch, ServiceFn &&service,
                                 clio::run::u32 max_rounds = 0) {
    clio::run::u32 rounds = 0;
    while (Round(launch)) {
      ++rounds;
      if (max_rounds != 0 && rounds >= max_rounds) {
        hit_round_cap_ = true;
        // LOUD ON PURPOSE. Hitting the cap means blocks kept parking and
        // nothing ever satisfied them -- a livelock. It is not a slow run.
        // Silently returning here let a gpu_vector test report CORRECT DATA
        // while spinning 200,000 rounds, because the pages that happened to
        // be resident still read back fine; the run looked like a pass. Any
        // caller that reaches this must treat it as a failure, and now cannot
        // miss it.
        std::fprintf(stderr,
                     "[yieldable] ROUND CAP HIT after %u rounds with %u block(s) "
                     "still suspended -- faults are not being satisfied. This is "
                     "a livelock, not a slow run.\n",
                     rounds, num_pending_);
        std::fflush(stderr);
        break;
      }
      if constexpr (std::is_same_v<decltype(service()), bool>) {
        if (!service()) {
          aborted_ = true;
          return rounds;
        }
      } else {
        service();
      }
    }
    if (num_pending_ == 0) {
      ++rounds;  // count the round that finished the last block
    }
    return rounds;
  }

  /** True when a service callback returned false and stopped the loop. Lets
   *  the caller tell "finished" from "gave up" without inspecting rounds. */
  bool Aborted() const { return aborted_; }

  /**
   * True when a run stopped because it hit `max_rounds` -- i.e. blocks were
   * STILL PARKED and their remaining work never ran.
   *
   * This has to be askable, because the cap is otherwise INVISIBLE:
   * RunToCompletion returns normally either way, so a livelocked kernel is
   * handed back looking exactly like a finished one and the caller computes
   * on a partially executed kernel. That is not hypothetical -- an
   * out-of-core MD step came back having run 17 of its 22 page iterations,
   * with no error anywhere, and the wrong answer looked plausible.
   * `Aborted()` does NOT cover this; it only reports the service-callback
   * path. STICKY: set once, never cleared by Reset(), so a caller can check
   * it after a whole phase rather than after every round.
   */
  bool HitRoundCap() const { return hit_round_cap_; }

  /** Recompute the pending set from device state (used after a partial round). */
  void RestorePendingFromState() {
    clio::run::u32 n = 0;
    for (clio::run::u32 i = 0; i < nblocks_; ++i) {
      if (host_yield_[i].status_ == kYieldSuspended) {
        host_pending_[n++] = i;
      }
    }
    num_pending_ = n;
    UploadPending();
  }

  /** RunToCompletion with a host-side readiness test; see Round(). */
  template <typename LaunchFn, typename ServiceFn, typename ResumeWhenFn>
  clio::run::u32 RunToCompletion(LaunchFn &&launch, ServiceFn &&service,
                                 clio::run::u32 max_rounds,
                                 ResumeWhenFn &&resume_when) {
    clio::run::u32 rounds = 0;
    while (Round(launch, resume_when)) {
      ++rounds;
      if (max_rounds != 0 && rounds >= max_rounds) {
        hit_round_cap_ = true;   // see HitRoundCap(): never silent again
        std::fprintf(stderr,
                     "[yieldable] ROUND CAP HIT after %u rounds with %u block(s) "
                     "still suspended -- faults are not being satisfied. This is "
                     "a livelock, not a slow run.\n",
                     rounds, num_pending_);
        std::fflush(stderr);
        break;
      }
      service();
    }
    if (num_pending_ == 0) {
      ++rounds;
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
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    // GpuApi rather than cuda* directly -- see the note in yield_stack.h.
    d_yield_ = ctp::GpuApi::Malloc<YieldBlockState>(
        nblocks_ * sizeof(YieldBlockState));
    d_user_ = ctp::GpuApi::Malloc<StateT>(nblocks_ * sizeof(StateT));
    ctp::GpuApi::Memset(d_user_, 0, nblocks_ * sizeof(StateT));
    d_pending_ = ctp::GpuApi::Malloc<clio::run::u32>(
        nblocks_ * sizeof(clio::run::u32));
#endif
  }

  void Free() {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    if (d_yield_ != nullptr) ctp::GpuApi::Free(d_yield_);
    if (d_user_ != nullptr) ctp::GpuApi::Free(d_user_);
    if (d_pending_ != nullptr) ctp::GpuApi::Free(d_pending_);
#endif
    d_yield_ = nullptr;
    d_user_ = nullptr;
    d_pending_ = nullptr;
  }

  void Upload() {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    ctp::GpuApi::Memcpy(d_yield_, host_yield_.data(),
                        nblocks_ * sizeof(YieldBlockState));
#endif
    UploadPending();
  }

  void UploadPending() {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    if (num_pending_ > 0) {
      ctp::GpuApi::Memcpy(d_pending_, host_pending_.data(),
                          num_pending_ * sizeof(clio::run::u32));
    }
#endif
  }

  clio::run::u32 nblocks_ = 0;
  clio::run::u32 nthreads_ = 0;
  clio::run::u32 num_pending_ = 0;
  /** Set when a service callback returned false; see RunToCompletion. */
  bool aborted_ = false;
  /** See HitRoundCap(). Sticky across Reset(). */
  bool hit_round_cap_ = false;
  YieldBlockState *d_yield_ = nullptr;
  StateT *d_user_ = nullptr;
  clio::run::u32 *d_pending_ = nullptr;
  std::vector<YieldBlockState> host_yield_;
  std::vector<clio::run::u32> host_pending_;
};

}  // namespace clio::run::gpu


#endif  // CLIO_RUNTIME_GPU_YIELDABLE_H_

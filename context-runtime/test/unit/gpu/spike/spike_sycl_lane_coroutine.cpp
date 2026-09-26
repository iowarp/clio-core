/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE (not built by default): the FULL yield_coro.h shape, in SYCL.
 *
 * spike_sycl_coroutine.cpp proved the primitive works (a coroutine frame in
 * our memory, resumed across kernel launches). This spike asks whether the
 * things the real machinery does on top of it also work under SYCL:
 *
 *   1. PER-LANE chains. One coroutine per work-item, frames bump-allocated
 *      from that work-item's own slice of a USM arena.
 *   2. BLOCK-COLLECTIVE suspend. Every suspend site is a vote --
 *      __syncthreads_or in CUDA, sycl::any_of_group here -- so the whole
 *      work-group suspends together or none of it does. Each lane's
 *      condition is different (lane L waits until pass >= L), so the vote is
 *      load-bearing: without it the group would diverge at the barrier.
 *   3. The TRAMPOLINE. A nested yieldable call is a child coroutine; control
 *      is handed over by recording "who runs next" in the lane header and
 *      RETURNING to the driver loop, never by resuming from inside the
 *      awaiter. This is the thing NVPTX forced on the CUDA implementation
 *      (no tail calls => a device stack frame per handoff), and it is what
 *      makes an arbitrary chain depth cost one resume of stack.
 *   4. A VALUE-RETURNING child (YCoroTaskT), including the `V &&`
 *      return_value signature that the CUDA port needs to stop a
 *      double-destroy of an RAII operand.
 *   5. A group BARRIER INSIDE a coroutine, crossed on every pass.
 *
 * Build (NVPTX; spir64 crashes DPC++ codegen -- see the sibling spike):
 *   clang++ -fsycl -std=c++20 -fsycl-targets=nvptx64-nvidia-cuda \
 *           --cuda-path=/usr/local/cuda spike_sycl_lane_coroutine.cpp -o s
 */
#include <coroutine>
#include <cstdio>

#include <sycl/sycl.hpp>

namespace {

constexpr int kLanes = 32;      // work-items per group
constexpr int kGroups = 4;
constexpr size_t kLaneBytes = 4096;

/** The per-lane header + frame arena, exactly the YieldLaneHeader shape:
 *  a bump cursor, the frame to resume next, and a park flag the driver's
 *  loop reads. Lives in USM device memory, so it survives kernel exit. */
struct LaneHeader {
  unsigned long sp_;        // bump cursor, bytes from `this`
  void *resume_;            // deepest suspended frame
  void *top_;               // the lane's YCoroMain frame
  unsigned park_;           // set by a suspend: the driver loop must stop
  unsigned pad_;
};

inline LaneHeader *Lane(char *arena, int gid, int lid) {
  return reinterpret_cast<LaneHeader *>(
      arena + (static_cast<size_t>(gid) * kLanes + lid) * kLaneBytes);
}

inline void *LaneAlloc(LaneHeader *l, size_t n) {
  const unsigned long off = (l->sp_ + 15u) & ~15ul;
  const unsigned long end = off + ((n + 15) & ~15ul);
  if (end > kLaneBytes) return nullptr;
  l->sp_ = end;
  return reinterpret_cast<char *>(l) + off;
}

inline void LaneFree(LaneHeader *l, void *p) {
  l->sp_ = static_cast<unsigned long>(reinterpret_cast<char *>(p) -
                                      reinterpret_cast<char *>(l));
}

/** Everything a coroutine needs that CUDA gets from implicit globals: the
 *  lane header (for the arena and the trampoline slot) and the group (for
 *  the barrier and the vote). SYCL has no threadIdx, so this rides along as
 *  the first parameter of every yieldable function -- which is also what
 *  puts it in front of promise_type::operator new. */
struct Ctx {
  LaneHeader *lane_;
  sycl::group<1> grp_;
};

/* --------------------------- coroutine types --------------------------- */

/** The lane's top-level coroutine (YCoroMain). */
struct CoMain {
  struct promise_type {
    LaneHeader *lane_;
    template <typename... A>
    static void *operator new(size_t n, Ctx c, A &&...) {
      return LaneAlloc(c.lane_, n);
    }
    static void operator delete(void *, size_t) {}
    template <typename... A>
    promise_type(Ctx c, A &&...) : lane_(c.lane_) {}
    CoMain get_return_object() {
      return CoMain{
          std::coroutine_handle<promise_type>::from_promise(*this).address()};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() {}
  };
  void *frame_;
};

/** A value-returning child (YCoroTaskT<V>): `V v = co_await Child(...)`. */
template <typename V>
struct CoTask {
  struct promise_type {
    LaneHeader *lane_;
    void *parent_ = nullptr;
    V value_{};
    template <typename... A>
    static void *operator new(size_t n, Ctx c, A &&...) {
      return LaneAlloc(c.lane_, n);
    }
    static void operator delete(void *p, size_t) {}
    template <typename... A>
    promise_type(Ctx c, A &&...) : lane_(c.lane_) {}
    CoTask get_return_object() {
      return CoTask{
          std::coroutine_handle<promise_type>::from_promise(*this).address()};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    /** Trampoline home: record the parent as "who runs next" and return to
     *  the driver loop. NOT a symmetric transfer. */
    struct FinalXfer {
      bool await_ready() const noexcept { return false; }
      void await_suspend(std::coroutine_handle<promise_type> h) noexcept {
        auto &p = h.promise();
        p.lane_->resume_ = p.parent_;
      }
      void await_resume() const noexcept {}
    };
    FinalXfer final_suspend() noexcept { return {}; }
    // BY RVALUE REFERENCE -- see yield_coro.h: a by-value parameter here
    // gets bitwise-copied out of the frame with no move ctor, so an RAII
    // operand ends up with two live owners.
    void return_value(V &&v) noexcept { value_ = static_cast<V &&>(v); }
    void unhandled_exception() {}
  };

  void *frame_;

  bool await_ready() const noexcept { return false; }
  template <typename P>
  void await_suspend(std::coroutine_handle<P> parent) noexcept {
    auto h = std::coroutine_handle<promise_type>::from_address(frame_);
    h.promise().parent_ = parent.address();
    // Trampoline: queue the child, return to the driver.
    h.promise().lane_->resume_ = frame_;
  }
  V await_resume() noexcept {
    auto h = std::coroutine_handle<promise_type>::from_address(frame_);
    V v = static_cast<V &&>(h.promise().value_);
    LaneHeader *l = h.promise().lane_;
    h.destroy();
    LaneFree(l, frame_);
    return v;
  }
};

/** The suspend point (YCoroSuspend): park the lane and tell the driver. */
struct Park {
  LaneHeader *lane_;
  bool await_ready() const noexcept { return false; }
  template <typename P>
  void await_suspend(std::coroutine_handle<P> h) noexcept {
    lane_->resume_ = h.address();
    lane_->park_ = 1u;
  }
  void await_resume() const noexcept {}
};

/** CLIO_CO_YIELD_WHEN: block-collective conditional yield. Every work-item
 *  votes; the group suspends if ANY is still waiting; a resume RE-EVALUATES,
 *  so this is a retry loop with the host in the middle. */
#define SYCL_CO_YIELD_WHEN(ctx, cond)                          \
  for (;;) {                                                   \
    sycl::group_barrier((ctx).grp_);                           \
    if (!sycl::any_of_group((ctx).grp_, (cond) ? true : false)) \
      break;                                                   \
    co_await Park{(ctx).lane_};                                \
  }

/* ------------------------------ the workload --------------------------- */

/** A nested yieldable call that RETURNS A VALUE. Suspends inside itself, so
 *  a resume must land in the CHILD, not the parent. */
CoTask<int> Child(Ctx ctx, int lane_id, const int *pass, int round) {
  // Wait until the host has run enough passes for this lane. Ordinary
  // locals across the suspend: `lane_id`, `round`, and the loop state.
  SYCL_CO_YIELD_WHEN(ctx, *pass < lane_id + round);
  int acc = 0;
  for (int i = 0; i <= round; ++i) {
    acc += i;                     // live across the barrier below
    sycl::group_barrier(ctx.grp_);  // a barrier INSIDE a coroutine
  }
  co_return acc * 100 + lane_id;
}

/** The lane's top-level chain: two nested rounds, each ending in a child. */
CoMain Walk(Ctx ctx, int lane_id, const int *pass, int *out) {
  int total = 0;
  for (int round = 0; round < 2; ++round) {
    // A child coroutine, awaited: trampoline down, trampoline back.
    total += co_await Child(ctx, lane_id, pass, round);
  }
  out[lane_id] = total;
  co_return;
}

}  // namespace

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  const size_t arena_bytes = static_cast<size_t>(kGroups) * kLanes * kLaneBytes;
  auto *arena = static_cast<char *>(sycl::malloc_device(arena_bytes, q));
  auto *out = sycl::malloc_shared<int>(kGroups * kLanes, q);
  auto *pass = sycl::malloc_shared<int>(1, q);
  auto *alive = sycl::malloc_shared<int>(1, q);
  for (int i = 0; i < kGroups * kLanes; ++i) out[i] = -1;
  *pass = 0;

  q.memset(arena, 0, arena_bytes).wait();

  int launches = 0;
  for (int p = 0; p < 128; ++p) {
    *pass = p;
    *alive = 0;
    ++launches;
    q.parallel_for(
         sycl::nd_range<1>{sycl::range<1>(kGroups * kLanes),
                           sycl::range<1>(kLanes)},
         [=](sycl::nd_item<1> it) {
           const int gid = static_cast<int>(it.get_group(0));
           const int lid = static_cast<int>(it.get_local_id(0));
           LaneHeader *lane = Lane(arena, gid, lid);
           Ctx ctx{lane, it.get_group()};

           if (lane->top_ == nullptr && lane->sp_ == 0) {
             lane->sp_ = sizeof(LaneHeader);
             CoMain t = Walk(ctx, lid, pass, out + gid * kLanes);
             lane->top_ = t.frame_;
             lane->resume_ = t.frame_;
           }
           sycl::group_barrier(ctx.grp_);
           if (lane->top_ == nullptr) return;  // lane already finished

           // THE DRIVER LOOP. Resume the deepest frame; a trampoline handoff
           // returns here and loops, so chain depth costs one frame of stack.
           for (;;) {
             std::coroutine_handle<>::from_address(lane->resume_).resume();
             if (lane->park_ != 0u ||
                 std::coroutine_handle<>::from_address(lane->top_).done())
               break;
           }
           lane->park_ = 0u;
           sycl::group_barrier(ctx.grp_);
           if (std::coroutine_handle<>::from_address(lane->top_).done()) {
             std::coroutine_handle<>::from_address(lane->top_).destroy();
             lane->top_ = nullptr;
             lane->resume_ = nullptr;
             lane->sp_ = sizeof(LaneHeader);
           } else {
             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                              sycl::memory_scope::device>(alive[0])
                 .fetch_add(1);
           }
         })
        .wait();
    if (*alive == 0) break;
  }

  // Lane L waits for pass >= L in round 0 and pass >= L+1 in round 1, then
  // returns (sum 0..round)*100 + L per round: (0*100 + L) + (1*100 + L).
  bool ok = true;
  for (int g = 0; g < kGroups; ++g) {
    for (int l = 0; l < kLanes; ++l) {
      const int want = 0 * 100 + l + 1 * 100 + l;
      if (out[g * kLanes + l] != want) {
        std::printf("group %d lane %d: got %d want %d\n", g, l,
                    out[g * kLanes + l], want);
        ok = false;
      }
    }
  }
  std::printf("launches=%d out[0]=%d out[31]=%d\n", launches, out[0],
              out[kLanes - 1]);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(arena, q);
  sycl::free(out, q);
  sycl::free(pass, q);
  sycl::free(alive, q);
  return ok ? 0 : 1;
}

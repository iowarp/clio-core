/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE (not built by default): can the SYCL port be MACRO SUBSTITUTION?
 *
 * spike_sycl_lane_coroutine.cpp proved the machinery works, but paid for it
 * with a `Ctx` parameter threaded through every yieldable function -- SYCL
 * has no threadIdx, and promise_type::operator new only sees the
 * coroutine's own parameters, so the arena had to arrive that way. That
 * would mean touching every signature in device_vector.h.
 *
 * It does not have to. Two DPC++ features remove the parameter entirely:
 *
 *   - sycl::ext::oneapi::this_work_item::get_nd_item<1>() -- a FREE FUNCTION
 *     work-item query, so threadIdx / blockIdx / __syncthreads become plain
 *     macros over it with no signature change anywhere.
 *   - device_global -- a device-side global the host sets once, so the
 *     arena base is reachable from operator new(size_t) with no parameters.
 *
 * Which means promise_type::operator new is the ORDINARY one-argument form,
 * exactly as in yield_coro.h, and a yieldable function keeps its signature.
 *
 * Build:
 *   clang++ -fsycl -std=c++20 -fsycl-targets=nvptx64-nvidia-cuda \
 *           --cuda-path=/usr/local/cuda spike_sycl_implicit_lane.cpp -o s
 */
#include <coroutine>
#include <cstdio>

#include <sycl/sycl.hpp>

namespace exp = sycl::ext::oneapi::experimental;
namespace twi = sycl::ext::oneapi::this_work_item;

namespace {

constexpr int kLanes = 32;
constexpr int kGroups = 2;
constexpr size_t kLaneBytes = 4096;

/* The CUDA spellings, as macros over the free-function queries. This is the
 * whole of what yield_stack.h / device_vector.h would need at the top. */
#define SY_THREADIDX_X (twi::get_nd_item<1>().get_local_id(0))
#define SY_BLOCKIDX_X (twi::get_nd_item<1>().get_group(0))
#define SY_BLOCKDIM_X (twi::get_nd_item<1>().get_local_range(0))
#define SY_SYNCTHREADS() sycl::group_barrier(twi::get_work_group<1>())
#define SY_SYNCTHREADS_OR(c) \
  sycl::any_of_group(twi::get_work_group<1>(), (c) ? true : false)

struct LaneHeader {
  unsigned long sp_;
  void *resume_;
  void *top_;
  unsigned park_;
  unsigned pad_;
};

/** The arena base, set once by the host. This is what replaces the implicit
 *  reachability CUDA gets from __device__ globals. */
exp::device_global<char *> g_arena;

/** YieldLane(): the CURRENT work-item's header, from nothing but the
 *  free-function queries -- no parameter, callable from operator new. */
inline LaneHeader *Lane() {
  const size_t idx = SY_BLOCKIDX_X * kLanes + SY_THREADIDX_X;
  return reinterpret_cast<LaneHeader *>(g_arena.get() + idx * kLaneBytes);
}

inline void *LaneAlloc(size_t n) {
  LaneHeader *l = Lane();
  const unsigned long off = (l->sp_ + 15u) & ~15ul;
  const unsigned long end = off + ((n + 15) & ~15ul);
  if (end > kLaneBytes) return nullptr;
  l->sp_ = end;
  return reinterpret_cast<char *>(l) + off;
}

inline void LaneFree(void *p) {
  LaneHeader *l = Lane();
  l->sp_ = static_cast<unsigned long>(reinterpret_cast<char *>(p) -
                                      reinterpret_cast<char *>(l));
}

/* --- the coroutine types, now with ORDINARY operator new (cf. yield_coro.h) */

struct CoMain {
  struct promise_type {
    static void *operator new(size_t n) { return LaneAlloc(n); }
    static void operator delete(void *p, size_t) { LaneFree(p); }
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

template <typename V>
struct CoTask {
  struct promise_type {
    void *parent_ = nullptr;
    V value_{};
    static void *operator new(size_t n) { return LaneAlloc(n); }
    static void operator delete(void *p, size_t) { LaneFree(p); }
    CoTask get_return_object() {
      return CoTask{
          std::coroutine_handle<promise_type>::from_promise(*this).address()};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    struct FinalXfer {
      bool await_ready() const noexcept { return false; }
      void await_suspend(std::coroutine_handle<promise_type> h) noexcept {
        Lane()->resume_ = h.promise().parent_;
      }
      void await_resume() const noexcept {}
    };
    FinalXfer final_suspend() noexcept { return {}; }
    void return_value(V &&v) noexcept { value_ = static_cast<V &&>(v); }
    void unhandled_exception() {}
  };
  void *frame_;
  bool await_ready() const noexcept { return false; }
  template <typename P>
  void await_suspend(std::coroutine_handle<P> parent) noexcept {
    auto h = std::coroutine_handle<promise_type>::from_address(frame_);
    h.promise().parent_ = parent.address();
    Lane()->resume_ = frame_;
  }
  V await_resume() noexcept {
    auto h = std::coroutine_handle<promise_type>::from_address(frame_);
    V v = static_cast<V &&>(h.promise().value_);
    h.destroy();
    LaneFree(frame_);
    return v;
  }
};

struct Park {
  bool await_ready() const noexcept { return false; }
  template <typename P>
  void await_suspend(std::coroutine_handle<P> h) noexcept {
    LaneHeader *l = Lane();
    l->resume_ = h.address();
    l->park_ = 1u;
  }
  void await_resume() const noexcept {}
};

/** Byte-for-byte the shape of CLIO_CO_YIELD_WHEN, with no ctx argument. */
#define SY_CO_YIELD_WHEN(cond)     \
  for (;;) {                       \
    SY_SYNCTHREADS();              \
    if (!SY_SYNCTHREADS_OR(cond))  \
      break;                       \
    co_await Park{};               \
  }

/* Yieldable functions with their ORIGINAL signatures -- no Ctx, no nd_item. */
CoTask<int> Child(const int *pass, int round) {
  const int lane_id = static_cast<int>(SY_THREADIDX_X);
  SY_CO_YIELD_WHEN(*pass < lane_id + round);
  int acc = 0;
  for (int i = 0; i <= round; ++i) {
    acc += i;
    SY_SYNCTHREADS();
  }
  co_return acc * 100 + lane_id;
}

CoMain Walk(const int *pass, int *out) {
  int total = 0;
  for (int round = 0; round < 2; ++round) {
    total += co_await Child(pass, round);
  }
  out[SY_THREADIDX_X] = total;
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
  q.memset(arena, 0, arena_bytes).wait();
  q.copy(&arena, g_arena, 1).wait();  // publish the arena base to the device

  int launches = 0;
  for (int p = 0; p < 128; ++p) {
    *pass = p;
    *alive = 0;
    ++launches;
    q.parallel_for(sycl::nd_range<1>{sycl::range<1>(kGroups * kLanes),
                                     sycl::range<1>(kLanes)},
                   [=](sycl::nd_item<1> it) {
                     LaneHeader *lane = Lane();
                     const int gid = static_cast<int>(it.get_group(0));
                     if (lane->top_ == nullptr && lane->sp_ == 0) {
                       lane->sp_ = sizeof(LaneHeader);
                       CoMain t = Walk(pass, out + gid * kLanes);
                       lane->top_ = t.frame_;
                       lane->resume_ = t.frame_;
                     }
                     SY_SYNCTHREADS();
                     if (lane->top_ == nullptr) return;
                     for (;;) {
                       std::coroutine_handle<>::from_address(lane->resume_)
                           .resume();
                       if (lane->park_ != 0u ||
                           std::coroutine_handle<>::from_address(lane->top_)
                               .done())
                         break;
                     }
                     lane->park_ = 0u;
                     SY_SYNCTHREADS();
                     if (std::coroutine_handle<>::from_address(lane->top_)
                             .done()) {
                       std::coroutine_handle<>::from_address(lane->top_)
                           .destroy();
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

  bool ok = true;
  for (int g = 0; g < kGroups; ++g) {
    for (int l = 0; l < kLanes; ++l) {
      const int want = l + 100 + l;
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

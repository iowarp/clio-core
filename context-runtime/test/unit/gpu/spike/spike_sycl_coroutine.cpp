/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE (not built by default): C++20 coroutines running in SYCL DEVICE code.
 *
 * The SYCL twin of spike_device_coroutine.cu. Same question, same shape:
 * can a coroutine frame live in OUR memory, suspend inside a loop, and
 * resume ACROSS SEPARATE KERNEL LAUNCHES with its ordinary locals intact?
 *
 * Differences from the CUDA spike, all in our favour:
 *   - no <coroutine> shim needed. SYCL is single-source with no host/device
 *     function attributes, so libstdc++'s std::coroutine_handle members are
 *     callable from device code as written -- the exact thing that forced
 *     the CUDA spike to re-declare the whole type with __device__ on every
 *     member and go through __builtin_coro_* by hand.
 *   - the frame arena is plain USM device memory (sycl::malloc_device).
 *
 * The thing under test is whether the SYCL device backend can LOWER a
 * coroutine at all: resume/destroy are indirect calls through pointers
 * stored in the frame, and that is precisely what SPIR-V forbids without
 * SPV_INTEL_function_pointers. On the CUDA (nvptx64) target there is no
 * SPIR-V in the path, so this is expected to work; the spir64 target is the
 * interesting failure. Build both:
 *
 *   icpx -fsycl -std=c++20 -fsycl-targets=nvptx64-nvidia-cuda \
 *        spike_sycl_coroutine.cpp -o spike_sycl_coro
 *   icpx -fsycl -std=c++20 -fsycl-targets=spir64 -c spike_sycl_coroutine.cpp
 */
#include <coroutine>
#include <cstdio>

#include <sycl/sycl.hpp>

namespace {

/** The lane's frame arena, in USM. Bump-allocated, never freed except LIFO --
 *  the same contract as YCoroAlloc/YCoroFree in yield_coro.h. `cursor_` lives
 *  in the arena itself so it survives between kernel launches. */
struct Arena {
  unsigned long cursor_;
  unsigned long bytes_;
  // frames follow
};

Arena *g_arena = nullptr;  // device pointer, captured by value into kernels

inline void *ArenaAlloc(Arena *a, size_t n) {
  const unsigned long off = (a->cursor_ + 15u) & ~15ul;
  const unsigned long end = off + ((n + 15) & ~15ul);
  if (end > a->bytes_) {
    return nullptr;  // a real port traps here; the spike just poisons
  }
  a->cursor_ = end;
  return reinterpret_cast<char *>(a) + off;
}

inline void ArenaFree(Arena *a, void *p) {
  a->cursor_ = static_cast<unsigned long>(reinterpret_cast<char *>(p) -
                                          reinterpret_cast<char *>(a));
}

/* ------------------------------------------------------------------ */
/* The coroutine types. Deliberately the same shape as YCoroMain /     */
/* YCoroSuspend in clio_runtime/gpu/yield_coro.h.                      */
/* ------------------------------------------------------------------ */

/** Where the arena lives for the promise's operator new. A coroutine's
 *  operator new cannot take our arena pointer unless every coroutine
 *  function takes it as a parameter (the promise's operator new is handed
 *  the coroutine's own arguments) -- so pass it, exactly as a real port
 *  would pass the nd_item. */
struct LaneCtx {
  Arena *arena_;
};

struct CoroMain {
  struct promise_type {
    Arena *arena_ = nullptr;

    // operator new receives the coroutine's parameters after the size. This
    // is how the frame gets into OUR memory without any device malloc.
    // The signature must match the coroutine's parameters EXACTLY: the
    // C-variadic spelling the host would accept ("size_t, LaneCtx, ...")
    // makes SYCL reject the whole coroutine as a variadic function.
    static void *operator new(size_t n, LaneCtx ctx, int *, void **) {
      return ArenaAlloc(ctx.arena_, n);
    }
    static void operator delete(void *p, size_t) { /* LIFO, see Reset */ }

    template <typename... Args>
    promise_type(LaneCtx ctx, Args &&...) : arena_(ctx.arena_) {}

    CoroMain get_return_object() {
      return CoroMain{
          std::coroutine_handle<promise_type>::from_promise(*this).address()};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() {}
  };
  void *frame_;
};

/** The suspend point: record the deepest frame and park. In the real system
 *  this also publishes a wait tag to the host; here the handle IS the
 *  published state. */
struct Park {
  void **resume_slot_;
  bool await_ready() const noexcept { return false; }
  template <typename P>
  void await_suspend(std::coroutine_handle<P> h) noexcept {
    *resume_slot_ = h.address();
  }
  void await_resume() const noexcept {}
};

/** THE TEST BODY. Ordinary locals -- `i` and `acc` are captured by the
 *  compiler into the frame, and must survive a kernel exit. */
CoroMain Walk(LaneCtx ctx, int *out, void **resume_slot) {
  int acc = 0;
  for (int i = 0; i < 5; ++i) {
    acc += i * 10;
    out[i] = acc;
    co_await Park{resume_slot};
  }
  out[5] = 777;
  co_return;
}

}  // namespace

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  constexpr size_t kArenaBytes = 64 * 1024;
  auto *arena = static_cast<Arena *>(sycl::malloc_device(kArenaBytes, q));
  auto *out = sycl::malloc_shared<int>(6, q);
  // The lane header: [0] = handle to resume, [1] = top handle (for done()).
  auto *slots = sycl::malloc_shared<void *>(2, q);
  // done() must be evaluated ON DEVICE: the frame lives in device memory, so
  // the host cannot read the resume pointer the intrinsic dereferences.
  auto *done_flag = sycl::malloc_shared<int>(1, q);
  for (int i = 0; i < 6; ++i) out[i] = -1;
  slots[0] = nullptr;
  slots[1] = nullptr;
  *done_flag = 0;

  // Initialise the arena in device memory.
  q.single_task([=] {
     arena->cursor_ = sizeof(Arena);
     arena->bytes_ = kArenaBytes;
   }).wait();

  int launches = 0;
  bool done = false;
  for (int pass = 0; pass < 16 && !done; ++pass) {
    ++launches;
    q.single_task([=] {
       if (slots[1] == nullptr) {
         // First entry: create the chain. The task object is discarded; the
         // frame pointer is what persists.
         CoroMain t = Walk(LaneCtx{arena}, out, &slots[0]);
         slots[1] = t.frame_;
         slots[0] = t.frame_;
       }
       auto h = std::coroutine_handle<>::from_address(slots[0]);
       h.resume();
       done_flag[0] =
           std::coroutine_handle<>::from_address(slots[1]).done() ? 1 : 0;
     }).wait();
    done = *done_flag != 0;
  }

  std::printf("launches=%d done=%d out=", launches, done ? 1 : 0);
  for (int i = 0; i < 6; ++i) std::printf(" %d", out[i]);
  std::printf("\n");

  const bool ok = done && out[0] == 0 && out[1] == 10 && out[2] == 30 &&
                  out[3] == 60 && out[4] == 100 && out[5] == 777 &&
                  launches == 6;
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(arena, q);
  sycl::free(out, q);
  sycl::free(slots, q);
  return ok ? 0 : 1;
}

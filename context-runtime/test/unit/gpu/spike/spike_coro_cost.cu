/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE 5: does co_await pay the frame cost that CLIO_YLOCAL does?
 *
 * The macro measurement said a YLOCAL used inside a loop costs 28.8x, and
 * 49.6x with a suspend point in the loop, because the value is a reference
 * into device memory that nvcc will not promote. A coroutine's locals are
 * ordinary locals, so the compiler is free to keep them in registers and
 * spill only where a suspend can actually happen. This checks whether it does.
 *
 *   PLAIN        ordinary kernel, register accumulator (the floor)
 *   CORO_NEVER   coroutine whose await never fires (await_ready() == true),
 *                the analogue of CLIO_YIELD_IF(false)
 *   CORO_INLOOP  same, but the await sits inside the loop
 *   CORO_RESUMED the loop runs in the OUTLINED body, after a real suspend
 */
#include <cstdio>
#include <cstddef>

namespace std {
template <typename Promise = void> struct coroutine_handle;
template <> struct coroutine_handle<void> {
  void *ptr_ = nullptr;
  __host__ __device__ constexpr coroutine_handle() noexcept {}
  __host__ __device__ constexpr coroutine_handle(decltype(nullptr)) noexcept {}
  __host__ __device__ static coroutine_handle from_address(void *a) noexcept {
    coroutine_handle h; h.ptr_ = a; return h; }
  __host__ __device__ void *address() const noexcept { return ptr_; }
  __host__ __device__ void resume() const { __builtin_coro_resume(ptr_); }
  __host__ __device__ bool done() const { return __builtin_coro_done(ptr_); }
};
template <typename Promise> struct coroutine_handle : coroutine_handle<void> {
  __host__ __device__ static coroutine_handle from_promise(Promise &p) {
    coroutine_handle h;
    h.ptr_ = __builtin_coro_promise(reinterpret_cast<char *>(&p), alignof(Promise), true);
    return h; }
  __host__ __device__ static coroutine_handle from_address(void *a) noexcept {
    coroutine_handle h; h.ptr_ = a; return h; }
};
template <typename R, typename...> struct coroutine_traits { using promise_type = typename R::promise_type; };
struct suspend_always {
  __host__ __device__ bool await_ready() const noexcept { return false; }
  __host__ __device__ void await_suspend(coroutine_handle<>) const noexcept {}
  __host__ __device__ void await_resume() const noexcept {}
};
}  // namespace std

/** Suspends only when `ready` is false -- the analogue of CLIO_YIELD_IF. */
struct MaybeSuspend {
  bool ready_;
  __device__ bool await_ready() const noexcept { return ready_; }
  __device__ void await_suspend(std::coroutine_handle<>) const noexcept {}
  __device__ void await_resume() const noexcept {}
};

#define BLOCKS 64
#define THREADS 256
#define NLANE (BLOCKS * THREADS)
#define LANE_BYTES 256

__device__ char g_arena[(size_t)NLANE * LANE_BYTES];
__device__ unsigned g_off[NLANE];
__device__ __forceinline__ unsigned Gid() { return blockIdx.x * THREADS + threadIdx.x; }

struct Task {
  struct promise_type {
    __device__ Task get_return_object() {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    __device__ std::suspend_always final_suspend() noexcept { return {}; }
    __device__ void return_void() {}
    __device__ void unhandled_exception() {}
    __device__ void *operator new(size_t n) {
      const unsigned l = Gid(); const unsigned o = g_off[l];
      g_off[l] = o + (unsigned)((n + 15) & ~15u);
      return g_arena + (size_t)l * LANE_BYTES + o; }
    __device__ void operator delete(void *, size_t) {}
  };
  std::coroutine_handle<promise_type> h_;
};

__global__ void KPlain(const unsigned *data, unsigned n, unsigned long long *out) {
  unsigned long long acc = 0;
  for (unsigned e = threadIdx.x; e < n; e += THREADS) acc += data[e];
  out[Gid()] = acc;
}

__device__ Task CoroNever(const unsigned *data, unsigned n, unsigned long long *out) {
  unsigned long long acc = 0;
  // Opaque: never true at run time, but the compiler cannot fold it, so the
  // suspend point stays live -- the honest analogue of CLIO_YIELD_IF(false).
  co_await MaybeSuspend{data[0] != 0xDEADBEEFu};
  for (unsigned e = threadIdx.x; e < n; e += THREADS) acc += data[e];
  out[Gid()] = acc;
}

__device__ Task CoroInLoop(const unsigned *data, unsigned n, unsigned long long *out) {
  unsigned long long acc = 0;
  for (unsigned e = threadIdx.x; e < n; e += THREADS) {
    acc += data[e];
    co_await MaybeSuspend{data[e] != 0xDEADBEEFu};   // per-ITERATION, unhoistable
  }
  out[Gid()] = acc;
}

__device__ Task CoroSuspendFirst(const unsigned *data, unsigned n, unsigned long long *out) {
  unsigned long long acc = 0;
  co_await std::suspend_always{};                 // really suspends
  for (unsigned e = threadIdx.x; e < n; e += THREADS) acc += data[e];
  out[Gid()] = acc;
}

template <Task (*FN)(const unsigned *, unsigned, unsigned long long *)>
__global__ void KRunToEnd(const unsigned *data, unsigned n, unsigned long long *out) {
  g_off[Gid()] = 0;
  Task t = FN(data, n, out);
  t.h_.resume();
}

__global__ void KStart(const unsigned *data, unsigned n, unsigned long long *out, void **saved) {
  g_off[Gid()] = 0;
  Task t = CoroSuspendFirst(data, n, out);
  saved[Gid()] = t.h_.address();
  t.h_.resume();                                   // runs to the suspend
}
__global__ void KResume(void **saved) {
  auto h = std::coroutine_handle<>::from_address(saved[Gid()]);
  if (!h.done()) h.resume();                       // the LOOP runs here
}

int main() {
  const unsigned n = 1u << 20;
  unsigned *d; unsigned long long *o; void **saved;
  cudaMalloc(&d, n * sizeof(unsigned)); cudaMemset(d, 1, n * sizeof(unsigned));
  cudaMalloc(&o, NLANE * sizeof(unsigned long long));
  cudaMalloc(&saved, NLANE * sizeof(void *));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int R = 20;
  float ms;

  KPlain<<<BLOCKS,THREADS>>>(d, n, o); cudaDeviceSynchronize();
  cudaEventRecord(a); for (int r=0;r<R;++r) KPlain<<<BLOCKS,THREADS>>>(d, n, o);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  const float plain = ms / R;

  auto bench = [&](const char *name, void (*launch)()) {
    launch(); cudaDeviceSynchronize();
    cudaEventRecord(a); for (int r=0;r<R;++r) launch();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float t = 0; cudaEventElapsedTime(&t, a, b); t /= R;
    printf("  %-14s %8.3f ms  %6.2fx plain\n", name, t, t / plain);
  };

  printf("PLAIN %.3f ms (1.00x)\n", plain);
  static const unsigned *sd; static unsigned sn; static unsigned long long *so;
  static void **ss;
  sd = d; sn = n; so = o; ss = saved;
  bench("coro_never", []{ KRunToEnd<CoroNever><<<BLOCKS,THREADS>>>(sd, sn, so); });
  bench("coro_inloop", []{ KRunToEnd<CoroInLoop><<<BLOCKS,THREADS>>>(sd, sn, so); });

  // resumed: time ONLY the resume, where the loop actually runs
  KStart<<<BLOCKS,THREADS>>>(d, n, o, saved); cudaDeviceSynchronize();
  KResume<<<BLOCKS,THREADS>>>(saved); cudaDeviceSynchronize();
  float total = 0;
  for (int r = 0; r < R; ++r) {
    KStart<<<BLOCKS,THREADS>>>(d, n, o, saved); cudaDeviceSynchronize();
    cudaEventRecord(a); KResume<<<BLOCKS,THREADS>>>(saved); cudaEventRecord(b);
    cudaEventSynchronize(b); float t=0; cudaEventElapsedTime(&t,a,b); total += t;
  }
  printf("  %-14s %8.3f ms  %6.2fx plain\n", "coro_resumed", total/R, (total/R)/plain);
  return 0;
}

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE 2: one coroutine PER LANE, frames from a per-lane arena, resumed
 * across kernel launches -- and each lane yielding a DIFFERENT number of
 * times.
 *
 * That last part is the question. A macro yield contains __syncthreads, so
 * control flow at a yield must be uniform across the block and per-lane loop
 * counts are illegal. A coroutine suspends only the lane that co_awaits, and
 * the block barrier moves OUT to the kernel, which is a strictly larger set of
 * programs. This checks whether that actually holds on device.
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
  __host__ __device__ void destroy() const { __builtin_coro_destroy(ptr_); }
  __host__ __device__ bool done() const { return __builtin_coro_done(ptr_); }
  __host__ __device__ explicit operator bool() const noexcept { return ptr_ != nullptr; }
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

#define LANES 64
#define BLOCKS 4
#define LANE_BYTES 1024

// Per-lane arena: this is where the existing YieldStack would plug in.
__device__ char g_arena[BLOCKS * LANES * LANE_BYTES];
__device__ unsigned g_off[BLOCKS * LANES];

__device__ __forceinline__ unsigned LaneId() {
  return blockIdx.x * LANES + threadIdx.x;
}

struct LaneTask {
  struct promise_type {
    __device__ LaneTask get_return_object() {
      return LaneTask{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    __device__ std::suspend_always final_suspend() noexcept { return {}; }
    __device__ void return_void() {}
    __device__ void unhandled_exception() {}
    // Frame comes from THIS LANE's slice. No global allocator, no contention.
    __device__ void *operator new(size_t n) {
      const unsigned lane = LaneId();
      const unsigned off = g_off[lane];
      g_off[lane] = off + (unsigned)((n + 15) & ~15u);
      return g_arena + (size_t)lane * LANE_BYTES + off;
    }
    __device__ void operator delete(void *, size_t) {}
  };
  std::coroutine_handle<promise_type> h_;
};

/**
 * NON-UNIFORM on purpose: lane 0 yields once, lane 63 yields many times.
 * `i` and `acc` are ordinary locals captured by the compiler.
 */
__device__ LaneTask LaneWork(unsigned iters, unsigned long long *out) {
  unsigned long long acc = 0;
  for (unsigned i = 0; i < iters; ++i) {
    acc += threadIdx.x + i;
    co_await std::suspend_always{};
  }
  out[LaneId()] = acc;
}

__global__ void StartK(const unsigned *iters, unsigned long long *out, void **saved) {
  g_off[LaneId()] = 0;
  LaneTask t = LaneWork(iters[LaneId()], out);
  saved[LaneId()] = t.h_.address();
  t.h_.resume();
}

/** Resume every lane; the BLOCK exits when any lane is still unfinished. */
__global__ void ResumeK(void **saved, int *block_pending) {
  auto h = std::coroutine_handle<>::from_address(saved[LaneId()]);
  if (!h.done()) h.resume();          // per-lane, freely divergent
  const int still = h.done() ? 0 : 1;
  const int any = __syncthreads_or(still);   // barrier OUTSIDE the coroutine
  if (threadIdx.x == 0) block_pending[blockIdx.x] = any;
}

int main() {
  const int n = BLOCKS * LANES;
  unsigned h_iters[n];
  for (int i = 0; i < n; ++i) h_iters[i] = (i % 4) + 1;   // 1..4, non-uniform

  unsigned *d_iters; unsigned long long *d_out; void **d_saved; int *d_pend;
  cudaMalloc(&d_iters, n * sizeof(unsigned));
  cudaMemcpy(d_iters, h_iters, n * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, n * sizeof(unsigned long long));
  cudaMemset(d_out, 0, n * sizeof(unsigned long long));
  cudaMalloc(&d_saved, n * sizeof(void *));
  cudaMalloc(&d_pend, BLOCKS * sizeof(int));

  StartK<<<BLOCKS, LANES>>>(d_iters, d_out, d_saved);
  printf("start: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

  int rounds = 0, pending = 1;
  while (pending && rounds < 32) {
    ResumeK<<<BLOCKS, LANES>>>(d_saved, d_pend);
    ++rounds;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("resume err: %s\n", cudaGetErrorString(e)); return 1; }
    int p[BLOCKS]; cudaMemcpy(p, d_pend, sizeof(p), cudaMemcpyDeviceToHost);
    pending = 0; for (int b = 0; b < BLOCKS; ++b) pending |= p[b];
  }

  unsigned long long out[n]; cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    unsigned long long want = 0;
    for (unsigned k = 0; k < h_iters[i]; ++k) want += (i % LANES) + k;
    if (out[i] != want) { if (bad < 3) printf("lane %d: got %llu want %llu\n", i, out[i], want); ++bad; }
  }
  printf("rounds=%d mismatches=%d (lanes yield 1..4 times, non-uniformly)\n", rounds, bad);
  return bad != 0;
}

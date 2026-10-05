/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE (not built by default): C++20 coroutines running in CUDA DEVICE code.
 *
 * The macro system in yield_stack.h asks the author to hoist every local that
 * lives across a yield, and nothing diagnoses a miss. The compiler already
 * solves that problem for coroutines -- it splits the function into a state
 * machine AND works out what belongs in the frame -- so the question is
 * whether that machinery is reachable from device code.
 *
 * It is, with clang. Measured on this machine:
 *
 *     launches=5 done=1 out= 0 10 30 60 100 777
 *
 * Walk() below is written with ORDINARY locals. `i` and `acc` are captured by
 * the compiler, the coroutine suspends inside a loop, and it resumes across
 * FIVE SEPARATE KERNEL LAUNCHES with acc continuing correctly (running sum of
 * i*10) before running off the end of the loop to write 777.
 *
 * Two things had to be true and both are:
 *   - the frame can live in OUR memory (promise_type::operator new), which is
 *     where a per-lane stack plugs in;
 *   - a handle survives between launches, because it is just a pointer:
 *     coroutine_handle<>::from_address() rebuilds it.
 *
 * WHAT IT COSTS
 *
 *   - nvcc CANNOT compile this: "device code does not support coroutines"
 *     (verified). Only the TUs holding yieldable kernels would need clang.
 *   - <coroutine> is unusable: libstdc++ marks those members __host__ only,
 *     so the shim below re-declares the same names with __device__ on every
 *     member, over clang's __builtin_coro_* intrinsics. Defining these in
 *     namespace std is what freestanding implementations do, but this file
 *     must never see a real <coroutine> as well.
 *   - resume() is an indirect call through a pointer in the frame, which may
 *     inhibit inlining; unmeasured.
 *   - this spike is ONE thread. Coroutines are per-thread, so block-collective
 *     yielding still needs the __syncthreads_or vote, and each lane needs its
 *     own frame -- which is what the existing YieldStack would supply.
 *
 * BUILD
 *   clang++ -x cuda -std=c++20 --cuda-gpu-arch=sm_89 \
 *           --cuda-path=/usr/local/cuda -Wno-unknown-cuda-version \
 *           -c spike_device_coroutine.cu -o spike.o
 *   g++ spike.o -o spike -L/usr/local/cuda/lib64 -lcudart
 * (conda's clang could not drive the final link here; g++ links fine.)
 */
// Device-callable coroutine support. <coroutine> is NOT included: libstdc++'s
// types are host-only, so we supply the same names with __device__ on every
// member. The operations themselves are clang builtins, which work on device.
#include <cstdio>
#include <cstddef>

namespace std {

template <typename Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  void *ptr_ = nullptr;
  __host__ __device__ constexpr coroutine_handle() noexcept {}
  __host__ __device__ constexpr coroutine_handle(decltype(nullptr)) noexcept {}
  __host__ __device__ static coroutine_handle from_address(void *a) noexcept {
    coroutine_handle h; h.ptr_ = a; return h;
  }
  __host__ __device__ void *address() const noexcept { return ptr_; }
  __host__ __device__ void resume() const { __builtin_coro_resume(ptr_); }
  __host__ __device__ void destroy() const { __builtin_coro_destroy(ptr_); }
  __host__ __device__ bool done() const { return __builtin_coro_done(ptr_); }
  __host__ __device__ explicit operator bool() const noexcept { return ptr_ != nullptr; }
};

template <typename Promise>
struct coroutine_handle : coroutine_handle<void> {
  __host__ __device__ static coroutine_handle from_promise(Promise &p) {
    coroutine_handle h;
    h.ptr_ = __builtin_coro_promise(reinterpret_cast<char *>(&p),
                                    alignof(Promise), true);
    return h;
  }
  __host__ __device__ static coroutine_handle from_address(void *a) noexcept {
    coroutine_handle h; h.ptr_ = a; return h;
  }
  __host__ __device__ Promise &promise() const {
    return *reinterpret_cast<Promise *>(
        __builtin_coro_promise(ptr_, alignof(Promise), false));
  }
};

template <typename R, typename...>
struct coroutine_traits { using promise_type = typename R::promise_type; };

struct suspend_always {
  __host__ __device__ bool await_ready() const noexcept { return false; }
  __host__ __device__ void await_suspend(coroutine_handle<>) const noexcept {}
  __host__ __device__ void await_resume() const noexcept {}
};

}  // namespace std

__device__ char g_arena[1 << 16];
__device__ unsigned g_off;

struct DevTask {
  struct promise_type {
    __device__ DevTask get_return_object() {
      return DevTask{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    __device__ std::suspend_always final_suspend() noexcept { return {}; }
    __device__ void return_void() {}
    __device__ void unhandled_exception() {}
    __device__ void *operator new(size_t n) {
      unsigned off = atomicAdd(&g_off, (unsigned)((n + 15) & ~15u));
      return g_arena + off;
    }
    __device__ void operator delete(void *, size_t) {}
  };
  std::coroutine_handle<promise_type> h_;
};

// Ordinary locals. Nothing is hoisted by hand -- the COMPILER decides what
// lives in the frame.
__device__ DevTask Walk(int n, int *out) {
  int acc = 0;
  for (int i = 0; i < n; ++i) {
    acc += i * 10;
    out[i] = acc;
    co_await std::suspend_always{};
  }
  out[n] = 777;
}

__global__ void StartK(int n, int *out, void **saved) {
  g_off = 0;
  DevTask t = Walk(n, out);
  *saved = t.h_.address();
  t.h_.resume();
}

__global__ void ResumeK(void **saved, int *done) {
  auto h = std::coroutine_handle<>::from_address(*saved);
  if (!h.done()) h.resume();
  *done = h.done() ? 1 : 0;
}

int main() {
  const int n = 5;
  int *d_out = nullptr, *d_done = nullptr;
  void **d_saved = nullptr;
  cudaMalloc(&d_out, (n + 1) * sizeof(int));
  cudaMemset(d_out, 0, (n + 1) * sizeof(int));
  cudaMalloc(&d_done, sizeof(int));
  cudaMalloc(&d_saved, sizeof(void *));

  StartK<<<1, 1>>>(n, d_out, d_saved);
  printf("start: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

  int done = 0, launches = 0;
  for (int k = 0; k < 20 && !done; ++k) {
    ResumeK<<<1, 1>>>(d_saved, d_done);
    ++launches;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("resume err: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(&done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
  }

  int out[16] = {0};
  cudaMemcpy(out, d_out, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  printf("launches=%d done=%d out=", launches, done);
  for (int i = 0; i <= n; ++i) printf(" %d", out[i]);
  printf("\nwant=          0 10 30 60 100 777\n");
  return 0;
}

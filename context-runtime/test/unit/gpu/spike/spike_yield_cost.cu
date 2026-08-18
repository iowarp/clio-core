/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE 3: what does making a kernel yieldable COST per iteration?
 *
 * Three variants of one workload, all compiled by the same compiler in the
 * same binary, so the comparison is mechanism vs mechanism:
 *
 *   PLAIN   ordinary kernel, no yield machinery at all (the floor)
 *   SWITCH  the macro shape: locals live in a per-lane frame in device memory,
 *           resume via switch on a saved point
 *   CORO    a coroutine: ordinary locals, the compiler decides what to spill
 *
 * The number that matters for gpu_vector is the NEVER-YIELDS case. HoldPage is
 * called constantly and almost always hits; if being yieldable taxes every
 * iteration, it is unusable on the hot path no matter how good the suspend is.
 */
#include <cstdio>
#include <cstddef>
#include <cstdlib>

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

#define BLOCKS 64
#define THREADS 256
#define NLANE (BLOCKS * THREADS)
#define LANE_BYTES 256

__device__ __forceinline__ unsigned Gid() { return blockIdx.x * THREADS + threadIdx.x; }
/** Cheap, dependent arithmetic so the loop cannot be optimized away. */
__device__ unsigned g_workw = 1;   // dependent ops per iteration
__device__ __forceinline__ unsigned long long Work(unsigned i) {
  unsigned long long v = (unsigned long long)i * 2654435761u + (i >> 3);
  for (unsigned w = 1; w < g_workw; ++w) v = v * 6364136223846793005ull + 1442695040888963407ull;
  return v;
}

// ---------------- PLAIN ----------------
__global__ void KPlain(unsigned iters, unsigned long long *out) {
  unsigned long long acc = 0;
  for (unsigned i = 0; i < iters; ++i) acc += Work(i);
  out[Gid()] = acc;
}

// ---------------- SWITCH (macro shape) ----------------
struct Frame { unsigned point_; unsigned i_; unsigned long long acc_; };
__device__ Frame g_frames[NLANE];

__global__ void KSwitch(unsigned iters, unsigned every, unsigned long long *out,
                        int *pending) {
  Frame *f = &g_frames[Gid()];
  int suspended = 0;
  switch (f->point_) {
    case 0:
      f->i_ = 0; f->acc_ = 0;
      for (; f->i_ < iters; ++f->i_) {
        f->acc_ += Work(f->i_);            // locals live in device memory
        if (every && (f->i_ % every) == every - 1) {
          f->point_ = 1;
          suspended = 1;
          goto done;
    case 1: ;
        }
      }
      out[Gid()] = f->acc_;
      f->point_ = 2;
  }
done:
  { const int any = __syncthreads_or(suspended);
    if (threadIdx.x == 0) pending[blockIdx.x] = any; }
}

// ---------------- CORO ----------------
__device__ char g_arena[(size_t)NLANE * LANE_BYTES];
__device__ unsigned g_stride = LANE_BYTES;
__device__ unsigned g_off[NLANE];

struct LaneTask {
  struct promise_type {
    __device__ LaneTask get_return_object() {
      return LaneTask{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    __device__ std::suspend_always initial_suspend() noexcept { return {}; }
    __device__ std::suspend_always final_suspend() noexcept { return {}; }
    __device__ void return_void() {}
    __device__ void unhandled_exception() {}
    __device__ void *operator new(size_t n) {
      const unsigned l = Gid(); const unsigned o = g_off[l];
      g_off[l] = o + (unsigned)((n + 15) & ~15u);
      return g_arena + (size_t)l * g_stride + o; }
    __device__ void operator delete(void *, size_t) {}
  };
  std::coroutine_handle<promise_type> h_;
};

__device__ LaneTask CoroWork(unsigned iters, unsigned every, unsigned long long *out) {
  unsigned long long acc = 0;                 // ordinary locals
  for (unsigned i = 0; i < iters; ++i) {
    acc += Work(i);
    if (every && (i % every) == every - 1) co_await std::suspend_always{};
  }
  out[Gid()] = acc;
}

__global__ void KCoroStart(unsigned iters, unsigned every, unsigned long long *out,
                           void **saved, int *pending) {
  g_off[Gid()] = 0;
  LaneTask t = CoroWork(iters, every, out);
  saved[Gid()] = t.h_.address();
  t.h_.resume();
  const int any = __syncthreads_or(t.h_.done() ? 0 : 1);
  if (threadIdx.x == 0) pending[blockIdx.x] = any;
}

__global__ void KCoroResume(void **saved, int *pending) {
  auto h = std::coroutine_handle<>::from_address(saved[Gid()]);
  if (!h.done()) h.resume();
  const int any = __syncthreads_or(h.done() ? 0 : 1);
  if (threadIdx.x == 0) pending[blockIdx.x] = any;
}

// ---------------- driver ----------------
static double RunMs(void (*fn)(), int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  fn(); cudaDeviceSynchronize();                 // warm
  cudaEventRecord(a);
  for (int r = 0; r < reps; ++r) fn();
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / reps;
}

static unsigned g_iters, g_every;
static unsigned long long *g_out; static void **g_saved; static int *g_pend;


int main(int argc, char **argv) {
  const unsigned iters = 20000, every = 2000, work = 64;
  cudaMemcpyToSymbol(g_workw, &work, sizeof(unsigned));
  cudaMalloc(&g_out, NLANE * sizeof(unsigned long long));
  cudaMalloc(&g_saved, NLANE * sizeof(void *));
  cudaMalloc(&g_pend, BLOCKS * sizeof(int));
  g_iters = iters; g_every = every;

  // Reference: the plain kernel's answer.
  KPlain<<<BLOCKS, THREADS>>>(iters, g_out);
  cudaDeviceSynchronize();
  unsigned long long *ref = new unsigned long long[NLANE];
  cudaMemcpy(ref, g_out, NLANE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float ms_plain = 0;
  cudaEventRecord(a);
  for (int r = 0; r < 10; ++r) KPlain<<<BLOCKS, THREADS>>>(iters, g_out);
  cudaEventRecord(b); cudaEventSynchronize(b);
  cudaEventElapsedTime(&ms_plain, a, b); ms_plain /= 10;
  printf("PLAIN %.3f ms   (work=%u, iters=%u, yields=%u)\n\n",
         ms_plain, work, iters, iters / every - 1);
  printf("%-8s %10s %8s %s\n", "stride", "ms", "vs plain", "correct?");

  const unsigned only = (argc > 1) ? (unsigned)atoi(argv[1]) : 256u;
  const unsigned strides[] = {only};
  for (unsigned si = 0; si < 1; ++si) {
    const unsigned st = strides[si];
    cudaMemcpyToSymbol(g_stride, &st, sizeof(unsigned));
    cudaMemset(g_out, 0, NLANE * sizeof(unsigned long long));

    auto run = [&]{
      KCoroStart<<<BLOCKS, THREADS>>>(g_iters, g_every, g_out, g_saved, g_pend);
      int p[BLOCKS], pending = 0, guard = 0;
      cudaMemcpy(p, g_pend, sizeof(p), cudaMemcpyDeviceToHost);
      for (int i2 = 0; i2 < BLOCKS; ++i2) pending |= p[i2];
      while (pending && guard++ < 64) {
        KCoroResume<<<BLOCKS, THREADS>>>(g_saved, g_pend);
        cudaMemcpy(p, g_pend, sizeof(p), cudaMemcpyDeviceToHost);
        pending = 0; for (int i2 = 0; i2 < BLOCKS; ++i2) pending |= p[i2];
      }
    };
    run(); cudaDeviceSynchronize();

    unsigned long long *got = new unsigned long long[NLANE];
    cudaMemcpy(got, g_out, NLANE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i2 = 0; i2 < NLANE; ++i2) if (got[i2] != ref[i2]) ++bad;
    delete[] got;

    cudaEventRecord(a);
    for (int r = 0; r < 5; ++r) run();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 5;

    printf("%-8u %10.3f %7.2fx %s\n", st, ms, ms / ms_plain,
           bad == 0 ? "yes" : "NO -- frames overlap, timing meaningless");
  }
  return 0;
}

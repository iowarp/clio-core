/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE 4: the realistic shape -- streaming paged data with a fault per page.
 *
 * Spike 3 measured a loop whose body was a few arithmetic ops, which makes the
 * yield mechanism look enormous because there is nothing else in the frame.
 * This is the workload that actually matters: 256 KB pages, every page faults
 * on first touch, the host services the fault between rounds, and the body
 * does page-sized memory work.
 *
 *   PLAIN   everything already resident, no yield machinery (the floor)
 *   SWITCH  macro shape: values live across a yield sit in a per-lane frame,
 *           the inner loop keeps a register copy (what a careful author writes)
 *   CORO    coroutine with ordinary locals
 *
 * Both yielding variants re-check residency after resuming, so the fault path
 * is a retry loop in both -- the same semantics CLIO_YIELD_IF has.
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
#define PAGES 16
#define PAGE_ELEMS 65536            // 256 KB of u32, the real page size
#define NLANE (BLOCKS * THREADS)
#define LANE_BYTES 128

__device__ __forceinline__ unsigned Gid() { return blockIdx.x * THREADS + threadIdx.x; }
__device__ __forceinline__ size_t PageBase(unsigned p) {
  return ((size_t)blockIdx.x * PAGES + p) * PAGE_ELEMS;
}

// ---------------- PLAIN: no faults, no machinery ----------------
__global__ void KPlain(const unsigned *data, unsigned long long *out) {
  unsigned long long acc = 0;
  for (unsigned p = 0; p < PAGES; ++p) {
    const size_t base = PageBase(p);
    for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS) acc += data[base + e];
  }
  out[Gid()] = acc;
}

// ---------------- SWITCH ----------------
struct Frame { unsigned point_; unsigned p_; unsigned long long acc_; };
__device__ Frame g_frames[NLANE];

__global__ void KSwitch(const unsigned *data, const unsigned *resident,
                        unsigned long long *out, int *pending) {
  Frame *f = &g_frames[Gid()];
  int suspended = 0;
  switch (f->point_) {
    case 0:
      f->p_ = 0; f->acc_ = 0;
      for (; f->p_ < PAGES; ++f->p_) {
        for (;;) {                       // retry loop: re-check after resume
          if (resident[f->p_]) break;
          f->point_ = 1; suspended = 1; goto done;
    case 1: ;
        }
        {
          const size_t base = PageBase(f->p_);
          unsigned long long acc = f->acc_;      // register copy for the body
          for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS) acc += data[base + e];
          f->acc_ = acc;
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
      return g_arena + (size_t)l * LANE_BYTES + o; }
    __device__ void operator delete(void *, size_t) {}
  };
  std::coroutine_handle<promise_type> h_;
};

__device__ LaneTask CoroStream(const unsigned *data, const unsigned *resident,
                               unsigned long long *out) {
  unsigned long long acc = 0;
  for (unsigned p = 0; p < PAGES; ++p) {
    while (!resident[p]) co_await std::suspend_always{};
    const size_t base = PageBase(p);
    for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS) acc += data[base + e];
  }
  out[Gid()] = acc;
}

__global__ void KCoroStart(const unsigned *data, const unsigned *resident,
                           unsigned long long *out, void **saved, int *pending) {
  g_off[Gid()] = 0;
  LaneTask t = CoroStream(data, resident, out);
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

int main() {
  const size_t nelem = (size_t)BLOCKS * PAGES * PAGE_ELEMS;
  printf("stream: %d blocks x %d pages x %d KB = %.0f MB, fault per page\n",
         BLOCKS, PAGES, (int)(PAGE_ELEMS * 4 / 1024),
         nelem * 4.0 / (1024 * 1024));

  unsigned *d_data; unsigned long long *d_out; unsigned *d_res;
  void **d_saved; int *d_pend;
  cudaMalloc(&d_data, nelem * sizeof(unsigned));
  cudaMemset(d_data, 1, nelem * sizeof(unsigned));
  cudaMalloc(&d_out, NLANE * sizeof(unsigned long long));
  cudaMalloc(&d_res, PAGES * sizeof(unsigned));
  cudaMalloc(&d_saved, NLANE * sizeof(void *));
  cudaMalloc(&d_pend, BLOCKS * sizeof(int));
  unsigned h_res[PAGES];
  Frame *frames = nullptr; cudaGetSymbolAddress((void **)&frames, g_frames);

  auto all_resident = [&]{ for (int i = 0; i < PAGES; ++i) h_res[i] = 1;
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice); };
  auto none_resident = [&]{ for (int i = 0; i < PAGES; ++i) h_res[i] = 0;
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice); };
  auto service = [&](int upto){ for (int i = 0; i <= upto && i < PAGES; ++i) h_res[i] = 1;
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice); };

  // reference
  all_resident();
  KPlain<<<BLOCKS, THREADS>>>(d_data, d_out);
  cudaDeviceSynchronize();
  unsigned long long *ref = new unsigned long long[NLANE];
  cudaMemcpy(ref, d_out, NLANE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  auto check = [&](const char *who){
    unsigned long long *got = new unsigned long long[NLANE];
    cudaMemcpy(got, d_out, NLANE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < NLANE; ++i) if (got[i] != ref[i]) ++bad;
    delete[] got;
    if (bad) printf("  %s: %d WRONG lanes\n", who, bad);
    return bad == 0; };

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int reps = 5;

  // PLAIN
  cudaEventRecord(a);
  for (int r = 0; r < reps; ++r) KPlain<<<BLOCKS, THREADS>>>(d_data, d_out);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms_plain = 0; cudaEventElapsedTime(&ms_plain, a, b); ms_plain /= reps;

  // SWITCH
  int sw_rounds = 0;
  auto run_switch = [&]{
    cudaMemset(frames, 0, NLANE * sizeof(Frame));
    none_resident();
    int pending = 1, guard = 0; sw_rounds = 0;
    while (pending && guard < 64) {
      KSwitch<<<BLOCKS, THREADS>>>(d_data, d_res, d_out, d_pend);
      ++sw_rounds;
      int p[BLOCKS]; cudaMemcpy(p, d_pend, sizeof(p), cudaMemcpyDeviceToHost);
      pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= p[i];
      if (pending) service(guard);       // host services the fault
      ++guard;
    } };
  run_switch(); cudaDeviceSynchronize();
  const bool sw_ok = check("SWITCH");
  cudaEventRecord(a);
  for (int r = 0; r < reps; ++r) run_switch();
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms_sw = 0; cudaEventElapsedTime(&ms_sw, a, b); ms_sw /= reps;

  // CORO
  int co_rounds = 0;
  auto run_coro = [&]{
    none_resident();
    KCoroStart<<<BLOCKS, THREADS>>>(d_data, d_res, d_out, d_saved, d_pend);
    co_rounds = 1;
    int p[BLOCKS], pending = 0, guard = 0;
    cudaMemcpy(p, d_pend, sizeof(p), cudaMemcpyDeviceToHost);
    for (int i = 0; i < BLOCKS; ++i) pending |= p[i];
    while (pending && guard < 64) {
      if (pending) service(guard);
      KCoroResume<<<BLOCKS, THREADS>>>(d_saved, d_pend);
      ++co_rounds;
      cudaMemcpy(p, d_pend, sizeof(p), cudaMemcpyDeviceToHost);
      pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= p[i];
      ++guard;
    } };
  run_coro(); cudaDeviceSynchronize();
  const bool co_ok = check("CORO");
  cudaEventRecord(a);
  for (int r = 0; r < reps; ++r) run_coro();
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms_co = 0; cudaEventElapsedTime(&ms_co, a, b); ms_co /= reps;

  const double gb = nelem * 4.0 / (1024.0 * 1024.0 * 1024.0);
  printf("\n%-8s %10s %10s %9s %8s %s\n", "variant", "ms", "GB/s", "vs plain", "rounds", "correct");
  printf("%-8s %10.3f %10.1f %9s %8d %s\n", "PLAIN", ms_plain, gb / (ms_plain / 1000), "1.00x", 1, "ref");
  printf("%-8s %10.3f %10.1f %8.2fx %8d %s\n", "SWITCH", ms_sw, gb / (ms_sw / 1000), ms_sw / ms_plain, sw_rounds, sw_ok ? "yes" : "NO");
  printf("%-8s %10.3f %10.1f %8.2fx %8d %s\n", "CORO", ms_co, gb / (ms_co / 1000), ms_co / ms_plain, co_rounds, co_ok ? "yes" : "NO");
  return 0;
}

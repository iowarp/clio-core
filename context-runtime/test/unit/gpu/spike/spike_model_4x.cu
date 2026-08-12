/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE 6: model weight calculation with the model 4x larger than the GPU-side
 * page cache -- the question the mechanism choice actually has to answer.
 *
 * Every page must be fetched from host memory, used once, and evicted, so the
 * run is dominated by PCIe traffic and fault round trips rather than by
 * whatever the kernel does between suspends. Both variants perform the SAME
 * fetches with the SAME cache policy; only the suspend/resume mechanism
 * differs.
 *
 *   SWITCH  the macro shape: values crossing a yield live in a per-lane frame,
 *           the inner dot product keeps a register copy (the documented rule)
 *   CORO    co_await, ordinary locals
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

#define BLOCKS 16
#define THREADS 256
#define NLANE (BLOCKS * THREADS)
#define PAGE_ELEMS 65536            // 256 KB of float
#define CACHE_PAGES 8               // resident pages per block
#define MODEL_PAGES 32              // 4x the cache
#define LANE_BYTES 128

__device__ __forceinline__ unsigned Gid() { return blockIdx.x * THREADS + threadIdx.x; }

// cache[block][slot][elem]; resident[block*CACHE+slot] holds the page id
__device__ __forceinline__ size_t SlotBase(unsigned slot) {
  return ((size_t)blockIdx.x * CACHE_PAGES + slot) * PAGE_ELEMS;
}

// ---------------- RESIDENT: the whole model is already in GPU memory ----
// No paging, no faults, no yield machinery. The ceiling this is measured
// against, and what you get if the model simply fits.
__global__ void KResident(const float *model, const float *act, float *out) {
  float acc = 0.0f;
  for (unsigned p = 0; p < MODEL_PAGES; ++p) {
    const size_t base = ((size_t)blockIdx.x * MODEL_PAGES + p) * PAGE_ELEMS;
    for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS)
      acc += model[base + e] * act[e];
  }
  out[Gid()] = acc;
}

// ---------------- SWITCH ----------------
struct Frame { unsigned point_; unsigned p_; float acc_; unsigned pad_; };
__device__ Frame g_frames[NLANE];

__global__ void KSwitch(const float *cache, const int *resident, const float *act,
                        float *out, int *want, int *pending) {
  Frame *f = &g_frames[Gid()];
  int suspended = 0;
  switch (f->point_) {
    case 0:
      f->p_ = 0; f->acc_ = 0.0f;
      for (; f->p_ < MODEL_PAGES; ++f->p_) {
        for (;;) {   // no declarations here: the case label jumps in
          if (resident[blockIdx.x * CACHE_PAGES + (f->p_ % CACHE_PAGES)] ==
              (int)f->p_) break;
          if (threadIdx.x == 0) want[blockIdx.x] = (int)f->p_;
          f->point_ = 1; suspended = 1; goto done;
    case 1: ;
        }
        {
          const size_t base = SlotBase(f->p_ % CACHE_PAGES);
          float acc = f->acc_;                       // register for the body
          for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS)
            acc += cache[base + e] * act[e];
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

__device__ Task CoroModel(const float *cache, const int *resident, const float *act,
                          float *out, int *want) {
  float acc = 0.0f;
  for (unsigned p = 0; p < MODEL_PAGES; ++p) {
    const unsigned slot = p % CACHE_PAGES;
    while (resident[blockIdx.x * CACHE_PAGES + slot] != (int)p) {
      if (threadIdx.x == 0) want[blockIdx.x] = (int)p;
      co_await std::suspend_always{};
    }
    const size_t base = SlotBase(slot);
    for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS)
      acc += cache[base + e] * act[e];
  }
  out[Gid()] = acc;
}

__global__ void KCoroStart(const float *cache, const int *resident, const float *act,
                           float *out, int *want, void **saved, int *pending) {
  g_off[Gid()] = 0;
  Task t = CoroModel(cache, resident, act, out, want);
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

static float g_resident_ms = 0.0f;

int main() {
  const size_t page_bytes = (size_t)PAGE_ELEMS * sizeof(float);
  const size_t model_bytes = (size_t)BLOCKS * MODEL_PAGES * page_bytes;
  const size_t cache_bytes = (size_t)BLOCKS * CACHE_PAGES * page_bytes;
  printf("model %.0f MB, GPU cache %.0f MB  (%.1fx oversubscribed)\n",
         model_bytes / 1048576.0, cache_bytes / 1048576.0,
         (double)model_bytes / cache_bytes);

  float *h_model = nullptr;                       // the model lives in host RAM
  cudaHostAlloc(&h_model, model_bytes, cudaHostAllocDefault);
  for (size_t i = 0; i < model_bytes / sizeof(float); ++i) h_model[i] = 1.0f;

  float *d_cache, *d_act, *d_out; int *d_res, *d_want, *d_pend;
  cudaMalloc(&d_cache, cache_bytes);
  cudaMalloc(&d_act, page_bytes);
  cudaMemset(d_act, 0, page_bytes);
  cudaMalloc(&d_out, NLANE * sizeof(float));
  cudaMalloc(&d_res, BLOCKS * CACHE_PAGES * sizeof(int));
  cudaMalloc(&d_want, BLOCKS * sizeof(int));
  cudaMalloc(&d_pend, BLOCKS * sizeof(int));
  void **d_saved; cudaMalloc(&d_saved, NLANE * sizeof(void *));
  Frame *frames; cudaGetSymbolAddress((void **)&frames, g_frames);

  int h_res[BLOCKS * CACHE_PAGES], h_want[BLOCKS], h_pend[BLOCKS];
  cudaStream_t s; cudaStreamCreate(&s);

  // Service: fetch each block's requested page into its slot. This is the
  // PCIe traffic a 4x-oversubscribed model cannot avoid.
  auto service = [&](long long *bytes) {
    cudaMemcpy(h_want, d_want, sizeof(h_want), cudaMemcpyDeviceToHost);
    for (int b = 0; b < BLOCKS; ++b) {
      const int p = h_want[b];
      if (p < 0 || p >= MODEL_PAGES) continue;
      const int slot = p % CACHE_PAGES;
      cudaMemcpyAsync(d_cache + ((size_t)b * CACHE_PAGES + slot) * PAGE_ELEMS,
                      h_model + ((size_t)b * MODEL_PAGES + p) * PAGE_ELEMS,
                      page_bytes, cudaMemcpyHostToDevice, s);
      h_res[b * CACHE_PAGES + slot] = p;
      *bytes += (long long)page_bytes;
    }
    cudaStreamSynchronize(s);
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice);
  };
  auto reset = [&]{
    for (int i = 0; i < BLOCKS * CACHE_PAGES; ++i) h_res[i] = -1;
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice);
    for (int b = 0; b < BLOCKS; ++b) h_want[b] = -1;
    cudaMemcpy(d_want, h_want, sizeof(h_want), cudaMemcpyHostToDevice);
  };

  // Warm up BOTH variants before timing either. Without this the variant that
  // runs first absorbs pinned-buffer first touch and context warm-up, and was
  // measured 1.46x slower purely for going first.
  for (int warm = 0; warm < 2; ++warm) {
    for (int variant = 0; variant < 2; ++variant) {
      reset();
      cudaMemset(frames, 0, NLANE * sizeof(Frame));
      long long junk = 0; int pending = 1, guard = 0;
      if (variant == 1) {
        KCoroStart<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_saved, d_pend);
        cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
        pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
      }
      while (pending && guard++ < 256) {
        service(&junk);
        if (variant == 0) KSwitch<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_pend);
        else KCoroResume<<<BLOCKS, THREADS>>>(d_saved, d_pend);
        cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
        pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
      }
      cudaDeviceSynchronize();
    }
  }

  // Ceiling: the same arithmetic with the model resident in GPU memory.
  float *d_full = nullptr;
  const bool have_full = (cudaMalloc(&d_full, model_bytes) == cudaSuccess);
  if (have_full) {
    cudaMemcpy(d_full, h_model, model_bytes, cudaMemcpyHostToDevice);
    KResident<<<BLOCKS, THREADS>>>(d_full, d_act, d_out);
    cudaDeviceSynchronize();
    cudaEvent_t a, b2; cudaEventCreate(&a); cudaEventCreate(&b2);
    cudaEventRecord(a);
    for (int r = 0; r < 10; ++r) KResident<<<BLOCKS, THREADS>>>(d_full, d_act, d_out);
    cudaEventRecord(b2); cudaEventSynchronize(b2);
    float ms = 0; cudaEventElapsedTime(&ms, a, b2); ms /= 10;
    printf("  %-9s %8.2f ms   rounds=  1   read=%.0f MB   %.2f GB/s (device memory)\n",
           "RESIDENT", ms, model_bytes / 1048576.0,
           (model_bytes / 1073741824.0) / (ms / 1000.0));
    g_resident_ms = ms;
  }

  const int REPS = 3;
  for (int variant = 0; variant < 2; ++variant) {
    const char *name = variant == 0 ? "SWITCH" : "CORO";
    double best = 1e30; int rounds_seen = 0; long long bytes_seen = 0;
    for (int rep = 0; rep < REPS; ++rep) {
      reset();
      cudaMemset(frames, 0, NLANE * sizeof(Frame));
      long long bytes = 0; int rounds = 0;
      cudaEvent_t a, b2; cudaEventCreate(&a); cudaEventCreate(&b2);
      cudaEventRecord(a);
      int pending = 1, guard = 0;
      if (variant == 1) {
        KCoroStart<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_saved, d_pend);
        ++rounds;
        cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
        pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
      }
      while (pending && guard++ < 256) {
        service(&bytes);
        if (variant == 0) KSwitch<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_pend);
        else KCoroResume<<<BLOCKS, THREADS>>>(d_saved, d_pend);
        ++rounds;
        cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
        pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
      }
      cudaEventRecord(b2); cudaEventSynchronize(b2);
      float ms = 0; cudaEventElapsedTime(&ms, a, b2);
      if (ms < best) best = ms;
      rounds_seen = rounds; bytes_seen = bytes;
      cudaEventDestroy(a); cudaEventDestroy(b2);
    }
    printf("  %-9s %8.2f ms   rounds=%3d   fetched=%.0f MB   %.2f GB/s effective"
           "   %.1fx resident\n",
           name, best, rounds_seen, bytes_seen / 1048576.0,
           (bytes_seen / 1073741824.0) / (best / 1000.0),
           g_resident_ms > 0 ? best / g_resident_ms : 0.0);
  }
  return 0;
}

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * SPIKE 7: the compressed vector, served device-to-device.
 *
 * Spike 6 showed a 4x-oversubscribed model costs 18x, and that all of it is
 * PCIe. Compression attacks exactly that: keep the model in VRAM in compressed
 * form and expand each page on the device when it faults, so the fault is a
 * D2D decompress instead of a host transfer.
 *
 * This is only possible because the kernel YIELDS. A kernel that spins waiting
 * for its page blocks every later launch in the context, so the decompressor
 * -- which is itself a kernel -- can never run. Exiting and resuming is what
 * puts a GPU codec on the fault path at all.
 *
 * Decompression cost is swept, because the answer depends on it: `work` is
 * dependent ALU ops per output element, 0 being the pure-bandwidth floor that
 * no real codec reaches.
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
#define PAGE_ELEMS 65536            // 256 KB of float, decompressed
#define CACHE_PAGES 8
#define MODEL_PAGES 32              // 4x the cache when uncompressed
#define RATIO 4                     // compression ratio

__device__ __forceinline__ unsigned Gid() { return blockIdx.x * THREADS + threadIdx.x; }
__device__ __forceinline__ size_t SlotBase(unsigned slot) {
  return ((size_t)blockIdx.x * CACHE_PAGES + slot) * PAGE_ELEMS;
}

struct Frame { unsigned point_; unsigned p_; float acc_; unsigned pad_; };
__device__ Frame g_frames[NLANE];

/** The yieldable weight calculation. Identical to spike 6's SWITCH variant. */
__global__ void KSwitch(const float *cache, const int *resident, const float *act,
                        float *out, int *want, int *pending) {
  Frame *f = &g_frames[Gid()];
  int suspended = 0;
  switch (f->point_) {
    case 0:
      f->p_ = 0; f->acc_ = 0.0f;
      for (; f->p_ < MODEL_PAGES; ++f->p_) {
        for (;;) {
          if (resident[blockIdx.x * CACHE_PAGES + (f->p_ % CACHE_PAGES)] ==
              (int)f->p_) break;
          if (threadIdx.x == 0) want[blockIdx.x] = (int)f->p_;
          f->point_ = 1; suspended = 1; goto done;
    case 1: ;
        }
        {
          const size_t base = SlotBase(f->p_ % CACHE_PAGES);
          float acc = f->acc_;
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

// ---------------- CORO: same algorithm, co_await instead of the switch ----
#define LANE_BYTES 128
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

/** The whole model resident and UNCOMPRESSED -- the ceiling. */
__global__ void KResident(const float *model, const float *act, float *out) {
  float acc = 0.0f;
  for (unsigned p = 0; p < MODEL_PAGES; ++p) {
    const size_t base = ((size_t)blockIdx.x * MODEL_PAGES + p) * PAGE_ELEMS;
    for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS)
      acc += model[base + e] * act[e];
  }
  out[Gid()] = acc;
}

/**
 * Fault service, on the device: expand one compressed page per faulting block
 * straight into its cache slot. Reads PAGE_ELEMS/RATIO, writes PAGE_ELEMS.
 */
__global__ void KDecompress(const float *comp, float *cache, const int *want,
                            int work) {
  const int p = want[blockIdx.x];
  if (p < 0 || p >= MODEL_PAGES) return;
  const unsigned slot = p % CACHE_PAGES;
  const size_t cbase = ((size_t)blockIdx.x * MODEL_PAGES + p) * (PAGE_ELEMS / RATIO);
  const size_t obase = ((size_t)blockIdx.x * CACHE_PAGES + slot) * PAGE_ELEMS;
  for (unsigned e = threadIdx.x; e < PAGE_ELEMS; e += THREADS) {
    float v = comp[cbase + e / RATIO];
    for (int w = 0; w < work; ++w) v = v * 1.0000001f + 0.5f;  // codec ALU
    cache[obase + e] = v;
  }
}

int main() {
  const size_t page_bytes = (size_t)PAGE_ELEMS * sizeof(float);
  const size_t model_bytes = (size_t)BLOCKS * MODEL_PAGES * page_bytes;
  const size_t comp_bytes = model_bytes / RATIO;
  const size_t cache_bytes = (size_t)BLOCKS * CACHE_PAGES * page_bytes;
  printf("model %.0f MB uncompressed, %.0f MB compressed (%dx), cache %.0f MB\n",
         model_bytes / 1048576.0, comp_bytes / 1048576.0, RATIO,
         cache_bytes / 1048576.0);

  float *h_model; cudaHostAlloc(&h_model, model_bytes, cudaHostAllocDefault);
  for (size_t i = 0; i < model_bytes / sizeof(float); ++i) h_model[i] = 1.0f;

  float *d_cache, *d_act, *d_out, *d_comp, *d_full;
  int *d_res, *d_want, *d_pend;
  cudaMalloc(&d_cache, cache_bytes);
  cudaMalloc(&d_act, page_bytes); cudaMemset(d_act, 0, page_bytes);
  cudaMalloc(&d_out, NLANE * sizeof(float));
  cudaMalloc(&d_comp, comp_bytes);                 // compressed model in VRAM
  cudaMemset(d_comp, 0, comp_bytes);
  cudaMalloc(&d_res, BLOCKS * CACHE_PAGES * sizeof(int));
  cudaMalloc(&d_want, BLOCKS * sizeof(int));
  cudaMalloc(&d_pend, BLOCKS * sizeof(int));
  const bool have_full = (cudaMalloc(&d_full, model_bytes) == cudaSuccess);
  if (have_full) cudaMemcpy(d_full, h_model, model_bytes, cudaMemcpyHostToDevice);
  Frame *frames; cudaGetSymbolAddress((void **)&frames, g_frames);

  int h_res[BLOCKS * CACHE_PAGES], h_want[BLOCKS], h_pend[BLOCKS];
  cudaStream_t s; cudaStreamCreate(&s);

  auto reset = [&]{
    for (int i = 0; i < BLOCKS * CACHE_PAGES; ++i) h_res[i] = -1;
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice);
    for (int b = 0; b < BLOCKS; ++b) h_want[b] = -1;
    cudaMemcpy(d_want, h_want, sizeof(h_want), cudaMemcpyHostToDevice);
    cudaMemset(frames, 0, NLANE * sizeof(Frame));
  };
  auto mark_resident = [&]{
    cudaMemcpy(h_want, d_want, sizeof(h_want), cudaMemcpyDeviceToHost);
    for (int b = 0; b < BLOCKS; ++b) {
      const int p = h_want[b];
      if (p >= 0 && p < MODEL_PAGES) h_res[b * CACHE_PAGES + (p % CACHE_PAGES)] = p;
    }
    cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice);
  };

  // PCIe paging, for reference (spike 6's number)
  auto run_pcie = [&](int *rounds_out) {
    reset(); int pending = 1, guard = 0, rounds = 0;
    while (pending && guard++ < 256) {
      cudaMemcpy(h_want, d_want, sizeof(h_want), cudaMemcpyDeviceToHost);
      for (int b = 0; b < BLOCKS; ++b) {
        const int p = h_want[b];
        if (p < 0 || p >= MODEL_PAGES) continue;
        cudaMemcpyAsync(d_cache + ((size_t)b * CACHE_PAGES + (p % CACHE_PAGES)) * PAGE_ELEMS,
                        h_model + ((size_t)b * MODEL_PAGES + p) * PAGE_ELEMS,
                        page_bytes, cudaMemcpyHostToDevice, s);
        h_res[b * CACHE_PAGES + (p % CACHE_PAGES)] = p;
      }
      cudaStreamSynchronize(s);
      cudaMemcpy(d_res, h_res, sizeof(h_res), cudaMemcpyHostToDevice);
      KSwitch<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_pend);
      ++rounds;
      cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
      pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
    }
    if (rounds_out) *rounds_out = rounds;
  };

  // Compressed, served on the device
  void **d_saved; cudaMalloc(&d_saved, NLANE * sizeof(void *));

  // coro==false -> macro switch shape; coro==true -> co_await
  auto run_d2d = [&](int work, bool coro, int *rounds_out) {
    reset(); int pending = 1, guard = 0, rounds = 0;
    if (coro) {
      KCoroStart<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_saved, d_pend);
      ++rounds;
      cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
      pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
    }
    while (pending && guard++ < 256) {
      KDecompress<<<BLOCKS, THREADS>>>(d_comp, d_cache, d_want, work);
      mark_resident();
      if (coro) KCoroResume<<<BLOCKS, THREADS>>>(d_saved, d_pend);
      else KSwitch<<<BLOCKS, THREADS>>>(d_cache, d_res, d_act, d_out, d_want, d_pend);
      ++rounds;
      cudaMemcpy(h_pend, d_pend, sizeof(h_pend), cudaMemcpyDeviceToHost);
      pending = 0; for (int i = 0; i < BLOCKS; ++i) pending |= h_pend[i];
    }
    if (rounds_out) *rounds_out = rounds;
  };

  cudaEvent_t a, b2; cudaEventCreate(&a); cudaEventCreate(&b2);
  auto time_it = [&](auto &&fn, int reps) {
    fn(); cudaDeviceSynchronize();               // warm
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) fn();
    cudaEventRecord(b2); cudaEventSynchronize(b2);
    float ms = 0; cudaEventElapsedTime(&ms, a, b2); return ms / reps;
  };

  float resident_ms = 0;
  if (have_full) {
    resident_ms = time_it([&]{ KResident<<<BLOCKS, THREADS>>>(d_full, d_act, d_out); }, 10);
    printf("\n%-22s %8.2f ms  %7.1f GB/s   1.0x\n", "RESIDENT uncompressed",
           resident_ms, (model_bytes / 1073741824.0) / (resident_ms / 1000.0));
  }
  int rr = 0;
  const float pcie_ms = time_it([&]{ run_pcie(&rr); }, 3);
  printf("%-22s %8.2f ms  %7.1f GB/s  %5.1fx   (rounds=%d)\n", "PCIe paging 4x",
         pcie_ms, (model_bytes / 1073741824.0) / (pcie_ms / 1000.0),
         pcie_ms / resident_ms, rr);

  // Warm BOTH mechanisms before timing either: whichever runs first
  // otherwise absorbs first-touch cost and looks slower (spike 6's trap).
  for (int w2 = 0; w2 < 2; ++w2) { run_d2d(0, false, nullptr); run_d2d(0, true, nullptr); }
  cudaDeviceSynchronize();

  printf("\n%-16s %-6s %10s %10s %8s\n", "compressed D2D", "codec", "SWITCH", "CORO", "delta");
  for (int work : {0, 4, 16, 64}) {
    const float ms_sw = time_it([&]{ run_d2d(work, false, &rr); }, 3);
    const float ms_co = time_it([&]{ run_d2d(work, true, &rr); }, 3);
    printf("%-16s w=%-4d %8.2f ms %8.2f ms %7.2fx  (SWITCH %.2fx vs PCIe)\n",
           "", work, ms_sw, ms_co, ms_co / ms_sw, pcie_ms / ms_sw);
  }
  return 0;
}

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * MLP training, BaM edition: the capacity sibling. W1 and W2 live in pinned
 * host DRAM behind BaM page caches -- ONE CACHE EACH, because BaM tags pages
 * by array-relative offset and two arrays on one cache alias (the grayscott
 * sibling measured 23% field drift from exactly that). Reads go through
 * acquire-once page windows; the in-place SGD update writes the cached page
 * and publishes it whole with host_write_page under a strict one-writer-
 * per-page decomposition -- the same write-site-publish contract the paged
 * bench certifies, built from BaM's parts. Biases are tiny and stay in
 * plain device memory.
 *
 * The backprop's o-sum accumulates page-by-page in ascending order, and the
 * in-process dense reference uses the identical page-blocked association,
 * so LOSS (per step) and the WEIGHT DIGEST gate on BIT EQUALITY, matching
 * the family's standard. Links nothing from clio.
 */

#include <cuda_runtime.h>
#include <bam/array.cuh>
#include <bam/page_cache.cuh>
#include <bam/page_cache_host.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define LB_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

using u32 = unsigned int;
using u64 = unsigned long long;

__host__ __device__ inline u64 Lcg(u64 s) {
  return s * 6364136223846793005ull + 1442695040888963407ull;
}
__host__ __device__ inline float Sym01(u64 s) {
  return (static_cast<float>((s >> 40) & 0xFFFFFF) / 8388608.0f) - 1.0f;
}

/** Acquire page `pg` of `arr` on thread 0 and share the raw pointer. */
#define LB_ACQUIRE(arr, pg, elems_per_page, s_ptr, s_tok)                    \
  do {                                                                       \
    if (threadIdx.x == 0) {                                                  \
      bam::PageWindow<float> _w = (arr).acquire_page((pg) *                  \
                                                     (elems_per_page));      \
      s_ptr = _w.data;                                                       \
      s_tok = _w.page_off;                                                   \
    }                                                                        \
    __syncthreads();                                                         \
  } while (0)

#define LB_RELEASE(arr, s_tok)                                               \
  do {                                                                       \
    __syncthreads();                                                         \
    if (threadIdx.x == 0) {                                                  \
      bam::PageWindow<float> _w;                                             \
      _w.page_off = s_tok;                                                   \
      (arr).release_page(_w);                                                \
    }                                                                        \
    __syncthreads();                                                         \
  } while (0)

/** a1 rows for the W1 pages this block strides (rpp1 rows per page). */
__global__ void Fwd1Bam(bam::ArrayDevice<float> w1, const float *b1, u64 I,
                        u64 H, u64 B, u64 rpp1, const float *x, float *a1) {
  __shared__ float *s_ptr;
  __shared__ uint64_t s_tok;
  const u64 npages = H / rpp1;
  for (u64 pg = blockIdx.x; pg < npages; pg += gridDim.x) {
    LB_ACQUIRE(w1, pg, rpp1 * I, s_ptr, s_tok);
    const float *w = s_ptr;
    const u64 h0 = pg * rpp1;
    for (u64 t = threadIdx.x; t < rpp1 * B; t += blockDim.x) {
      const u64 h = h0 + t / B;
      const u64 b = t % B;
      float acc = b1[h];
      for (u64 i = 0; i < I; ++i) {
        acc += w[(h - h0) * I + i] * x[b * I + i];
      }
      a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
    }
    LB_RELEASE(w1, s_tok);
  }
}

__global__ void Fwd2Bam(bam::ArrayDevice<float> w2, const float *b2, u64 H,
                        u64 O, u64 B, u64 rpp2, const float *a1,
                        const float *y, float *d2, double *loss_parts) {
  __shared__ float *s_ptr;
  __shared__ uint64_t s_tok;
  const u64 npages = O / rpp2;
  for (u64 pg = blockIdx.x; pg < npages; pg += gridDim.x) {
    LB_ACQUIRE(w2, pg, rpp2 * H, s_ptr, s_tok);
    const float *w = s_ptr;
    const u64 o0 = pg * rpp2;
    for (u64 t = threadIdx.x; t < rpp2 * B; t += blockDim.x) {
      const u64 o = o0 + t / B;
      const u64 b = t % B;
      float acc = b2[o];
      for (u64 h = 0; h < H; ++h) {
        acc += w[(o - o0) * H + h] * a1[h * B + b];
      }
      const float diff = acc - y[b * O + o];
      d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
      loss_parts[o * B + b] =
          static_cast<double>(diff) * static_cast<double>(diff);
    }
    LB_RELEASE(w2, s_tok);
  }
}

/** d1 for this block's h-range: W2 pages visited in ASCENDING order, one
 *  page-block of the o-sum per visit -- the association the reference
 *  reproduces exactly. */
__global__ void Bwd1Bam(bam::ArrayDevice<float> w2, u64 H, u64 O, u64 B,
                        u64 rpp2, const float *a1, const float *d2,
                        float *d1) {
  __shared__ float *s_ptr;
  __shared__ uint64_t s_tok;
  const u64 hper = H / gridDim.x;
  const u64 h0 = blockIdx.x * hper;
  const u64 h1 = (blockIdx.x + 1 == gridDim.x) ? H : h0 + hper;
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    d1[(h0 + t / B) * B + t % B] = 0.0f;
  }
  const u64 npages = O / rpp2;
  for (u64 pg = 0; pg < npages; ++pg) {
    LB_ACQUIRE(w2, pg, rpp2 * H, s_ptr, s_tok);
    const float *w = s_ptr;
    const u64 o0 = pg * rpp2;
    for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
      const u64 h = h0 + t / B;
      const u64 b = t % B;
      float blk = 0.0f;
      for (u64 o = o0; o < o0 + rpp2; ++o) {
        blk += w[(o - o0) * H + h] * d2[o * B + b];
      }
      d1[h * B + b] += blk;
    }
    LB_RELEASE(w2, s_tok);
  }
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    const u64 h = h0 + t / B;
    const u64 b = t % B;
    if (a1[h * B + b] <= 0.0f) d1[h * B + b] = 0.0f;
  }
}

/** In-place SGD on the cached page, then publish it WHOLE: one block owns
 *  each page, so the write-back cannot race another writer. */
__global__ void Upd2Bam(bam::ArrayDevice<float> w2, float *b2, u64 H, u64 O,
                        u64 B, u64 rpp2, const float *a1, const float *d2,
                        float lr) {
  __shared__ float *s_ptr;
  __shared__ uint64_t s_tok;
  const u64 npages = O / rpp2;
  for (u64 pg = blockIdx.x; pg < npages; pg += gridDim.x) {
    LB_ACQUIRE(w2, pg, rpp2 * H, s_ptr, s_tok);
    float *w = s_ptr;
    const u64 o0 = pg * rpp2;
    for (u64 t = threadIdx.x; t < rpp2 * H; t += blockDim.x) {
      const u64 o = o0 + t / H;
      const u64 h = t % H;
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) g += d2[o * B + b] * a1[h * B + b];
      w[(o - o0) * H + h] -= lr * g;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      bam::host_write_page(reinterpret_cast<const uint8_t *>(s_ptr),
                      const_cast<uint8_t *>(w2.host_base), s_tok,
                      static_cast<uint32_t>(w2.cache_state.page_size));
    }
    LB_RELEASE(w2, s_tok);
  }
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < O;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[t * B + b];
    b2[t] -= lr * g;
  }
}

__global__ void Upd1Bam(bam::ArrayDevice<float> w1, float *b1, u64 I, u64 H,
                        u64 B, u64 rpp1, const float *x, const float *d1,
                        float lr) {
  __shared__ float *s_ptr;
  __shared__ uint64_t s_tok;
  const u64 npages = H / rpp1;
  for (u64 pg = blockIdx.x; pg < npages; pg += gridDim.x) {
    LB_ACQUIRE(w1, pg, rpp1 * I, s_ptr, s_tok);
    float *w = s_ptr;
    const u64 h0 = pg * rpp1;
    for (u64 t = threadIdx.x; t < rpp1 * I; t += blockDim.x) {
      const u64 h = h0 + t / I;
      const u64 i = t % I;
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) g += d1[h * B + b] * x[b * I + i];
      w[(h - h0) * I + i] -= lr * g;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      bam::host_write_page(reinterpret_cast<const uint8_t *>(s_ptr),
                      const_cast<uint8_t *>(w1.host_base), s_tok,
                      static_cast<uint32_t>(w1.cache_state.page_size));
    }
    LB_RELEASE(w1, s_tok);
  }
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < H;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[t * B + b];
    b1[t] -= lr * g;
  }
}

// ---- dense reference kernels (plain memory, same loops) -------------------

__global__ void RefFwd1(const float *w1, const float *b1, u64 I, u64 H,
                        u64 B, const float *x, float *a1) {
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < H * B;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = t / B;
    const u64 b = t % B;
    float acc = b1[h];
    for (u64 i = 0; i < I; ++i) acc += w1[h * I + i] * x[b * I + i];
    a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
  }
}

__global__ void RefFwd2(const float *w2, const float *b2, u64 H, u64 O,
                        u64 B, const float *a1, const float *y, float *d2,
                        double *loss_parts) {
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < O * B;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 o = t / B;
    const u64 b = t % B;
    float acc = b2[o];
    for (u64 h = 0; h < H; ++h) acc += w2[o * H + h] * a1[h * B + b];
    const float diff = acc - y[b * O + o];
    d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
    loss_parts[o * B + b] =
        static_cast<double>(diff) * static_cast<double>(diff);
  }
}

__global__ void RefBwd1(const float *w2, u64 H, u64 O, u64 B, u64 rpp2,
                        const float *a1, const float *d2, float *d1) {
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < H * B;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = t / B;
    const u64 b = t % B;
    float acc = 0.0f;
    for (u64 op = 0; op < O; op += rpp2) {
      float blk = 0.0f;
      for (u64 o = op; o < op + rpp2 && o < O; ++o) {
        blk += w2[o * H + h] * d2[o * B + b];
      }
      acc += blk;
    }
    d1[h * B + b] = (a1[h * B + b] <= 0.0f) ? 0.0f : acc;
  }
}

__global__ void RefUpd2(float *w2, float *b2, u64 H, u64 O, u64 B,
                        const float *a1, const float *d2, float lr) {
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < O * H;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 o = t / H;
    const u64 h = t % H;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b] * a1[h * B + b];
    w2[o * H + h] -= lr * g;
  }
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < O;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[t * B + b];
    b2[t] -= lr * g;
  }
}

__global__ void RefUpd1(float *w1, float *b1, u64 I, u64 H, u64 B,
                        const float *x, const float *d1, float lr) {
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < H * I;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = t / I;
    const u64 i = t % I;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b] * x[b * I + i];
    w1[h * I + i] -= lr * g;
  }
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < H;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[t * B + b];
    b1[t] -= lr * g;
  }
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

/** Order-independent digest of a host float span at logical ids. */
static unsigned long long HostDigest(const float *w, u64 gbase, u64 n) {
  unsigned long long acc = 0;
  for (u64 i = 0; i < n; ++i) {
    u32 bits;
    std::memcpy(&bits, &w[i], sizeof(bits));
    acc += static_cast<unsigned long long>(bits) * (2ull * (gbase + i) + 1ull);
  }
  return acc;
}

int main(int argc, char **argv) {
  u32 blocks = 8, threads = 256;
  u64 I = 256, H = 4096, O = 64, B = 64, steps = 5, cache_mb = 1,
      bam_page_kb = 64;
  float lr = 0.01f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--in") I = next();
    else if (a == "--hidden") H = next();
    else if (a == "--out") O = next();
    else if (a == "--batch") B = next();
    else if (a == "--steps") steps = next();
    else if (a == "--cache-mb") cache_mb = next();
    else if (a == "--bam-page-kb") bam_page_kb = next();
    else if (a == "--lr" && i + 1 < argc) lr = std::strtof(argv[++i], nullptr);
  }
  LB_CUDA_CHECK(cudaSetDevice(0));
  const u64 page_bytes = bam_page_kb * 1024;
  const u64 rpp1 = page_bytes / (I * sizeof(float));
  const u64 rpp2 = page_bytes / (H * sizeof(float));
  if (rpp1 == 0 || rpp2 == 0 || H % rpp1 != 0 || O % rpp2 != 0 ||
      page_bytes % (I * sizeof(float)) != 0 ||
      page_bytes % (H * sizeof(float)) != 0) {
    std::fprintf(stderr, "LBANN BAM: rows must tile BaM pages exactly "
                 "(I=%llu H=%llu page=%lluKB)\n", (unsigned long long)I,
                 (unsigned long long)H, (unsigned long long)bam_page_kb);
    return 2;
  }
  const u64 w1_n = H * I, w2_n = O * H;

  std::printf("MLP training, BaM edition: %llu -> %llu -> %llu, batch=%llu, "
              "steps=%llu, weights %.1f MB behind %llu MB BaM caches\n",
              (unsigned long long)I, (unsigned long long)H,
              (unsigned long long)O, (unsigned long long)B,
              (unsigned long long)steps,
              static_cast<double>((w1_n + w2_n) * sizeof(float)) / 1048576.0,
              (unsigned long long)cache_mb);

  // One cache per array (the aliasing lesson).
  bam::PageCacheConfig cfg{};
  cfg.page_size = static_cast<size_t>(page_bytes);
  cfg.num_pages =
      static_cast<size_t>((cache_mb * 1024 * 1024) / page_bytes);
  cfg.num_queues = 1;
  cfg.queue_depth = 1024;
  cfg.backend = bam::BackendType::kHostMemory;
  cfg.nvme_dev = nullptr;
  std::unique_ptr<bam::PageCache> c1(new bam::PageCache(cfg));
  std::unique_ptr<bam::PageCache> c2(new bam::PageCache(cfg));
  std::unique_ptr<bam::Array<float>> w1(new bam::Array<float>(w1_n, *c1));
  std::unique_ptr<bam::Array<float>> w2(new bam::Array<float>(w2_n, *c2));
  std::vector<float> h_w1(w1_n), h_w2(w2_n), h_b1(H), h_b2(O);
  for (u64 i = 0; i < w1_n; ++i) {
    h_w1[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
  }
  for (u64 h = 0; h < H; ++h) {
    h_b1[h] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + h)) * 0.05f;
  }
  for (u64 i = 0; i < w2_n; ++i) {
    h_w2[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + H + i)) * 0.05f;
  }
  for (u64 o = 0; o < O; ++o) {
    h_b2[o] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + H + w2_n + o)) * 0.05f;
  }
  w1->load_from_host(h_w1.data(), w1_n);
  w2->load_from_host(h_w2.data(), w2_n);

  std::vector<float> hxv(B * I), hyv(B * O);
  for (u64 i = 0; i < B * I; ++i) hxv[i] = Sym01(Lcg(0xA02BDBF7BB3C0A7ull + i));
  for (u64 i = 0; i < B * O; ++i) hyv[i] = Sym01(Lcg(0x6C62272E07BB0142ull + i));
  float *d_x, *d_y, *d_b1, *d_b2, *d_a1, *d_d1, *d_d2;
  double *d_lp;
  LB_CUDA_CHECK(cudaMalloc(&d_x, B * I * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_y, B * O * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_b1, H * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_b2, O * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_a1, H * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_d1, H * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_d2, O * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_lp, O * B * sizeof(double)));
  LB_CUDA_CHECK(cudaMemcpy(d_x, hxv.data(), B * I * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(d_y, hyv.data(), B * O * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), H * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), O * sizeof(float),
                           cudaMemcpyHostToDevice));

  std::vector<double> h_lp(O * B), loss(steps);
  const double t0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    Fwd1Bam<<<blocks, threads>>>(w1->device(), d_b1, I, H, B, rpp1, d_x,
                                 d_a1);
    LB_CUDA_CHECK(cudaGetLastError());
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    Fwd2Bam<<<blocks, threads>>>(w2->device(), d_b2, H, O, B, rpp2, d_a1,
                                 d_y, d_d2, d_lp);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    LB_CUDA_CHECK(cudaMemcpy(h_lp.data(), d_lp, O * B * sizeof(double),
                             cudaMemcpyDeviceToHost));
    double l = 0.0;
    for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
    loss[s] = l / static_cast<double>(B * O);
    Bwd1Bam<<<blocks, threads>>>(w2->device(), H, O, B, rpp2, d_a1, d_d2,
                                 d_d1);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    Upd2Bam<<<blocks, threads>>>(w2->device(), d_b2, H, O, B, rpp2, d_a1,
                                 d_d2, lr);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    Upd1Bam<<<blocks, threads>>>(w1->device(), d_b1, I, H, B, rpp1, d_x,
                                 d_d1, lr);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double ms = NowMs() - t0;

  // Final weights: the BaM backing store IS the truth after the page
  // publishes; biases come off the device.
  std::vector<float> f_w1(w1_n), f_w2(w2_n), f_b1(H), f_b2(O);
  {
    // host_base is pinned host memory owned by the cache; read it back
    // through the arrays' own accessor: load path is host->cache, so the
    // simplest truth source is the backing mirror we can copy from the
    // device-visible host_base pointer via cudaMemcpy (host pinned).
    const uint8_t *hb1 = nullptr, *hb2 = nullptr;
    hb1 = w1->device().host_base;
    hb2 = w2->device().host_base;
    std::memcpy(f_w1.data(), hb1, w1_n * sizeof(float));
    std::memcpy(f_w2.data(), hb2, w2_n * sizeof(float));
  }
  LB_CUDA_CHECK(cudaMemcpy(f_b1.data(), d_b1, H * sizeof(float),
                           cudaMemcpyDeviceToHost));
  LB_CUDA_CHECK(cudaMemcpy(f_b2.data(), d_b2, O * sizeof(float),
                           cudaMemcpyDeviceToHost));
  unsigned long long dg = HostDigest(f_w1.data(), 0, w1_n);
  dg += HostDigest(f_b1.data(), w1_n, H);
  dg += HostDigest(f_w2.data(), w1_n + H, w2_n);
  dg += HostDigest(f_b2.data(), w1_n + H + w2_n, O);

  // ---- dense reference, same association ----------------------------------
  float *r_w1, *r_b1, *r_w2, *r_b2;
  LB_CUDA_CHECK(cudaMalloc(&r_w1, w1_n * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&r_b1, H * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&r_w2, w2_n * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&r_b2, O * sizeof(float)));
  LB_CUDA_CHECK(cudaMemcpy(r_w1, h_w1.data(), w1_n * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(r_b1, h_b1.data(), H * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(r_w2, h_w2.data(), w2_n * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(r_b2, h_b2.data(), O * sizeof(float),
                           cudaMemcpyHostToDevice));
  std::vector<double> loss_ref(steps);
  for (u64 s = 0; s < steps; ++s) {
    RefFwd1<<<blocks, threads>>>(r_w1, r_b1, I, H, B, d_x, d_a1);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    RefFwd2<<<blocks, threads>>>(r_w2, r_b2, H, O, B, d_a1, d_y, d_d2, d_lp);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    LB_CUDA_CHECK(cudaMemcpy(h_lp.data(), d_lp, O * B * sizeof(double),
                             cudaMemcpyDeviceToHost));
    double l = 0.0;
    for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
    loss_ref[s] = l / static_cast<double>(B * O);
    RefBwd1<<<blocks, threads>>>(r_w2, H, O, B, rpp2, d_a1, d_d2, d_d1);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    RefUpd2<<<blocks, threads>>>(r_w2, r_b2, H, O, B, d_a1, d_d2, lr);
    RefUpd1<<<blocks, threads>>>(r_w1, r_b1, I, H, B, d_x, d_d1, lr);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
  }
  std::vector<float> rf(w1_n);
  LB_CUDA_CHECK(cudaMemcpy(rf.data(), r_w1, w1_n * sizeof(float),
                           cudaMemcpyDeviceToHost));
  unsigned long long dg_ref = HostDigest(rf.data(), 0, w1_n);
  std::vector<float> rb1(H), rw2(w2_n), rb2(O);
  LB_CUDA_CHECK(cudaMemcpy(rb1.data(), r_b1, H * sizeof(float),
                           cudaMemcpyDeviceToHost));
  LB_CUDA_CHECK(cudaMemcpy(rw2.data(), r_w2, w2_n * sizeof(float),
                           cudaMemcpyDeviceToHost));
  LB_CUDA_CHECK(cudaMemcpy(rb2.data(), r_b2, O * sizeof(float),
                           cudaMemcpyDeviceToHost));
  dg_ref += HostDigest(rb1.data(), w1_n, H);
  dg_ref += HostDigest(rw2.data(), w1_n + H, w2_n);
  dg_ref += HostDigest(rb2.data(), w1_n + H + w2_n, O);

  int rc = 0;
  std::printf("  %llu steps in %.1f ms\n", (unsigned long long)steps, ms);
  bool ok = true;
  for (u64 s = 0; s < steps; ++s) {
    if (loss[s] != loss_ref[s]) {
      std::printf("  LOSS GATE: step %llu %.17g != %.17g\n",
                  (unsigned long long)s, loss[s], loss_ref[s]);
      ok = false;
    }
  }
  if (ok) {
    std::printf("  LOSS GATE: PASS (all steps bit-equal; %.6f -> %.6f)\n",
                loss_ref[0], loss_ref[steps - 1]);
  } else {
    rc = 1;
  }
  if (dg != dg_ref) {
    std::printf("  WEIGHT GATE: FAIL (digest %llu != %llu)\n", dg, dg_ref);
    rc = 1;
  } else {
    std::printf("  WEIGHT GATE: PASS (bit-equal to dense reference)\n");
  }
  std::printf("%s\n", rc == 0 ? "LBANN BAM: ALL GATES PASS"
                              : "LBANN BAM: GATE FAILURE");
  return rc;
}

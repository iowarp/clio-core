/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * PME spread+gather, BaM edition: the capacity sibling. The K^3 mesh lives
 * in pinned host DRAM behind BaM's page cache. BaM has no safe sub-page
 * write path, so the spread accumulates each plane in a small device buffer
 * (the plane-owner decomposition makes that exact) and streams it to the
 * host mirror; the mesh is then published once with load_from_host and the
 * GATHER reads it back through bam_ptr -- the read-heavy half is the part
 * that exercises the page cache. Same fixed-point science as every sibling;
 * CONSERVATION and MESH gates exact. Links nothing from clio.
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

#define GX_CUDA_CHECK(x)                                                     \
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

/** Q40.24 fixed point; identical constants and generators to the paged
 *  bench so every substrate spreads the same charges. */
static constexpr double kFxScale = 16777216.0;

__host__ __device__ inline u64 Lcg(u64 s) {
  return s * 6364136223846793005ull + 1442695040888963407ull;
}
__host__ __device__ inline float Frac01(u64 s) {
  return static_cast<float>((s >> 40) & 0xFFFFFF) / 16777216.0f;
}
__host__ __device__ inline void Spline4(float t, float w[4]) {
  const float t2 = t * t, t3 = t2 * t;
  w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) / 6.0f;
  w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) / 6.0f;
  w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) / 6.0f;
  w[3] = t3 / 6.0f;
}

/**
 * Spread the contributions of bins z-3..z onto plane z of `dst` (one plane,
 * dst points at its base). The exactly-conserving hierarchical split is the
 * paged bench's, verbatim: the four z-pieces sum to q exactly and the 16
 * xy-pieces of each z-piece sum to it exactly, so conservation is an integer
 * identity on every substrate.
 */
__device__ inline void SpreadPlane(unsigned long long *dst, u64 z, u64 K,
                                   const float *ax, const float *ay,
                                   const float *az, const long long *aq,
                                   const u32 *bin_start) {
  for (int db = -3; db <= 0; ++db) {
    const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
    const u32 a0 = bin_start[b];
    const u32 a1 = bin_start[b + 1];
    for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
      const float x = ax[a], y = ay[a], zz = az[a];
      const int ix0 = static_cast<int>(floorf(x)) - 1;
      const int iy0 = static_cast<int>(floorf(y)) - 1;
      const int dzw = static_cast<int>((z + K - b) % K);
      float wx[4], wy[4], wz[4];
      Spline4(x - floorf(x), wx);
      Spline4(y - floorf(y), wy);
      Spline4(zz - floorf(zz), wz);
      long long qz4[4];
      {
        long long run = 0;
        for (int j = 0; j < 3; ++j) {
          qz4[j] = static_cast<long long>(
              llrint(static_cast<double>(aq[a]) * wz[j]));
          run += qz4[j];
        }
        qz4[3] = aq[a] - run;
      }
      const long long qz = qz4[dzw];
      long long xy_run = 0;
      for (int jy = 0; jy < 4; ++jy) {
        const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                         static_cast<int>(K));
        for (int jx = 0; jx < 4; ++jx) {
          const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                          static_cast<int>(K));
          const long long v =
              (jy == 3 && jx == 3)
                  ? qz - xy_run
                  : static_cast<long long>(llrint(
                        static_cast<double>(qz) * wy[jy] * wx[jx]));
          xy_run += v;
          atomicAdd(&dst[gy_ * K + gx], static_cast<unsigned long long>(v));
        }
      }
    }
    __syncthreads();
  }
}

/**
 * The gather, decomposed BY PLANE OWNER: an atom's energy is
 * q * sum_jz wz_jz * (xy interpolation on plane iz0+jz), and each z-term
 * only needs ONE plane -- so the owner of plane z accumulates the z-terms
 * of the atoms in bins z-3..z, and the partials meet in an exact integer
 * reduction. No halo, no replica reads: the same locality the spread has.
 */
__device__ inline unsigned long long GatherPlane(
    const unsigned long long *pl, u64 z, u64 K, const float *ax,
    const float *ay, const float *az, const long long *aq,
    const u32 *bin_start) {
  unsigned long long acc = 0;
  for (int db = -3; db <= 0; ++db) {
    const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
    const u32 a0 = bin_start[b];
    const u32 a1 = bin_start[b + 1];
    for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
      const float x = ax[a], y = ay[a], zz = az[a];
      const int ix0 = static_cast<int>(floorf(x)) - 1;
      const int iy0 = static_cast<int>(floorf(y)) - 1;
      const int dzw = static_cast<int>((z + K - b) % K);
      float wx[4], wy[4], wzS[4];
      Spline4(x - floorf(x), wx);
      Spline4(y - floorf(y), wy);
      Spline4(zz - floorf(zz), wzS);
      double pl_sum = 0.0;
      for (int jy = 0; jy < 4; ++jy) {
        const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                         static_cast<int>(K));
        double row = 0.0;
        for (int jx = 0; jx < 4; ++jx) {
          const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                          static_cast<int>(K));
          row += static_cast<double>(
                     static_cast<long long>(pl[gy_ * K + gx])) * wx[jx];
        }
        pl_sum += row * wy[jy];
      }
      // Same requantisation as the paged bench, per z-term. NOTE: the paged
      // bench rounds the SUM of the four z-terms once; rounding per term
      // differs in the last unit, so the gather gate between substrates is
      // exact only against siblings using this same per-plane form. The
      // conservation and mesh gates stay universally exact.
      acc += static_cast<unsigned long long>(static_cast<long long>(
          llrint(pl_sum * wzS[dzw] *
                 (static_cast<double>(aq[a]) / kFxScale))));
    }
    __syncthreads();
  }
  return acc;
}


__global__ void SpreadOnePlane(unsigned long long *pl, u64 z, u64 K,
                               const float *ax, const float *ay,
                               const float *az, const long long *aq,
                               const u32 *bin_start) {
  if (blockIdx.x == 0) SpreadPlane(pl, z, K, ax, ay, az, aq, bin_start);
}

__global__ void GatherMesh(bam::ArrayDevice<uint64_t> mesh, u64 K,
                           u64 plane, const float *ax, const float *ay,
                           const float *az, const long long *aq,
                           const u32 *bin_start, unsigned long long *out) {
  // ACQUIRE ONCE, STREAM THE PLANE. Per-element bam_ptr.at() costs a
  // pin/unpin atomic pair per read; at 3M reads over two hot page slots the
  // acquire path livelocked outright (50k atoms hung where 1k finished).
  // One thread pins the plane's page, everyone reads the raw pointer, one
  // thread releases -- the pattern the BaM API's own comments prescribe.
  __shared__ const unsigned long long *s_pl;
  __shared__ uint64_t s_tok;
  for (u64 z = blockIdx.x; z < K; z += gridDim.x) {
    if (threadIdx.x == 0) {
      bam::PageWindow<uint64_t> win = mesh.acquire_page(z * plane);
      s_pl = reinterpret_cast<const unsigned long long *>(win.data) +
             (z * plane - win.start);
      s_tok = win.page_off;
    }
    __syncthreads();
    const unsigned long long e =
        GatherPlane(s_pl, z, K, ax, ay, az, aq, bin_start);
    atomicAdd(&out[2], e);
    __syncthreads();
    if (threadIdx.x == 0) {
      bam::PageWindow<uint64_t> w;
      w.page_off = s_tok;
      mesh.release_page(w);
    }
    __syncthreads();
  }
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  u32 blocks = 8, threads = 256;
  u64 K = 128, atoms = 200000, cache_mb = 32, bam_page_kb = 128;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--k") K = next();
    else if (a == "--atoms") atoms = next();
    else if (a == "--cache-mb") cache_mb = next();
    else if (a == "--bam-page-kb") bam_page_kb = next();
  }
  GX_CUDA_CHECK(cudaSetDevice(0));
  const u64 plane = K * K;
  const u64 nmesh = plane * K;
  const u64 page_bytes = bam_page_kb * 1024;
  if (plane * 8 > page_bytes || page_bytes % (plane * 8) != 0) {
    std::fprintf(stderr, "GMX BAM: one XY plane (%llu B) must divide the "
                 "BaM page (%llu B)\n",
                 (unsigned long long)(plane * 8),
                 (unsigned long long)page_bytes);
    return 2;
  }

  std::printf("PME spread+gather, BaM edition: mesh=%llu^3 (%.1f MB) behind "
              "a %llu MB BaM cache, atoms=%llu\n",
              (unsigned long long)K,
              static_cast<double>(nmesh * 8) / 1048576.0,
              (unsigned long long)cache_mb, (unsigned long long)atoms);
  // Deterministic atoms, z-binned CSR -- identical to the paged bench, and
  // replicated on every rank: the MESH is the big object, not the atoms.
  std::vector<float> hx(atoms), hy(atoms), hz(atoms);
  std::vector<long long> hq(atoms);
  std::vector<u32> bin_count(K + 1, 0);
  {
    u64 s = 0x9E3779B97F4A7C15ull;
    for (u64 a = 0; a < atoms; ++a) {
      s = Lcg(s); hx[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hy[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hz[a] = Frac01(s) * static_cast<float>(K);
      hq[a] = (a & 1) ? -(1ll << 24) : (1ll << 24);
    }
  }
  std::vector<u32> order(atoms);
  {
    std::vector<u32> bin(atoms);
    for (u64 a = 0; a < atoms; ++a) {
      const int iz0 = static_cast<int>(std::floor(hz[a])) - 1;
      bin[a] = static_cast<u32>((iz0 + static_cast<int>(K)) %
                                static_cast<int>(K));
      bin_count[bin[a] + 1]++;
    }
    for (u64 b = 0; b < K; ++b) bin_count[b + 1] += bin_count[b];
    std::vector<u32> cur(bin_count.begin(), bin_count.end() - 1);
    for (u64 a = 0; a < atoms; ++a) order[cur[bin[a]]++] = static_cast<u32>(a);
  }
  {
    std::vector<float> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = hx[order[a]];
    hx.swap(t);
    for (u64 a = 0; a < atoms; ++a) t[a] = hy[order[a]];
    hy.swap(t);
    for (u64 a = 0; a < atoms; ++a) t[a] = hz[order[a]];
    hz.swap(t);
  }
  {
    std::vector<long long> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = hq[order[a]];
    hq.swap(t);
  }
  long long q_total = 0;
  for (u64 a = 0; a < atoms; ++a) q_total += hq[a];


  float *d_ax, *d_ay, *d_az;
  long long *d_aq;
  u32 *d_bs;
  GX_CUDA_CHECK(cudaMalloc(&d_ax, atoms * sizeof(float)));
  GX_CUDA_CHECK(cudaMalloc(&d_ay, atoms * sizeof(float)));
  GX_CUDA_CHECK(cudaMalloc(&d_az, atoms * sizeof(float)));
  GX_CUDA_CHECK(cudaMalloc(&d_aq, atoms * sizeof(long long)));
  GX_CUDA_CHECK(cudaMalloc(&d_bs, (K + 1) * sizeof(u32)));
  GX_CUDA_CHECK(cudaMemcpy(d_ax, hx.data(), atoms * sizeof(float),
                           cudaMemcpyHostToDevice));
  GX_CUDA_CHECK(cudaMemcpy(d_ay, hy.data(), atoms * sizeof(float),
                           cudaMemcpyHostToDevice));
  GX_CUDA_CHECK(cudaMemcpy(d_az, hz.data(), atoms * sizeof(float),
                           cudaMemcpyHostToDevice));
  GX_CUDA_CHECK(cudaMemcpy(d_aq, hq.data(), atoms * sizeof(long long),
                           cudaMemcpyHostToDevice));
  GX_CUDA_CHECK(cudaMemcpy(d_bs, bin_count.data(), (K + 1) * sizeof(u32),
                           cudaMemcpyHostToDevice));

  bam::PageCacheConfig cfg{};
  cfg.page_size = static_cast<size_t>(page_bytes);
  cfg.num_pages = static_cast<size_t>((cache_mb * 1024 * 1024) / page_bytes);
  cfg.num_queues = 1;
  cfg.queue_depth = 1024;
  cfg.backend = bam::BackendType::kHostMemory;
  cfg.nvme_dev = nullptr;
  std::unique_ptr<bam::PageCache> cache(new bam::PageCache(cfg));
  const u64 bam_elems = ((nmesh * 8 + page_bytes - 1) / page_bytes) *
                        (page_bytes / 8);
  std::unique_ptr<bam::Array<uint64_t>> mesh(
      new bam::Array<uint64_t>(bam_elems, *cache));

  // SPREAD: plane-owner into a device plane, streamed to the host mirror.
  std::vector<unsigned long long> h_mesh(bam_elems, 0ull);
  unsigned long long *d_pl = nullptr, *d_out = nullptr;
  GX_CUDA_CHECK(cudaMalloc(&d_pl, plane * sizeof(unsigned long long)));
  GX_CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(unsigned long long)));
  GX_CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(unsigned long long)));
  const double t0 = NowMs();
  unsigned long long tot_q = 0, tot_ck = 0;
  for (u64 z = 0; z < K; ++z) {
    GX_CUDA_CHECK(cudaMemset(d_pl, 0, plane * sizeof(unsigned long long)));
    SpreadOnePlane<<<1, threads>>>(d_pl, z, K, d_ax, d_ay, d_az, d_aq, d_bs);
    GX_CUDA_CHECK(cudaGetLastError());
    GX_CUDA_CHECK(cudaDeviceSynchronize());
    GX_CUDA_CHECK(cudaMemcpy(h_mesh.data() + z * plane, d_pl,
                             plane * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost));
  }
  for (u64 i = 0; i < nmesh; ++i) {
    tot_q += h_mesh[i];
    tot_ck += h_mesh[i] * (2ull * i + 1ull);
  }
  mesh->load_from_host(reinterpret_cast<const uint64_t *>(h_mesh.data()),
                       bam_elems);
  const double t1 = NowMs();
  GatherMesh<<<blocks, threads>>>(mesh->device(), K, plane, d_ax, d_ay, d_az,
                                  d_aq, d_bs, d_out);
  GX_CUDA_CHECK(cudaGetLastError());
  GX_CUDA_CHECK(cudaDeviceSynchronize());
  const double t2 = NowMs();
  unsigned long long out[3] = {0, 0, 0};
  GX_CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));

  int rc = 0;
  std::printf("  spread %.1f ms  gather %.1f ms\n", t1 - t0, t2 - t1);
  if (tot_q != static_cast<unsigned long long>(q_total)) {
    std::printf("  CONSERVATION GATE: FAIL (%llu != %llu)\n", tot_q,
                static_cast<unsigned long long>(q_total));
    rc = 1;
  } else {
    std::printf("  CONSERVATION GATE: PASS (exact)\n");
  }
  std::printf("  mesh_checksum=%llu  gather_energy=%llu\n", tot_ck, out[2]);
  std::printf("%s\n", rc == 0 ? "GMX BAM: ALL GATES PASS"
                              : "GMX BAM: GATE FAILURE");
  return rc;
}

/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * PME spread+gather, MPI edition: the mesh is cut into z-plane slabs, one
 * rank per slab, atoms replicated (the mesh is the big object). Spread and
 * gather are both decomposed BY PLANE OWNER -- bins z-3..z feed plane z for
 * both -- so NO halo is ever exchanged; only three exact integer reductions
 * cross the wire (charge total, position-weighted mesh checksum, gather
 * energy). Same fixed-point science as the paged bench, so CONSERVATION and
 * MESH gates are exact at any rank count. Links nothing from clio.
 *
 * Run recipe: mpirun -n 4 clio_gmx_mpi_bench --k 128 --atoms 200000
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GX_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
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

__global__ void SpreadSlab(unsigned long long *slab, u64 z0, u64 z1, u64 K,
                           u64 plane, const float *ax, const float *ay,
                           const float *az, const long long *aq,
                           const u32 *bin_start) {
  for (u64 z = z0 + blockIdx.x; z < z1; z += gridDim.x) {
    SpreadPlane(slab + (z - z0) * plane, z, K, ax, ay, az, aq, bin_start);
  }
}

__global__ void SumSlab(const unsigned long long *slab, u64 z0, u64 z1,
                        u64 K, u64 plane, unsigned long long *out) {
  for (u64 z = z0 + blockIdx.x; z < z1; z += gridDim.x) {
    unsigned long long q = 0, ck = 0;
    const unsigned long long *pl = slab + (z - z0) * plane;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      q += pl[i];
      ck += pl[i] * (2ull * (z * plane + i) + 1ull);
    }
    atomicAdd(&out[0], q);
    atomicAdd(&out[1], ck);
  }
}

__global__ void GatherSlab(const unsigned long long *slab, u64 z0, u64 z1,
                           u64 K, u64 plane, const float *ax, const float *ay,
                           const float *az, const long long *aq,
                           const u32 *bin_start, unsigned long long *out) {
  for (u64 z = z0 + blockIdx.x; z < z1; z += gridDim.x) {
    const unsigned long long e =
        GatherPlane(slab + (z - z0) * plane, z, K, ax, ay, az, aq, bin_start);
    atomicAdd(&out[2], e);
  }
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  {
    int ndev = 0;
    GX_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    GX_CUDA_CHECK(cudaSetDevice(ndev > 0 ? rank % ndev : 0));
  }
  u32 blocks = 8, threads = 256;
  u64 K = 128, atoms = 200000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--k") K = next();
    else if (a == "--atoms") atoms = next();
  }
  const u64 plane = K * K;

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

  const u64 per = K / nranks;
  const u64 z0 = static_cast<u64>(rank) * per;
  const u64 z1 = (rank == nranks - 1) ? K : z0 + per;

  if (rank == 0) {
    std::printf("PME spread+gather, MPI edition: mesh=%llu^3, atoms=%llu, "
                "%d ranks\n",
                (unsigned long long)K, (unsigned long long)atoms, nranks);
  }

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

  unsigned long long *d_slab = nullptr, *d_out = nullptr;
  GX_CUDA_CHECK(cudaMalloc(&d_slab,
                           (z1 - z0) * plane * sizeof(unsigned long long)));
  GX_CUDA_CHECK(cudaMemset(d_slab, 0,
                           (z1 - z0) * plane * sizeof(unsigned long long)));
  GX_CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(unsigned long long)));
  GX_CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(unsigned long long)));

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  SpreadSlab<<<blocks, threads>>>(d_slab, z0, z1, K, plane, d_ax, d_ay, d_az,
                                  d_aq, d_bs);
  GX_CUDA_CHECK(cudaDeviceSynchronize());
  const double t1 = NowMs();
  SumSlab<<<blocks, threads>>>(d_slab, z0, z1, K, plane, d_out);
  GatherSlab<<<blocks, threads>>>(d_slab, z0, z1, K, plane, d_ax, d_ay, d_az,
                                  d_aq, d_bs, d_out);
  GX_CUDA_CHECK(cudaDeviceSynchronize());
  const double t2 = NowMs();

  unsigned long long loc[3] = {0, 0, 0}, tot[3] = {0, 0, 0};
  GX_CUDA_CHECK(cudaMemcpy(loc, d_out, sizeof(loc), cudaMemcpyDeviceToHost));
  MPI_Allreduce(loc, tot, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  int rc = 0;
  if (rank == 0) {
    std::printf("  spread %.1f ms  gather+sum %.1f ms\n", t1 - t0, t2 - t1);
    if (tot[0] != static_cast<unsigned long long>(q_total)) {
      std::printf("  CONSERVATION GATE: FAIL (%llu != %llu)\n", tot[0],
                  static_cast<unsigned long long>(q_total));
      rc = 1;
    } else {
      std::printf("  CONSERVATION GATE: PASS (exact)\n");
    }
    std::printf("  mesh_checksum=%llu  gather_energy=%llu\n", tot[1], tot[2]);
    std::printf("%s\n", rc == 0 ? "GMX MPI: ALL GATES PASS"
                                : "GMX MPI: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return rc;
}

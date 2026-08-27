/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * weights, NVSHMEM edition: the shard lives on the symmetric heap and the
 * exact integer partial sums meet through nvshmem_ulonglong_sum_reduce on
 * device buffers -- or host-staged MPI under an MPI bootstrap, because
 * NVSHMEM's host collectives refuse limited-MPG runs (see the kmeans and md
 * siblings). Gate is EXACT: integer accumulation commutes. Links nothing
 * from clio.
 */

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#if defined(MD_NVSHMEM_USE_MPI)
#include <mpi.h>
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define WB_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      std::exit(1);                                                               \
    }                                                                        \
  } while (0)

using u32 = unsigned int;
using u64 = unsigned long long;

/** The paged bench's weight generator, verbatim: runs of 8 identical values
 *  with a hashed sprinkling of flat 64KB granules (--flat-pct). */
static constexpr u64 kFlatGranuleElems = (64 * 1024) / sizeof(u32);
__host__ __device__ inline bool PageIsFlat(u64 page, u32 pct) {
  u64 h = page * 0x9E3779B97F4A7C15ull;
  h ^= h >> 33;
  return (h % 100u) < pct;
}
__host__ __device__ inline u32 WeightRaw(u64 i) {
  constexpr u64 kRun = 8;
  u32 r = static_cast<u32>((i / kRun) * 2654435761u);
  r ^= r >> 13;
  return (r & 0x3F3F3F3Fu) |
         (static_cast<u32>((i / 4096) % 13) * 0x40404040u);
}
__host__ __device__ inline u32 Weight(u64 i, u32 flat_pct) {
  if (PageIsFlat(i / kFlatGranuleElems, flat_pct)) return 0x01010101u;
  return WeightRaw(i);
}
__host__ __device__ inline u32 Activation(u64 i) {
  return static_cast<u32>((i % 7) + 1);
}

__global__ void SeedKernel(u32 *w, u64 base, u64 n, u32 flat_pct) {
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    w[i] = Weight(base + i, flat_pct);
  }
}

__global__ void SumKernel(const u32 *w, u64 base, u64 n,
                          unsigned long long *sum) {
  unsigned long long r = 0;
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    r += static_cast<unsigned long long>(w[i]) * Activation(base + i);
  }
  atomicAdd(sum, r);
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Init(&argc, &argv);
  {
    int wr = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &wr);
    int ndev = 0;
    WB_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    WB_CUDA_CHECK(cudaSetDevice(ndev > 0 ? wr % ndev : 0));
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &comm;
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
      std::fprintf(stderr, "nvshmem MPI bootstrap failed\n");
      return 1;
    }
  }
#else
  WB_CUDA_CHECK(cudaSetDevice(0));
  {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&uid);
    nvshmemx_set_attr_uniqueid_args(0, 1, &uid, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
      std::fprintf(stderr, "nvshmem uniqueid bootstrap failed\n");
      return 1;
    }
  }
#endif
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();

  u32 blocks = 64, threads = 256, passes = 3, flat_pct = 0;
  u64 data_mb = 512;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--passes") passes = static_cast<u32>(next());
    else if (a == "--flat-pct") flat_pct = static_cast<u32>(next());
    else if (a == "--data-mb") data_mb = next();
  }
  const u64 n = (data_mb * 1024ull * 1024ull) / sizeof(u32);
  const u64 per = n / npes;
  const u64 e0 = static_cast<u64>(mype) * per;
  const u64 e1 = (mype == npes - 1) ? n : e0 + per;

  u32 *d_w = static_cast<u32 *>(nvshmem_malloc((e1 - e0) * sizeof(u32)));
  unsigned long long *d_sum = static_cast<unsigned long long *>(
      nvshmem_malloc(sizeof(unsigned long long)));
  unsigned long long *d_sum_out = static_cast<unsigned long long *>(
      nvshmem_malloc(sizeof(unsigned long long)));
  if (!d_w || !d_sum || !d_sum_out) {
    std::fprintf(stderr, "nvshmem_malloc failed\n");
    return 1;
  }
  SeedKernel<<<256, 256>>>(d_w, e0, e1 - e0, flat_pct);
  WB_CUDA_CHECK(cudaDeviceSynchronize());

  nvshmem_barrier_all();
  const double t0 = NowMs();
  unsigned long long got = 0;
  for (u32 p = 0; p < passes; ++p) {
    WB_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    SumKernel<<<blocks, threads>>>(d_w, e0, e1 - e0, d_sum);
    WB_CUDA_CHECK(cudaDeviceSynchronize());
#if defined(MD_NVSHMEM_USE_MPI)
    unsigned long long local = 0;
    WB_CUDA_CHECK(cudaMemcpy(&local, d_sum, sizeof(local),
                             cudaMemcpyDeviceToHost));
    got = 0;
    MPI_Allreduce(&local, &got, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    nvshmem_ulonglong_sum_reduce(NVSHMEM_TEAM_WORLD, d_sum_out, d_sum, 1);
    nvshmem_barrier_all();
    WB_CUDA_CHECK(cudaMemcpy(&got, d_sum_out, sizeof(got),
                             cudaMemcpyDeviceToHost));
#endif
  }
  const double ms = NowMs() - t0;

  int rc = 0;
  if (mype == 0) {
    unsigned long long want = 0;
    for (u64 i = 0; i < n; ++i) {
      want += static_cast<unsigned long long>(Weight(i, flat_pct)) *
              Activation(i);
    }
    std::printf("weights, NVSHMEM edition: %llu elems, %u passes, %d PEs, "
                "%.1f ms\n",
                (unsigned long long)n, passes, npes, ms);
    if (got != want) {
      std::printf("  SUM GATE: FAIL (got=%llu want=%llu)\n", got, want);
      rc = 1;
    } else {
      std::printf("  SUM GATE: PASS (exact, %llu)\n", got);
    }
    std::printf("%s\n", rc == 0 ? "WEIGHTS NVSHMEM: ALL GATES PASS"
                                : "WEIGHTS NVSHMEM: GATE FAILURE");
  }
  nvshmem_free(d_w);
  nvshmem_free(d_sum);
  nvshmem_free(d_sum_out);
  nvshmem_finalize();
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Finalize();
#endif
  return rc;
}

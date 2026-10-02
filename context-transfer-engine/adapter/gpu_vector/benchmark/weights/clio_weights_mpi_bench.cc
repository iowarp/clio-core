/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * weights, MPI edition: the scale-out baseline for the paged weights bench.
 *
 * SAME SCIENCE, DIFFERENT DATA PLANE: the inference-shaped integer weighted
 * sum over the whole model, sharded by rank into plain device memory, the
 * partial sums combined with MPI_Allreduce. Integer accumulation commutes,
 * so the gate is EXACT: got must equal the host-computed want, bit for bit,
 * at any rank count. Links nothing from clio.
 *
 * Run recipe: mpirun -n 4 clio_weights_mpi_bench --data-mb 512 --passes 3
 */

#include <mpi.h>
#include <cuda_runtime.h>
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
      MPI_Abort(MPI_COMM_WORLD, 1);                                                               \
    }                                                                        \
  } while (0)

using u32 = unsigned int;
using u64 = unsigned long long;

// THE GENERATOR IS SHARED NOW, NOT COPIED.
//
// What was here claimed to be "the paged bench's weight generator, verbatim"
// and was not: its PageIsFlat used a different hash (64-bit 0x9E37... >> 33
// against the paged bench's 32-bit 2654435761 >> 15). At --flat-pct 25 the
// two disagreed about 37% of pages, so this baseline and the thing it is a
// baseline FOR were compressing different datasets, and their stored-size
// and residency numbers were not comparable.
//
// It went unnoticed because the default is --flat-pct 0, where both hashes
// select nothing and the data is identical. The divergence appears only in
// the compression sweep -- which is the measurement this benchmark exists to
// make.
#include "weights_math.h"

using clio_wt::Activation;
using clio_wt::PageIsFlat;
using clio_wt::Weight;


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
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  {
    int ndev = 0;
    WB_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    WB_CUDA_CHECK(cudaSetDevice(ndev > 0 ? rank % ndev : 0));
  }
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
  const u64 per = n / nranks;
  const u64 e0 = static_cast<u64>(rank) * per;
  const u64 e1 = (rank == nranks - 1) ? n : e0 + per;

  u32 *d_w = nullptr;
  unsigned long long *d_sum = nullptr;
  WB_CUDA_CHECK(cudaMalloc(&d_w, (e1 - e0) * sizeof(u32)));
  WB_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
  SeedKernel<<<256, 256>>>(d_w, e0, e1 - e0, flat_pct);
  WB_CUDA_CHECK(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  unsigned long long got = 0;
  for (u32 p = 0; p < passes; ++p) {
    WB_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    SumKernel<<<blocks, threads>>>(d_w, e0, e1 - e0, d_sum);
    WB_CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long local = 0;
    WB_CUDA_CHECK(cudaMemcpy(&local, d_sum, sizeof(local),
                             cudaMemcpyDeviceToHost));
    got = 0;
    MPI_Allreduce(&local, &got, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
  }
  const double ms = NowMs() - t0;

  int rc = 0;
  if (rank == 0) {
    unsigned long long want = 0;
    for (u64 i = 0; i < n; ++i) {
      want += static_cast<unsigned long long>(Weight(i, flat_pct)) *
              Activation(i);
    }
    std::printf("weights, MPI edition: %llu elems, %u passes, %d ranks, "
                "%.1f ms\n",
                (unsigned long long)n, passes, nranks, ms);
    if (got != want) {
      std::printf("  SUM GATE: FAIL (got=%llu want=%llu)\n", got, want);
      rc = 1;
    } else {
      std::printf("  SUM GATE: PASS (exact, %llu)\n", got);
    }
    std::printf("%s\n", rc == 0 ? "WEIGHTS MPI: ALL GATES PASS"
                                : "WEIGHTS MPI: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return rc;
}

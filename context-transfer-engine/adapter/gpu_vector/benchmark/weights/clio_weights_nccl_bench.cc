/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * weight streaming, NCCL edition: the GPU-direct scale-out baseline.
 *
 * SAME SCIENCE AS THE MPI EDITION. Each rank seeds and sums its contiguous
 * shard of the deterministic weight/activation product, and the partial sums
 * are combined so every rank ends with the global total. Integer
 * accumulation commutes, so the SUM GATE stays EXACT regardless of how the
 * reduction is ordered -- which is what makes this workload a clean read on
 * the wire and nothing else.
 *
 * WHAT NCCL CHANGES IS THE BOUNCE. The MPI edition copies the per-rank
 * scalar down to the host every pass and reduces there. ncclAllReduce
 * reduces the DEVICE scalar in place, so the value never leaves the GPU
 * until the gate reads it once at the end.

 * MPI IS LINKED FOR BOOTSTRAP ONLY: rank 0 calls ncclGetUniqueId and the id
 * is broadcast with MPI_Bcast before ncclCommInitRank. No MPI collective
 * carries workload data.
 *
 * MULTI-NODE NEEDS A NET PLUGIN. NCCL's built-in transports are NVLink/PCIe
 * and IB verbs; on a Slingshot machine it finds neither between nodes and
 * falls back to TCP sockets unless aws-ofi-nccl is loaded (module
 * aws-ofi-nccl, which sets NCCL_NET_PLUGIN=ofi).
 *
 * Run recipe:
 *   srun --mpi=pmi2 -n 2 --ntasks-per-node=1 clio_weights_nccl_bench --data-mb 512
 */

#include <mpi.h>
#include <nccl.h>
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

#define WB_NCCL_CHECK(x)                                                     \
  do {                                                                       \
    ncclResult_t _r = (x);                                                   \
    if (_r != ncclSuccess) {                                                 \
      std::fprintf(stderr, "NCCL error %s at %s:%d\n",                       \
                   ncclGetErrorString(_r), __FILE__, __LINE__);              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
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
  // BOOTSTRAP: the only place MPI carries anything. NCCL has no launcher of
  // its own -- rank 0 makes the unique id, MPI_Bcast distributes it, and
  // ncclCommInitRank is collective so every rank must reach it.
  // ONE GPU PER RANK, checked BEFORE ncclCommInitRank rather than letting it
  // die inside NCCL with "invalid usage" and an MPI_ABORT of 1. Exit 77: the
  // ctest entries carry SKIP_RETURN_CODE 77, so a single-GPU host reports
  // SKIP instead of a permanent failure -- the md edition has done this from
  // the start, and these five were red on every 1-GPU machine without it.
  int guard_ndev = 0;
  cudaGetDeviceCount(&guard_ndev);
  if (nranks > 1 && guard_ndev < nranks) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "NCCL needs one GPU per rank: %d ranks but %d visible "
                   "device(s); skipping (77).\n", nranks, guard_ndev);
    }
    MPI_Finalize();
    return 77;
  }
  ncclUniqueId nccl_id;
  if (rank == 0) WB_NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  WB_NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

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

  // NCCL WARMUP -- deliberately OUTSIDE the timed region.
  //
  // NCCL brings up its channels, starts its proxy threads and registers
  // buffers LAZILY, on the FIRST collective. The MPI_Barrier below warms
  // MPI's connections, so the MPI edition's setup already falls outside ITS
  // timer -- but a barrier does nothing for NCCL, and without this warmup
  // the first collective inside the timed loop carries all of that setup.
  // That is a property of where the timer starts, not of the substrate.
  //
  // MEASURED on weights (3 passes, 2 nodes, cray-mpich + aws-ofi-nccl):
  // a FLAT ~78 ms per run at EVERY --blocks rung -- 10% of the blocks=32
  // cell and 72% of the blocks=256 one. It made NCCL read 1.10x -> 1.70x
  // slower than MPI across the blocks sweep while the steady-state rates
  // were within a few percent; the ratio only "widened" because the fixed
  // cost stayed put while the compute shrank. kmeans and lbann hid it
  // entirely -- against 40-180 s runs of hundreds-of-MB collectives it is
  // noise, and there NCCL comes out slightly AHEAD of MPI.
  {
    unsigned long long *d_nccl_warm = nullptr;
    WB_CUDA_CHECK(cudaMalloc(&d_nccl_warm, sizeof(unsigned long long)));
    WB_CUDA_CHECK(cudaMemset(d_nccl_warm, 0, sizeof(unsigned long long)));
    WB_NCCL_CHECK(ncclAllReduce(d_nccl_warm, d_nccl_warm, 1, ncclUint64, ncclSum,
                      comm, 0));
    WB_CUDA_CHECK(cudaStreamSynchronize(0));
    WB_CUDA_CHECK(cudaFree(d_nccl_warm));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  unsigned long long got = 0;
  for (u32 p = 0; p < passes; ++p) {
    WB_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    SumKernel<<<blocks, threads>>>(d_w, e0, e1 - e0, d_sum);
    WB_CUDA_CHECK(cudaDeviceSynchronize());
    // THE DATA PLANE UNDER TEST: the device scalar is reduced in place, with
    // no host bounce. Only the final value is read back, once, below.
    WB_NCCL_CHECK(ncclAllReduce(d_sum, d_sum, 1, ncclUint64, ncclSum, comm,
                                0));
  }
  // NCCL is asynchronous on its stream; the reduced total has to land before
  // the timer stops and the gate reads it.
  WB_CUDA_CHECK(cudaStreamSynchronize(0));
  WB_CUDA_CHECK(cudaMemcpy(&got, d_sum, sizeof(got), cudaMemcpyDeviceToHost));
  const double ms = NowMs() - t0;

  int rc = 0;
  if (rank == 0) {
    unsigned long long want = 0;
    for (u64 i = 0; i < n; ++i) {
      want += static_cast<unsigned long long>(Weight(i, flat_pct)) *
              Activation(i);
    }
    std::printf("weights, NCCL edition: %llu elems, %u passes, %d ranks, "
                "%.1f ms\n",
                (unsigned long long)n, passes, nranks, ms);
    if (got != want) {
      std::printf("  SUM GATE: FAIL (got=%llu want=%llu)\n", got, want);
      rc = 1;
    } else {
      std::printf("  SUM GATE: PASS (exact, %llu)\n", got);
    }
    std::printf("%s\n", rc == 0 ? "WEIGHTS NCCL: ALL GATES PASS"
                                : "WEIGHTS NCCL: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  WB_NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return rc;
}

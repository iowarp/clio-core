/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * k-means, NCCL edition: the GPU-direct scale-out baseline for paged k-means.
 *
 * SAME SCIENCE AS THE MPI EDITION, DIFFERENT WIRE. The generator, the
 * assignment kernel and the centroid update are the same shared kmeans_math.h
 * the paged bench and the MPI baseline use, and the decomposition is the same
 * data-parallel Lloyd: each rank owns a contiguous shard of the deterministic
 * point cloud, assigns locally, and the per-cluster sums and counts are
 * combined so every rank computes the identical update.
 *
 * WHAT NCCL CHANGES IS THE BOUNCE. The MPI edition stages both reductions
 * through the host because that MPI reports no CUDA support -- device to
 * host, allreduce, host to device, every iteration. ncclAllReduce takes the
 * DEVICE pointers directly, so the partial sums never touch host memory.
 * That difference is the entire reason this baseline exists; everything else
 * is held constant so the two are comparable.
 *
 * MPI IS STILL LINKED, FOR BOOTSTRAP ONLY. NCCL has no launcher of its own:
 * rank 0 calls ncclGetUniqueId and the id is broadcast with MPI_Bcast before
 * ncclCommInitRank. No MPI collective carries workload data -- MPI_Bcast of
 * the 128-byte id and the final return code are the only ones in the file.
 * This mirrors clio_lammps_md_nccl_bench.cc, which bootstraps the same way.
 *
 * MULTI-NODE NEEDS A NET PLUGIN. NCCL's built-in transports are NVLink/PCIe
 * and IB verbs; on a Slingshot machine it finds neither between nodes and
 * falls back to TCP sockets unless aws-ofi-nccl is loaded (module
 * aws-ofi-nccl, which sets NCCL_NET_PLUGIN=ofi). Both work -- the plugin is
 * what makes the number worth quoting.
 *
 * GATES:
 *   COUNT   sum of cluster counts == total points, every iteration, exact --
 *           a lost or double-counted point cannot hide.
 *   CSUM    the final centroid checksum, printed always; --check-csum V
 *           compares against a reference with the documented relative
 *           tolerance -- atomicAdd + allreduce make float summation order
 *           layout-dependent, so bit equality is not expected between
 *           substrates.
 *
 * Run recipe:
 *   srun --mpi=pmi2 -n 2 --ntasks-per-node=1 clio_kmeans_nccl_bench \
 *        --data-mb 256 --iters 4
 */

#include <mpi.h>
#include <nccl.h>

#include <cuda_runtime.h>

// The science, shared with the paged bench and the SYCL baseline. Carries no
// clio dependency, so including it does not compromise "a baseline links
// nothing from clio".
#include "kmeans_math.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

#define KM_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                        \
  } while (0)

#define KM_NCCL_CHECK(x)                                                     \
  do {                                                                       \
    ncclResult_t _r = (x);                                                   \
    if (_r != ncclSuccess) {                                                 \
      std::fprintf(stderr, "NCCL error %s at %s:%d\n",                       \
                   ncclGetErrorString(_r), __FILE__, __LINE__);              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                        \
  } while (0)

// IDENTICAL to the paged bench and the SYCL baseline because it is the SAME
// CODE now, not a copy with a comment claiming so.
using clio_km::NearestCentroid;
using clio_km::PointVal;
using clio_km::UpdateCentroid;

__global__ void SeedKernel(float *pts, u64 base_idx, u64 n, u32 dims, u32 k) {
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    pts[i] = PointVal(base_idx + i, dims, k);
  }
}

/** The paged bench's assignment step, verbatim science. */
__global__ void AssignKernel(const float *pts, u64 npts, u32 dims, u32 k,
                             const float *cent, float *sums,
                             unsigned *counts) {
  for (u64 p = blockIdx.x * blockDim.x + threadIdx.x; p < npts;
       p += static_cast<u64>(gridDim.x) * blockDim.x) {
    const float *pt = pts + p * dims;
    const u32 bestk = NearestCentroid(pt, cent, dims, k);
    for (u32 i = 0; i < dims; ++i) {
      atomicAdd(&sums[bestk * dims + i], pt[i]);
    }
    atomicAdd(&counts[bestk], 1u);
  }
}

/** centroid = sum / count, leaving an empty cluster where it was. */
__global__ void UpdateKernel(float *cent, const float *sums,
                             const unsigned *counts, u32 dims, u32 k) {
  const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  UpdateCentroid(cent, sums, counts, dims, c);
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
    KM_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    KM_CUDA_CHECK(cudaSetDevice(ndev > 0 ? rank % ndev : 0));
  }

  // BOOTSTRAP: the only place MPI carries anything. ncclCommInitRank is
  // collective over the communicator, so every rank must reach it -- it is
  // built here, before the argument parse can early-return on --help (which
  // finalizes MPI and exits on every rank alike).
  ncclUniqueId nccl_id;
  if (rank == 0) KM_NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  KM_NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

  u32 blocks = 64, threads = 256, dims = 32, k = 16, iters = 4;
  u64 data_mb = 256;
  double check_csum = 0.0, check_tol = 1e-4;
  bool do_check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--dims") dims = static_cast<u32>(next());
    else if (a == "--clusters") k = static_cast<u32>(next());
    else if (a == "--iters") iters = static_cast<u32>(next());
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--check-csum" && i + 1 < argc) {
      check_csum = std::strtod(argv[++i], nullptr);
      do_check = true;
    } else if (a == "--check-tol" && i + 1 < argc) {
      check_tol = std::strtod(argv[++i], nullptr);
    } else if (a == "--help") {
      if (rank == 0) {
        std::printf("usage: srun -n N %s [--blocks N] [--threads N] "
                    "[--dims N] [--clusters N] [--iters N] [--data-mb N] "
                    "[--check-csum V] [--check-tol R]\n", argv[0]);
      }
      KM_NCCL_CHECK(ncclCommDestroy(comm));
      MPI_Finalize();
      return 0;
    }
  }

  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 npts = total_elems / dims;
  // Contiguous point shards; the LAST rank absorbs the remainder so every
  // point is owned exactly once.
  const u64 per = npts / nranks;
  const u64 p0 = static_cast<u64>(rank) * per;
  const u64 p1 = (rank == nranks - 1) ? npts : p0 + per;
  const u64 my_pts = p1 - p0;
  const u64 my_elems = my_pts * dims;

  if (rank == 0) {
    std::printf("k-means, NCCL edition\n"
                "  points=%llu dims=%u k=%u iters=%u  ranks=%d "
                "(%.1f MB/rank)\n",
                (unsigned long long)npts, dims, k, iters, nranks,
                static_cast<double>(my_elems * sizeof(float)) / 1048576.0);
  }

  float *d_pts = nullptr;
  KM_CUDA_CHECK(cudaMalloc(&d_pts, my_elems * sizeof(float)));
  SeedKernel<<<256, 256>>>(d_pts, p0 * dims, my_elems, dims, k);
  KM_CUDA_CHECK(cudaDeviceSynchronize());

  // Initial centroids: the first k points -- identical on every rank and to
  // the paged bench, computed straight from the generator.
  std::vector<float> h_cent(static_cast<size_t>(k) * dims);
  for (u64 i = 0; i < static_cast<u64>(k) * dims; ++i) {
    h_cent[i] = PointVal(i, dims, k);
  }
  float *d_cent = nullptr, *d_sums = nullptr;
  unsigned *d_counts = nullptr;
  KM_CUDA_CHECK(cudaMalloc(&d_cent, h_cent.size() * sizeof(float)));
  KM_CUDA_CHECK(cudaMalloc(&d_sums, h_cent.size() * sizeof(float)));
  KM_CUDA_CHECK(cudaMalloc(&d_counts, k * sizeof(unsigned)));
  KM_CUDA_CHECK(cudaMemcpy(d_cent, h_cent.data(),
                           h_cent.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

  // Sized like d_sums; the NCCL edition never stages through it.
  std::vector<float> h_sums(h_cent.size());
  std::vector<unsigned> h_counts(k);
  int rc = 0;
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
    KM_CUDA_CHECK(cudaMalloc(&d_nccl_warm, sizeof(unsigned long long)));
    KM_CUDA_CHECK(cudaMemset(d_nccl_warm, 0, sizeof(unsigned long long)));
    KM_NCCL_CHECK(ncclAllReduce(d_nccl_warm, d_nccl_warm, 1, ncclUint64, ncclSum,
                      comm, 0));
    KM_CUDA_CHECK(cudaStreamSynchronize(0));
    KM_CUDA_CHECK(cudaFree(d_nccl_warm));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  for (u32 it = 0; it < iters; ++it) {
    KM_CUDA_CHECK(cudaMemset(d_sums, 0, h_cent.size() * sizeof(float)));
    KM_CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(unsigned)));
    AssignKernel<<<blocks, threads>>>(d_pts, my_pts, dims, k, d_cent, d_sums,
                                      d_counts);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
    // THE DATA PLANE UNDER TEST: both reductions run in place on the DEVICE
    // buffers. No host bounce -- that is the whole difference from the MPI
    // edition, which copies down, reduces, and copies back every iteration.
    // Grouped so the two collectives are issued as one network operation.
    KM_NCCL_CHECK(ncclGroupStart());
    KM_NCCL_CHECK(ncclAllReduce(d_sums, d_sums, h_sums.size(), ncclFloat,
                                ncclSum, comm, 0));
    KM_NCCL_CHECK(ncclAllReduce(d_counts, d_counts, k, ncclUint32, ncclSum,
                                comm, 0));
    KM_NCCL_CHECK(ncclGroupEnd());
    // NCCL is asynchronous on the stream it was given (the default stream
    // here); the counts are read on the host below, so it has to land first.
    KM_CUDA_CHECK(cudaStreamSynchronize(0));
    // Only the COUNT GATE needs a host copy, and only of the counts.
    KM_CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                             k * sizeof(unsigned), cudaMemcpyDeviceToHost));
    // COUNT GATE, exact: a lost or doubled point cannot hide in a float.
    u64 sum_counts = 0;
    for (u32 c = 0; c < k; ++c) sum_counts += h_counts[c];
    if (sum_counts != npts) {
      if (rank == 0) {
        std::printf("  COUNT GATE: FAIL at iter %u (%llu != %llu)\n", it,
                    (unsigned long long)sum_counts, (unsigned long long)npts);
      }
      rc = 1;
    }
    UpdateKernel<<<(k + 63) / 64, 64>>>(d_cent, d_sums, d_counts, dims, k);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double ms = NowMs() - t0;

  KM_CUDA_CHECK(cudaMemcpy(h_cent.data(), d_cent,
                           h_cent.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
  double csum = 0.0;
  for (float f : h_cent) csum += static_cast<double>(f);

  if (rank == 0) {
    if (rc == 0) std::printf("  COUNT GATE: PASS (all %u iterations)\n", iters);
    std::printf("  %u iters in %.1f ms  centroid_checksum=%.6f\n", iters, ms,
                csum);
    if (do_check) {
      const double rel = std::fabs(csum - check_csum) /
                         (std::fabs(check_csum) > 0 ? std::fabs(check_csum)
                                                    : 1.0);
      if (rel > check_tol) {
        std::printf("  CSUM GATE: FAIL (%.6f vs %.6f, rel %.2e > %.0e)\n",
                    csum, check_csum, rel, check_tol);
        rc = 1;
      } else {
        std::printf("  CSUM GATE: PASS (rel %.2e)\n", rel);
      }
    }
    std::printf("%s\n", rc == 0 ? "KMEANS NCCL: ALL GATES PASS"
                                : "KMEANS NCCL: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  KM_NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return rc;
}

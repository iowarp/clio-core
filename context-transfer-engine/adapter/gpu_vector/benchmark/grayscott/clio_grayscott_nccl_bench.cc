/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott reaction-diffusion, NCCL edition: the GPU-direct baseline.
 *
 * SAME SCIENCE AS THE MPI EDITION. The domain is cut into contiguous
 * z-slabs, each rank owns one plus a one-plane halo on each side, and the
 * halo is exchanged every step before the stencil runs.
 *
 * WHAT NCCL CHANGES IS THE BOUNCE. The MPI edition stages each halo plane
 * through host memory -- device to host, Sendrecv, host to device, twice per
 * field per step. ncclSend/ncclRecv move the plane straight between DEVICE
 * buffers, which is the entire point of the comparison.
 *
 * THE EXCHANGE IS STILL TWO PHASES, matching the MPI edition's two tagged
 * Sendrecvs: up-then-down, each phase its own NCCL group. NCCL has no tags,
 * so the group boundary is what keeps the two phases from matching each
 * other's messages.

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
 *   srun --mpi=cray_shasta -n 2 --ntasks-per-node=1 clio_grayscott_nccl_bench --data-mb 64 --steps 4
 */

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GS_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                        \
  } while (0)

#define GS_NCCL_CHECK(x)                                                     \
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

/** Initial condition, verbatim from the paged bench: v seeded in a centred
 *  cube, u elsewhere, so every substrate starts from the identical field. */
__host__ __device__ inline float InitU(u64 x, u64 y, u64 z, u64 nx, u64 ny,
                                       u64 nz) {
  const bool in = (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 &&
                   y < 2 * ny / 3 && z > nz / 3 && z < 2 * nz / 3);
  return in ? 0.5f : 1.0f;
}
__host__ __device__ inline float InitV(u64 x, u64 y, u64 z, u64 nx, u64 ny,
                                       u64 nz) {
  const bool in = (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 &&
                   y < 2 * ny / 3 && z > nz / 3 && z < 2 * nz / 3);
  return in ? 0.25f : 0.0f;
}

/**
 * One Gray-Scott step for local planes [1, nzl+1) of an extended slab
 * [halo_lo | own planes | halo_hi]. The formula is the paged bench's,
 * verbatim; `gz0` maps a local plane to its global z so the fixed global
 * boundary (z == 0, z == nz-1) is honoured no matter how the slab is cut.
 */
__global__ void StepKernel(const float *u, const float *v, float *un,
                           float *vn, u64 plane, u64 nx, u64 ny, u64 nzl,
                           u64 gz0, u64 nz, float Du, float Dv, float F,
                           float K, float dt) {
  for (u64 lz = 1 + blockIdx.x; lz < nzl + 1; lz += gridDim.x) {
    const u64 gzz = gz0 + lz - 1;
    const bool interior = (gzz > 0 && gzz + 1 < nz);
    const float *uz = u + lz * plane;
    const float *uzm = interior ? uz - plane : uz;
    const float *uzp = interior ? uz + plane : uz;
    const float *vz = v + lz * plane;
    const float *vzm = interior ? vz - plane : vz;
    const float *vzp = interior ? vz + plane : vz;
    float *unx = un + lz * plane;
    float *vnx = vn + lz * plane;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const u64 x = i % nx, y = i / nx;
      const float uu = uz[i];
      const float vv = vz[i];
      float lu, lv;
      if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
        lu = 0.0f; lv = 0.0f;
      } else {
        lu = uz[i - 1] + uz[i + 1] + uz[i - nx] + uz[i + nx] + uzm[i] +
             uzp[i] - 6.0f * uu;
        lv = vz[i - 1] + vz[i + 1] + vz[i - nx] + vz[i + nx] + vzm[i] +
             vzp[i] - 6.0f * vv;
      }
      const float uvv = uu * vv * vv;
      unx[i] = uu + dt * (Du * lu - uvv + F * (1.0f - uu));
      vnx[i] = vv + dt * (Dv * lv + uvv - (F + K) * vv);
    }
  }
}

__global__ void SeedSlab(float *u, float *v, u64 plane, u64 nx, u64 ny,
                         u64 nzl, u64 gz0, u64 nz) {
  for (u64 lz = 1 + blockIdx.x; lz < nzl + 1; lz += gridDim.x) {
    const u64 gzz = gz0 + lz - 1;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      u[lz * plane + i] = InitU(i % nx, i / nx, gzz, nx, ny, nz);
      v[lz * plane + i] = InitV(i % nx, i / nx, gzz, nx, ny, nz);
    }
  }
}

__global__ void SumV(const float *v, u64 plane, u64 nzl, double *out) {
  double acc = 0.0;
  for (u64 lz = 1 + blockIdx.x; lz < nzl + 1; lz += gridDim.x) {
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      acc += static_cast<double>(v[lz * plane + i]);
    }
  }
  atomicAdd(out, acc);
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
    GS_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    GS_CUDA_CHECK(cudaSetDevice(ndev > 0 ? rank % ndev : 0));
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
  if (rank == 0) GS_NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  GS_NCCL_CHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));


  u32 blocks = 64, threads = 256, steps = 4;
  u64 page_kb = 1024, data_mb = 2048;
  float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 1.0f;
  double check_csum = 0.0, check_tol = 1e-3;
  bool do_check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--steps") steps = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--check-csum" && i + 1 < argc) {
      check_csum = std::strtod(argv[++i], nullptr);
      do_check = true;
    } else if (a == "--check-tol" && i + 1 < argc) {
      check_tol = std::strtod(argv[++i], nullptr);
    }
  }
  // SAME GRID DERIVATION as the paged bench: one page is one XY plane, four
  // regions share the dataset budget -- so checksums are comparable.
  const u64 plane = (page_kb * 1024) / sizeof(float);
  u64 nx = 1, ny = plane;
  while (nx * 2 <= ny) { nx *= 2; ny /= 2; }
  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 nz = total_elems / (4 * plane);

  const u64 per = nz / nranks;
  const u64 gz0 = static_cast<u64>(rank) * per;
  const u64 gz1 = (rank == nranks - 1) ? nz : gz0 + per;
  const u64 nzl = gz1 - gz0;
  const u64 ext = (nzl + 2) * plane;

  if (rank == 0) {
    std::printf("Gray-Scott, NCCL edition: %llux%llux%llu, %u steps, %d "
                "ranks\n",
                (unsigned long long)nx, (unsigned long long)ny,
                (unsigned long long)nz, steps, nranks);
  }

  float *u, *v, *un, *vn;
  GS_CUDA_CHECK(cudaMalloc(&u, ext * sizeof(float)));
  GS_CUDA_CHECK(cudaMalloc(&v, ext * sizeof(float)));
  GS_CUDA_CHECK(cudaMalloc(&un, ext * sizeof(float)));
  GS_CUDA_CHECK(cudaMalloc(&vn, ext * sizeof(float)));
  SeedSlab<<<blocks, threads>>>(u, v, plane, nx, ny, nzl, gz0, nz);
  SeedSlab<<<blocks, threads>>>(un, vn, plane, nx, ny, nzl, gz0, nz);
  GS_CUDA_CHECK(cudaDeviceSynchronize());

  // -1 rather than MPI_PROC_NULL: NCCL has no null peer, so an edge rank
  // simply issues no send/recv in that direction.
  const int up = (rank + 1 < nranks) ? rank + 1 : -1;
  const int dn = (rank > 0) ? rank - 1 : -1;
  auto exchange = [&](float *fld) {
    // THE DATA PLANE UNDER TEST. Phase 1: my TOP own plane goes up, my LOW
    // halo comes from below. Phase 2: the reverse. Device pointers straight
    // onto the wire -- no host staging anywhere in here.
    GS_NCCL_CHECK(ncclGroupStart());
    if (up >= 0) {
      GS_NCCL_CHECK(ncclSend(fld + nzl * plane, plane, ncclFloat, up, comm,
                             0));
    }
    if (dn >= 0) {
      GS_NCCL_CHECK(ncclRecv(fld, plane, ncclFloat, dn, comm, 0));
    }
    GS_NCCL_CHECK(ncclGroupEnd());
    GS_NCCL_CHECK(ncclGroupStart());
    if (dn >= 0) {
      GS_NCCL_CHECK(ncclSend(fld + plane, plane, ncclFloat, dn, comm, 0));
    }
    if (up >= 0) {
      GS_NCCL_CHECK(ncclRecv(fld + (nzl + 1) * plane, plane, ncclFloat, up,
                             comm, 0));
    }
    GS_NCCL_CHECK(ncclGroupEnd());
    // The stencil kernel below reads the halo, so the transfers must land.
    GS_CUDA_CHECK(cudaStreamSynchronize(0));
  };

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
    GS_CUDA_CHECK(cudaMalloc(&d_nccl_warm, sizeof(unsigned long long)));
    GS_CUDA_CHECK(cudaMemset(d_nccl_warm, 0, sizeof(unsigned long long)));
    GS_NCCL_CHECK(ncclAllReduce(d_nccl_warm, d_nccl_warm, 1, ncclUint64, ncclSum,
                      comm, 0));
    GS_CUDA_CHECK(cudaStreamSynchronize(0));
    GS_CUDA_CHECK(cudaFree(d_nccl_warm));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  for (u32 s = 0; s < steps; ++s) {
    exchange(u);
    exchange(v);
    StepKernel<<<blocks, threads>>>(u, v, un, vn, plane, nx, ny, nzl, gz0,
                                    nz, Du, Dv, F, K, dt);
    GS_CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(u, un);
    std::swap(v, vn);
  }
  const double ms = NowMs() - t0;

  double *d_sum = nullptr;
  GS_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
  GS_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
  SumV<<<blocks, threads>>>(v, plane, nzl, d_sum);
  GS_CUDA_CHECK(cudaDeviceSynchronize());
  // The checksum reduction is device-side too, read back once at the end.
  double csum = 0.0;
  GS_NCCL_CHECK(ncclAllReduce(d_sum, d_sum, 1, ncclDouble, ncclSum, comm, 0));
  GS_CUDA_CHECK(cudaStreamSynchronize(0));
  GS_CUDA_CHECK(cudaMemcpy(&csum, d_sum, sizeof(double),
                           cudaMemcpyDeviceToHost));

  int rc = 0;
  if (rank == 0) {
    std::printf("  %u steps in %.1f ms  v_checksum=%.6f\n", steps, ms, csum);

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

    std::printf("%s\n", rc == 0 ? "GRAYSCOTT NCCL: ALL GATES PASS"
                                : "GRAYSCOTT NCCL: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GS_NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return rc;
}

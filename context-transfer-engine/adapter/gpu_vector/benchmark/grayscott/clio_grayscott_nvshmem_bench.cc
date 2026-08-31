/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott, NVSHMEM edition: the slabs live on the symmetric heap and the
 * one-plane halo is PULLED one-sided from inside a small device kernel
 * (nvshmemx_getmem_block) before each step -- no host staging, no
 * two-sided pairing; the barrier between steps is the only synchronisation.
 * Stencil, seeding and constants are the paged bench's, verbatim. Links
 * nothing from clio.
 */

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#if defined(MD_NVSHMEM_USE_MPI)
#include <mpi.h>
#endif
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
      std::exit(1);                                                          \
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

/** Pull my halo planes from the neighbours' symmetric slabs, one-sided. */
__global__ void HaloPull(float *fld, u64 plane, u64 nzl_mine, u64 nzl_dn,
                         int pe_dn, int pe_up, int have_dn, int have_up) {
  if (blockIdx.x == 0 && have_dn) {
    // My low halo = the DOWN neighbour's top own plane (its local index
    // nzl_dn, in ITS extended layout).
    nvshmemx_getmem_block(fld, fld + nzl_dn * plane,
                          plane * sizeof(float), pe_dn);
  }
  if (blockIdx.x == (have_dn ? 1 : 0) && have_up) {
    // My high halo = the UP neighbour's first own plane (its local 1).
    nvshmemx_getmem_block(fld + (nzl_mine + 1) * plane, fld + plane,
                          plane * sizeof(float), pe_up);
  }
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
    GS_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    GS_CUDA_CHECK(cudaSetDevice(ndev > 0 ? wr % ndev : 0));
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &comm;
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
      std::fprintf(stderr, "nvshmem MPI bootstrap failed\n");
      return 1;
    }
  }
#else
  GS_CUDA_CHECK(cudaSetDevice(0));
  {
    // BOOTSTRAP THROUGH THE LAUNCHER, not through a hardcoded 1-PE

    // unique ID. set_attr_uniqueid_args(0, 1, ...) pinned this path to a

    // SINGLE PE, so a build without MPI could never be distributed --

    // which is the only build that runs on a Cray machine, because

    // NVSHMEM\'s MPI bootstrap plugin is built against OpenMPI and

    // Delta\'s system MPI is cray-mpich.

    //

    // nvshmem_init() honours NVSHMEM_BOOTSTRAP, so the launcher decides:

    //   NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2 + srun --mpi=pmi2

    // is what works multi-node here, and is what NVIDIA\'s own perftest

    // binaries use. A bare run with no launcher still gets 1 PE, which

    // is what the old code hardcoded.

    nvshmem_init();
  }
#endif
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();

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

  // EQUAL SLABS: the symmetric heap hands every PE the same allocation, and
  // the halo pull reads the peer's planes at fixed symmetric offsets -- an
  // uneven remainder slab would break that address math, so it is refused.
  if (nz % npes != 0) {
    if (mype == 0) {
      std::fprintf(stderr, "GRAYSCOTT NVSHMEM: nz=%llu not divisible by %d "
                   "PEs\n", (unsigned long long)nz, npes);
    }
    return 2;
  }
  const u64 nzl = nz / npes;
  const u64 gz0 = static_cast<u64>(mype) * nzl;
  const u64 ext = (nzl + 2) * plane;
  const int pe_dn = mype - 1, pe_up = mype + 1;
  const int have_dn = mype > 0, have_up = mype + 1 < npes;

  if (mype == 0) {
    std::printf("Gray-Scott, NVSHMEM edition: %llux%llux%llu, %u steps, %d "
                "PEs\n",
                (unsigned long long)nx, (unsigned long long)ny,
                (unsigned long long)nz, steps, npes);
  }

  float *u = static_cast<float *>(nvshmem_malloc(ext * sizeof(float)));
  float *v = static_cast<float *>(nvshmem_malloc(ext * sizeof(float)));
  float *un = static_cast<float *>(nvshmem_malloc(ext * sizeof(float)));
  float *vn = static_cast<float *>(nvshmem_malloc(ext * sizeof(float)));
  if (!u || !v || !un || !vn) {
    std::fprintf(stderr, "nvshmem_malloc failed\n");
    return 1;
  }
  SeedSlab<<<blocks, threads>>>(u, v, plane, nx, ny, nzl, gz0, nz);
  SeedSlab<<<blocks, threads>>>(un, vn, plane, nx, ny, nzl, gz0, nz);
  GS_CUDA_CHECK(cudaDeviceSynchronize());

  nvshmem_barrier_all();
  const double t0 = NowMs();
  for (u32 s = 0; s < steps; ++s) {
    HaloPull<<<2, 1>>>(u, plane, nzl, nzl, pe_dn, pe_up, have_dn, have_up);
    HaloPull<<<2, 1>>>(v, plane, nzl, nzl, pe_dn, pe_up, have_dn, have_up);
    GS_CUDA_CHECK(cudaDeviceSynchronize());
    StepKernel<<<blocks, threads>>>(u, v, un, vn, plane, nx, ny, nzl, gz0,
                                    nz, Du, Dv, F, K, dt);
    GS_CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(u, un);
    std::swap(v, vn);
    // The swap changes which SYMMETRIC buffer holds the field, and the next
    // step's pull reads the PEER's current field -- every PE must have
    // swapped before anyone pulls.
    nvshmem_barrier_all();
  }
  const double ms = NowMs() - t0;

  double *d_sum = nullptr;
  GS_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
  GS_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
  SumV<<<blocks, threads>>>(v, plane, nzl, d_sum);
  GS_CUDA_CHECK(cudaDeviceSynchronize());
  double local = 0.0, csum = 0.0;
  GS_CUDA_CHECK(cudaMemcpy(&local, d_sum, sizeof(double),
                           cudaMemcpyDeviceToHost));
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Allreduce(&local, &csum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  {
      // NOT "single PE" ANY MORE. This branch was only reachable under
      // the hardcoded 1-PE unique-id bootstrap, so copying the local value
      // was correct. Switching the bootstrap to nvshmem_init() made it the
      // MULTI-PE path, and the copy then reported one shard as the whole
      // problem. Reduce on the symmetric heap, as the MPI arm does.
    double *s = (double *)nvshmem_malloc(sizeof(double));
    double *d = (double *)nvshmem_malloc(sizeof(double));
    GS_CUDA_CHECK(cudaMemcpy(s, &local, sizeof(double), cudaMemcpyHostToDevice));
    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, d, s, 1);
    GS_CUDA_CHECK(cudaMemcpy(&csum, d, sizeof(double), cudaMemcpyDeviceToHost));
    nvshmem_free(s);
    nvshmem_free(d);
  }
#endif

  int rc = 0;
  if (mype == 0) {
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

    std::printf("%s\n", rc == 0 ? "GRAYSCOTT NVSHMEM: ALL GATES PASS"
                                : "GRAYSCOTT NVSHMEM: GATE FAILURE");
  }
  nvshmem_free(u); nvshmem_free(v); nvshmem_free(un); nvshmem_free(vn);
  nvshmem_finalize();
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Finalize();
#endif
  return rc;
}

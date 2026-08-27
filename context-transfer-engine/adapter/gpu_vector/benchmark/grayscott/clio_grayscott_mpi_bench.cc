/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott, MPI edition: z-slab decomposition with a one-plane halo
 * exchanged by MPI_Sendrecv (host-staged, as this MPI is built) before
 * every step. Stencil, seeding and constants are the paged bench's,
 * verbatim; the fixed global boundary is honoured through the local->global
 * plane map. The checksum is sum(v), allreduced; compare against the paged
 * bench with --check-csum and a loose tolerance (the paged bench itself
 * documents a run-to-run spread). Links nothing from clio.
 *
 * Run recipe: mpirun -n 4 clio_grayscott_mpi_bench --page-kb 64
 *             --data-mb 256 --steps 4
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

#define GS_CUDA_CHECK(x)                                                     \
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
    std::printf("Gray-Scott, MPI edition: %llux%llux%llu, %u steps, %d "
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

  const int up = (rank + 1 < nranks) ? rank + 1 : MPI_PROC_NULL;
  const int dn = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  std::vector<float> h_send(plane), h_recv(plane);
  auto exchange = [&](float *fld) {
    // Send my TOP own plane up, receive my LOW halo from below; then the
    // reverse. Host-staged, as the md sibling documents.
    GS_CUDA_CHECK(cudaMemcpy(h_send.data(), fld + nzl * plane,
                             plane * sizeof(float), cudaMemcpyDeviceToHost));
    MPI_Sendrecv(h_send.data(), static_cast<int>(plane), MPI_FLOAT, up, 11,
                 h_recv.data(), static_cast<int>(plane), MPI_FLOAT, dn, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (dn != MPI_PROC_NULL) {
      GS_CUDA_CHECK(cudaMemcpy(fld, h_recv.data(), plane * sizeof(float),
                               cudaMemcpyHostToDevice));
    }
    GS_CUDA_CHECK(cudaMemcpy(h_send.data(), fld + plane,
                             plane * sizeof(float), cudaMemcpyDeviceToHost));
    MPI_Sendrecv(h_send.data(), static_cast<int>(plane), MPI_FLOAT, dn, 12,
                 h_recv.data(), static_cast<int>(plane), MPI_FLOAT, up, 12,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up != MPI_PROC_NULL) {
      GS_CUDA_CHECK(cudaMemcpy(fld + (nzl + 1) * plane, h_recv.data(),
                               plane * sizeof(float),
                               cudaMemcpyHostToDevice));
    }
  };

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
  double local = 0.0, csum = 0.0;
  GS_CUDA_CHECK(cudaMemcpy(&local, d_sum, sizeof(double),
                           cudaMemcpyDeviceToHost));
  MPI_Allreduce(&local, &csum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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

    std::printf("%s\n", rc == 0 ? "GRAYSCOTT MPI: ALL GATES PASS"
                                : "GRAYSCOTT MPI: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return rc;
}

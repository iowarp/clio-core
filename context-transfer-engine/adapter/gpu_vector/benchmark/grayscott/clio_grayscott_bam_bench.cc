/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott, BaM edition: the capacity sibling. The INPUT fields live in
 * pinned host DRAM behind BaM's page cache (one page per XY plane) and the
 * stencil reads them through bam_ptr. BaM has no safe sub-page write path
 * (see the md sibling), so outputs stream to a host mirror one plane at a
 * time and the step ends with load_from_host + a cache invalidation -- the
 * same publish-then-invalidate move the md bench makes after each list
 * rebuild. Stencil, seeding and constants are the paged bench's, verbatim.
 * Links nothing from clio.
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

/** One output plane of the step, inputs read through BaM. */
__global__ void StepPlane(bam::ArrayDevice<float> u, bam::ArrayDevice<float> v,
                          float *unx, float *vnx, u64 plane, u64 nx, u64 ny,
                          u64 z, u64 nz) {
  const bool interior = (z > 0 && z + 1 < nz);
  const u64 zm = interior ? (z - 1) : z;
  const u64 zp = interior ? (z + 1) : z;
  bam::bam_ptr<float> up(&u);
  bam::bam_ptr<float> vp(&v);
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < plane;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 x = i % nx, y = i / nx;
    const float uu = up.at(z * plane + i);
    const float vv = vp.at(z * plane + i);
    float lu, lv;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
      lu = 0.0f; lv = 0.0f;
    } else {
      lu = up.at(z * plane + i - 1) + up.at(z * plane + i + 1) +
           up.at(z * plane + i - nx) + up.at(z * plane + i + nx) +
           up.at(zm * plane + i) + up.at(zp * plane + i) - 6.0f * uu;
      lv = vp.at(z * plane + i - 1) + vp.at(z * plane + i + 1) +
           vp.at(z * plane + i - nx) + vp.at(z * plane + i + nx) +
           vp.at(zm * plane + i) + vp.at(zp * plane + i) - 6.0f * vv;
    }
    const float uvv = uu * vv * vv;
    unx[i] = uu + 1.0f * (0.2f * lu - uvv + 0.02f * (1.0f - uu));
    vnx[i] = vv + 1.0f * (0.1f * lv + uvv - (0.02f + 0.048f) * vv);
  }
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, steps = 4;
  u64 page_kb = 64, data_mb = 256, cache_mb = 32;
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
    else if (a == "--cache-mb") cache_mb = next();
    else if (a == "--check-csum" && i + 1 < argc) {
      check_csum = std::strtod(argv[++i], nullptr);
      do_check = true;
    } else if (a == "--check-tol" && i + 1 < argc) {
      check_tol = std::strtod(argv[++i], nullptr);
    }
  }
  GS_CUDA_CHECK(cudaSetDevice(0));
  const u64 page_bytes = page_kb * 1024;
  const u64 plane = page_bytes / sizeof(float);
  u64 nx = 1, ny = plane;
  while (nx * 2 <= ny) { nx *= 2; ny /= 2; }
  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 nz = total_elems / (4 * plane);
  const u64 n = plane * nz;

  std::printf("Gray-Scott, BaM edition: %llux%llux%llu, %u steps, inputs "
              "behind a %llu MB BaM cache\n",
              (unsigned long long)nx, (unsigned long long)ny,
              (unsigned long long)nz, steps, (unsigned long long)cache_mb);

  bam::PageCacheConfig cfg{};
  cfg.page_size = static_cast<size_t>(page_bytes);
  cfg.num_pages = static_cast<size_t>((cache_mb * 1024 * 1024) / page_bytes);
  cfg.num_queues = 1;
  cfg.queue_depth = 1024;
  cfg.backend = bam::BackendType::kHostMemory;
  cfg.nvme_dev = nullptr;
  // ONE CACHE PER ARRAY. BaM tags pages by ARRAY-RELATIVE offset while the
  // cache state is shared, so two arrays on one cache alias: u's page N and
  // v's page N carry the same tag, a hit serves the OTHER field's bytes,
  // and the field drifts 23% in four steps with nothing else wrong.
  cfg.num_pages /= 2;
  const size_t cfg_num_pages = cfg.num_pages;
  std::unique_ptr<bam::PageCache> u_cache(new bam::PageCache(cfg));
  std::unique_ptr<bam::PageCache> v_cache(new bam::PageCache(cfg));
  std::unique_ptr<bam::Array<float>> u(new bam::Array<float>(n, *u_cache));
  std::unique_ptr<bam::Array<float>> v(new bam::Array<float>(n, *v_cache));
  std::vector<float> h_u(n), h_v(n), h_un(n), h_vn(n);
  for (u64 z = 0; z < nz; ++z) {
    for (u64 i = 0; i < plane; ++i) {
      h_u[z * plane + i] = InitU(i % nx, i / nx, z, nx, ny, nz);
      h_v[z * plane + i] = InitV(i % nx, i / nx, z, nx, ny, nz);
    }
  }
  u->load_from_host(h_u.data(), n);
  v->load_from_host(h_v.data(), n);

  float *d_un = nullptr, *d_vn = nullptr;
  GS_CUDA_CHECK(cudaMalloc(&d_un, plane * sizeof(float)));
  GS_CUDA_CHECK(cudaMalloc(&d_vn, plane * sizeof(float)));

  const double t0 = NowMs();
  for (u32 s = 0; s < steps; ++s) {
    for (u64 z = 0; z < nz; ++z) {
      StepPlane<<<blocks, threads>>>(u->device(), v->device(), d_un, d_vn,
                                     plane, nx, ny, z, nz);
      GS_CUDA_CHECK(cudaGetLastError());
      GS_CUDA_CHECK(cudaDeviceSynchronize());
      GS_CUDA_CHECK(cudaMemcpy(h_un.data() + z * plane, d_un,
                               plane * sizeof(float),
                               cudaMemcpyDeviceToHost));
      GS_CUDA_CHECK(cudaMemcpy(h_vn.data() + z * plane, d_vn,
                               plane * sizeof(float),
                               cudaMemcpyDeviceToHost));
    }
    // Publish the new fields into the backing store and INVALIDATE the
    // cache: BaM's pages are tagged by offset and stay falsely valid when
    // the store is rewritten underneath them (the md sibling's trap).
    u->load_from_host(h_un.data(), n);
    v->load_from_host(h_vn.data(), n);
    for (bam::PageCache *pc : {u_cache.get(), v_cache.get()}) {
      const bam::PageCacheDeviceState st = pc->device_state();
      GS_CUDA_CHECK(cudaMemset(st.page_tags, 0xFF,
                               cfg_num_pages * sizeof(uint64_t)));
      GS_CUDA_CHECK(cudaMemset(st.page_states,
                               static_cast<int>(bam::PageState::kInvalid),
                               cfg_num_pages * sizeof(uint32_t)));
    }
    std::swap(h_u, h_un);
    std::swap(h_v, h_vn);
  }
  const double ms = NowMs() - t0;

  double csum = 0.0;
  for (u64 i = 0; i < n; ++i) csum += static_cast<double>(h_v[i]);

  int rc = 0;
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

  std::printf("%s\n", rc == 0 ? "GRAYSCOTT BAM: ALL GATES PASS"
                              : "GRAYSCOTT BAM: GATE FAILURE");
  return rc;
}

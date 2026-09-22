/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * k-means, Aurora baseline editions (MPI / oneCCL / Intel SHMEM), one source.
 *
 * SAME SCIENCE, DIFFERENT DATA PLANE. The point generator, the assignment
 * step and the centroid update are the paged bench's, through
 * kmeans_math.h; each rank generates its GLOBAL-INDEX shard of the
 * deterministic point cloud into device memory, assigns locally, and the
 * per-cluster sums and counts are combined by the substrate's allreduce
 * (gv_comm.h). Every rank then computes the identical centroid update, so
 * the model is replicated and only the reductions cross the wire -- classic
 * data-parallel Lloyd, as in the CUDA MPI edition.
 *
 * GATES, as the CUDA editions:
 *   COUNT   sum of cluster counts == total points, every iteration, exact.
 *   CSUM    the final centroid checksum, printed always; --check-csum V
 *           compares against a reference with the documented tolerance
 *           (atomic float sums are order-dependent, so bit equality between
 *           substrates is not expected).
 *
 * Run recipe (one rank per node, one tile each):
 *   mpiexec -n 4 --ppn 1 clio_kmeans_<sub>_bench --data-mb 1024 --iters 4
 */

#include "gv_comm.h"
#include "../kmeans/kmeans_math.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using gvc::u32;
using gvc::u64;
using clio_km::NearestCentroid;
using clio_km::PointVal;
using clio_km::UpdateCentroid;

namespace {

/** Fill this rank's shard with the deterministic point cloud.
 *  @param q       the device queue
 *  @param pts     device buffer of n floats
 *  @param base    global element index of pts[0]
 *  @param n       elements to write
 *  @param dims,k  the generator's parameters */
void Seed(sycl::queue &q, float *pts, u64 base, u64 n, u32 dims, u32 k) {
  // Grid-stride: a 32 GB shard is 8.6 G elements, more than a range<1>
  // of 32-bit ids can name.
  const size_t g = 1024 * 256;
  q.parallel_for(sycl::nd_range<1>(g, 256), [=](sycl::nd_item<1> it) {
     for (u64 i = it.get_global_id(0); i < n; i += g) {
       pts[i] = PointVal(base + i, dims, k);
     }
   }).wait();
}

/** The assignment step: each point joins its nearest centroid's sum and
 *  count. Grid-stride over points, atomics into sums/counts, as the CUDA
 *  kernel. */
void Assign(sycl::queue &q, u32 blocks, u32 threads, const float *pts,
            u64 npts, u32 dims, u32 k, const float *cent, float *sums,
            unsigned *counts) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     for (u64 p = it.get_global_id(0); p < npts; p += g) {
       const float *pt = pts + p * dims;
       const u32 bestk = NearestCentroid(pt, cent, dims, k);
       for (u32 i = 0; i < dims; ++i) {
         gvc::AtomicAdd(&sums[bestk * dims + i], pt[i]);
       }
       gvc::AtomicAdd(&counts[bestk], 1u);
     }
   }).wait();
}

/** centroid = sum / count, leaving an empty cluster where it was. */
void Update(sycl::queue &q, float *cent, const float *sums,
            const unsigned *counts, u32 dims, u32 k) {
  q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> c) {
     UpdateCentroid(cent, sums, counts, dims, static_cast<u32>(c));
   }).wait();
}

}  // namespace

int main(int argc, char **argv) {
  gvc::Comm comm;
  comm.Init(&argc, &argv);
  const int rank = comm.rank, nranks = comm.nranks;
  sycl::queue &q = comm.q;

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
        std::printf("usage: mpiexec -n N %s [--blocks N] [--threads N] "
                    "[--dims N] [--clusters N] [--iters N] [--data-mb N] "
                    "[--check-csum V] [--check-tol R]\n", argv[0]);
      }
      comm.Finalize();
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
    std::printf("k-means, %s edition (SYCL)\n"
                "  points=%llu dims=%u k=%u iters=%u  ranks=%d "
                "(%.1f MB/rank) device=%s\n",
                gvc::Comm::Name(), (unsigned long long)npts, dims, k, iters,
                nranks,
                static_cast<double>(my_elems * sizeof(float)) / 1048576.0,
                q.get_device().get_info<sycl::info::device::name>().c_str());
  }

  float *d_pts = comm.AllocLocal<float>(my_elems);
  Seed(q, d_pts, p0 * dims, my_elems, dims, k);

  // Initial centroids: the first k points -- identical on every rank and to
  // the paged bench, computed straight from the generator.
  const u64 kd = static_cast<u64>(k) * dims;
  std::vector<float> h_cent(static_cast<size_t>(kd));
  for (u64 i = 0; i < kd; ++i) h_cent[i] = PointVal(i, dims, k);
  float *d_cent = comm.Alloc<float>(kd);
  float *d_sums = comm.Alloc<float>(kd);
  unsigned *d_counts = comm.Alloc<unsigned>(k);
  q.memcpy(d_cent, h_cent.data(), kd * sizeof(float)).wait();

  std::vector<unsigned> h_counts(k);
  int rc = 0;
  comm.Barrier();
  const double t0 = gvc::NowMs();
  double t_comm = 0.0;
  for (u32 it = 0; it < iters; ++it) {
    q.memset(d_sums, 0, kd * sizeof(float));
    q.memset(d_counts, 0, k * sizeof(unsigned));
    Assign(q, blocks, threads, d_pts, my_pts, dims, k, d_cent, d_sums,
           d_counts);
    // THE DATA PLANE UNDER TEST: partial sums meet on the substrate's
    // allreduce, device buffers in and out.
    const double c0 = gvc::NowMs();
    comm.AllreduceSum(d_sums, kd);
    comm.AllreduceSum(d_counts, static_cast<u64>(k));
    t_comm += gvc::NowMs() - c0;
    // COUNT GATE, exact: a lost or doubled point cannot hide in a float.
    q.memcpy(h_counts.data(), d_counts, k * sizeof(unsigned)).wait();
    u64 sum_counts = 0;
    for (u32 c = 0; c < k; ++c) sum_counts += h_counts[c];
    if (sum_counts != npts) {
      if (rank == 0) {
        std::printf("  COUNT GATE: FAIL at iter %u (%llu != %llu)\n", it,
                    (unsigned long long)sum_counts, (unsigned long long)npts);
      }
      rc = 1;
    }
    Update(q, d_cent, d_sums, d_counts, dims, k);
    // Progress, so a run that overruns its cap shows whether it stalled or
    // merely slowed (a 12-iteration deck timed out where 4 took 79 s).
    if (rank == 0) {
      std::fprintf(stderr, "  iter %u done at %.1f ms (comm so far %.1f ms)\n",
                   it, gvc::NowMs() - t0, t_comm);
    }
  }
  const double ms = gvc::NowMs() - t0;

  q.memcpy(h_cent.data(), d_cent, kd * sizeof(float)).wait();
  double csum = 0.0;
  for (float f : h_cent) csum += static_cast<double>(f);

  if (rank == 0) {
    if (rc == 0) std::printf("  COUNT GATE: PASS (all %u iterations)\n", iters);
    std::printf("  %u iters in %.1f ms (comm %.1f ms)  centroid_checksum=%.6f\n",
                iters, ms, t_comm, csum);
    std::printf("KMEANS %s: points=%llu dims=%u k=%u iters=%u ranks=%d "
                "data_mb=%llu ms=%.1f comm_ms=%.1f ms_per_iter=%.2f "
                "centroid_checksum=%.6f\n",
                gvc::Comm::Name(), (unsigned long long)npts, dims, k, iters,
                nranks, (unsigned long long)data_mb, ms, t_comm, ms / iters,
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
    std::printf("KMEANS %s: %s\n", gvc::Comm::Name(),
                rc == 0 ? "ALL GATES PASS" : "GATE FAILURE");
  }
  rc = comm.Verdict(rc);
  comm.FreeLocal(d_pts);
  comm.Free(d_cent);
  comm.Free(d_sums);
  comm.Free(d_counts);
  comm.Finalize();
  return rc;
}

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
 * How much does a host-initiated 64 KB USM copy cost WHILE a persistent
 * kernel occupies the tile?
 *
 * The runtime stages every device-resident bulk buffer through the host with
 * `sycl::queue::memcpy(...).wait()` from a host thread, and on a two-node
 * kmeans that measured ~2 ms per 64 KB page (NETTRACE sendin ser), against
 * ~10 us for the same copy on an idle tile. This probe times that copy from
 * the host under a spinning kernel, for several queue configurations, so the
 * runtime can pick the one that does not serialise behind the kernel:
 *
 *   ooo       default out-of-order queue (what GpuApi::SyclQueue() is)
 *   inorder   in_order queue
 *   ooo_ctx   out-of-order queue on a SEPARATE context (own L0 command queue)
 *
 * and for each: D2H and H2D of 64 KB, 200 reps, median and mean in ms,
 * first with the tile idle and then with the kernel running.
 *
 * Build (login node):  icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc"
 *                      -O2 sycl_copy_probe.cc -o sycl_copy_probe
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/**
 * Time `reps` copies of `bytes` through `q`, returning median and mean in ms.
 * @param q Queue the copy is submitted to
 * @param dst Destination pointer (host or device USM)
 * @param src Source pointer (host or device USM)
 * @param bytes Bytes per copy
 * @param reps Number of copies timed
 * @param median_ms Output: median copy time in ms
 * @param mean_ms Output: mean copy time in ms
 */
void TimeCopies(sycl::queue &q, void *dst, const void *src, size_t bytes,
                int reps, double *median_ms, double *mean_ms) {
  std::vector<double> t;
  t.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(dst, src, bytes).wait();
    auto t1 = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(t.begin(), t.end());
  double sum = 0;
  for (double x : t) sum += x;
  *median_ms = t[t.size() / 2];
  *mean_ms = sum / t.size();
}

/**
 * Run the D2H and H2D timings for one queue and print one line per direction.
 * @param tag Label for the configuration
 * @param state "idle" or "busy" (whether the spin kernel is running)
 * @param q Queue to time
 * @param dev Device USM buffer
 * @param host Host USM buffer
 * @param bytes Bytes per copy
 */
void Report(const char *tag, const char *state, sycl::queue &q, void *dev,
            void *host, size_t bytes) {
  double med, mean;
  TimeCopies(q, host, dev, bytes, 200, &med, &mean);
  std::printf("PROBE %-8s %-4s D2H median=%.3fms mean=%.3fms\n", tag, state,
              med, mean);
  TimeCopies(q, dev, host, bytes, 200, &med, &mean);
  std::printf("PROBE %-8s %-4s H2D median=%.3fms mean=%.3fms\n", tag, state,
              med, mean);
  std::fflush(stdout);
}

}  // namespace

int main() {
  const size_t kBytes = 64 * 1024;
  sycl::device dev_sel{sycl::gpu_selector_v};
  sycl::queue qk{dev_sel, sycl::property::queue::in_order()};
  sycl::queue q_ooo{dev_sel};
  sycl::queue q_in{dev_sel, sycl::property::queue::in_order()};
  sycl::context ctx2{dev_sel};
  sycl::queue q_ctx{ctx2, dev_sel};

  // Allocations in the kernel queue's (default) context; the separate-context
  // queue copies between them through host memory it can see.
  char *dev = sycl::malloc_device<char>(kBytes, qk);
  char *host = sycl::malloc_host<char>(kBytes, qk);
  char *dev2 = sycl::malloc_device<char>(kBytes, q_ctx);
  char *host2 = sycl::malloc_host<char>(kBytes, q_ctx);
  // PAGEABLE host memory, what the runtime actually copies into: a heap
  // buffer (SaveTaskArchive staging), a std::vector (the transfer engine's
  // bounce), a memfd-backed segment (bdev buffers). Pinned USM is the
  // fast case; this is the question.
  char *pageable = new char[kBytes];
  std::vector<char> vec(kBytes, 1);
  int *flag = sycl::malloc_device<int>(1, qk);  // device USM: kernel polls it
  std::memset(host, 1, kBytes);
  std::memset(host2, 1, kBytes);
  std::memset(pageable, 1, kBytes);
  int zero = 0;
  qk.memcpy(flag, &zero, sizeof(int)).wait();
  qk.memcpy(dev, host, kBytes).wait();
  q_ctx.memcpy(dev2, host2, kBytes).wait();

  std::printf("device: %s\n",
              dev_sel.get_info<sycl::info::device::name>().c_str());

  // CALIBRATE THE COUNTED SPIN. sycl_cuda_compat's __nanosleep(ns) is
  // `for (i < ns) sink = sink + 1` on a volatile, and device_vector.h backs
  // off with it up to 1<<20 per retry under page-cache pressure. Time that
  // loop from the host: one work-group of 256 items each running it (the
  // shape it runs in), and 64 groups, kernel-only.
  int *sink = sycl::malloc_device<int>(64 * 256, qk);
  for (unsigned iters : {1u << 10, 1u << 14, 1u << 17, 1u << 20}) {
    for (int groups : {1, 64}) {
      auto t0 = std::chrono::steady_clock::now();
      qk.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 256),
                              sycl::range<1>(256)),
            [=](sycl::nd_item<1> it) {
              volatile unsigned s = 0;
              for (unsigned i = 0; i < iters; ++i) s = s + 1u;
              sink[it.get_global_id(0)] = static_cast<int>(s);
            })
          .wait();
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::printf("SPIN iters=%u groups=%d total=%.3fms per_iter=%.2fns\n",
                  iters, groups, ms, ms * 1e6 / iters);
    }
  }
  std::fflush(stdout);

  Report("ooo", "idle", q_ooo, dev, host, kBytes);
  Report("inorder", "idle", q_in, dev, host, kBytes);
  Report("ooo_ctx", "idle", q_ctx, dev2, host2, kBytes);
  Report("ooo_pg", "idle", q_ooo, dev, pageable, kBytes);
  Report("ooo_vec", "idle", q_ooo, dev, vec.data(), kBytes);

  // The persistent kernel: 64 work-groups of 256, every item spinning on the
  // host flag -- the shape of the benchmarks' block-per-group kernels.
  auto ev = qk.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(64 * 256), sycl::range<1>(256)),
      [=](sycl::nd_item<1>) {
        // A plain volatile load of DEVICE memory: PVC has no atomics on
        // host-resident USM (an atomic there is an AtomicAccessViolation
        // from the GPU), and a volatile load of host USM never observed the
        // host's store (the first run of this probe timed out on it).
        volatile int *f = flag;
        while (*f == 0) {
        }
      });
  // Give the kernel time to be resident before timing.
  auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) {
  }
  Report("ooo", "busy", q_ooo, dev, host, kBytes);
  Report("inorder", "busy", q_in, dev, host, kBytes);
  Report("ooo_ctx", "busy", q_ctx, dev2, host2, kBytes);
  Report("ooo_pg", "busy", q_ooo, dev, pageable, kBytes);
  Report("ooo_vec", "busy", q_ooo, dev, vec.data(), kBytes);

  // Stop the kernel through the out-of-order queue (an in-order copy would
  // queue behind the kernel it is meant to release).
  int one = 1;
  q_ooo.memcpy(flag, &one, sizeof(int)).wait();
  ev.wait();
  std::printf("PROBE done\n");
  sycl::free(dev, qk);
  sycl::free(host, qk);
  sycl::free(dev2, q_ctx);
  sycl::free(host2, q_ctx);
  sycl::free(flag, qk);
  delete[] pageable;
  return 0;
}

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
 * Does this device support ATOMICS ON SHARED USM?
 *
 * Every ported benchmark dies on Aurora's Max 1550 with
 *
 *   Segmentation fault from GPU ... type: 2 (AtomicAccessViolation),
 *   access: 2 (Atomic)
 *
 * and gpu_vector's page table is reached through CTP's MallocManaged, which is
 * sycl::malloc_shared on SYCL and cudaMallocManaged on CUDA. Managed memory
 * takes device atomics on NVIDIA; shared USM takes them on SPIR-V only if the
 * device reports aspect::usm_atomic_shared_allocations.
 *
 * This probe separates the two claims -- what the device SAYS it supports, and
 * what it DOES -- for each of the three allocation kinds, so the answer is a
 * measurement rather than an inference from a fault address. Run it in a job;
 * it needs a real GPU and nothing else.
 *
 *   icpx -fsycl -std=c++20 -O2 usm_atomic_probe.cc -o usm_atomic_probe
 */
#include <cstdio>
#include <utility>

#include <sycl/sycl.hpp>

namespace {

/** One work-item per lane, all hammering the same word. If the device cannot
 *  do atomics on this allocation the kernel faults here rather than returning
 *  a wrong answer, which is exactly the signature the benchmarks show. */
bool TryAtomic(sycl::queue &q, const char *what, int *p) {
  if (p == nullptr) {
    std::printf("  %-14s ALLOCATION FAILED\n", what);
    return false;
  }
  *p = 0;
  try {
    q.parallel_for(sycl::nd_range<1>{sycl::range<1>(256), sycl::range<1>(64)},
                   [=](sycl::nd_item<1>) {
                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>
                         a(*p);
                     a.fetch_add(1);
                   })
        .wait_and_throw();
  } catch (const sycl::exception &e) {
    std::printf("  %-14s THREW: %s\n", what, e.what());
    return false;
  }
  const bool ok = (*p == 256);
  std::printf("  %-14s %s (value=%d, expected 256)\n", what,
              ok ? "OK" : "WRONG", *p);
  return ok;
}

}  // namespace

int main() {
  sycl::queue q;
  const auto &d = q.get_device();
  std::printf("device: %s\n", d.get_info<sycl::info::device::name>().c_str());

  std::printf("declared aspects:\n");
  std::printf("  usm_shared_allocations        %d\n",
              d.has(sycl::aspect::usm_shared_allocations));
  std::printf("  usm_atomic_shared_allocations %d\n",
              d.has(sycl::aspect::usm_atomic_shared_allocations));
  std::printf("  usm_host_allocations          %d\n",
              d.has(sycl::aspect::usm_host_allocations));
  std::printf("  usm_atomic_host_allocations   %d\n",
              d.has(sycl::aspect::usm_atomic_host_allocations));
  std::printf("  usm_device_allocations        %d\n",
              d.has(sycl::aspect::usm_device_allocations));

  std::printf("measured, device-scope fetch_add from 256 work-items:\n");
  int *dev = sycl::malloc_device<int>(1, q);
  int *shared = sycl::malloc_shared<int>(1, q);
  int *host = sycl::malloc_host<int>(1, q);

  // Device memory cannot be initialised by a host store, so it is seeded and
  // read back through the queue instead.
  if (dev != nullptr) {
    int zero = 0;
    q.memcpy(dev, &zero, sizeof(int)).wait();
    try {
      q.parallel_for(sycl::nd_range<1>{sycl::range<1>(256), sycl::range<1>(64)},
                     [=](sycl::nd_item<1>) {
                       sycl::atomic_ref<
                           int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
                           a(*dev);
                       a.fetch_add(1);
                     })
          .wait_and_throw();
      int got = -1;
      q.memcpy(&got, dev, sizeof(int)).wait();
      std::printf("  %-14s %s (value=%d, expected 256)\n", "malloc_device",
                  got == 256 ? "OK" : "WRONG", got);
    } catch (const sycl::exception &e) {
      std::printf("  %-14s THREW: %s\n", "malloc_device", e.what());
    }
  }

  TryAtomic(q, "malloc_shared", shared);
  TryAtomic(q, "malloc_host", host);

  // The cheapest possible fix, if it works: keep malloc_shared -- so the host
  // can still dereference the page table directly -- and merely advise the
  // runtime about placement. Tested separately because "shared fails" and
  // "shared fails even when advised" call for very different fixes.
  //
  // queue::mem_advise takes a BACKEND-SPECIFIC int, so these are the Level Zero
  // ze_memory_advice_t values. Two of them bear on this directly:
  //   2 = SET_PREFERRED_LOCATION      make it device-resident
  //   5 = CLEAR_NON_ATOMIC_MOSTLY     withdraw the "atomics are rare" hint
  for (const auto &adv : {std::pair<int, const char *>{2, "advise:prefloc"},
                          std::pair<int, const char *>{5, "advise:atomic"}}) {
    int *a = sycl::malloc_shared<int>(1, q);
    if (a == nullptr) continue;
    try {
      q.mem_advise(a, sizeof(int), adv.first).wait();
    } catch (const sycl::exception &e) {
      std::printf("  %-14s mem_advise(%d) threw: %s\n", adv.second, adv.first,
                  e.what());
      sycl::free(a, q);
      continue;
    }
    TryAtomic(q, adv.second, a);
    sycl::free(a, q);
  }

  sycl::free(dev, q);
  sycl::free(shared, q);
  sycl::free(host, q);
  std::printf("PROBE DONE\n");
  return 0;
}

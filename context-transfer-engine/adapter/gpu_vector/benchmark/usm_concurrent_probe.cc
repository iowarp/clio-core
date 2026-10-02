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
 * Can the HOST poll a word WHILE a kernel is writing it, and can the kernel
 * see a host store while it runs? Per allocation kind.
 *
 * This is the gpu2cpu ring's access pattern: the device pushes with atomics,
 * the CPU worker polls concurrently, and each side waits on the other. The
 * ring moved from malloc_host (device atomics fault on PVC) to malloc_shared
 * (device atomics work) -- and the first benchmark then produced no output at
 * all for 90s. If a shared page the device has touched stalls host access
 * until the kernel ends, that is a deadlock by construction, and the fix is a
 * different allocation, not a different runtime.
 *
 * For each kind: a kernel spins incrementing a counter until it sees the host
 * write a stop flag, or until it has spun a bounded number of times. The host
 * meanwhile polls the counter for up to 3 seconds and reports how many
 * distinct values it saw. "1 value" means the host was blind (or blocked)
 * while the kernel ran; "many" means concurrent access works.
 *
 * Also prints Level Zero's own access-capability flags for host and shared
 * allocations, which is the device's declaration of exactly this.
 */
#include <chrono>
#include <cstdio>
#include <thread>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

namespace {

struct Words {
  int counter;
  int stop;
};

/** Returns the number of distinct counter values the host observed while the
 *  kernel ran, and whether the kernel saw the stop flag (vs. spinning out). */
void Trial(sycl::queue &q, const char *what, Words *w) {
  if (w == nullptr) {
    std::printf("  %-14s ALLOCATION FAILED\n", what);
    return;
  }
  // Seed through the queue: device memory is not host-writable.
  Words zero{0, 0};
  q.memcpy(w, &zero, sizeof(Words)).wait();

  auto ev = q.single_task([=]() {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        c(w->counter), s(w->stop);
    for (int i = 0; i < 200000000 && s.load() == 0; ++i) {
      c.fetch_add(1);
    }
  });

  int distinct = 0;
  int last = -1;
  bool host_readable = true;
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    int v = -1;
    try {
      // On a device allocation this line is the fault; catch nothing, it
      // would be a SIGSEGV -- so device memory is polled through a copy.
      v = *reinterpret_cast<volatile int *>(&w->counter);
    } catch (...) {
      host_readable = false;
      break;
    }
    if (v != last) {
      ++distinct;
      last = v;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (ms > 3000.0) break;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  // Tell the kernel to stop, the way the CPU worker acknowledges a push.
  *reinterpret_cast<volatile int *>(&w->stop) = 1;
  const auto t1 = std::chrono::steady_clock::now();
  ev.wait();
  const double wait_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t1)
                             .count();
  Words fin{};
  q.memcpy(&fin, w, sizeof(Words)).wait();
  std::printf("  %-14s host saw %d distinct values in 3s; kernel ended %.0f ms "
              "after host stop (final counter %d) %s\n",
              what, distinct, wait_ms, fin.counter,
              host_readable ? "" : "[host read faulted]");
}

void PrintZeCaps(sycl::queue &q) {
  auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_device());
  ze_device_memory_access_properties_t p{};
  p.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
  if (zeDeviceGetMemoryAccessProperties(zdev, &p) != ZE_RESULT_SUCCESS) {
    std::printf("zeDeviceGetMemoryAccessProperties failed\n");
    return;
  }
  auto show = [](const char *n, ze_memory_access_cap_flags_t f) {
    std::printf("  %-28s rw=%d atomic=%d concurrent=%d concurrent_atomic=%d\n",
                n, !!(f & ZE_MEMORY_ACCESS_CAP_FLAG_RW),
                !!(f & ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC),
                !!(f & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT),
                !!(f & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC));
  };
  std::printf("Level Zero access capabilities:\n");
  show("host allocations", p.hostAllocCapabilities);
  show("device allocations", p.deviceAllocCapabilities);
  show("shared single-device", p.sharedSingleDeviceAllocCapabilities);
  show("shared cross-device", p.sharedCrossDeviceAllocCapabilities);
  show("shared system", p.sharedSystemAllocCapabilities);
}

/** Plain device stores to host USM, polled concurrently by the host: the
 *  gpu2cpu device ring's access pattern for entries_/ready_. Atomics on host
 *  memory fault on this device (usm_atomic_probe), so the kernel uses stores. */
void HostTrial(sycl::queue &q) {
  Words *host = sycl::malloc_host<Words>(1, q);
  if (host != nullptr) {
    host->counter = 0;
    host->stop = 0;
    auto ev = q.single_task([=]() {
      for (int i = 0; i < 200000000; ++i) {
        if (*reinterpret_cast<volatile int *>(&host->stop) != 0) break;
        *reinterpret_cast<volatile int *>(&host->counter) = i;
      }
    });
    int distinct = 0, last = -1;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
               .count() < 3000.0) {
      const int v = *reinterpret_cast<volatile int *>(&host->counter);
      if (v != last) { ++distinct; last = v; }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    *reinterpret_cast<volatile int *>(&host->stop) = 1;
    const auto t1 = std::chrono::steady_clock::now();
    ev.wait();
    std::printf("  %-14s host saw %d distinct values in 3s (plain stores); "
                "kernel ended %.0f ms after host stop\n",
                "malloc_host", distinct,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t1)
                    .count());
    sycl::free(host, q);
  }
}

}  // namespace

int main() {
  // UNBUFFERED. The first run of this probe hung inside a trial and was
  // killed by the job cap, and every line it had printed died in the stdio
  // buffer -- the log showed nothing at all. A probe that can hang must flush
  // as it goes, or the hang erases the very evidence it was gathering.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  sycl::queue q;
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  PrintZeCaps(q);

  // HOST MEMORY FIRST: it is the pattern the device ring depends on, and it
  // goes before the shared-memory trial -- the one suspected of hanging -- so
  // that trial cannot prevent this answer from being recorded.
  HostTrial(q);

  std::printf("concurrent host poll vs device atomic increment:\n");
  Words *shared = sycl::malloc_shared<Words>(1, q);
  Trial(q, "malloc_shared", shared);
  sycl::free(shared, q);

  std::printf("PROBE DONE\n");
  return 0;
}

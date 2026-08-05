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

// Targeted never-miss-wakeup test for the gated signalling paths.
//
// Producers now signal a worker ONLY when it is parked (Worker::SuspendMe
// publishes SetActive(false)/SetShardParked before a fenced re-check;
// IpcManager::AwakenWorker and the SHM transport skip the tgkill for a
// running consumer). The failure mode this guards against is a MISSED wake:
// a producer that skips the signal while the worker commits to epoll. A miss
// is invisible to throughput tests (the worker's max_sleep cap re-polls
// within ~50 ms) but shows up as exactly a ~50 ms latency step on an
// otherwise-microsecond operation.
//
// So this test measures COLD-SUBMIT LATENCY: idle long enough for every
// worker to park (past first_busy_wait), then time one synchronous PutBlob.
// Repeated over many iterations:
//   - a HANG (wake lost AND self-heal broken) trips the ctest timeout;
//   - a SYSTEMATIC miss (handshake wrong) makes ~every cold op pay ~50 ms,
//     tripping the slow-op budget below;
//   - the healthy path completes each op in tens..hundreds of us, with only
//     scheduling-noise stragglers.
//
// Env knobs:
//   WAKEUP_ITERS    (800)   cold-submit iterations
//   WAKEUP_IDLE_US  (3000)  idle time before each submit (must exceed
//                           first_busy_wait so workers actually park)
//   WAKEUP_SLOW_US  (40000) an op slower than this bears the missed-wake
//                           (max_sleep self-heal) signature

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "wakeup_never_miss_target";
constexpr clio::run::u64 kTargetSize = 512ULL * 1024 * 1024;  // 512 MiB

int FromEnv(const char *name, int dflt) {
  if (const char *e = std::getenv(name)) {
    int n = std::atoi(e);
    if (n > 0) return n;
  }
  return dflt;
}

class Fixture {
 public:
  bool initialized_ = false;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(902, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(kTargetName,
                                        clio::run::bdev::BdevType::kRam,
                                        kTargetSize, clio::run::PoolQuery::Local(),
                                        bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

}  // namespace

TEST_CASE("WakeupNeverMiss - cold submits after every worker parks complete "
          "without the ~50ms missed-wake latency signature",
          "[runtime][wakeup][signalling]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  const int kIters = FromEnv("WAKEUP_ITERS", 800);
  const int kIdleUs = FromEnv("WAKEUP_IDLE_US", 3000);
  const int kSlowUs = FromEnv("WAKEUP_SLOW_US", 40000);
  constexpr clio::run::u64 kIoSize = 4096;

  clio::cte::core::Tag tag("wakeup_never_miss_tag");
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kIoSize);
  REQUIRE_FALSE(buf.IsNull());
  std::memset(buf.ptr_, 0x5A, kIoSize);

  // Warm the blob once (metadata + first allocation) so the timed loop
  // measures the wake path, not first-touch setup.
  {
    auto p = tag.AsyncPutBlob("wake_blob", buf.shm_.template Cast<void>(),
                              kIoSize, 0, 1.0f);
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  }

  int slow_ops = 0;
  double max_us = 0.0, sum_us = 0.0;
  for (int i = 0; i < kIters; ++i) {
    // Idle past first_busy_wait so every worker parks (epoll) — the next
    // submit must then WIN the park/signal handshake to complete promptly.
    std::this_thread::sleep_for(std::chrono::microseconds(kIdleUs));

    auto t0 = std::chrono::steady_clock::now();
    auto p = tag.AsyncPutBlob("wake_blob", buf.shm_.template Cast<void>(),
                              kIoSize, 0, 1.0f);
    p.Wait();
    double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    REQUIRE(p->GetReturnCode() == 0);

    sum_us += us;
    if (us > max_us) max_us = us;
    if (us > kSlowUs) ++slow_ops;
  }

  HLOG(kInfo,
       "[wakeup] iters={} idle_us={} mean_us={} max_us={} slow(>{}us)={}",
       kIters, kIdleUs, static_cast<int>(sum_us / kIters),
       static_cast<int>(max_us), kSlowUs, slow_ops);

  // A systematic missed wake makes essentially EVERY cold op pay the
  // ~max_sleep (50 ms) self-heal. Healthy runs see only scheduling-noise
  // stragglers. Budget: <2% of ops may exceed the slow threshold (CI boxes
  // are noisy; a real handshake bug fails this by 50x, not by a whisker).
  const int budget = std::max(2, kIters / 50);
  REQUIRE(slow_ops < budget);

  ipc->FreeBuffer(buf);
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}

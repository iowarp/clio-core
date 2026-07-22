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
 * BDEV LEAK STRESS TEST
 *
 * Hammer PutBlob + DelBlob of the same key in a tight loop for a wall-clock
 * duration (default 60 s) against a single small RAM tier, then assert that
 * NOTHING leaked at two independent layers:
 *
 *   1. BDEV BLOCK LEVEL — the bdev target's used capacity must be back to zero:
 *      GetTargetInfo(remaining_space_) after the loop equals the steady-state
 *      baseline captured after a warm-up cycle. Every PutBlob draws blocks from
 *      the bdev allocator and every DelBlob must return them, so after tens of
 *      thousands of balanced cycles the live set is still one blob and the
 *      allocator's outstanding bytes are 0. A block leak shows up as
 *      remaining_after < remaining_before.
 *
 *   2. RUNTIME ALLOCATOR LEVEL — the runtime leak checker metric
 *      (IpcManager::GetRuntimeHeapAllocatedBytes = CTP_MALLOC private heap +
 *      every owned SHM MultiProcessAllocator) must not have grown beyond a
 *      small tolerance across the loop. Because that counter tracks handed-out
 *      bytes (balanced alloc/free nets to zero), a per-op runtime leak that the
 *      coarse per-test gate might tolerate accumulates over the minute and trips
 *      here. Returns 0 in non-leak builds, so the assertion is a no-op there.
 *
 * Both leak checkers added alongside this test run simultaneously: the runtime
 * scan (IpcManager::ReportRuntimeLeaks) fires at ServerFinalize and the
 * allocator-destructor checker (ctp::ipc::AllocatorLeakChecker) fires at static
 * teardown when the process exits — this test simply drives enough churn to make
 * a leak, if present, unmistakable.
 *
 * Duration is overridable via CTE_LEAK_STRESS_SECONDS (the force-net variant
 * routes every op over the loopback net path, so it uses a shorter window).
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "clio_ctp/memory/allocator/leak_checker.h"
#include "simple_test.h"

// 1 MiB per blob: draws from the bdev's 1 MiB size-class partition, the same
// path the block-reuse regression exercises.
static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;

// Explicitly-registered RAM bdev target for this test. GetTargetInfo() matches
// on this exact name, so registering it ourselves (rather than via a compose
// config) removes any name-resolution ambiguity.
static const char *kTargetName = "leak_stress_target";
static constexpr clio::run::u64 kTargetSize = 256ULL * 1024 * 1024;  // 256 MiB

// Loop duration. Default 60 s (the requested "1 minute" stress); overridable so
// the loopback force-net variant can use a shorter, still-representative window.
static int StressSecondsFromEnv() {
  if (const char *e = std::getenv("CTE_LEAK_STRESS_SECONDS")) {
    int n = std::atoi(e);
    if (n > 0) return n;
  }
  return 60;
}
static const int kStressSeconds = StressSecondsFromEnv();

class BdevLeakStressFixture {
 public:
  bool initialized_ = false;

  BdevLeakStressFixture() {
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    SetupTarget();
    initialized_ = true;
  }

  ~BdevLeakStressFixture() = default;

  // Create a single 256 MiB RAM bdev and register it with the CTE under
  // kTargetName. Small on purpose: only ~1 blob is ever live, so it is ample for
  // a clean run but would be exhausted quickly by a block leak.
  void SetupTarget() {
    auto *cte = CLIO_CTE_CLIENT;

    clio::run::PoolId bdev_pool_id(901, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();

    auto reg = cte->AsyncRegisterTarget(
        kTargetName, clio::run::bdev::BdevType::kRam, kTargetSize,
        clio::run::PoolQuery::Local(), bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
  }
};

static BdevLeakStressFixture *g_fixture = nullptr;

/**
 * Query the bdev target's remaining allocatable bytes via the CTE client.
 *
 * GetTargetInfo does NOT query the allocator — it returns the cached
 * TargetInfo::remaining_space_, which is only refreshed when a StatTargets
 * task runs. That task is periodic at config_.performance_.stat_targets_
 * period_ms_, default *5000 ms*. Reading GetTargetInfo on its own therefore
 * reports whatever the allocator looked like up to five seconds ago.
 *
 * So force a one-shot StatTargets and wait for it before reading. Without
 * this, the value below is a stale snapshot and any leak assertion built on
 * it is a coin flip.
 */
static clio::run::u64 TargetRemaining() {
  auto *cte = CLIO_CTE_CLIENT;
  // period_ms = 0 -> one-shot: refresh the cache now rather than waiting for
  // the next periodic tick.
  auto refresh = cte->AsyncStatTargets();
  refresh.Wait();
  REQUIRE(refresh->GetReturnCode() == 0);

  auto info = cte->AsyncGetTargetInfo(kTargetName);
  info.Wait();
  REQUIRE(info->GetReturnCode() == 0);
  return info->remaining_space_;
}

// Poll the runtime-heap counter until it stops changing (so async server-side
// frees that lag the client's Wait() have settled), or a short bound elapses.
/**
 * Read the bdev target's remaining bytes, waiting for it to stop moving first.
 *
 * DelBlob's block free completes asynchronously, so sampling immediately after
 * the loop can catch the tail of the last cycle still in flight.
 *
 * Mirrors StabilizeRuntimeHeap: poll until two consecutive reads agree, with a
 * bounded wait so a genuine leak still fails rather than hanging. This is only
 * meaningful because TargetRemaining() forces a StatTargets refresh per read —
 * polling the cached value alone would "stabilize" instantly on a stale number
 * that nothing was updating, since the periodic refresh (5 s) is far longer
 * than this loop's 1 s bound.
 *
 * NOTE: an earlier revision of this comment claimed the Windows force-net
 * failure was simply this settling race, with a "constant 65536-byte residue"
 * as the tell. That was wrong — adding the stabilize loop did not fix it (see
 * issue #791). The stale-cache problem described above is a real defect in how
 * this test measures, but whether it fully accounts for #791 is unconfirmed.
 */
static clio::run::u64 StabilizeTargetRemaining() {
  clio::run::u64 prev = TargetRemaining();
  for (int i = 0; i < 50; ++i) {  // up to ~1 s
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    clio::run::u64 cur = TargetRemaining();
    if (cur == prev) return cur;
    prev = cur;
  }
  return prev;
}

static size_t StabilizeRuntimeHeap() {
  auto *ipc = CLIO_IPC;
  size_t prev = ipc->GetRuntimeHeapAllocatedBytes();
  for (int i = 0; i < 50; ++i) {  // up to ~1 s
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    size_t cur = ipc->GetRuntimeHeapAllocatedBytes();
    if (cur == prev) return cur;
    prev = cur;
  }
  return prev;
}

/**
 * Put + Del the same key for kStressSeconds, then assert no leak at the bdev
 * block level (used capacity back to 0) and the runtime allocator level (heap
 * counter flat).
 */
TEST_CASE("BdevLeakStress - PutBlob/DelBlob loop leaves no leaks",
          "[bdev][leak][stress]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte = CLIO_CTE_CLIENT;
  REQUIRE(cte != nullptr);

  clio::cte::core::Tag tag("leak_stress_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob_name = "leak_stress_blob";

  auto shm = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm.IsNull());
  std::memset(shm.ptr_, 0xAB, kBlobSize);
  ctp::ipc::ShmPtr<> shm_ptr = shm.shm_.template Cast<void>();

  // Warm-up cycle: force first-touch lazy allocations (SHM segment growth,
  // target metadata) so they are folded into the baseline, not counted as a
  // leak.
  {
    auto put = tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, 1.0f);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    auto del = cte->AsyncDelBlob(tag_id, blob_name);
    del.Wait();
    REQUIRE(del->GetReturnCode() == 0);
  }

  const clio::run::u64 remaining_before = StabilizeTargetRemaining();
  const size_t heap_before = StabilizeRuntimeHeap();

  INFO("Baseline: bdev remaining=" << remaining_before
       << " runtime-heap=" << heap_before
       << " running Put/Del for " << kStressSeconds << "s");

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kStressSeconds);
  long iters = 0;
  int put_fail = 0;
  int del_fail = 0;
  // Sampled once, after the very first completed cycle. If a residue is
  // present here it is a one-time cost paid at loop entry; if it only shows
  // up in the final reading it accrues later. Issue #791 sees a residue of
  // exactly 65536 on Windows whose origin is not yet identified, and this
  // narrows where to look without needing another CI round-trip. Sampled
  // once rather than per-iteration because TargetRemaining() now forces a
  // StatTargets round-trip and would otherwise distort the loop.
  clio::run::u64 remaining_after_first = 0;

  while (std::chrono::steady_clock::now() < deadline) {
    auto put = tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, 1.0f);
    put.Wait();
    if (put->GetReturnCode() != 0) {
      ++put_fail;
      INFO("PutBlob failed at iter " << iters
                                     << " rc=" << put->GetReturnCode());
      break;
    }

    auto del = cte->AsyncDelBlob(tag_id, blob_name);
    del.Wait();
    if (del->GetReturnCode() != 0) {
      ++del_fail;
      INFO("DelBlob failed at iter " << iters
                                     << " rc=" << del->GetReturnCode());
      break;
    }
    ++iters;
    if (iters == 1) remaining_after_first = TargetRemaining();
  }

  CLIO_IPC->FreeBuffer(shm);

  // Let any trailing async frees settle before measuring. This MUST stabilize
  // rather than sample once: the previous code took this reading before
  // StabilizeRuntimeHeap() ran, so nothing had actually settled despite the
  // comment saying so.
  const clio::run::u64 remaining_after = StabilizeTargetRemaining();
  const size_t heap_after = StabilizeRuntimeHeap();

  const clio::run::u64 bdev_used =
      (remaining_before > remaining_after) ? (remaining_before - remaining_after)
                                           : 0;
  const size_t heap_growth =
      (heap_after > heap_before) ? (heap_after - heap_before) : 0;

  INFO("Completed " << iters << " Put/Del cycles ("
       << (static_cast<double>(iters) * kBlobSize / (1024.0 * 1024.0))
       << " MiB cumulative); bdev_used_after=" << bdev_used
       << " remaining_before=" << remaining_before
       << " remaining_after_first_cycle=" << remaining_after_first
       << " remaining_after=" << remaining_after
       << " runtime_heap_growth=" << heap_growth
       << " allocator_leak_checker_total="
       << ctp::ipc::AllocatorLeakChecker::Get().TotalLeakedBytes());

  // Every op must have succeeded (a block leak would exhaust the 256 MiB tier).
  REQUIRE(put_fail == 0);
  REQUIRE(del_fail == 0);
  REQUIRE(iters > 0);

  // (1) BDEV BLOCK LEVEL: used capacity back to zero — every DelBlob returned
  //     its blocks. Exact: block alloc/free is deterministic.
  REQUIRE(remaining_after == remaining_before);

  // (2) RUNTIME ALLOCATOR LEVEL: the leak-checker heap counter did not grow.
  //     Allow a couple of in-flight buffers of slack; a real per-op leak dwarfs
  //     this after tens of thousands of cycles. No-op in non-leak builds
  //     (counter is a constant 0).
  const size_t kHeapTolerance = 2 * kBlobSize;
  REQUIRE(heap_growth <= kHeapTolerance);

  auto del_tag = cte->AsyncDelTag("leak_stress_tag");
  del_tag.Wait();
}

int main(int argc, char **argv) {
  g_fixture = new BdevLeakStressFixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return result;
}

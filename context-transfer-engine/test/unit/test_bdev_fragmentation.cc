/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

// bdev fragmentation reuse (issue #820).
//
// Fragmentation avoidance moved OUT of the CTE (ExtendBlob used to cap every
// block at 64 KB) and INTO the bdev allocator, which now assembles a large
// request from several smaller freed extents. This is the decisive regression
// test for that move -- the scenario xfstests 074/521/522 guard.
//
// The trap the old design fell into: the free list is bucketed by size
// category, so a 1 MiB ask only ever probes the 1 MiB bucket and can never see
// freed 8 KiB blocks. So:
//   1. Fill a SMALL target to near-capacity with 8 KiB blobs (heap ~= cap).
//   2. Delete them all -> the free pool is now ~all of the target, but entirely
//      in 8 KiB extents; the bump-only heap watermark stays at capacity.
//   3. Allocate LARGE blobs (1 MiB) and KEEP them (no large-to-large reuse to
//      mask the effect). Their bytes can ONLY come from the freed 8 KiB extents.
//      Total large bytes far exceed the heap headroom.
// Without the fix, step 3 EIOs once the couple-MiB of headroom is gone. With it,
// every large put succeeds and reads back intact.

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "frag_small_target";
// Deliberately SMALL so "heap near capacity" is reachable in a few thousand
// small blobs rather than a million.
constexpr clio::run::u64 kTargetSize = 32ULL * 1024 * 1024;  // 32 MiB

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
    clio::run::PoolId bdev_pool_id(931, 0);
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

TEST_CASE("Large alloc served from fragmented small free extents (#820)",
          "[cte][bdev][frag][820]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag("frag_small_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  const size_t kSmall = 8 * 1024;
  const size_t kLarge = 1024 * 1024;

  // 1. Fill to near-capacity with 8 KiB blobs.
  auto sb = ipc->AllocateBuffer(kSmall);
  REQUIRE(!sb.IsNull());
  std::memset(sb.ptr_, 'x', kSmall);
  int filled = 0;
  // Fill until the tier actually REFUSES a put -- that is what drives the
  // bump-only heap watermark to capacity, which is the precondition for the
  // regression (a later large ask can't heap-bump and must reuse). The cap is a
  // generous upper bound; the RAM bdev's real capacity decides where it stops.
  const int kMaxSmall = 200000;
  for (int i = 0; i < kMaxSmall; ++i) {
    const std::string nm = "f_" + std::to_string(i);
    auto p = cte->AsyncPutBlob(tag_id, nm, 0, kSmall,
                               ctp::ipc::ShmPtr<>(sb.shm_));
    p.Wait();
    if (p->GetReturnCode() != 0) break;  // tier full
    ++filled;
  }
  ipc->FreeBuffer(sb);
  REQUIRE(filled > 0);
  // We must have driven the heap near capacity -- most of the target is now
  // consumed by 8 KiB blobs.
  REQUIRE(static_cast<clio::run::u64>(filled) * kSmall >= kTargetSize / 2);

  // 2. Delete them all -> ~all of the target free, in 8 KiB extents; heap
  //    watermark stays high.
  for (int i = 0; i < filled; ++i) {
    const std::string nm = "f_" + std::to_string(i);
    auto d = cte->AsyncDelBlob(tag_id, nm);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
  }

  // 3. Allocate LARGE blobs, keeping them. Total far exceeds any plausible heap
  //    headroom, so success requires reusing the small freed extents.
  std::vector<char> lbuf(kLarge);
  for (size_t j = 0; j < kLarge; ++j) lbuf[j] = static_cast<char>((j * 13) % 251);
  const int kNumLarge = 20;  // 20 MiB, >> the ~few MiB of heap headroom
  int large_ok = 0;
  for (int i = 0; i < kNumLarge; ++i) {
    auto lb = ipc->AllocateBuffer(kLarge);
    REQUIRE(!lb.IsNull());
    std::memcpy(lb.ptr_, lbuf.data(), kLarge);
    const std::string nm = "L_" + std::to_string(i);
    auto p = cte->AsyncPutBlob(tag_id, nm, 0, kLarge,
                               ctp::ipc::ShmPtr<>(lb.shm_));
    p.Wait();
    ipc->FreeBuffer(lb);
    if (p->GetReturnCode() != 0) break;
    ++large_ok;

    // Read back: a large blob assembled from many small physical blocks must
    // reconstruct exactly.
    auto rd = ipc->AllocateBuffer(kLarge);
    REQUIRE(!rd.IsNull());
    std::memset(rd.ptr_, 0, kLarge);
    auto g = cte->AsyncGetBlob(tag_id, nm, 0, kLarge, /*flags=*/0,
                               ctp::ipc::ShmPtr<>(rd.shm_));
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(std::memcmp(rd.ptr_, lbuf.data(), kLarge) == 0);
    ipc->FreeBuffer(rd);
  }

  std::printf("[#820] filled %d x 8KiB, freed, then %d/%d x 1MiB large allocs "
              "succeeded from fragmented free space\n",
              filled, large_ok, kNumLarge);
  // The whole point: every large alloc had to be assembled from freed 8 KiB
  // extents, because the heap had no room and no large contiguous free block
  // existed.
  REQUIRE(large_ok == kNumLarge);
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}

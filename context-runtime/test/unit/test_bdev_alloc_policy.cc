/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Unit tests for bdev::AllocPolicy (page-allocation policy for memory-backed
 * bdevs) and the eager preallocation it drives in MemBdevTransport.
 *
 * Coverage:
 *   A. CreateParams serialization round-trip (alloc_policy_ crosses IPC).
 *   B. CreateParams::LoadConfig YAML parsing of the `alloc:` key.
 *   C. Policy resolution + preallocation behaviour in MemBdevTransport::Init,
 *      observed through IsRamPageCommitted / CommittedRamBytes.
 *   D. The read contract preallocation has to keep intact: a region that was
 *      never written reads back as zeros. Driven through the public Read/Write
 *      path, not through the transport's internals.
 *
 * These are pure host tests: MemBdevTransport is exercised directly, with tasks
 * pointing at ordinary host buffers (no server, no shared memory, no GPU
 * required), so they run in the default suite.
 */

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <string>
#include <vector>

#include "clio_ctp/data_structures/serialization/local_serialize.h"
#include "clio_runtime/bdev/bdev_tasks.h"
#include "clio_runtime/bdev/transports/mem_bdev_transport.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/ipc_manager.h"
#include "simple_test.h"

using clio::run::bdev::AllocPolicy;
using clio::run::bdev::BdevType;
using clio::run::bdev::Block;
using clio::run::bdev::CreateParams;
using clio::run::bdev::MemBdevTransport;
using clio::run::bdev::PersistenceLevel;
using clio::run::bdev::ReadTask;
using clio::run::bdev::WriteTask;

namespace {

// Small sized pool: one 1 GiB RAM page once preallocated.
constexpr clio::run::u64 kPoolSize = 16 * 1024 * 1024;  // 16 MB

/** Bytes a pool of `capacity` commits once preallocated: capacity rounded up to
 * whole RAM pages (a committed page costs a full page however little is used). */
clio::run::u64 PreallocatedBytes(clio::run::u64 capacity) {
  const clio::run::u64 page = MemBdevTransport::RamPageSizeBytes();
  return ((capacity + page - 1) / page) * page;
}

/** Build a PoolConfig whose YAML body is `body`. */
clio::run::PoolConfig MakePoolConfig(const std::string &body) {
  clio::run::PoolConfig cfg;
  cfg.mod_name_ = "clio_bdev";
  cfg.pool_name_ = "alloc_policy_test";
  cfg.config_ = body;
  return cfg;
}

/** Serialize `src` and deserialize it into `dst` using the runtime's archive. */
void RoundTrip(CreateParams &src, CreateParams &dst) {
  std::string buf;
  ctp::ipc::LocalSerialize<std::string> save(buf);
  save(src);
  save.Finalize();
  REQUIRE(!buf.empty());
  ctp::ipc::LocalDeserialize<std::string> load(buf);
  load(dst);
}

/** Build a CreateParams without tripping the ambiguous 4-arg ctor overload. */
CreateParams MakeParams(BdevType type, clio::run::u64 size,
                        clio::run::u32 io_depth, clio::run::u32 alignment) {
  CreateParams p;
  p.bdev_type_ = type;
  p.total_size_ = size;
  p.io_depth_ = io_depth;
  p.alignment_ = alignment;
  return p;
}

/** A CreateParams whose every field differs from a default-constructed one, so
 * a round-trip that silently drops a field is visible. */
CreateParams PoisonedParams() {
  CreateParams p = MakeParams(BdevType::kPinned, /*total_size=*/12345,
                              /*io_depth=*/7, /*alignment=*/512);
  p.perf_metrics_.read_bandwidth_mbps_ = 1.5;
  p.perf_metrics_.write_bandwidth_mbps_ = 2.5;
  p.perf_metrics_.read_latency_us_ = 3.5;
  p.perf_metrics_.write_latency_us_ = 4.5;
  p.perf_metrics_.iops_ = 5.5;
  p.persistence_level_ = PersistenceLevel::kLongTerm;
  p.alloc_policy_ = AllocPolicy::kLazy;
  return p;
}

/** Init a MemBdevTransport with the given type/size/policy. */
void InitTransport(MemBdevTransport &t, BdevType type, clio::run::u64 size,
                   AllocPolicy policy) {
  CreateParams params = MakeParams(type, size, /*io_depth=*/32,
                                   /*alignment=*/4096);
  params.alloc_policy_ = policy;
  REQUIRE(t.Init(params, "alloc_policy_test", nullptr));
}

// --- Driving the public Read/Write path without a runtime -------------------
//
// WriteBlocks/ReadBlocks are C++20 coroutines that a worker would normally
// start. Both resolve their data buffer through CLIO_IPC->ToFullPtr(ShmPtr),
// whose documented "private memory" case — a null AllocatorId, with the raw
// address carried in the offset — is exactly what FullPtr's raw-pointer
// constructor produces. So a host buffer can be handed to a task directly, and
// the coroutine (which never awaits on the host-buffer path: see
// Read/WriteBlocksCpu) runs to completion in a single resume(). No server, no
// shared memory, no GPU.

/** One block covering [offset, offset + size) of the pool. */
clio::run::priv::vector<Block> OneBlock(clio::run::u64 offset,
                                        clio::run::u64 size) {
  clio::run::priv::vector<Block> blocks(CLIO_PRIV_ALLOC);
  blocks.push_back(Block(offset, size, /*block_type=*/0u));
  return blocks;
}

/** Write `data` to the pool at `offset` through the public WriteBlocks path. */
void WriteRegion(MemBdevTransport &t, clio::run::u64 offset,
                 const std::vector<char> &data) {
  WriteTask task;
  task.blocks_ = OneBlock(offset, data.size());
  task.data_ = ctp::ipc::FullPtr<char>(const_cast<char *>(data.data()))
                   .shm_.template Cast<void>();
  task.length_ = data.size();
  task.return_code_ = 1;  // poison: an un-run task must not look successful

  clio::run::TaskResume resume =
      t.WriteBlocks(ctp::ipc::FullPtr<WriteTask>(&task));
  resume.resume();
  REQUIRE(resume.done());
  REQUIRE(task.return_code_ == 0);
  REQUIRE(task.bytes_written_ == data.size());
}

/** Read `size` bytes from the pool at `offset` through the public ReadBlocks
 * path. The destination is pre-poisoned with 0xEE, so a read that silently
 * copies nothing cannot masquerade as a read that returned zeros. */
std::vector<char> ReadRegion(MemBdevTransport &t, clio::run::u64 offset,
                             size_t size) {
  std::vector<char> out(size, static_cast<char>(0xEE));

  ReadTask task;
  task.blocks_ = OneBlock(offset, size);
  task.data_ = ctp::ipc::FullPtr<char>(out.data()).shm_.template Cast<void>();
  task.length_ = size;
  task.return_code_ = 1;  // poison

  clio::run::TaskResume resume =
      t.ReadBlocks(ctp::ipc::FullPtr<ReadTask>(&task));
  resume.resume();
  REQUIRE(resume.done());
  REQUIRE(task.return_code_ == 0);
  REQUIRE(task.bytes_read_ == size);
  return out;
}

/** True iff every byte of `buf` in [from, to) equals `want`. */
bool AllBytesAre(const std::vector<char> &buf, size_t from, size_t to,
                 unsigned char want) {
  for (size_t i = from; i < to; ++i) {
    if (static_cast<unsigned char>(buf[i]) != want) return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// A. Serialization round-trip. CreateParams crosses IPC; if alloc_policy_ is
//    ever dropped from serialize()'s ar(...), everything still compiles and the
//    policy silently never reaches the server. The destination is pre-poisoned
//    with a *different* policy so a dropped field cannot be masked by the
//    receiver's default.
// ---------------------------------------------------------------------------
TEST_CASE("BdevAllocPolicy - CreateParams round-trips alloc_policy_",
          "[bdev][alloc_policy][serialize]") {
  const AllocPolicy kPolicies[] = {AllocPolicy::kAuto, AllocPolicy::kEager,
                                   AllocPolicy::kLazy};
  for (AllocPolicy policy : kPolicies) {
    CreateParams src = MakeParams(BdevType::kRam, kPoolSize, /*io_depth=*/16,
                                  /*alignment=*/512);
    src.persistence_level_ = PersistenceLevel::kTemporaryNonVolatile;
    src.alloc_policy_ = policy;

    // Poison every field of the destination: a dropped field leaves the poison
    // value behind, which cannot equal the source's value.
    CreateParams dst = PoisonedParams();
    dst.alloc_policy_ = (policy == AllocPolicy::kLazy) ? AllocPolicy::kEager
                                                       : AllocPolicy::kLazy;
    REQUIRE(dst.alloc_policy_ != policy);

    RoundTrip(src, dst);

    REQUIRE(dst.alloc_policy_ == policy);
    // The test doubles as a guard on the rest of the struct.
    REQUIRE(dst.bdev_type_ == BdevType::kRam);
    REQUIRE(dst.total_size_ == kPoolSize);
    REQUIRE(dst.io_depth_ == 16u);
    REQUIRE(dst.alignment_ == 512u);
    REQUIRE(dst.persistence_level_ == PersistenceLevel::kTemporaryNonVolatile);
    REQUIRE(dst.perf_metrics_.read_bandwidth_mbps_ ==
            src.perf_metrics_.read_bandwidth_mbps_);
    REQUIRE(dst.perf_metrics_.write_bandwidth_mbps_ ==
            src.perf_metrics_.write_bandwidth_mbps_);
    REQUIRE(dst.perf_metrics_.read_latency_us_ ==
            src.perf_metrics_.read_latency_us_);
    REQUIRE(dst.perf_metrics_.write_latency_us_ ==
            src.perf_metrics_.write_latency_us_);
    REQUIRE(dst.perf_metrics_.iops_ == src.perf_metrics_.iops_);
  }
}

// ---------------------------------------------------------------------------
// B. Config parsing: `alloc: auto|eager|lazy`, case-insensitive, unknown -> kAuto,
//    absent -> kAuto (the default).
// ---------------------------------------------------------------------------
TEST_CASE("BdevAllocPolicy - LoadConfig parses the alloc key",
          "[bdev][alloc_policy][config]") {
  SECTION("alloc: auto");
  {
    CreateParams p;
    p.alloc_policy_ = AllocPolicy::kLazy;  // poison
    p.LoadConfig(MakePoolConfig("bdev_type: ram\ncapacity: 16MB\nalloc: auto\n"));
    REQUIRE(p.alloc_policy_ == AllocPolicy::kAuto);
    REQUIRE(p.bdev_type_ == BdevType::kRam);
  }

  SECTION("alloc: eager");
  {
    CreateParams p;
    p.LoadConfig(MakePoolConfig("bdev_type: ram\ncapacity: 16MB\nalloc: eager\n"));
    REQUIRE(p.alloc_policy_ == AllocPolicy::kEager);
  }

  SECTION("alloc: lazy");
  {
    CreateParams p;
    p.LoadConfig(MakePoolConfig("bdev_type: ram\ncapacity: 16MB\nalloc: lazy\n"));
    REQUIRE(p.alloc_policy_ == AllocPolicy::kLazy);
  }

  SECTION("alloc is case-insensitive");
  {
    CreateParams p;
    p.LoadConfig(MakePoolConfig("alloc: EAGER\n"));
    REQUIRE(p.alloc_policy_ == AllocPolicy::kEager);

    CreateParams p2;
    p2.LoadConfig(MakePoolConfig("alloc: LaZy\n"));
    REQUIRE(p2.alloc_policy_ == AllocPolicy::kLazy);
  }

  SECTION("unknown alloc value falls back to auto");
  {
    CreateParams p;
    p.alloc_policy_ = AllocPolicy::kLazy;  // poison
    p.LoadConfig(MakePoolConfig("bdev_type: ram\nalloc: bogus\n"));
    REQUIRE(p.alloc_policy_ == AllocPolicy::kAuto);
  }

  SECTION("absent alloc key leaves the default (auto)");
  {
    CreateParams p;
    REQUIRE(p.alloc_policy_ == AllocPolicy::kAuto);  // struct default
    p.LoadConfig(MakePoolConfig("bdev_type: ram\ncapacity: 16MB\n"));
    REQUIRE(p.alloc_policy_ == AllocPolicy::kAuto);
    REQUIRE(p.total_size_ == 16ull * 1024 * 1024);
  }
}

// ---------------------------------------------------------------------------
// C. Policy resolution + preallocation, observed through the transport's
//    introspection API: IsRamPageCommitted(0) says whether Init() committed the
//    pool's first page before any I/O ran, and CommittedRamBytes() says how much
//    host RAM the pool took in total — 0 for a pool that has not preallocated,
//    the declared capacity rounded up to whole pages for one that has.
// ---------------------------------------------------------------------------
TEST_CASE("BdevAllocPolicy - eager + explicit size preallocates at Init",
          "[bdev][alloc_policy][prealloc]") {
  MemBdevTransport t;
  InitTransport(t, BdevType::kRam, kPoolSize, AllocPolicy::kEager);
  REQUIRE(t.IsRamPageCommitted(0));
  REQUIRE(t.CommittedRamBytes() == PreallocatedBytes(kPoolSize));
}

TEST_CASE("BdevAllocPolicy - auto + explicit size preallocates at Init",
          "[bdev][alloc_policy][prealloc]") {
  // The default path. Must not have changed.
  MemBdevTransport t;
  InitTransport(t, BdevType::kRam, kPoolSize, AllocPolicy::kAuto);
  REQUIRE(t.IsRamPageCommitted(0));
  REQUIRE(t.CommittedRamBytes() == PreallocatedBytes(kPoolSize));
}

TEST_CASE("BdevAllocPolicy - auto + unsized pool does not preallocate",
          "[bdev][alloc_policy][prealloc]") {
  // total_size_ == 0 -> capacity falls back to DefaultRamCapacityBytes()
  // (~80% of DRAM), which must never be committed eagerly. CommittedRamBytes()
  // is the direct statement of that: the pool holds nothing at Init.
  MemBdevTransport t;
  InitTransport(t, BdevType::kRam, /*size=*/0, AllocPolicy::kAuto);
  REQUIRE(!t.IsRamPageCommitted(0));
  REQUIRE(t.CommittedRamBytes() == 0u);
}

TEST_CASE("BdevAllocPolicy - lazy + explicit size does not preallocate",
          "[bdev][alloc_policy][prealloc]") {
  MemBdevTransport t;
  InitTransport(t, BdevType::kRam, kPoolSize, AllocPolicy::kLazy);
  REQUIRE(!t.IsRamPageCommitted(0));
  REQUIRE(t.CommittedRamBytes() == 0u);
}

TEST_CASE("BdevAllocPolicy - eager + unsized pool degrades to lazy (safety)",
          "[bdev][alloc_policy][prealloc]") {
  // Safety property: eager cannot know the size here, and the kRam fallback is
  // ~80% of DRAM. Committing that would OOM the node, so eager must degrade to
  // lazy rather than preallocate. CommittedRamBytes() == 0 is the assertion
  // that no such commit happened.
  MemBdevTransport t;
  InitTransport(t, BdevType::kRam, /*size=*/0, AllocPolicy::kEager);
  REQUIRE(!t.IsRamPageCommitted(0));
  REQUIRE(t.CommittedRamBytes() == 0u);
}

TEST_CASE("BdevAllocPolicy - a lazy pool commits a page on first write",
          "[bdev][alloc_policy][prealloc]") {
  // The other half of the lazy contract: it does not preallocate, but it must
  // still commit the page when the I/O actually needs it.
  MemBdevTransport t;
  InitTransport(t, BdevType::kRam, kPoolSize, AllocPolicy::kLazy);
  REQUIRE(t.CommittedRamBytes() == 0u);

  WriteRegion(t, /*offset=*/0, std::vector<char>(4096, static_cast<char>(0x5A)));

  REQUIRE(t.IsRamPageCommitted(0));
  REQUIRE(t.CommittedRamBytes() == MemBdevTransport::RamPageSizeBytes());
}

// ---------------------------------------------------------------------------
// D. The zero-fill invariant, stated as the user-visible contract it actually
//    is: a read of a region that was never written returns zeros.
//
//    This used to be free — an untouched page was a null pointer, and the read
//    path memsets the destination when a page is missing. Preallocation removes
//    that: the page is committed at Init, so an unwritten read now copies out of
//    the page itself, and for kPinned that page came from cudaMallocHost, which
//    does NOT zero-initialize. Without the memset in PreallocateRamPages the
//    read below hands back whatever the allocator last had there.
//
//    Each test therefore reads a window that straddles a written and an
//    unwritten region: the written half proves the read really is coming out of
//    the committed page (so the unwritten half cannot be passing vacuously via
//    the missing-page memset), and the unwritten half is the invariant.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t kHalf = 4096;
constexpr size_t kWindow = 2 * kHalf;
constexpr unsigned char kMarker = 0xA5;

/** Under eager preallocation, [0, kHalf) is never written and must read as
 * zeros, while [kHalf, kWindow) is written and must read back the marker. */
void CheckUnwrittenReadsAsZeros(BdevType type) {
  MemBdevTransport t;
  InitTransport(t, type, kPoolSize, AllocPolicy::kEager);
  // Precondition: the page really is committed, so the read below goes through
  // the page and not through the read path's missing-page zero fill.
  REQUIRE(t.IsRamPageCommitted(0));

  WriteRegion(t, /*offset=*/kHalf,
              std::vector<char>(kHalf, static_cast<char>(kMarker)));

  std::vector<char> got = ReadRegion(t, /*offset=*/0, kWindow);
  REQUIRE(AllBytesAre(got, kHalf, kWindow, kMarker));  // the read is live
  REQUIRE(AllBytesAre(got, 0, kHalf, 0x00));           // the invariant
}

}  // namespace

TEST_CASE("BdevAllocPolicy - eager kRam: an unwritten region reads back zeros",
          "[bdev][alloc_policy][zerofill]") {
  CheckUnwrittenReadsAsZeros(BdevType::kRam);
}

TEST_CASE("BdevAllocPolicy - eager kPinned: an unwritten region reads back zeros",
          "[bdev][alloc_policy][zerofill]") {
  // On a GPU build this page is cudaMallocHost memory (uninitialized); on a
  // non-GPU build MallocHost returns nullptr and the page falls back to
  // pageable `new char[]` (also uninitialized). Either way the invariant is
  // the memset in PreallocateRamPages.
  CheckUnwrittenReadsAsZeros(BdevType::kPinned);
}

SIMPLE_TEST_MAIN()

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
 * Daemon-level end-to-end test for the safe_bdev ChiMod (issue #543).
 *
 * Exercises the RAID-0-data + dedicated-parity model over RAM-backed member
 * bdevs:
 *   1. Roundtrip + recovery: k=3 data + 1 parity (max_failures=1); write a
 *      multi-chunk striped pattern; read it back; flush parity; fault a data
 *      member; degraded read reconstructs; RecoverBdev onto a fresh member;
 *      read back.
 *   2. RAID-0 striping: a > k*kChunkLen write physically lands on DIFFERENT
 *      data members (verified by reading members directly at mapped offsets).
 *   3. Reclaim: allocate, free, allocate again -> the space is reused
 *      (allocator is not a bump pointer).
 *   4. Superblock reattach + foreign-refuse.
 *   5. Partial-chunk write/read.
 */

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

using namespace std::chrono_literals;

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

// Reused bdev client (members are plain bdev pools).
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

// safe_bdev client and tasks.
#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>

// Admin client for pool management.
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

// msgpack for parsing Monitor "stats" output (reattached_members).
#include <clio_ctp/serialize/msgpack_wrapper.h>

namespace {

bool g_initialized = false;

// Per-member RAM size: must hold the reserved EC region. The runtime reserves
// num_rows * kChunkLen per member where kChunkLen = 64KiB. 4 MiB comfortably
// holds many rows.
constexpr clio::run::u64 kMemberRamSize = 4 * 1024 * 1024;
constexpr clio::run::u64 kChunkLen = 65536;        // mirrors Runtime::kChunkLen
constexpr clio::run::u64 kSuperblockSize = 65536;  // mirrors Runtime::kSuperblockSize

/** Initialize Chimaera once for the test suite. */
void EnsureInit() {
  if (!g_initialized) {
    HLOG(kInfo, "Initializing Chimaera (safe_bdev test)...");
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    if (success) {
      g_initialized = true;
      SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
      std::this_thread::sleep_for(500ms);
      HLOG(kInfo, "Chimaera initialization successful");
    } else {
      HLOG(kInfo, "Failed to initialize Chimaera");
    }
  }
}

/** Create a RAM-backed member bdev pool. */
bool CreateRamMember(clio::run::bdev::Client &client,
                     const std::string &pool_name,
                     const clio::run::PoolId &pool_id) {
  auto create_task = client.AsyncCreate(clio::run::PoolQuery::Dynamic(), pool_name,
                                        pool_id, clio::run::bdev::BdevType::kRam,
                                        kMemberRamSize);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  client.return_code_ = create_task->return_code_;
  return create_task->GetReturnCode() == 0;
}

/**
 * Create a FILE-backed member bdev pool over `file_path`. File-backed members
 * persist their bytes across a safe-bdev pool destroy, which is what makes the
 * restart/recovery test meaningful: re-creating a bdev pool over the SAME file
 * re-exposes the SAME data (including the safe-bdev superblock).
 */
bool CreateFileMember(clio::run::bdev::Client &client,
                      const std::string &file_path,
                      const clio::run::PoolId &pool_id) {
  auto create_task = client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), file_path, pool_id,
      clio::run::bdev::BdevType::kFile, kMemberRamSize);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  client.return_code_ = create_task->return_code_;
  return create_task->GetReturnCode() == 0;
}

/** Whether two [offset,size) logical ranges overlap. */
bool RangesOverlap(clio::run::u64 a_off, clio::run::u64 a_size, clio::run::u64 b_off,
                   clio::run::u64 b_size) {
  return a_off < b_off + b_size && b_off < a_off + a_size;
}

/** A repeating, position-dependent byte pattern. */
std::vector<ctp::u8> MakePattern(size_t size, ctp::u8 seed) {
  std::vector<ctp::u8> v(size);
  for (size_t i = 0; i < size; ++i) {
    v[i] = static_cast<ctp::u8>((seed + i * 7 + (i >> 8)) & 0xFF);
  }
  return v;
}

// Forward decl (defined later in this TU's anonymous namespace): query a
// scalar field from safe_bdev Monitor("stats"). Used here to assert the
// recovery-observability counters after RecoverBdev.
long QueryStatField(clio::run::safe_bdev::Client &safe, const char *field);

// --- Multi-block I/O helpers -------------------------------------------------
// The dynamic view-group layout allocates PER-CHUNK round-robin, so a request
// of N chunks comes back as N blocks banded across the data members (NOT one
// contiguous block). Tests must therefore carry the FULL block list through
// write + read; these helpers do that (the pattern buffer maps to the blocks in
// order, exactly like the CTE uses a bdev).

/** Sum of a block list's sizes (== total logical bytes it covers). */
clio::run::u64 BlocksLen(const std::vector<clio::run::bdev::Block> &blocks) {
  clio::run::u64 n = 0;
  for (const auto &b : blocks) n += b.size_;
  return n;
}

/** Rebuild a runtime priv::vector<Block> from a std::vector<Block>. */
clio::run::priv::vector<clio::run::bdev::Block> ToPriv(
    const std::vector<clio::run::bdev::Block> &blocks) {
  clio::run::priv::vector<clio::run::bdev::Block> v(CTP_MALLOC);
  for (const auto &b : blocks) v.push_back(b);
  return v;
}

/** Allocate `io_len` bytes; return the FULL block list (across members). */
std::vector<clio::run::bdev::Block> AllocBlocks(clio::run::safe_bdev::Client &safe,
                                                clio::run::u64 io_len) {
  auto alloc = safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), io_len);
  alloc.Wait();
  REQUIRE(alloc->GetReturnCode() == 0);
  REQUIRE(alloc->blocks_.size() > 0);
  std::vector<clio::run::bdev::Block> v;
  for (size_t i = 0; i < alloc->blocks_.size(); ++i) v.push_back(alloc->blocks_[i]);
  return v;
}

/** Write `pattern` (== BlocksLen(blocks)) across the block list. */
void WriteBlocks(clio::run::safe_bdev::Client &safe,
                 const std::vector<clio::run::bdev::Block> &blocks,
                 const std::vector<ctp::u8> &pattern) {
  const clio::run::u64 len = pattern.size();
  auto wb = ToPriv(blocks);
  auto wbuf = CLIO_IPC->AllocateBuffer(len);
  REQUIRE_FALSE(wbuf.IsNull());
  memcpy(wbuf.ptr_, pattern.data(), len);
  auto wt = safe.AsyncWrite(clio::run::PoolQuery::Dynamic(), wb,
                            wbuf.shm_.template Cast<void>(), len);
  wt.Wait();
  REQUIRE(wt->GetReturnCode() == 0);
  REQUIRE(wt->bytes_written_ == len);
  CLIO_IPC->FreeBuffer(wbuf);
}

/** Read the block list back into `out` (sized to BlocksLen(blocks)). */
void ReadBlocks(clio::run::safe_bdev::Client &safe,
                const std::vector<clio::run::bdev::Block> &blocks,
                std::vector<ctp::u8> &out) {
  const clio::run::u64 len = BlocksLen(blocks);
  auto rb = ToPriv(blocks);
  auto rbuf = CLIO_IPC->AllocateBuffer(len);
  REQUIRE_FALSE(rbuf.IsNull());
  memset(rbuf.ptr_, 0, len);
  auto rt = safe.AsyncRead(clio::run::PoolQuery::Dynamic(), rb,
                           rbuf.shm_.template Cast<void>(), len);
  rt.Wait();
  REQUIRE(rt->GetReturnCode() == 0);
  REQUIRE(rt->bytes_read_ == len);
  out.resize(len);
  memcpy(out.data(), rbuf.ptr_, len);
  CLIO_IPC->FreeBuffer(rbuf);
}

/** Flush the async parity builder as a durability barrier. */
void FlushParity(clio::run::safe_bdev::Client &safe) {
  auto flush = safe.AsyncBuildParity(clio::run::PoolQuery::Dynamic(), 0);
  flush.Wait();
  REQUIRE(flush->GetReturnCode() == 0);
}

}  // namespace

TEST_CASE("safe_bdev_ec_roundtrip_recovery", "[safe_bdev][ec][recovery]") {
  EnsureInit();
  REQUIRE(g_initialized);

  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "safe_member_" + std::to_string(getpid()) + "_" +
           std::to_string(idx);
  };

  // --- Create k=3 RAM data member pools + 1 parity. ---
  const int k = 3;
  std::vector<clio::run::PoolId> data_ids;
  std::vector<clio::run::bdev::Client> data_clients;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(9100 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
    data_clients.push_back(client);
  }

  // Parity member.
  clio::run::PoolId parity_id(static_cast<clio::run::u32>(9200 + pidsalt), 0);
  clio::run::bdev::Client parity_client(parity_id);
  REQUIRE(CreateRamMember(parity_client, member_name(100), parity_id));
  parity_id = parity_client.pool_id_;

  // --- Create safe_bdev over the 3 data members, max_failures=1. ---
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(9300 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);

  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "safe_bdev_ec_pool", safe_id,
                                      /*max_failures=*/1, members);
  create_task.Wait();
  safe.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->GetReturnCode() == 0);

  // --- AddBdev the parity member so parity_level becomes 1. ---
  auto add_task = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(),
                                    member_name(100), /*node_id=*/0, parity_id,
                                    /*as_parity=*/1);
  add_task.Wait();
  REQUIRE(add_task->GetReturnCode() == 0);

  // --- Allocate a logical block spanning MULTIPLE chunks so it stripes across
  //     all data members, write a known pattern, read it back. ---
  const clio::run::u64 io_len = 2 * static_cast<clio::run::u64>(k) * kChunkLen;  // 6 chunks
  std::vector<clio::run::bdev::Block> blocks = AllocBlocks(safe, io_len);

  std::vector<ctp::u8> pattern = MakePattern(io_len, 0x5A);
  WriteBlocks(safe, blocks, pattern);

  auto read_back = [&](std::vector<ctp::u8> &out) {
    ReadBlocks(safe, blocks, out);
  };

  // (1) Happy-path roundtrip.
  {
    std::vector<ctp::u8> got;
    read_back(got);
    REQUIRE(got == pattern);
    HLOG(kInfo, "safe_bdev EC: happy-path roundtrip OK ({} bytes)", io_len);
  }

  // (2) Degraded read: fault data member 1, reconstruct on read.
  {
    // Parity is computed asynchronously off the write path (the background
    // BuildParity task). Flush it as a durability barrier so the rows are
    // protected before we induce a failure.
    auto flush = safe.AsyncBuildParity(clio::run::PoolQuery::Dynamic(),
                                       /*max_batch=*/0);
    flush.Wait();
    REQUIRE(flush->GetReturnCode() == 0);

    auto rm = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), data_ids[1],
                                   /*was_faulty=*/1);
    rm.Wait();
    REQUIRE(rm->GetReturnCode() == 0);

    std::vector<ctp::u8> got;
    read_back(got);
    REQUIRE(got == pattern);
    HLOG(kInfo, "safe_bdev EC: degraded read (member 1 faulty) reconstructed OK");
  }

  // (3) Recovery onto a fresh member pool.
  {
    clio::run::PoolId recover_id(static_cast<clio::run::u32>(9400 + pidsalt), 0);
    clio::run::bdev::Client recover_client(recover_id);
    REQUIRE(CreateRamMember(recover_client, member_name(200), recover_id));
    recover_id = recover_client.pool_id_;

    auto rec = safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(), data_ids[1],
                                     member_name(200), /*node_id=*/0,
                                     recover_id);
    rec.Wait();
    REQUIRE(rec->GetReturnCode() == 0);

    // Recovery observability: after a successful RecoverBdev the Monitor
    // counters must show a finished rebuild -- a non-zero total (rows were
    // rebuilt), completed == total, and no active/in-flight work remaining.
    const long rec_total = QueryStatField(safe, "recovery_ops_total");
    const long rec_done = QueryStatField(safe, "recovery_ops_completed");
    const long rec_remaining = QueryStatField(safe, "recovery_ops_remaining");
    const long rec_inflight = QueryStatField(safe, "recovery_ops_in_flight");
    const long rec_active = QueryStatField(safe, "recovery_active");
    HLOG(kInfo,
         "safe_bdev EC: recovery counters total={} completed={} remaining={} "
         "in_flight={} active={}",
         rec_total, rec_done, rec_remaining, rec_inflight, rec_active);
    REQUIRE(rec_total > 0);
    REQUIRE(rec_done == rec_total);
    REQUIRE(rec_remaining == 0);
    REQUIRE(rec_inflight == 0);
    REQUIRE(rec_active == 0);

    std::vector<ctp::u8> got;
    read_back(got);
    REQUIRE(got == pattern);
    HLOG(kInfo, "safe_bdev EC: post-recovery read OK");

    // Confirm redundancy restored: fault a DIFFERENT member, still readable.
    auto rm2 = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), data_ids[0],
                                    /*was_faulty=*/1);
    rm2.Wait();
    REQUIRE(rm2->GetReturnCode() == 0);

    std::vector<ctp::u8> got2;
    read_back(got2);
    REQUIRE(got2 == pattern);
    HLOG(kInfo, "safe_bdev EC: degraded read after recovery (member 0 faulty) OK");
  }

  HLOG(kInfo, "safe_bdev EC end-to-end test PASSED");
}

TEST_CASE("safe_bdev_raid0_striping", "[safe_bdev][ec][striping]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "st_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  // --- k=3 data members, no parity needed for this test. ---
  const int k = 3;
  std::vector<clio::run::PoolId> data_ids;
  std::vector<clio::run::bdev::Client> data_clients;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(9800 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
    data_clients.push_back(client);
  }

  clio::run::PoolId safe_id(static_cast<clio::run::u32>(9850 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "safe_bdev_st_pool", safe_id,
                                      /*max_failures=*/1, members);
  create_task.Wait();
  safe.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->GetReturnCode() == 0);

  // Allocate/write a buffer > k*kChunkLen so it lands on different members.
  const clio::run::u64 io_len = 2 * static_cast<clio::run::u64>(k) * kChunkLen;  // 6 chunks
  std::vector<clio::run::bdev::Block> blocks = AllocBlocks(safe, io_len);
  // First alloc, first chunk -> data member 0, slot 0 -> banded logical off 0.
  REQUIRE(blocks[0].offset_ == 0);

  std::vector<ctp::u8> pattern = MakePattern(io_len, 0x3C);
  WriteBlocks(safe, blocks, pattern);

  // Verify physically: chunk ci lands on data member (ci % k) at member offset
  // kSuperblockSize + (ci/k)*kChunkLen. Read each data member directly via its
  // bdev client and confirm the chunk equals the corresponding slice of the
  // pattern. Because offset 0 -> chunk 0 -> member 0, chunk 1 -> member 1, etc,
  // distinct chunks land on DIFFERENT members.
  const clio::run::u64 num_chunks = io_len / kChunkLen;  // 6
  for (clio::run::u64 ci = 0; ci < num_chunks; ++ci) {
    const int data_col = static_cast<int>(ci % static_cast<clio::run::u64>(k));
    const clio::run::u64 row = ci / static_cast<clio::run::u64>(k);
    const clio::run::u64 phys = kSuperblockSize + row * kChunkLen;

    clio::run::priv::vector<clio::run::bdev::Block> rb(CTP_MALLOC);
    rb.push_back(clio::run::bdev::Block(phys, kChunkLen, 0));
    auto rbuf = CLIO_IPC->AllocateBuffer(kChunkLen);
    REQUIRE_FALSE(rbuf.IsNull());
    memset(rbuf.ptr_, 0, kChunkLen);
    auto rt = data_clients[data_col].AsyncRead(
        clio::run::PoolQuery::Local(), rb, rbuf.shm_.template Cast<void>(),
        kChunkLen);
    rt.Wait();
    REQUIRE(rt->GetReturnCode() == 0);
    std::vector<ctp::u8> got(kChunkLen);
    memcpy(got.data(), rbuf.ptr_, kChunkLen);
    std::vector<ctp::u8> expect(pattern.begin() + ci * kChunkLen,
                                pattern.begin() + (ci + 1) * kChunkLen);
    REQUIRE(got == expect);
    CLIO_IPC->FreeBuffer(rbuf);
  }
  HLOG(kInfo,
       "safe_bdev RAID-0: {} chunks verified striped across {} data members",
       num_chunks, k);
}

TEST_CASE("safe_bdev_reclaim", "[safe_bdev][ec][reclaim]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "rc_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  const int k = 3;
  std::vector<clio::run::PoolId> data_ids;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(9900 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
  }
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(9950 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "safe_bdev_rc_pool", safe_id,
                                      /*max_failures=*/1, members);
  create_task.Wait();
  safe.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->GetReturnCode() == 0);

  const clio::run::u64 sz = static_cast<clio::run::u64>(k) * kChunkLen;

  // Allocate, capture the offset, free, allocate again: the SAME region should
  // be reused (the allocator is a free-list, not a bump pointer).
  auto a1 = safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), sz);
  a1.Wait();
  REQUIRE(a1->GetReturnCode() == 0);
  REQUIRE(a1->blocks_.size() > 0);
  const clio::run::u64 off1 = a1->blocks_[0].offset_;

  clio::run::priv::vector<clio::run::bdev::Block> fblocks(CTP_MALLOC);
  fblocks.push_back(a1->blocks_[0]);
  auto fr = safe.AsyncFreeBlocks(clio::run::PoolQuery::Dynamic(), fblocks);
  fr.Wait();
  REQUIRE(fr->GetReturnCode() == 0);

  auto a2 = safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), sz);
  a2.Wait();
  REQUIRE(a2->GetReturnCode() == 0);
  REQUIRE(a2->blocks_.size() > 0);
  const clio::run::u64 off2 = a2->blocks_[0].offset_;

  REQUIRE(off1 == off2);
  HLOG(kInfo,
       "safe_bdev reclaim: freed region at offset {} reused by next alloc "
       "(offset {})",
       off1, off2);
}

TEST_CASE("safe_bdev_partial_chunk", "[safe_bdev][ec][partial]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "pc_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  const int k = 3;
  std::vector<clio::run::PoolId> data_ids;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(10000 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
  }
  clio::run::PoolId parity_id(static_cast<clio::run::u32>(10060 + pidsalt), 0);
  clio::run::bdev::Client parity_client(parity_id);
  REQUIRE(CreateRamMember(parity_client, member_name(50), parity_id));
  parity_id = parity_client.pool_id_;

  clio::run::PoolId safe_id(static_cast<clio::run::u32>(10080 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "safe_bdev_pc_pool", safe_id,
                                      /*max_failures=*/1, members);
  create_task.Wait();
  safe.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->GetReturnCode() == 0);

  auto add = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), member_name(50),
                               /*node_id=*/0, parity_id, /*as_parity=*/1);
  add.Wait();
  REQUIRE(add->GetReturnCode() == 0);

  // A partial-chunk write: 100000 bytes spans chunk 0 (full 64KiB) + part of
  // chunk 1, i.e. an arbitrary, non-chunk-aligned length.
  const clio::run::u64 io_len = 100000;
  std::vector<clio::run::bdev::Block> blocks = AllocBlocks(safe, io_len);
  std::vector<ctp::u8> pattern = MakePattern(io_len, 0x9E);
  WriteBlocks(safe, blocks, pattern);

  // Flush parity, fault a data member, degraded read of the partial range.
  FlushParity(safe);

  auto rm = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), data_ids[1],
                                 /*was_faulty=*/1);
  rm.Wait();
  REQUIRE(rm->GetReturnCode() == 0);

  std::vector<ctp::u8> got;
  ReadBlocks(safe, blocks, got);
  REQUIRE(got == pattern);
  HLOG(kInfo, "safe_bdev partial-chunk write/degraded-read OK ({} bytes)",
       io_len);
}

namespace {

/**
 * Query safe_bdev Monitor("stats") and extract an unsigned-integer field by
 * key. Returns -1 if the field/blob could not be parsed.
 */
long QueryStatField(clio::run::safe_bdev::Client &safe, const char *field) {
  auto mon = safe.AsyncMonitor(clio::run::PoolQuery::Dynamic(), "stats");
  mon.Wait();
  if (mon->GetReturnCode() != 0) {
    return -1;
  }
  for (const auto &kv_blob : mon->results_) {
    const std::string &blob = kv_blob.second;
    if (blob.empty()) {
      continue;
    }
    msgpack::object_handle oh = msgpack::unpack(blob.data(), blob.size());
    const msgpack::object &obj = oh.get();
    if (obj.type != msgpack::type::MAP) {
      continue;
    }
    for (uint32_t j = 0; j < obj.via.map.size; ++j) {
      const auto &kv = obj.via.map.ptr[j];
      std::string key;
      kv.key.convert(key);
      if (key == field) {
        clio::run::u64 v = 0;
        kv.val.convert(v);
        return static_cast<long>(v);
      }
    }
  }
  return -1;
}

/** Back-compat shim for the reattach test. */
long QueryReattachedMembers(clio::run::safe_bdev::Client &safe) {
  return QueryStatField(safe, "reattached_members");
}

}  // namespace

TEST_CASE("safe_bdev_superblock_reattach", "[safe_bdev][superblock][reattach]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "sb_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  // --- Create k=3 fresh RAM data members. ---
  const int k = 3;
  std::vector<clio::run::PoolId> data_ids;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(9600 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
  }

  // --- Phase 1: create safe_bdev pool X over the fresh members. ---
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(9650 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  {
    auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                        "safe_bdev_sb_poolX", safe_id,
                                        /*max_failures=*/1, members);
    create_task.Wait();
    safe.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->GetReturnCode() == 0);
  }

  // Fresh members must NOT be reattached.
  REQUIRE(QueryReattachedMembers(safe) == 0);

  // Write + flush + read a multi-chunk block so the members hold real data.
  const clio::run::u64 io_len = static_cast<clio::run::u64>(k) * kChunkLen;
  std::vector<clio::run::bdev::Block> blocks = AllocBlocks(safe, io_len);
  std::vector<ctp::u8> pattern = MakePattern(io_len, 0x77);
  WriteBlocks(safe, blocks, pattern);
  FlushParity(safe);
  {
    std::vector<ctp::u8> got;
    ReadBlocks(safe, blocks, got);
    REQUIRE(got == pattern);
  }
  HLOG(kInfo, "safe_bdev SB: phase 1 (fresh create + roundtrip) OK");

  // --- Tear down array X (members + their superblocks persist as independent
  //     bdev pools). Destroying first forces a genuine re-attach that re-reads
  //     each member's superblock. ---
  {
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto destroy = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                          safe.pool_id_);
    destroy.Wait();
    REQUIRE(destroy->GetReturnCode() == 0);
    std::this_thread::sleep_for(100ms);
  }

  // --- Phase 2: create AGAIN with the SAME pool id X over the SAME members.
  //     Every member already carries our superblock => all re-attached. ---
  clio::run::PoolId safe_id2 = safe_id;  // same identity
  clio::run::safe_bdev::Client safe2(safe_id2);
  {
    auto create_task = safe2.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                         "safe_bdev_sb_poolX", safe_id2,
                                         /*max_failures=*/1, members);
    create_task.Wait();
    safe2.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->GetReturnCode() == 0);
  }
  long reattached = QueryReattachedMembers(safe2);
  HLOG(kInfo, "safe_bdev SB: re-attach reported reattached_members={}",
       reattached);
  REQUIRE(reattached > 0);
  REQUIRE(reattached == k);
  HLOG(kInfo, "safe_bdev superblock reattach test PASSED");
}

TEST_CASE("safe_bdev_superblock_foreign_refuse",
          "[safe_bdev][superblock][foreign]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "fr_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  // --- Create k=3 fresh RAM data members. ---
  const int k = 3;
  std::vector<clio::run::PoolId> data_ids;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(9700 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
  }
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }

  // --- Array X claims the members (writes superblocks). ---
  clio::run::PoolId safe_idX(static_cast<clio::run::u32>(9750 + pidsalt), 0);
  clio::run::safe_bdev::Client safeX(safe_idX);
  {
    auto create_task = safeX.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                         "safe_bdev_fr_poolX", safe_idX,
                                         /*max_failures=*/1, members);
    create_task.Wait();
    safeX.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->GetReturnCode() == 0);
  }

  // --- Array Y (DIFFERENT pool id) over the SAME members => must REFUSE. ---
  clio::run::PoolId safe_idY(static_cast<clio::run::u32>(9760 + pidsalt), 0);
  clio::run::safe_bdev::Client safeY(safe_idY);
  {
    auto create_task = safeY.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                         "safe_bdev_fr_poolY", safe_idY,
                                         /*max_failures=*/1, members);
    create_task.Wait();
    HLOG(kInfo, "safe_bdev FR: foreign create returned rc={}",
         create_task->GetReturnCode());
    REQUIRE(create_task->GetReturnCode() != 0);
  }
  HLOG(kInfo, "safe_bdev superblock foreign-refuse test PASSED");
}

TEST_CASE("safe_bdev_add_data_no_reshuffle",
          "[safe_bdev][ec][add_data][groups]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "ad_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  // --- Group 0: k=2 data + 1 parity (max_failures=1). ---
  const int k0 = 2;
  std::vector<clio::run::PoolId> data_ids;
  for (int c = 0; c < k0; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(10100 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
  }
  clio::run::PoolId parity_id(static_cast<clio::run::u32>(10160 + pidsalt), 0);
  clio::run::bdev::Client parity_client(parity_id);
  REQUIRE(CreateRamMember(parity_client, member_name(50), parity_id));
  parity_id = parity_client.pool_id_;

  clio::run::PoolId safe_id(static_cast<clio::run::u32>(10180 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k0; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "safe_bdev_ad_pool", safe_id,
                                      /*max_failures=*/1, members);
  create_task.Wait();
  safe.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->GetReturnCode() == 0);

  auto add_par = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), member_name(50),
                                   /*node_id=*/0, parity_id, /*as_parity=*/1);
  add_par.Wait();
  REQUIRE(add_par->GetReturnCode() == 0);
  REQUIRE(QueryStatField(safe, "data_count") == 2);

  auto write_pattern = [&](clio::run::u64 io_len, ctp::u8 seed,
                           std::vector<clio::run::bdev::Block> &out_blocks,
                           std::vector<ctp::u8> &out_pattern) {
    out_blocks = AllocBlocks(safe, io_len);
    out_pattern = MakePattern(io_len, seed);
    WriteBlocks(safe, out_blocks, out_pattern);
    FlushParity(safe);
  };

  auto read_verify = [&](const std::vector<clio::run::bdev::Block> &blocks,
                         const std::vector<ctp::u8> &expect) {
    std::vector<ctp::u8> got;
    ReadBlocks(safe, blocks, got);
    REQUIRE(got == expect);
  };

  // --- Pattern A written while k=2 (round-robin over 2 data members). ---
  const clio::run::u64 lenA = 2 * static_cast<clio::run::u64>(k0) * kChunkLen;  // 4 chunks
  std::vector<clio::run::bdev::Block> blockA;
  std::vector<ctp::u8> patternA;
  write_pattern(lenA, 0xA1, blockA, patternA);
  read_verify(blockA, patternA);
  HLOG(kInfo, "safe_bdev add-data: pattern A written+verified with 2 data "
              "members ({} bytes)", lenA);

  // --- AddBdev a 3rd DATA member: succeeds with NO data movement; the
  //     round-robin now spreads new writes over 3 members. ---
  clio::run::PoolId data3_id(static_cast<clio::run::u32>(10170 + pidsalt), 0);
  clio::run::bdev::Client data3_client(data3_id);
  REQUIRE(CreateRamMember(data3_client, member_name(2), data3_id));
  data3_id = data3_client.pool_id_;
  data_ids.push_back(data3_id);

  auto add_data = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), member_name(2),
                                    /*node_id=*/0, data3_id, /*as_parity=*/0);
  add_data.Wait();
  REQUIRE(add_data->GetReturnCode() == 0);
  REQUIRE(QueryStatField(safe, "data_count") == 3);
  HLOG(kInfo, "safe_bdev add-data: 3rd data member added (data_count=3)");

  // --- Pattern B written now that k=3 (spreads over 3 data drives). ---
  const clio::run::u64 lenB = 2 * 3 * kChunkLen;  // 6 chunks across 3 data drives
  std::vector<clio::run::bdev::Block> blockB;
  std::vector<ctp::u8> patternB;
  write_pattern(lenB, 0xB2, blockB, patternB);
  read_verify(blockB, patternB);
  HLOG(kInfo, "safe_bdev add-data: pattern B written+verified with 3 data "
              "members ({} bytes)", lenB);

  // --- NO RESHUFFLE: re-read pattern A; it must be UNCHANGED. ---
  read_verify(blockA, patternA);
  HLOG(kInfo, "safe_bdev add-data: pattern A UNCHANGED after add (no reshuffle)");

  // --- Fault data member 0 and confirm degraded reads reconstruct BOTH
  //     patterns (each stripe under its own variable width). ---
  auto rm = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), data_ids[0],
                                 /*was_faulty=*/1);
  rm.Wait();
  REQUIRE(rm->GetReturnCode() == 0);

  read_verify(blockA, patternA);  // pattern A degraded
  HLOG(kInfo, "safe_bdev add-data: degraded read of pattern A OK");
  read_verify(blockB, patternB);  // pattern B degraded
  HLOG(kInfo, "safe_bdev add-data: degraded read of pattern B OK");

  HLOG(kInfo, "safe_bdev add-data no-reshuffle test PASSED");
}

TEST_CASE("safe_bdev_alloc_log_restart_recovery",
          "[safe_bdev][alloc_log][recover][groups]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  // Use the platform temp dir (not a hardcoded /tmp, which does not exist on
  // Windows) so the file-backed members + on-disk WAL work cross-platform.
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  auto member_file = [&](int idx) {
    return (tmp_dir / ("safe_alog_member_" + std::to_string(getpid()) + "_" +
                       std::to_string(idx) + ".bin"))
        .string();
  };
  const std::string log_path =
      (tmp_dir / ("safe_alog_" + std::to_string(getpid()) + ".alog")).string();

  // Fresh start: remove any stale member files + alloc log. Best-effort: use
  // the non-throwing overload so a still-open handle (on Windows the in-process
  // runtime keeps file-bdev members open; POSIX allows unlinking open files)
  // does not abort the test.
  std::error_code cleanup_ec;
  for (int i = 0; i < 4; ++i) {
    std::filesystem::remove(member_file(i), cleanup_ec);
  }
  std::filesystem::remove(log_path, cleanup_ec);

  // The safe-bdev pool id is STABLE across the restart so the recovered array
  // recognizes its own superblocks (reattach, not foreign).
  const clio::run::PoolId safe_id(static_cast<clio::run::u32>(10500 + pidsalt), 0);

  // k=2 data + 1 parity. Group 0 has k=2; after AddBdev a 3rd data member,
  // group 1 has k=3.
  const int k0 = 2;

  // Patterns: A lands in group 0 (k=2, 4 chunks); B lands in group 1 (k=3,
  // 6 chunks). Captured here so phase 2 can verify them post-recovery.
  const clio::run::u64 lenA = 2 * static_cast<clio::run::u64>(k0) * kChunkLen;  // 4 chunks
  const clio::run::u64 lenB = 2 * 3 * kChunkLen;                          // 6 chunks
  std::vector<ctp::u8> patternA = MakePattern(lenA, 0xA1);
  std::vector<ctp::u8> patternB = MakePattern(lenB, 0xB2);
  std::vector<clio::run::bdev::Block> blockA;
  std::vector<clio::run::bdev::Block> blockB;

  // Member pool ids (reused verbatim in phase 2 so each bdev pool re-opens the
  // SAME backing file).
  std::vector<clio::run::PoolId> data_ids;
  clio::run::PoolId parity_id(static_cast<clio::run::u32>(10560 + pidsalt), 0);
  clio::run::PoolId data3_id(static_cast<clio::run::u32>(10570 + pidsalt), 0);

  // A write+flush(parity) helper bound to a given safe client.
  auto write_pattern = [&](clio::run::safe_bdev::Client &safe, clio::run::u64 io_len,
                           const std::vector<ctp::u8> &pattern,
                           std::vector<clio::run::bdev::Block> &out_blocks) {
    out_blocks = AllocBlocks(safe, io_len);
    WriteBlocks(safe, out_blocks, pattern);
    FlushParity(safe);
  };

  auto read_verify = [&](clio::run::safe_bdev::Client &safe,
                         const std::vector<clio::run::bdev::Block> &blocks,
                         const std::vector<ctp::u8> &expect) {
    std::vector<ctp::u8> got;
    ReadBlocks(safe, blocks, got);
    REQUIRE(got == expect);
  };

  // ======================================================================
  // PHASE 1: create, write A (group 0), add a 3rd data member (open group 1),
  //          write B (group 1), flush the alloc log, then destroy the pool.
  // ======================================================================
  {
    std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
    for (int c = 0; c < k0; ++c) {
      clio::run::PoolId id(static_cast<clio::run::u32>(10510 + pidsalt + c), 0);
      clio::run::bdev::Client client(id);
      REQUIRE(CreateFileMember(client, member_file(c), id));
      data_ids.push_back(client.pool_id_);
      members.emplace_back(member_file(c), /*node_id=*/0, client.pool_id_);
    }
    clio::run::bdev::Client parity_client(parity_id);
    REQUIRE(CreateFileMember(parity_client, member_file(3), parity_id));
    parity_id = parity_client.pool_id_;

    clio::run::safe_bdev::Client safe(safe_id);
    auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                        "safe_bdev_alog_pool", safe_id,
                                        /*max_failures=*/1, members, log_path);
    create_task.Wait();
    safe.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->GetReturnCode() == 0);

    auto add_par = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), member_file(3),
                                     /*node_id=*/0, parity_id, /*as_parity=*/1);
    add_par.Wait();
    REQUIRE(add_par->GetReturnCode() == 0);
    REQUIRE(QueryStatField(safe, "data_count") == 2);

    // Pattern A written while k=2.
    write_pattern(safe, lenA, patternA, blockA);
    read_verify(safe, blockA, patternA);
    HLOG(kInfo, "safe_bdev alog: phase1 pattern A written (k=2)");

    // Add a 3rd DATA member (no data movement; new writes now span 3 members).
    clio::run::bdev::Client data3_client(data3_id);
    REQUIRE(CreateFileMember(data3_client, member_file(2), data3_id));
    data3_id = data3_client.pool_id_;
    data_ids.push_back(data3_id);
    auto add_data = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), member_file(2),
                                      /*node_id=*/0, data3_id, /*as_parity=*/0);
    add_data.Wait();
    REQUIRE(add_data->GetReturnCode() == 0);
    REQUIRE(QueryStatField(safe, "data_count") == 3);

    // Pattern B written now that k=3.
    write_pattern(safe, lenB, patternB, blockB);
    read_verify(safe, blockB, patternB);
    HLOG(kInfo, "safe_bdev alog: phase1 pattern B written (k=3)");

    // Explicit one-shot alloc-log flush barrier (mirrors bdev's WAL test): the
    // PoolManager DestroyPool does not invoke the container Destroy handler, so
    // flush the buffered records before tearing the pool down.
    {
      auto fl = safe.AsyncFlushAllocLog(clio::run::PoolQuery::Dynamic(), /*period=*/0);
      fl.Wait();
      REQUIRE(fl->GetReturnCode() == 0);
    }
    {
      auto fl2 = safe.AsyncBuildParity(clio::run::PoolQuery::Dynamic(), 0);
      fl2.Wait();
      REQUIRE(fl2->GetReturnCode() == 0);
    }

    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto destroy = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                          safe.pool_id_);
    destroy.Wait();
    REQUIRE(destroy->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
  }

  // ======================================================================
  // PHASE 2 (RESTART): create a NEW safe-bdev pool over the SAME member files
  //          (re-opened bdev pools) + SAME alloc_log_path + SAME safe pool id.
  //          Assert recovery: data_count==3, A and B intact, and a fresh alloc
  //          does NOT collide with the recovered live blocks.
  // ======================================================================
  {
    // Re-open the bdev member pools over the SAME backing files (fresh bdev
    // pool ids; the files — and thus the bytes + superblocks — persist).
    std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
    for (int c = 0; c < 3; ++c) {
      clio::run::PoolId id(static_cast<clio::run::u32>(10580 + pidsalt + c), 0);
      clio::run::bdev::Client client(id);
      REQUIRE(CreateFileMember(client, member_file(c), id));
      members.emplace_back(member_file(c), /*node_id=*/0, client.pool_id_);
    }
    // NOTE: parity is no longer re-created/re-added here -- Create restores the
    // runtime-added parity member from the durable member manifest (its bdev
    // pool from phase 1 is still resident in this in-process runtime).

    clio::run::safe_bdev::Client safe2(safe_id);
    auto create_task = safe2.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                         "safe_bdev_alog_pool", safe_id,
                                         /*max_failures=*/1, members, log_path);
    create_task.Wait();
    safe2.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->GetReturnCode() == 0);

    // (a) Membership recovered: 3 data members, all re-attached; parity was
    //     restored from the durable member manifest without a manual re-add.
    REQUIRE(QueryStatField(safe2, "data_count") == 3);
    REQUIRE(QueryReattachedMembers(safe2) == 3);
    REQUIRE(QueryStatField(safe2, "parity_level") == 1);

    // (b) Pattern A AND pattern B read back correctly -- data intact on the file
    //     members + correct per-member slot-allocator recovery from the WAL.
    read_verify(safe2, blockA, patternA);
    read_verify(safe2, blockB, patternB);
    HLOG(kInfo, "safe_bdev alog: RESTART patterns A+B intact after recovery");

    // (c) A fresh allocation must NOT overlap any recovered live chunk of A/B.
    std::vector<clio::run::bdev::Block> nb = AllocBlocks(safe2, kChunkLen);
    for (const auto &fb : nb) {
      for (const auto &ab : blockA) {
        REQUIRE_FALSE(RangesOverlap(fb.offset_, fb.size_, ab.offset_, ab.size_));
      }
      for (const auto &bb : blockB) {
        REQUIRE_FALSE(RangesOverlap(fb.offset_, fb.size_, bb.offset_, bb.size_));
      }
    }
    HLOG(kInfo, "safe_bdev alog: RESTART fresh alloc does NOT collide with A/B");

    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto destroy = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                          safe2.pool_id_);
    destroy.Wait();
  }

  // Cleanup backing files + log. Best-effort (non-throwing): on Windows the
  // still-running in-process runtime may hold the file-bdev member handles
  // open, which would make a throwing remove() abort this otherwise-passing
  // test; leftover temp files are pid-namespaced and harmless. (Reuses the
  // cleanup_ec declared for the pre-test cleanup above.)
  for (int i = 0; i < 4; ++i) {
    std::filesystem::remove(member_file(i), cleanup_ec);
  }
  std::filesystem::remove(log_path, cleanup_ec);
  HLOG(kInfo, "safe_bdev alloc-log restart/recovery test PASSED");
}

// Consistency across a reboot WITH an interrupted recovery. Exercises the
// durable member manifest + recovery-resume path:
//   1. write a working set through the safe-bdev (a CTE over this safe-bdev
//      stores its blob bytes HERE, so safe-bdev durability == CTE data
//      durability),
//   2. shut down + reboot (destroy + re-create the safe pool over the SAME,
//      still-resident member bdevs -- like disks surviving a reboot) -> data
//      intact,
//   3. fault a data member and recover onto a spare, but
//   4. shut down BEFORE the rebuild finishes (CLIO_SAFE_BDEV_RECOVER_MAX_ROWS
//      interrupt hook),
//   5. restart, and
//   6. verify the data is FULLY recovered (redundancy restored).
// Recovery rebuilds exactly the faulted member's live slots, so the working set
// is sized (many small writes -> ~60 slots/member) so the 40-slot interrupt
// leaves the rebuild incomplete and resumable on restart; the 256 MB
// through-the-CTE path is covered by the containerized safe-bdev test.
TEST_CASE("safe_bdev_consistency_restart_interrupted_recovery",
          "[safe_bdev][consistency][recover][restart]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  auto mfile = [&](int i) {
    return (tmp_dir / ("safe_cons_" + std::to_string(getpid()) + "_" +
                       std::to_string(i) + ".bin"))
        .string();
  };
  const std::string log_path =
      (tmp_dir / ("safe_cons_" + std::to_string(getpid()) + ".alog")).string();

  std::error_code ec;
  for (int i = 0; i < 5; ++i) std::filesystem::remove(mfile(i), ec);
  std::filesystem::remove(log_path, ec);
  std::filesystem::remove(log_path + ".members", ec);

  const clio::run::u64 kMember = 16 * 1024 * 1024;  // 16 MB members => 255 slots
  const int k0 = 3;
  // Recovery now rebuilds exactly the faulted member's LIVE slots (not a whole
  // fixed group), so the working set must be large enough that the faulted data
  // member owns > 40 slots -- otherwise the 40-slot interrupt hook would finish
  // the rebuild instead of leaving it incomplete. We build that up with MANY
  // SMALL writes rather than one huge I/O: a single multi-MB allocate/write/read
  // would exhaust the IPC buffer pool on some platforms (macOS). kNumWrites x
  // kWriteChunks(6) = 180 chunks spread round-robin over 3 members => ~60 slots
  // per data member, while every individual I/O stays as small as the other
  // passing tests.
  const int kNumWrites = 30;
  const clio::run::u64 kWriteLen =
      2 * static_cast<clio::run::u64>(k0) * kChunkLen;  // 6 chunks per write

  // Stable pool ids: the member bdevs stay resident across the safe-pool
  // reboots (disks persist), so their ids + backing files are stable.
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(70000 + pidsalt), 0);
  std::vector<clio::run::PoolId> d_id;
  for (int c = 0; c < k0; ++c) {
    d_id.emplace_back(static_cast<clio::run::u32>(71000 + pidsalt + c), 0);
  }
  clio::run::PoolId parity_id(static_cast<clio::run::u32>(72000 + pidsalt), 0);
  clio::run::PoolId spare_id(static_cast<clio::run::u32>(73000 + pidsalt), 0);

  auto make_member = [&](const clio::run::PoolId &id, const std::string &path) {
    clio::run::bdev::Client c(id);
    auto t = c.AsyncCreate(clio::run::PoolQuery::Dynamic(), path, id,
                           clio::run::bdev::BdevType::kFile, kMember);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
  };

  // Each entry: the blocks of one small write + the exact bytes written, so
  // verify_all can read every write back and byte-compare (mirrors the capacity
  // test's per-block store).
  std::vector<std::pair<std::vector<clio::run::bdev::Block>, std::vector<ctp::u8>>>
      stored;

  auto write_all = [&](clio::run::safe_bdev::Client &safe) {
    stored.clear();
    for (int i = 0; i < kNumWrites; ++i) {
      auto blocks = AllocBlocks(safe, kWriteLen);
      auto pat = MakePattern(kWriteLen, static_cast<ctp::u8>(0xC7 + i));
      WriteBlocks(safe, blocks, pat);
      stored.emplace_back(std::move(blocks), std::move(pat));
    }
    FlushParity(safe);
  };

  auto verify_all = [&](clio::run::safe_bdev::Client &safe) {
    for (auto &sp : stored) {
      std::vector<ctp::u8> got;
      ReadBlocks(safe, sp.first, got);
      REQUIRE(got == sp.second);
    }
  };

  auto destroy_safe = [&](clio::run::safe_bdev::Client &safe) {
    auto fl = safe.AsyncFlushAllocLog(clio::run::PoolQuery::Dynamic(), 0);
    fl.Wait();
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto d =
        admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(), safe.pool_id_);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
  };

  auto make_members = [&](std::initializer_list<clio::run::PoolId> ids,
                          std::initializer_list<int> files) {
    std::vector<clio::run::safe_bdev::MemberBdevDesc> m;
    auto it = ids.begin();
    auto fi = files.begin();
    for (; it != ids.end(); ++it, ++fi) {
      m.emplace_back(mfile(*fi), 0, *it);
    }
    return m;
  };

  // Persistent member bdevs (survive the reboots).
  make_member(d_id[0], mfile(0));
  make_member(d_id[1], mfile(1));
  make_member(d_id[2], mfile(2));
  make_member(parity_id, mfile(3));
  make_member(spare_id, mfile(4));

  // ===== PHASE 1: write the working set, add parity, shut down. =====
  {
    auto members = make_members({d_id[0], d_id[1], d_id[2]}, {0, 1, 2});
    clio::run::safe_bdev::Client safe(safe_id);
    auto ct = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), "safe_cons",
                               safe_id, /*max_failures=*/1, members, log_path);
    ct.Wait();
    safe.pool_id_ = ct->new_pool_id_;
    REQUIRE(ct->GetReturnCode() == 0);
    auto ap = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), mfile(3), 0,
                                parity_id, /*as_parity=*/1);
    ap.Wait();
    REQUIRE(ap->GetReturnCode() == 0);
    write_all(safe);
    verify_all(safe);
    HLOG(kInfo, "safe_bdev consistency: phase1 wrote + verified working set");
    destroy_safe(safe);
  }

  // ===== PHASE 2 (REBOOT): data intact; fault d1 + recover-onto-spare, but
  //       INTERRUPT the rebuild; shut down before it finishes. =====
  {
    auto members = make_members({d_id[0], d_id[1], d_id[2]}, {0, 1, 2});
    clio::run::safe_bdev::Client safe(safe_id);
    auto ct = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), "safe_cons",
                               safe_id, /*max_failures=*/1, members, log_path);
    ct.Wait();
    safe.pool_id_ = ct->new_pool_id_;
    REQUIRE(ct->GetReturnCode() == 0);
    // Parity was restored from the durable manifest (no manual re-add).
    REQUIRE(QueryStatField(safe, "parity_level") == 1);
    // STEP 2: data survived the clean reboot.
    verify_all(safe);
    HLOG(kInfo, "safe_bdev consistency: STEP2 data intact after clean reboot");

    // STEP 3: fault data member 1 and recover onto the spare, but INTERRUPT the
    // rebuild after 40 of its ~60 live slots (simulated crash mid-recovery).
    auto rm = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), d_id[1],
                                   /*was_faulty=*/1);
    rm.Wait();
    REQUIRE(rm->GetReturnCode() == 0);
    ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_RECOVER_MAX_ROWS", "40", 1);
    auto rec = safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(), d_id[1],
                                     mfile(4), 0, spare_id);
    rec.Wait();
    REQUIRE(rec->GetReturnCode() == 0);  // partial but consistent
    const long done = QueryStatField(safe, "recovery_ops_completed");
    const long total = QueryStatField(safe, "recovery_ops_total");
    HLOG(kInfo, "safe_bdev consistency: interrupted recovery {}/{} rows", done,
         total);
    REQUIRE(total > done);  // genuinely incomplete
    REQUIRE(done > 0);
    // Even mid-recovery, degraded reads (spare excluded) serve correct data.
    verify_all(safe);
    HLOG(kInfo, "safe_bdev consistency: data intact mid-recovery (degraded)");

    // STEP 4: shut down before the recovery finishes.
    destroy_safe(safe);
  }

  // ===== PHASE 3 (RESTART): resume recovery, verify data FULLY recovered. =====
  {
    ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_RECOVER_MAX_ROWS", "0", 1);  // no cap
    // Current membership: the spare replaced data member 1 (column 1).
    auto members = make_members({d_id[0], spare_id, d_id[2]}, {0, 4, 2});
    clio::run::safe_bdev::Client safe(safe_id);
    auto ct = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), "safe_cons",
                               safe_id, /*max_failures=*/1, members, log_path);
    ct.Wait();
    safe.pool_id_ = ct->new_pool_id_;
    // STEP 5: Create restored membership + RESUMED and finished the recovery.
    REQUIRE(ct->GetReturnCode() == 0);
    REQUIRE(QueryStatField(safe, "parity_level") == 1);

    // STEP 6: data is fully readable after the resumed recovery.
    verify_all(safe);
    HLOG(kInfo,
         "safe_bdev consistency: STEP6 data intact after restart + resume");

    // Redundancy FULLY restored: fault a DIFFERENT member (d0) and confirm the
    // degraded read still reconstructs -- only possible if the spare (column 1)
    // was completely rebuilt during the resumed recovery.
    auto rm = safe.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), d_id[0],
                                   /*was_faulty=*/1);
    rm.Wait();
    REQUIRE(rm->GetReturnCode() == 0);
    verify_all(safe);
    HLOG(kInfo,
         "safe_bdev consistency: redundancy restored (degraded read after "
         "faulting a 2nd member OK) -- recovery was durable + complete");

    destroy_safe(safe);
  }

  ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_RECOVER_MAX_ROWS", "0", 1);
  for (int i = 0; i < 5; ++i) std::filesystem::remove(mfile(i), ec);
  std::filesystem::remove(log_path, ec);
  std::filesystem::remove(log_path + ".members", ec);
  HLOG(kInfo,
       "safe_bdev consistency restart + interrupted-recovery test PASSED");
}

// Capacity behaviour of the dynamic view-group layout. Scenario (scaled): start
// with ONE data drive, fill it heavily (~84% -- mirrors 216 MB of a 256 MB
// drive), add a second data drive, write the same again, add parity, then verify
// everything reads back. This asserts the redesign's WIN: adding a data drive
// makes its FULL capacity available (per-chunk round-robin, no group freeze), so
// the second heavy write fits entirely -- unlike the old append-only layout,
// which stranded the added drive (fitting only ~5/13 blocks in phase 2).
TEST_CASE("safe_bdev_capacity_add_data_then_parity",
          "[safe_bdev][capacity]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  auto mfile = [&](int i) {
    return (tmp_dir / ("safe_cap_" + std::to_string(getpid()) + "_" +
                       std::to_string(i) + ".bin"))
        .string();
  };
  const std::string log_path =
      (tmp_dir / ("safe_cap_" + std::to_string(getpid()) + ".alog")).string();
  std::error_code ec;
  for (int i = 0; i < 3; ++i) std::filesystem::remove(mfile(i), ec);
  std::filesystem::remove(log_path, ec);
  std::filesystem::remove(log_path + ".members", ec);

  // Members are kMemberRamSize (4 MB) => ~63 usable chunk-rows each. We write in
  // 256 KB (4-chunk) single blocks and target ~84% of a drive per phase.
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(80000 + pidsalt), 0);
  clio::run::PoolId d0_id(static_cast<clio::run::u32>(81000 + pidsalt), 0);
  clio::run::PoolId d1_id(static_cast<clio::run::u32>(81001 + pidsalt), 0);
  clio::run::PoolId par_id(static_cast<clio::run::u32>(82000 + pidsalt), 0);

  clio::run::bdev::Client c0(d0_id), c1(d1_id), cp(par_id);
  REQUIRE(CreateFileMember(c0, mfile(0), d0_id));
  REQUIRE(CreateFileMember(c1, mfile(1), d1_id));
  REQUIRE(CreateFileMember(cp, mfile(2), par_id));

  clio::run::safe_bdev::Client safe(safe_id);
  {
    std::vector<clio::run::safe_bdev::MemberBdevDesc> m;
    m.emplace_back(mfile(0), 0, d0_id);
    auto ct = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), "safe_cap",
                               safe_id, /*max_failures=*/1, m, log_path);
    ct.Wait();
    safe.pool_id_ = ct->new_pool_id_;
    REQUIRE(ct->GetReturnCode() == 0);
  }

  const clio::run::u64 BLK = 4 * kChunkLen;  // 256 KB == 4 chunks (round-robin)
  std::vector<std::pair<std::vector<clio::run::bdev::Block>,
                        std::vector<ctp::u8>>>
      stored;
  int seed = 0;

  auto try_write = [&]() -> bool {
    // Non-asserting alloc: this loop deliberately probes until the array is
    // full, so a failed allocation is an expected outcome, not a test failure.
    auto alloc = safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), BLK);
    alloc.Wait();
    if (alloc->GetReturnCode() != 0 || alloc->blocks_.empty()) return false;
    std::vector<clio::run::bdev::Block> blocks;
    for (size_t i = 0; i < alloc->blocks_.size(); ++i) {
      blocks.push_back(alloc->blocks_[i]);
    }
    auto pat = MakePattern(BLK, static_cast<ctp::u8>(0x40 + (seed++ & 0x3F)));
    auto wb = ToPriv(blocks);
    auto wbuf = CLIO_IPC->AllocateBuffer(BLK);
    if (wbuf.IsNull()) return false;
    memcpy(wbuf.ptr_, pat.data(), BLK);
    auto wt = safe.AsyncWrite(clio::run::PoolQuery::Dynamic(), wb,
                              wbuf.shm_.template Cast<void>(), BLK);
    wt.Wait();
    const bool ok = (wt->GetReturnCode() == 0 && wt->bytes_written_ == BLK);
    CLIO_IPC->FreeBuffer(wbuf);
    if (ok) stored.emplace_back(std::move(blocks), std::move(pat));
    return ok;
  };

  const int target = 13;  // ~84% of a 63-slot drive at 4 chunks/block

  // PHASE 1: one data drive. All `target` blocks fit (< a full drive).
  int s1 = 0;
  for (int i = 0; i < target; ++i) {
    if (try_write()) ++s1;
  }
  HLOG(kInfo, "safe_bdev capacity: phase1 (1 data drive) stored {}/{} blocks",
       s1, target);
  REQUIRE(s1 == target);

  // Add a second data drive. NO group freeze / NO stranding: the per-chunk
  // round-robin now fills the new drive too, so the FULL aggregate capacity
  // (Sum of per-drive capacity) becomes available with no data movement.
  auto add_d = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), mfile(1), 0,
                                 d1_id, /*as_parity=*/0);
  add_d.Wait();
  REQUIRE(add_d->GetReturnCode() == 0);

  // PHASE 2: write the same amount again. Under the dynamic view-group model it
  // ALL fits -- the second drive's space is fully usable. (The old append-only
  // layout stranded it, fitting only ~5/13 blocks here.)
  int s2 = 0;
  for (int i = 0; i < target; ++i) {
    if (try_write()) ++s2;
  }
  HLOG(kInfo, "safe_bdev capacity: phase2 (2 data drives) stored {}/{} blocks",
       s2, target);
  REQUIRE(s2 == target);  // FULL capacity, no stranding -- the redesign's win

  // Add a parity drive and build parity over everything written.
  auto add_p = safe.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), mfile(2), 0,
                                 par_id, /*as_parity=*/1);
  add_p.Wait();
  REQUIRE(add_p->GetReturnCode() == 0);
  auto bp = safe.AsyncBuildParity(clio::run::PoolQuery::Dynamic(), 0);
  bp.Wait();
  REQUIRE(bp->GetReturnCode() == 0);

  // Everything that WAS stored must read back byte-for-byte.
  for (auto &blk_pat : stored) {
    std::vector<ctp::u8> got;
    ReadBlocks(safe, blk_pat.first, got);
    REQUIRE(got == blk_pat.second);
  }
  HLOG(kInfo,
       "safe_bdev capacity: stored+verified {} of {} attempted blocks "
       "(phase1={}, phase2={})",
       stored.size(), 2 * target, s1, s2);

  clio::run::admin::Client admin(clio::run::kAdminPoolId);
  auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(), safe.pool_id_);
  d.Wait();
  for (int i = 0; i < 3; ++i) std::filesystem::remove(mfile(i), ec);
  std::filesystem::remove(log_path, ec);
  std::filesystem::remove(log_path + ".members", ec);
}

SIMPLE_TEST_MAIN()

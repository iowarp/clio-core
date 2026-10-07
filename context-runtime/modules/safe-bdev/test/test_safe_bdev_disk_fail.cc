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
 * Member disks that FAIL AT RUNTIME (issue #1117).
 *
 * The file bdev's test-only fault injection makes a member's every I/O fail
 * while "<backing file>.fail" exists. Over a 6-disk array (4 data + 2 parity,
 * max_failures 2, composed from YAML with `parity: true` members) these tests
 * check that a disk dying mid-run is marked faulty by the I/O that hits it,
 * that the request is retried degraded (reads reconstruct, writes go to the
 * survivors + parity), that a restart keeps the member faulty, that
 * RecoverBdev restores full redundancy, and that past max_failures requests
 * fail -- never return wrong bytes.
 */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

using namespace std::chrono_literals;

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>
#include <clio_runtime/singletons.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/serialize/msgpack_wrapper.h>

namespace {

namespace fs = std::filesystem;
using clio::run::bdev::Block;
using clio::run::safe_bdev::MemberBdevDesc;

bool g_initialized = false;

constexpr clio::run::u64 kChunkLen = 65536;               // Runtime::kChunkLen
constexpr clio::run::u64 kMemberSize = 4 * 1024 * 1024;   // per member disk
constexpr int kDataMembers = 4;
constexpr int kParityMembers = 2;
constexpr int kMembers = kDataMembers + kParityMembers;
/** Longer than the file bdev's 100 ms marker re-check period. */
constexpr auto kMarkerSettle = 300ms;

/** Start the embedded runtime once per process. */
void EnsureInit() {
  if (g_initialized) return;
  g_initialized = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (g_initialized) {
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
    std::this_thread::sleep_for(500ms);
  }
}

/**
 * Per-user, per-process scratch directory for member files and logs.
 * $CLIO_SAFE_STRESS_DIR overrides the system temp directory.
 * @param tag names the test's own subdirectory
 * @return the directory (created)
 */
fs::path ScratchDir(const std::string &tag) {
  const char *e = std::getenv("CLIO_SAFE_STRESS_DIR");
  const char *user = std::getenv("USER");
  fs::path d = (e != nullptr ? fs::path(e) : fs::temp_directory_path()) /
               ("safe_disk_fail_" + std::string(user ? user : "u") + "_" +
                std::to_string(getpid()) + "_" + tag);
  fs::create_directories(d);
  return d;
}

/** Make member file `path` fail every I/O (dead disk). */
void KillDisk(const std::string &path) {
  std::ofstream(path + ".fail").put('x');
  std::this_thread::sleep_for(kMarkerSettle);
}

/** Make member file `path` work again. */
void ReviveDisk(const std::string &path) {
  std::error_code ec;
  fs::remove(path + ".fail", ec);
  std::this_thread::sleep_for(kMarkerSettle);
}

/**
 * Create a file-backed bdev pool (a "disk").
 * @param path backing file (also the pool name)
 * @param id pool id to use
 * @return the created pool id (null on failure)
 */
clio::run::PoolId CreateDisk(const std::string &path,
                             const clio::run::PoolId &id) {
  clio::run::bdev::Client client(id);
  // A 4 KiB growth unit keeps the backing file only as long as what was
  // written, as a large member (capacity beyond the default 1 GiB unit) is
  // in a real deployment: reads of never-written chunk tails then land past
  // EOF and must still come back as zeros, not as a failing disk.
  auto t = client.AsyncCreate(clio::run::PoolQuery::Dynamic(), path, id,
                              clio::run::bdev::BdevType::kFile, kMemberSize,
                              /*io_depth=*/32, /*alignment=*/4096,
                              /*perf_metrics=*/nullptr, /*alloc_log_path=*/"",
                              /*growth_unit=*/4096);
  t.Wait();
  if (t->GetReturnCode() != 0) return clio::run::PoolId();
  return t->new_pool_id_;
}

/** Copy a block list into a runtime priv::vector. */
clio::run::priv::vector<Block> ToPriv(const std::vector<Block> &blocks) {
  clio::run::priv::vector<Block> v(CTP_MALLOC);
  for (const auto &b : blocks) v.push_back(b);
  return v;
}

/** @return total bytes covered by `blocks`. */
clio::run::u64 BlocksLen(const std::vector<Block> &blocks) {
  clio::run::u64 n = 0;
  for (const auto &b : blocks) n += b.size_;
  return n;
}

/** Deterministic pattern. */
std::vector<ctp::u8> Pattern(size_t n, ctp::u8 seed) {
  std::vector<ctp::u8> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<ctp::u8>((seed * 31u + i * 7u + (i >> 10)) & 0xFF);
  }
  return v;
}

/** Allocate `len` bytes on the array. */
std::vector<Block> Alloc(clio::run::safe_bdev::Client &safe,
                         clio::run::u64 len) {
  auto t = safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), len);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  std::vector<Block> v;
  for (size_t i = 0; i < t->blocks_.size(); ++i) v.push_back(t->blocks_[i]);
  return v;
}

/**
 * Write `data` across `blocks`.
 * @return the write's return code (0 = every byte stored)
 */
clio::run::u32 Write(clio::run::safe_bdev::Client &safe,
                     const std::vector<Block> &blocks,
                     const std::vector<ctp::u8> &data) {
  auto buf = CLIO_IPC->AllocateBuffer(data.size());
  REQUIRE_FALSE(buf.IsNull());
  memcpy(buf.ptr_, data.data(), data.size());
  auto t = safe.AsyncWrite(clio::run::PoolQuery::Dynamic(), ToPriv(blocks),
                           buf.shm_.template Cast<void>(), data.size());
  t.Wait();
  clio::run::u32 rc = t->GetReturnCode();
  if (rc == 0 && t->bytes_written_ != data.size()) rc = 999;
  CLIO_IPC->FreeBuffer(buf);
  return rc;
}

/**
 * Read `blocks` into `out`.
 * @return the read's return code (0 = every byte returned)
 */
clio::run::u32 Read(clio::run::safe_bdev::Client &safe,
                    const std::vector<Block> &blocks,
                    std::vector<ctp::u8> &out) {
  const clio::run::u64 len = BlocksLen(blocks);
  auto buf = CLIO_IPC->AllocateBuffer(len);
  REQUIRE_FALSE(buf.IsNull());
  memset(buf.ptr_, 0, len);
  auto t = safe.AsyncRead(clio::run::PoolQuery::Dynamic(), ToPriv(blocks),
                          buf.shm_.template Cast<void>(), len);
  t.Wait();
  clio::run::u32 rc = t->GetReturnCode();
  if (rc == 0 && t->bytes_read_ != len) rc = 999;
  out.assign(buf.ptr_, buf.ptr_ + len);
  CLIO_IPC->FreeBuffer(buf);
  return rc;
}

/** Drain the async parity builder (durability barrier). */
void FlushParity(clio::run::safe_bdev::Client &safe) {
  auto t = safe.AsyncBuildParity(clio::run::PoolQuery::Dynamic(), 0);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
}

/** One member as Monitor("stats") reports it. */
struct MemberView {
  std::string role;
  clio::run::u32 index = 0;
  std::string state;
};

/** Array state from Monitor("stats"). */
struct ArrayView {
  int64_t data_count = -1;
  int64_t parity_level = -1;
  int64_t faulty_members = -1;
  int64_t dirty_slots = -1;
  std::vector<MemberView> members;
};

/** Query Monitor("stats"). */
ArrayView QueryArray(clio::run::safe_bdev::Client &safe) {
  ArrayView v;
  auto mon = safe.AsyncMonitor(clio::run::PoolQuery::Dynamic(), "stats");
  mon.Wait();
  REQUIRE(mon->GetReturnCode() == 0);
  for (const auto &kv : mon->results_) {
    if (kv.second.empty()) continue;
    msgpack::object_handle oh = msgpack::unpack(kv.second.data(),
                                                kv.second.size());
    const msgpack::object &obj = oh.get();
    if (obj.type != msgpack::type::MAP) continue;
    for (uint32_t j = 0; j < obj.via.map.size; ++j) {
      const auto &e = obj.via.map.ptr[j];
      std::string key;
      e.key.convert(key);
      if (key == "data_count") e.val.convert(v.data_count);
      if (key == "parity_level") e.val.convert(v.parity_level);
      if (key == "faulty_members") e.val.convert(v.faulty_members);
      if (key == "dirty_slots") e.val.convert(v.dirty_slots);
      if (key != "members" || e.val.type != msgpack::type::ARRAY) continue;
      for (uint32_t m = 0; m < e.val.via.array.size; ++m) {
        const msgpack::object &mo = e.val.via.array.ptr[m];
        MemberView mv;
        for (uint32_t f = 0; f < mo.via.map.size; ++f) {
          std::string fk;
          mo.via.map.ptr[f].key.convert(fk);
          if (fk == "role") mo.via.map.ptr[f].val.convert(mv.role);
          if (fk == "index") mo.via.map.ptr[f].val.convert(mv.index);
          if (fk == "state") mo.via.map.ptr[f].val.convert(mv.state);
        }
        v.members.push_back(mv);
      }
    }
  }
  return v;
}

/** @return Monitor's state string for one member ("" if absent). */
std::string MemberState(const ArrayView &v, const std::string &role,
                        clio::run::u32 index) {
  for (const auto &m : v.members) {
    if (m.role == role && m.index == index) return m.state;
  }
  return "";
}

/**
 * Quote a path as a single-quoted YAML scalar. A double-quoted scalar treats
 * the backslashes in Windows paths as escapes (#1156); a single-quoted one
 * only needs its own quotes doubled.
 * @param s the string to quote
 * @return the YAML scalar
 */
static std::string YamlQuote(const std::string &s) {
  std::string out = "'";
  for (char c : s) {
    out += c;
    if (c == '\'') out += '\'';
  }
  return out + "'";
}

/** The 6-disk array under test and the bytes it should hold. */
struct Rig {
  fs::path dir;
  std::vector<std::string> paths;          // data 0..3, parity 0..1
  std::vector<clio::run::PoolId> ids;      // index-aligned with paths
  clio::run::PoolId safe_id;
  std::string alloc_log;
  clio::run::safe_bdev::CreateParams params;
  clio::run::safe_bdev::Client safe;
  /** Everything written so far: (blocks, bytes). */
  std::vector<std::pair<std::vector<Block>, std::vector<ctp::u8>>> sets;

  /**
   * Create the disks and compose the array from YAML (4 data members and
   * 2 `parity: true` members, max_failures 2).
   * @param tag distinguishes rigs (file names, pool ids)
   * @param id_base pool-id major of the first disk
   */
  void Build(const std::string &tag, clio::run::u32 id_base) {
    dir = ScratchDir(tag);
    alloc_log = (dir / (tag + ".alog")).string();
    std::error_code ec;
    fs::remove(alloc_log, ec);
    fs::remove(alloc_log + ".members", ec);
    fs::remove(alloc_log + ".journal", ec);
    std::string yaml = "max_failures: 2\nalloc_log: " + YamlQuote(alloc_log) +
                       "\nmembers:\n";
    for (int i = 0; i < kMembers; ++i) {
      const std::string p = (dir / (tag + "_disk" + std::to_string(i) +
                                    ".bin")).string();
      fs::remove(p, ec);
      fs::remove(p + ".fail", ec);
      fs::remove(p + ".alloc_log", ec);
      const clio::run::PoolId id = CreateDisk(
          p, clio::run::PoolId(id_base + static_cast<clio::run::u32>(i), 0));
      REQUIRE_FALSE(id.IsNull());
      paths.push_back(p);
      ids.push_back(id);
      yaml += "  - pool_name: " + YamlQuote(p) + "\n    node_id: 0\n" +
              "    pool_id_major: " + std::to_string(id.major_) + "\n" +
              "    pool_id_minor: " + std::to_string(id.minor_) + "\n";
      if (i >= kDataMembers) yaml += "    parity: true\n";
    }
    clio::run::PoolConfig pc;
    pc.config_ = yaml;
    params.LoadConfig(pc);
    int n_parity = 0;
    for (const auto &m : params.members_) n_parity += m.parity_ ? 1 : 0;
    REQUIRE(params.members_.size() == static_cast<size_t>(kMembers));
    REQUIRE(n_parity == kParityMembers);
    REQUIRE(params.max_failures_ == 2u);
    REQUIRE(params.alloc_log_path_ == alloc_log);
    safe_id = clio::run::PoolId(id_base + 50, 0);
    Create();
  }

  /** (Re)create the safe_bdev pool from `params`. */
  void Create() {
    safe = clio::run::safe_bdev::Client(safe_id);
    auto t = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                              "safe_disk_fail_" + safe_id.ToString(), safe_id,
                              params.max_failures_, params.members_,
                              params.alloc_log_path_);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    safe.pool_id_ = t->new_pool_id_;
  }

  /** Stop the array (its member disks stay), as a shutdown would. */
  void Shutdown() {
    auto fl = safe.AsyncFlushAllocLog(clio::run::PoolQuery::Dynamic(), 0);
    fl.Wait();
    REQUIRE(fl->GetReturnCode() == 0);
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                    safe.pool_id_);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
  }

  /** Write a fresh allocation of `len` bytes and remember it. */
  void WriteNew(clio::run::u64 len, ctp::u8 seed) {
    std::vector<Block> b = Alloc(safe, len);
    std::vector<ctp::u8> d = Pattern(len, seed);
    REQUIRE(Write(safe, b, d) == 0);
    sets.emplace_back(b, d);
  }

  /** Overwrite remembered set `i` with new bytes. */
  void Rewrite(size_t i, ctp::u8 seed) {
    std::vector<ctp::u8> d = Pattern(sets[i].second.size(), seed);
    REQUIRE(Write(safe, sets[i].first, d) == 0);
    sets[i].second = d;
  }

  /** Every remembered byte reads back exactly. */
  void VerifyAll() {
    for (const auto &s : sets) {
      std::vector<ctp::u8> got;
      REQUIRE(Read(safe, s.first, got) == 0);
      REQUIRE(got == s.second);
    }
  }

  /** Remove every fault marker and every file this rig made. */
  void Cleanup() {
    std::error_code ec;
    for (const auto &p : paths) fs::remove(p + ".fail", ec);
    fs::remove_all(dir, ec);
  }
};

/** Bytes per write set: four full stripes plus a half chunk. */
constexpr clio::run::u64 kSetLen = 16 * kChunkLen + kChunkLen / 2;

}  // namespace

TEST_CASE("safe_bdev_disk_fail_degraded_and_recover",
          "[safe_bdev][disk_fail]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      21000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("deg", base);
  ArrayView v = QueryArray(rig.safe);
  REQUIRE(v.data_count == kDataMembers);
  REQUIRE(v.parity_level == kParityMembers);
  REQUIRE(v.faulty_members == 0);

  rig.WriteNew(kSetLen, 1);
  rig.WriteNew(kSetLen, 2);
  FlushParity(rig.safe);
  rig.VerifyAll();

  // --- Disk 1 (data) dies. The read that hits it faults it and is served
  //     from the survivors + parity. ---
  KillDisk(rig.paths[1]);
  rig.VerifyAll();
  v = QueryArray(rig.safe);
  REQUIRE(v.faulty_members == 1);
  REQUIRE(MemberState(v, "data", 1) == "faulty");
  // Writes go on, degraded: overwrite a set (stripes with the dead disk)
  // and write new data.
  rig.Rewrite(0, 11);
  rig.WriteNew(kSetLen, 3);
  FlushParity(rig.safe);
  rig.VerifyAll();
  HLOG(kInfo, "disk_fail: one data disk down, all bytes served");

  // --- Disk 3 (data) dies; this time a WRITE hits it first. ---
  KillDisk(rig.paths[3]);
  rig.Rewrite(1, 12);
  v = QueryArray(rig.safe);
  REQUIRE(v.faulty_members == 2);
  REQUIRE(MemberState(v, "data", 3) == "faulty");
  rig.WriteNew(kSetLen, 4);
  FlushParity(rig.safe);
  rig.VerifyAll();
  HLOG(kInfo, "disk_fail: two data disks down, all bytes served");

  // --- The disks are replaced: rebuild both onto fresh disks. ---
  ReviveDisk(rig.paths[1]);
  ReviveDisk(rig.paths[3]);
  for (int dead : {1, 3}) {
    const std::string np = (rig.dir / ("deg_new" + std::to_string(dead) +
                                       ".bin")).string();
    const clio::run::PoolId nid = CreateDisk(
        np, clio::run::PoolId(base + 10 + static_cast<clio::run::u32>(dead),
                              0));
    REQUIRE_FALSE(nid.IsNull());
    auto rec = rig.safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(),
                                         rig.ids[dead], np, 0, nid);
    rec.Wait();
    REQUIRE(rec->GetReturnCode() == 0);
    rig.paths[dead] = np;
    rig.ids[dead] = nid;
  }
  v = QueryArray(rig.safe);
  REQUIRE(v.faulty_members == 0);
  rig.VerifyAll();

  // Full redundancy again: any two disks may die, here a data and a parity.
  KillDisk(rig.paths[0]);
  KillDisk(rig.paths[kDataMembers]);  // parity row 0
  rig.VerifyAll();
  v = QueryArray(rig.safe);
  REQUIRE(v.faulty_members == 2);
  REQUIRE(MemberState(v, "data", 0) == "faulty");
  REQUIRE(MemberState(v, "parity", 0) == "faulty");
  rig.WriteNew(kSetLen, 6);
  FlushParity(rig.safe);
  rig.VerifyAll();
  HLOG(kInfo, "disk_fail: recovered array survives two more failures");

  // --- A third disk dies: past max_failures. Requests that need it fail;
  //     none returns wrong bytes. ---
  KillDisk(rig.paths[2]);
  int failed_reads = 0;
  for (const auto &s : rig.sets) {
    std::vector<ctp::u8> got;
    if (Read(rig.safe, s.first, got) == 0) {
      REQUIRE(got == s.second);
    } else {
      ++failed_reads;
    }
  }
  REQUIRE(failed_reads > 0);
  // Every chunk still on a live disk reads back exactly, one block at a time.
  int good_chunks = 0;
  for (const auto &s : rig.sets) {
    clio::run::u64 pos = 0;
    for (const Block &b : s.first) {
      std::vector<ctp::u8> got;
      if (Read(rig.safe, {b}, got) == 0) {
        REQUIRE(std::equal(got.begin(), got.end(), s.second.begin() + pos));
        ++good_chunks;
      }
      pos += b.size_;
    }
  }
  REQUIRE(good_chunks > 0);
  // A write into stripes that lost three members fails.
  REQUIRE(Write(rig.safe, rig.sets[0].first,
                Pattern(rig.sets[0].second.size(), 99)) != 0);
  HLOG(kInfo, "disk_fail: 3 of 2 tolerated failures -> errors, never wrong "
       "bytes ({} whole-set reads failed, {} chunks still served)",
       failed_reads, good_chunks);
  rig.Cleanup();
}

TEST_CASE("safe_bdev_disk_fail_restart_keeps_faulty",
          "[safe_bdev][disk_fail][restart]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      26000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("rst", base);
  rig.WriteNew(kSetLen, 21);
  rig.WriteNew(kSetLen, 22);
  FlushParity(rig.safe);

  // Disk 2 dies at runtime and the first request to hit it is a WRITE to
  // healthy stripes: its member write fails after the other members' writes
  // landed. The write still succeeds -- its stripes are redone degraded --
  // and every byte, old and new, reads back.
  KillDisk(rig.paths[2]);
  rig.WriteNew(kSetLen, 23);
  REQUIRE(MemberState(QueryArray(rig.safe), "data", 2) == "faulty");
  FlushParity(rig.safe);
  rig.VerifyAll();

  // Restart: the array comes back with the member still faulty, from the
  // member manifest -- even though its disk now answers again.
  rig.Shutdown();
  ReviveDisk(rig.paths[2]);
  rig.Create();
  ArrayView v = QueryArray(rig.safe);
  REQUIRE(v.data_count == kDataMembers);
  REQUIRE(v.parity_level == kParityMembers);
  REQUIRE(v.faulty_members == 1);
  REQUIRE(MemberState(v, "data", 2) == "faulty");
  rig.VerifyAll();
  rig.WriteNew(kSetLen, 24);
  FlushParity(rig.safe);
  rig.VerifyAll();

  // Replacing it restores the array.
  const std::string np = (rig.dir / "rst_new2.bin").string();
  const clio::run::PoolId nid =
      CreateDisk(np, clio::run::PoolId(base + 20, 0));
  REQUIRE_FALSE(nid.IsNull());
  auto rec = rig.safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(),
                                       rig.ids[2], np, 0, nid);
  rec.Wait();
  REQUIRE(rec->GetReturnCode() == 0);
  REQUIRE(QueryArray(rig.safe).faulty_members == 0);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_compose_parity_members_limit",
          "[safe_bdev][disk_fail][compose]") {
  EnsureInit();
  REQUIRE(g_initialized);
  // More `parity: true` members than max_failures: Create refuses.
  const clio::run::u32 base =
      31000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  const fs::path dir = ScratchDir("lim");
  std::vector<MemberBdevDesc> members;
  for (int i = 0; i < 3; ++i) {
    const std::string p =
        (dir / ("lim_disk" + std::to_string(i) + ".bin")).string();
    const clio::run::PoolId id =
        CreateDisk(p, clio::run::PoolId(base + static_cast<clio::run::u32>(i),
                                        0));
    REQUIRE_FALSE(id.IsNull());
    members.emplace_back(p, 0, id, /*parity=*/i > 0);
  }
  const clio::run::PoolId safe_id(base + 3, 0);
  clio::run::safe_bdev::Client safe(safe_id);
  auto t = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), "safe_lim",
                            safe_id, /*max_failures=*/1, members);
  t.Wait();
  REQUIRE(t->GetReturnCode() != 0);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("safe_bdev_sync_through_plain_bdev_client",
          "[safe_bdev][disk_fail][sync]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      36000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("syn", base);
  rig.WriteNew(kSetLen, 31);
  rig.WriteNew(kSetLen, 32);
  // The CTE fsyncs a tier through the PLAIN bdev client (#1120): method 18
  // must be the array's Sync, not one of its management methods.
  clio::run::bdev::Client plain(rig.safe.pool_id_);
  auto t = plain.AsyncSync(clio::run::PoolQuery::Dynamic());
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  ArrayView v = QueryArray(rig.safe);
  REQUIRE(v.dirty_slots == 0);
  REQUIRE(v.faulty_members == 0);
  REQUIRE(v.data_count == kDataMembers);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_parity_current_when_write_acks",
          "[safe_bdev][disk_fail][sync]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      41000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("ack", base);
  // No FlushParity anywhere: every acked write must already be protected.
  for (int i = 0; i < 4; ++i) {
    rig.WriteNew(kSetLen, static_cast<ctp::u8>(40 + i));
    REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  }
  // Single-chunk writes from several threads at once. Consecutive chunks go
  // to different members' same slot, so these share stripes and race on
  // their parity (#1121).
  std::vector<std::vector<Block>> blocks;
  std::vector<std::vector<ctp::u8>> datas;
  for (int i = 0; i < 16; ++i) {
    blocks.push_back(Alloc(rig.safe, kChunkLen));
    datas.push_back(Pattern(kChunkLen, static_cast<ctp::u8>(60 + i)));
  }
  std::vector<std::thread> th;
  std::vector<clio::run::u32> rcs(blocks.size(), 1);
  for (size_t i = 0; i < blocks.size(); ++i) {
    th.emplace_back([&, i] { rcs[i] = Write(rig.safe, blocks[i], datas[i]); });
  }
  for (auto &x : th) x.join();
  for (size_t i = 0; i < blocks.size(); ++i) {
    REQUIRE(rcs[i] == 0);
    rig.sets.emplace_back(blocks[i], datas[i]);
  }
  REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  // Two disks die right away: everything acked must come back exactly.
  KillDisk(rig.paths[0]);
  KillDisk(rig.paths[kDataMembers + 1]);  // parity row 1
  rig.VerifyAll();
  // Reads touch parity only to rebuild the dead data disk, so the dead
  // parity disk shows up at the next write; that write must succeed too.
  rig.WriteNew(kSetLen, 90);
  REQUIRE(QueryArray(rig.safe).faulty_members == 2);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_crash_between_data_and_parity",
          "[safe_bdev][disk_fail][restart][sync]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      46000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("crs", base);
  rig.WriteNew(kSetLen, 71);
  rig.WriteNew(kSetLen, 72);
  // Overwrite set 0, "crashing" after its data lands but before its parity
  // does: the stripes' on-disk parity still encodes the OLD bytes.
  // (The fault also holds off the background builder until the "crash".)
  ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY", "1", 1);
  rig.Rewrite(0, 73);
  REQUIRE(QueryArray(rig.safe).dirty_slots > 0);
  rig.Shutdown();
  ctp::SystemInfo::Unsetenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY");
  rig.Create();
  // Restart re-encodes the stripes the intent log names. Had it trusted the
  // stale parity, losing a disk would now decode set 0 to wrong bytes.
  FlushParity(rig.safe);
  REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  KillDisk(rig.paths[1]);
  KillDisk(rig.paths[2]);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_crash_during_degraded_write",
          "[safe_bdev][disk_fail][restart]") {
  // #1137 (the degraded write hole): with a data member dead, its chunks
  // exist only through their stripes' parity. "Crash" a rewrite after its
  // data landed on the live members but before that parity did, and restart
  // with the member still dead. The stripes must come back encodable (the
  // down column's chunk was journaled before the write), every set must
  // read back, further writes to those stripes must succeed, and losing a
  // second disk afterwards must still decode everything.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      78000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("cdw", base);
  rig.WriteNew(kSetLen, 81);
  rig.WriteNew(kSetLen, 82);
  KillDisk(rig.paths[0]);  // a data member
  rig.VerifyAll();         // its first failed read marks it down
  ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY", "1", 1);
  rig.Rewrite(0, 83);
  REQUIRE(QueryArray(rig.safe).dirty_slots > 0);
  rig.Shutdown();
  ctp::SystemInfo::Unsetenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY");
  rig.Create();
  FlushParity(rig.safe);
  INFO("dirty stripes after restart: " +
       std::to_string(QueryArray(rig.safe).dirty_slots));
  REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  rig.VerifyAll();
  rig.Rewrite(0, 84);  // refused before the fix: "stale parity"
  rig.Rewrite(1, 85);
  rig.VerifyAll();
  KillDisk(rig.paths[1]);  // max_failures now
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_full_stripe_writes_protected",
          "[safe_bdev][disk_fail]") {
  // #1126: a write that covers whole stripes encodes their parity from its
  // own bytes, never reading the members. Fresh full-stripe writes,
  // full-stripe rewrites and a mixed write (whole stripes plus a partial
  // one) must all leave parity that decodes every byte once max_failures
  // data disks die -- with no flush in between.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      79000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("fsw", base);
  for (ctp::u8 k = 0; k < 4; ++k) rig.WriteNew(16 * kChunkLen, 90 + k);
  rig.WriteNew(kSetLen, 95);  // whole stripes plus half a chunk
  for (size_t i = 0; i < 4; ++i) rig.Rewrite(i, static_cast<ctp::u8>(100 + i));
  rig.Rewrite(4, 105);
  REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  KillDisk(rig.paths[0]);
  KillDisk(rig.paths[2]);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_free_keeps_stripes_protected",
          "[safe_bdev][disk_fail][restart][sync]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      51000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("fre", base);
  for (int i = 0; i < 6; ++i) rig.WriteNew(kSetLen, static_cast<ctp::u8>(80 + i));
  // Free every other set: the stripes they shared with the kept sets narrow.
  // Nothing may leave those stripes unprotected, not even for a moment.
  std::vector<std::pair<std::vector<Block>, std::vector<ctp::u8>>> kept;
  for (size_t i = 0; i < rig.sets.size(); ++i) {
    if (i % 2 == 0) {
      auto fr = rig.safe.AsyncFreeBlocks(clio::run::PoolQuery::Dynamic(),
                                         ToPriv(rig.sets[i].first));
      fr.Wait();
      REQUIRE(fr->GetReturnCode() == 0);
    } else {
      kept.push_back(rig.sets[i]);
    }
  }
  rig.sets = kept;
  REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  // Restart (the narrowed stripes come back from the intent log), then two
  // disks die at once with no parity flush in between.
  rig.Shutdown();
  rig.Create();
  FlushParity(rig.safe);
  KillDisk(rig.paths[1]);
  KillDisk(rig.paths[3]);
  rig.VerifyAll();
  // And without a restart: free more, kill nothing new, overwrite degraded.
  rig.Rewrite(0, 95);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_write_throughput",
          "[safe_bdev][disk_fail][perf]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      56000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("thr", base);
  // Concurrent 128 KiB writes from several threads into separate
  // allocations, the shape the CTE puts on a tier (#1121 made parity
  // synchronous; this reports what that costs). Rounds of allocate / write /
  // verify a sample / free keep it inside the small test members.
  constexpr int kThreads = 8;
  constexpr int kRounds = 10;
  // CLIO_SAFE_THR_CHUNKS: chunks per write (default 2 = 128 KiB; 16 = the
  // CTE's 1 MiB page, whole stripes). Fewer writes per thread for larger
  // ones, to stay inside the small test members.
  const char *ce = std::getenv("CLIO_SAFE_THR_CHUNKS");
  const clio::run::u64 chunks =
      ce != nullptr ? std::max<clio::run::u64>(1, std::strtoull(ce, nullptr, 10))
                    : 2;
  const clio::run::u64 kLen = chunks * kChunkLen;
  const int kPerThread = static_cast<int>(std::max<clio::run::u64>(1, 12 / chunks));
  double write_ms = 0;
  for (int r = 0; r < kRounds; ++r) {
    std::vector<std::vector<std::vector<Block>>> blocks(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      for (int i = 0; i < kPerThread; ++i) {
        blocks[t].push_back(Alloc(rig.safe, kLen));
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    std::vector<int> bad(kThreads, 0);
    for (int t = 0; t < kThreads; ++t) {
      th.emplace_back([&, t] {
        for (int i = 0; i < kPerThread; ++i) {
          if (Write(rig.safe, blocks[t][i],
                    Pattern(kLen, static_cast<ctp::u8>(r + t * 31 + i))) != 0) {
            ++bad[t];
          }
        }
      });
    }
    for (auto &x : th) x.join();
    write_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    for (int t = 0; t < kThreads; ++t) {
      REQUIRE(bad[t] == 0);
      std::vector<ctp::u8> got;
      REQUIRE(Read(rig.safe, blocks[t][0], got) == 0);
      REQUIRE(got == Pattern(kLen, static_cast<ctp::u8>(r + t * 31)));
      for (auto &b : blocks[t]) {
        auto fr = rig.safe.AsyncFreeBlocks(clio::run::PoolQuery::Dynamic(),
                                           ToPriv(b));
        fr.Wait();
        REQUIRE(fr->GetReturnCode() == 0);
      }
    }
  }
  const int nwrites = kThreads * kPerThread * kRounds;
  const double mib = nwrites * kLen / 1048576.0;
  HLOG(kInfo, "safe_bdev_write_throughput: {} x {} KiB writes from {} "
       "threads in {} ms = {} MiB/s ({} ms per write)",
       nwrites, kLen / 1024, kThreads, write_ms, mib / (write_ms / 1000.0),
       write_ms / nwrites);
  rig.Cleanup();
}

TEST_CASE("safe_bdev_small_rewrites_delta_parity",
          "[safe_bdev][disk_fail][sync]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      61000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("dlt", base);
  rig.WriteNew(kSetLen, 101);
  rig.WriteNew(kSetLen, 102);
  // Small rewrites inside already-encoded stripes take the delta path
  // (parity updated from old ^ new over the written range only, #1126).
  // Mix single-chunk pieces, chunk-crossing spans and unaligned edges.
  std::mt19937 rng(7);
  for (int i = 0; i < 60; ++i) {
    auto &set = rig.sets[static_cast<size_t>(i) % rig.sets.size()];
    const clio::run::u64 total = set.second.size();
    const clio::run::u64 len = 1 + rng() % (3 * 4096);
    const clio::run::u64 off = rng() % (total - len);
    // The byte range [off, off+len) of the set, as device blocks.
    std::vector<Block> sub;
    clio::run::u64 pos = 0;
    for (const Block &b : set.first) {
      const clio::run::u64 lo = std::max(pos, off);
      const clio::run::u64 hi = std::min(pos + b.size_, off + len);
      if (lo < hi) sub.push_back(Block(b.offset_ + (lo - pos), hi - lo, 0));
      pos += b.size_;
    }
    std::vector<ctp::u8> bytes = Pattern(len, static_cast<ctp::u8>(200 + i));
    REQUIRE(Write(rig.safe, sub, bytes) == 0);
    std::copy(bytes.begin(), bytes.end(), set.second.begin() + off);
  }
  REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  // No parity flush: two disks die right away. Every byte must reconstruct.
  KillDisk(rig.paths[0]);
  KillDisk(rig.paths[2]);
  rig.VerifyAll();
  rig.Cleanup();
}

TEST_CASE("safe_bdev_sync_with_dying_disks",
          "[safe_bdev][disk_fail][sync]") {
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      66000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("snc", base);
  rig.WriteNew(kSetLen, 111);
  clio::run::bdev::Client plain(rig.safe.pool_id_);
  auto sync = [&]() {
    auto t = plain.AsyncSync(clio::run::PoolQuery::Dynamic());
    t.Wait();
    return t->GetReturnCode();
  };
  // A disk that dies is found by the fsync itself: it must be faulted and
  // the fsync must still succeed while the array is within max_failures.
  KillDisk(rig.paths[1]);
  REQUIRE(sync() == 0);
  REQUIRE(QueryArray(rig.safe).faulty_members == 1);
  KillDisk(rig.paths[kDataMembers]);  // a parity disk
  rig.WriteNew(kSetLen, 112);
  REQUIRE(sync() == 0);
  REQUIRE(QueryArray(rig.safe).faulty_members == 2);
  rig.VerifyAll();
  // A third disk is past max_failures: now fsync reports it.
  KillDisk(rig.paths[2]);
  REQUIRE(sync() != 0);
  rig.Cleanup();
}


SIMPLE_TEST_MAIN()

namespace {

/**
 * Bytes that name their writer: every 8-byte word is (tag << 32 | word
 * index), so a mismatch says whose bytes came back and from which offset.
 * @param n length (bytes)
 * @param tag the writer's (thread, op) id
 * @return the pattern
 */
std::vector<ctp::u8> TaggedPattern(size_t n, clio::run::u32 tag) {
  std::vector<ctp::u8> v(n);
  for (size_t i = 0; i < n; i += 8) {
    const clio::run::u64 w =
        (static_cast<clio::run::u64>(tag) << 32) | static_cast<clio::run::u64>(i / 8);
    std::memcpy(v.data() + i, &w, std::min<size_t>(8, n - i));
  }
  return v;
}

/** One churn thread's live allocations and the bytes each must hold. */
struct ChurnSet {
  std::vector<Block> blocks;
  std::vector<ctp::u8> bytes;
};

/**
 * Sub-range [off, off+len) of an allocation, as device blocks.
 * @param blocks the allocation
 * @param off byte offset into it
 * @param len bytes
 * @return the device blocks covering exactly that range
 */
std::vector<Block> SubBlocks(const std::vector<Block> &blocks,
                             clio::run::u64 off, clio::run::u64 len) {
  std::vector<Block> sub;
  clio::run::u64 pos = 0;
  for (const Block &b : blocks) {
    const clio::run::u64 lo = std::max(pos, off);
    const clio::run::u64 hi = std::min(pos + b.size_, off + len);
    if (lo < hi) sub.push_back(Block(b.offset_ + (lo - pos), hi - lo, 0));
    pos += b.size_;
  }
  return sub;
}

/**
 * Describe the first mismatch between what a set must hold and what was
 * read: the offset, and whose (tag, word) the read bytes carry.
 * @param want expected bytes
 * @param got bytes read
 * @return a one-line description ("" when equal)
 */
std::string DescribeMismatch(const std::vector<ctp::u8> &want,
                             const std::vector<ctp::u8> &got) {
  if (want == got) return "";
  size_t i = 0;
  while (i < want.size() && i < got.size() && want[i] == got[i]) ++i;
  const size_t w0 = (i / 8) * 8;
  clio::run::u64 gw = 0, ww = 0;
  std::memcpy(&gw, got.data() + w0, std::min<size_t>(8, got.size() - w0));
  std::memcpy(&ww, want.data() + w0, std::min<size_t>(8, want.size() - w0));
  size_t bad = 0;
  for (size_t k = 0; k < want.size() && k < got.size(); ++k) {
    bad += want[k] != got[k] ? 1 : 0;
  }
  return "first diff at byte " + std::to_string(i) + " of " +
         std::to_string(want.size()) + " (" + std::to_string(bad) +
         " bytes differ): want tag " + std::to_string(ww >> 32) + " word " +
         std::to_string(ww & 0xffffffffULL) + ", got tag " +
         std::to_string(gw >> 32) + " word " +
         std::to_string(gw & 0xffffffffULL);
}

/**
 * One churn step on a thread's sets: write a fresh allocation, rewrite a
 * sub-range of a live one, or free one (keeping at most `max_sets`).
 * @param safe the array
 * @param sets this thread's live sets
 * @param rng this thread's generator
 * @param tag unique id for the bytes this step writes
 * @param max_sets cap on live sets
 * @return a failure description ("" on success)
 */
std::string ChurnStep(clio::run::safe_bdev::Client &safe,
                      std::vector<ChurnSet> &sets, std::mt19937 &rng,
                      clio::run::u32 tag, size_t max_sets) {
  const int op = static_cast<int>(rng() % 3);
  if ((op == 0 && !sets.empty()) || sets.size() >= max_sets) {
    // Free a random set: its stripes narrow, its chunks become reusable.
    const size_t i = rng() % sets.size();
    auto fr = safe.AsyncFreeBlocks(clio::run::PoolQuery::Dynamic(),
                                   ToPriv(sets[i].blocks));
    fr.Wait();
    if (fr->GetReturnCode() != 0) return "free failed";
    sets.erase(sets.begin() + static_cast<long>(i));
    return "";
  }
  if (op == 1 && !sets.empty()) {
    // Rewrite an unaligned sub-range in place.
    ChurnSet &s = sets[rng() % sets.size()];
    const clio::run::u64 total = s.bytes.size();
    const clio::run::u64 len = 1 + rng() % std::min<clio::run::u64>(total, 3 * kChunkLen / 2);
    const clio::run::u64 off = rng() % (total - len + 1);
    std::vector<ctp::u8> d = TaggedPattern(len, tag);
    if (Write(safe, SubBlocks(s.blocks, off, len), d) != 0) {
      return "rewrite failed";
    }
    std::copy(d.begin(), d.end(), s.bytes.begin() + static_cast<long>(off));
    return "";
  }
  // A fresh allocation of 1..3 chunks plus an unaligned tail.
  const clio::run::u64 len = (1 + rng() % 3) * kChunkLen + rng() % kChunkLen;
  auto t = safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), len);
  t.Wait();
  if (t->GetReturnCode() != 0) return "";  // full: not this test's concern
  ChurnSet s;
  for (size_t i = 0; i < t->blocks_.size(); ++i) s.blocks.push_back(t->blocks_[i]);
  s.bytes = TaggedPattern(len, tag);
  if (Write(safe, s.blocks, s.bytes) != 0) return "fresh write failed";
  sets.push_back(std::move(s));
  return "";
}

/**
 * Run `steps` churn steps on every thread concurrently.
 * @param safe the array
 * @param sets per-thread live sets
 * @param rngs per-thread generators
 * @param next_tag per-thread op counters (tags are thread << 20 | op)
 * @param steps steps per thread
 * @return failures seen ("" if none)
 */
std::string ChurnPhase(clio::run::safe_bdev::Client &safe,
                       std::vector<std::vector<ChurnSet>> &sets,
                       std::vector<std::mt19937> &rngs,
                       std::vector<clio::run::u32> &next_tag, int steps) {
  std::vector<std::string> err(sets.size());
  std::vector<std::thread> th;
  for (size_t t = 0; t < sets.size(); ++t) {
    th.emplace_back([&, t] {
      for (int i = 0; i < steps && err[t].empty(); ++i) {
        const clio::run::u32 tag =
            (static_cast<clio::run::u32>(t + 1) << 20) | next_tag[t]++;
        err[t] = ChurnStep(safe, sets[t], rngs[t], tag, /*max_sets=*/6);
      }
    });
  }
  for (auto &x : th) x.join();
  std::string all;
  for (size_t t = 0; t < err.size(); ++t) {
    if (!err[t].empty()) all += "thread " + std::to_string(t) + ": " + err[t] + "; ";
  }
  return all;
}

/**
 * Read every live set back.
 * @param safe the array
 * @param sets per-thread live sets
 * @return mismatches and read failures, described ("" if all exact)
 */
std::string VerifyChurn(clio::run::safe_bdev::Client &safe,
                        const std::vector<std::vector<ChurnSet>> &sets) {
  std::string out;
  int n = 0;
  for (size_t t = 0; t < sets.size(); ++t) {
    for (size_t i = 0; i < sets[t].size(); ++i) {
      std::vector<ctp::u8> got;
      const clio::run::u32 rc = Read(safe, sets[t][i].blocks, got);
      std::string d = rc != 0 ? "read rc " + std::to_string(rc)
                              : DescribeMismatch(sets[t][i].bytes, got);
      if (!d.empty() && n++ < 6) {
        out += "[thread " + std::to_string(t) + " set " + std::to_string(i) +
               ": " + d + "] ";
      }
    }
  }
  if (n > 6) out += "(+" + std::to_string(n - 6) + " more)";
  return out;
}

}  // namespace

TEST_CASE("safe_bdev_degraded_churn_verify",
          "[safe_bdev][disk_fail][restart]") {
  // #1131: after a data disk died under concurrent writes, came back, and
  // the array restarted, a 64 KiB read silently returned another
  // allocation's OLD bytes. Churn allocations (fresh writes, unaligned
  // rewrites, frees that let chunks be reused) from several threads across
  // that whole sequence and check every live byte -- each word names its
  // writer, so a wrong read says whose bytes it got.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      71000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("chn", base);
  constexpr size_t kThreads = 4;
  std::vector<std::vector<ChurnSet>> sets(kThreads);
  std::vector<std::mt19937> rngs;
  for (size_t t = 0; t < kThreads; ++t) rngs.emplace_back(1131 + t);
  std::vector<clio::run::u32> next_tag(kThreads, 0);
  auto phase = [&](int steps, const char *when) {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, steps);
    INFO(std::string("churn ") + when + ": " + e);
    REQUIRE(e.empty());
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO(std::string("verify ") + when + ": " + v);
    REQUIRE(v.empty());
  };
  phase(40, "healthy");
  KillDisk(rig.paths[0]);
  phase(60, "data disk 0 dead");
  ReviveDisk(rig.paths[0]);  // it answers again; the array must not trust it
  phase(40, "disk 0 answering again");
  rig.Shutdown();
  rig.Create();
  {
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO("verify after restart: " + v);
    REQUIRE(v.empty());
  }
  phase(40, "after restart");
  rig.Cleanup();
}

TEST_CASE("safe_bdev_two_down_churn_across_restart",
          "[safe_bdev][disk_fail][restart]") {
  // #1131 (traced): after a crash restart with a data AND a parity member
  // of an array dead (max_failures), a node-local cache copy written to the
  // array afterwards read back with one 64 KiB chunk holding an older
  // allocation's bytes. Churn with both members down, restart the array
  // (the crash-like way: no alloc-log flush first), keep churning with them
  // still dead, and check every live byte throughout.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      76000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("tdn", base);
  constexpr size_t kThreads = 4;
  std::vector<std::vector<ChurnSet>> sets(kThreads);
  std::vector<std::mt19937> rngs;
  for (size_t t = 0; t < kThreads; ++t) rngs.emplace_back(2131 + t);
  std::vector<clio::run::u32> next_tag(kThreads, 0);
  auto phase = [&](int steps, const char *when) {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, steps);
    INFO(std::string("churn ") + when + ": " + e);
    REQUIRE(e.empty());
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO(std::string("verify ") + when + ": " + v);
    REQUIRE(v.empty());
  };
  phase(40, "healthy");
  KillDisk(rig.paths[0]);                // a data member
  KillDisk(rig.paths[kMembers - 1]);     // a parity member
  phase(60, "data 0 + parity 1 dead");
  // Crash-like restart: destroy the pool without flushing its alloc log.
  {
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                    rig.safe.pool_id_);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
  }
  rig.Create();
  {
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO("verify after restart: " + v);
    REQUIRE(v.empty());
  }
  phase(80, "after restart, both still dead");
  rig.Cleanup();
}

TEST_CASE("safe_bdev_concurrent_degraded_crash",
          "[safe_bdev][disk_fail][restart]") {
  // #1137 under concurrency (seen on the 6-node rolling restart: 20 stripes
  // stranded after a crash): four threads churn (allocate, rewrite parts,
  // free) with a data disk dead, "crash" with every write's parity skipped,
  // and restart with the disk still dead. Every stripe must come back
  // encodable and every live byte intact.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      80000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("cdc", base);
  constexpr size_t kThreads = 4;
  std::vector<std::vector<ChurnSet>> sets(kThreads);
  std::vector<std::mt19937> rngs;
  for (size_t t = 0; t < kThreads; ++t) rngs.emplace_back(1137 + t);
  std::vector<clio::run::u32> next_tag(kThreads, 0);
  auto phase = [&](int steps, const char *when) {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, steps);
    INFO(std::string("churn ") + when + ": " + e);
    REQUIRE(e.empty());
  };
  phase(30, "healthy");
  KillDisk(rig.paths[0]);
  {
    const std::string v = VerifyChurn(rig.safe, sets);  // faults member 0
    INFO("verify after the disk died: " + v);
    REQUIRE(v.empty());
  }
  phase(30, "degraded");
  ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY", "1", 1);
  phase(30, "degraded, parity skipped");
  const int64_t dirty_before = QueryArray(rig.safe).dirty_slots;
  {
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                    rig.safe.pool_id_);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
  }
  ctp::SystemInfo::Unsetenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY");
  rig.Create();
  FlushParity(rig.safe);
  const int64_t dirty_after = QueryArray(rig.safe).dirty_slots;
  INFO("dirty stripes before the crash " + std::to_string(dirty_before) +
       ", after restart + flush " + std::to_string(dirty_after));
  REQUIRE(dirty_before > 0);
  REQUIRE(dirty_after == 0);
  {
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO("verify after restart: " + v);
    REQUIRE(v.empty());
  }
  phase(30, "after restart, disk still dead");
  {
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO("verify at the end: " + v);
    REQUIRE(v.empty());
  }
  rig.Cleanup();
}

TEST_CASE("safe_bdev_degraded_crash_cycles",
          "[safe_bdev][disk_fail][restart]") {
  // #1145: repeated crashes while a data disk is dead, with allocation
  // churn (frees and reuse) between them -- the shape of the 6-node rolling
  // restart that served another file's bytes. Each cycle churns with every
  // write's parity skipped, "crashes", restarts with the disk still dead,
  // and checks every live byte.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      81000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("dcc", base);
  constexpr size_t kThreads = 4;
  std::vector<std::vector<ChurnSet>> sets(kThreads);
  std::vector<std::mt19937> rngs;
  for (size_t t = 0; t < kThreads; ++t) rngs.emplace_back(1145 + t);
  std::vector<clio::run::u32> next_tag(kThreads, 0);
  auto phase = [&](int steps, const std::string &when) {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, steps);
    INFO("churn " + when + ": " + e);
    REQUIRE(e.empty());
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO("verify " + when + ": " + v);
    REQUIRE(v.empty());
  };
  phase(30, "healthy");
  KillDisk(rig.paths[0]);
  phase(20, "degraded");
  for (int cycle = 0; cycle < 6; ++cycle) {
    const std::string c = "cycle " + std::to_string(cycle);
    phase(15, c + " before the crash");
    ctp::SystemInfo::Setenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY", "1", 1);
    {
      const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, 10);
      INFO("churn " + c + " parity skipped: " + e);
      REQUIRE(e.empty());
    }
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                    rig.safe.pool_id_);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
    ctp::SystemInfo::Unsetenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY");
    rig.Create();
    {
      const std::string v = VerifyChurn(rig.safe, sets);
      INFO("verify " + c + " after restart: " + v);
      REQUIRE(v.empty());
    }
    FlushParity(rig.safe);
    INFO(c + " dirty after flush: " +
         std::to_string(QueryArray(rig.safe).dirty_slots));
    REQUIRE(QueryArray(rig.safe).dirty_slots == 0);
  }
  KillDisk(rig.paths[kMembers - 1]);  // a parity disk too: max_failures
  phase(20, "two down");
  rig.Cleanup();
}

TEST_CASE("safe_bdev_rebuild_under_load_then_lose_two",
          "[safe_bdev][disk_fail][recover]") {
  // #1146: a data disk dies, is replaced and rebuilt WHILE writers churn
  // (allocate, rewrite parts, free -- so freed chunks hold old bytes the
  // fixed-width parity still encodes); then a data and a parity disk die
  // (max_failures again, behind the rebuild). Every live byte must decode.
  EnsureInit();
  REQUIRE(g_initialized);
  const clio::run::u32 base =
      82000 + static_cast<clio::run::u32>(getpid() & 0x3FF) * 4;
  Rig rig;
  rig.Build("rul", base);
  constexpr size_t kThreads = 4;
  std::vector<std::vector<ChurnSet>> sets(kThreads);
  std::vector<std::mt19937> rngs;
  for (size_t t = 0; t < kThreads; ++t) rngs.emplace_back(1146 + t);
  std::vector<clio::run::u32> next_tag(kThreads, 0);
  auto verify = [&](const std::string &when) {
    const std::string v = VerifyChurn(rig.safe, sets);
    INFO("verify " + when + ": " + v);
    REQUIRE(v.empty());
  };
  {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, 40);
    INFO("churn healthy: " + e);
    REQUIRE(e.empty());
  }
  KillDisk(rig.paths[0]);
  verify("after the disk died");
  {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, 20);
    INFO("churn degraded: " + e);
    REQUIRE(e.empty());
  }
  const std::string np = (rig.dir / "rul_new0.bin").string();
  const clio::run::PoolId nid = CreateDisk(np, clio::run::PoolId(base + 30, 0));
  REQUIRE_FALSE(nid.IsNull());
  // Churn during the rebuild.
  std::string churn_err;
  std::atomic<bool> rebuilding{true};
  std::thread churner([&] {
    while (rebuilding.load() && churn_err.empty()) {
      churn_err = ChurnPhase(rig.safe, sets, rngs, next_tag, 2);
    }
  });
  auto rec = rig.safe.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(),
                                       rig.ids[0], np, 0, nid);
  rec.Wait();
  rebuilding.store(false);
  churner.join();
  INFO("churn during the rebuild: " + churn_err);
  REQUIRE(churn_err.empty());
  REQUIRE(rec->GetReturnCode() == 0);
  REQUIRE(QueryArray(rig.safe).faulty_members == 0);
  rig.paths[0] = np;
  rig.ids[0] = nid;
  verify("after the rebuild");
  {
    // Restart as a crash does: the array comes back BEFORE any pool added
    // at runtime -- here, the replacement disk's pool is gone entirely.
    // Create must re-attach it from its backing file.
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    for (const clio::run::PoolId &p : {rig.safe.pool_id_, nid}) {
      auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(), p);
      d.Wait();
      REQUIRE(d->GetReturnCode() == 0);
    }
    std::this_thread::sleep_for(150ms);
    rig.Create();
    REQUIRE(QueryArray(rig.safe).faulty_members == 0);
    verify("after a restart without the replacement disk's pool");
  }
  {
    const std::string e = ChurnPhase(rig.safe, sets, rngs, next_tag, 20);
    INFO("churn after the rebuild: " + e);
    REQUIRE(e.empty());
  }
  KillDisk(rig.paths[1]);
  KillDisk(rig.paths[kMembers - 1]);
  verify("after two more disks died behind the rebuild");
  rig.Cleanup();
}

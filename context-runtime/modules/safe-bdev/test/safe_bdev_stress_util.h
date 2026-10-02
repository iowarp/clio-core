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
 * Shared bookkeeping for the safe_bdev stress tests: disks (file bdev
 * pools), an array wrapper that records every block written, and byte-exact
 * verification. Used by the single-node stress suite and the distributed
 * driver.
 */
#ifndef CLIO_SAFE_BDEV_TEST_SAFE_BDEV_STRESS_UTIL_H_
#define CLIO_SAFE_BDEV_TEST_SAFE_BDEV_STRESS_UTIL_H_

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "simple_test.h"

#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>

namespace sbstress {

using namespace std::chrono_literals;

namespace sb = clio::run::safe_bdev;
namespace bd = clio::run::bdev;

/** Member size: holds the superblock plus ~250 64 KiB chunk rows. */
constexpr clio::run::u64 kMemberSize = 16ull << 20;
/** One logical write: 4 chunks, spread round-robin over the data members. */
constexpr clio::run::u64 kIoLen = 4 * 65536;
/** Blocks the live-churn case keeps (4 data disks x ~250 slots / 4 chunks,
 *  well under full so allocation failures mean a bug, not a full array). */
constexpr size_t kLiveCap = 150;

inline bool g_initialized = false;

/** Start the embedded runtime once per process. */
inline void EnsureInit() {
  if (g_initialized) return;
  g_initialized = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (g_initialized) {
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
    std::this_thread::sleep_for(500ms);
  }
}

/** @return directory for member backing files. */
inline std::filesystem::path StressDir() {
  const char *e = std::getenv("CLIO_SAFE_STRESS_DIR");
  std::filesystem::path d =
      e != nullptr ? std::filesystem::path(e)
                   : std::filesystem::temp_directory_path();
  std::filesystem::create_directories(d);
  return d;
}

/**
 * Deterministic pattern for one logical write.
 * @param seed distinguishes writes
 * @return kIoLen bytes
 */
inline std::vector<ctp::u8> Pattern(clio::run::u32 seed) {
  std::vector<ctp::u8> v(kIoLen);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<ctp::u8>((seed * 131u + i * 7u + (i >> 9)) & 0xFF);
  }
  return v;
}

/** A disk: a file bdev pool with its backing path. */
struct Disk {
  std::string path;  // bdev pool name (== backing file path)
  clio::run::PoolId id;
  clio::run::u32 node = 0;  // node holding it (distributed arrays)
};

/** The array under test plus every byte it should hold. */
class Array {
 public:
  /**
   * @param tag short name used in pool ids and file names
   * @param base pool-id major of the first disk (disks count up from it)
   */
  Array(const std::string &tag, clio::run::u32 base)
      : tag_(tag), base_(base),
        safe_(clio::run::PoolId(base, 0)) {}

  /**
   * Create a fresh file bdev (a new disk).
   * @param d receives the disk
   * @return true on success
   */
  bool NewDisk(Disk *d) {
    const clio::run::u32 n = next_disk_++;
    d->path = (StressDir() / (tag_ + "_" + std::to_string(ctp::SystemInfo::GetPid()) + "_d" +
                              std::to_string(n) + ".bin"))
                  .string();
    std::error_code ec;
    std::filesystem::remove(d->path, ec);
    d->id = clio::run::PoolId(base_ + 1 + n, 0);
    bd::Client c(d->id);
    auto t = c.AsyncCreate(clio::run::PoolQuery::Dynamic(), d->path, d->id,
                           bd::BdevType::kFile, kMemberSize);
    t.Wait();
    return t->GetReturnCode() == 0;
  }

  /**
   * Create the array on one data disk.
   * @param first the data disk
   * @param max_failures parity members the array may take
   * @return create rc
   */
  clio::run::u32 Create(const Disk &first, clio::run::u32 max_failures,
                        bool distributed = false) {
    first_ = first;
    max_failures_ = max_failures;
    distributed_ = distributed;
    safe_.SetPersistent(distributed);  // a real multi-node array is durable
    std::vector<sb::MemberBdevDesc> m;
    m.emplace_back(first.path, first.node, first.id);
    log_ = (StressDir() / (tag_ + "_" + std::to_string(ctp::SystemInfo::GetPid()) + ".alog"))
               .string();
    std::error_code ec;
    std::filesystem::remove(log_, ec);
    std::filesystem::remove(log_ + ".members", ec);
    auto t = safe_.AsyncCreate(clio::run::PoolQuery::Dynamic(), tag_,
                               safe_.pool_id_, max_failures, m, log_,
                               distributed);
    t.Wait();
    safe_.pool_id_ = t->new_pool_id_;
    return t->GetReturnCode();
  }

  /**
   * Stop the array (flush its logs, destroy the pool) and start it again
   * from its ORIGINAL configuration -- the one disk it was created with --
   * as a reboot would. Everything else must come back from its own logs.
   * @return the re-create rc
   */
  clio::run::u32 Restart() {
    auto fl = safe_.AsyncFlushAllocLog(clio::run::PoolQuery::Dynamic(), 0);
    fl.Wait();
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto d = admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(),
                                    safe_.pool_id_);
    d.Wait();
    if (d->GetReturnCode() != 0) return 100 + d->GetReturnCode();
    std::this_thread::sleep_for(200ms);
    std::vector<sb::MemberBdevDesc> m;
    m.emplace_back(first_.path, first_.node, first_.id);
    auto t = safe_.AsyncCreate(clio::run::PoolQuery::Dynamic(), tag_,
                               safe_.pool_id_, max_failures_, m, log_,
                               distributed_);
    t.Wait();
    return t->GetReturnCode();
  }

  /** @return AddBdev rc. */
  clio::run::u32 Add(const Disk &d, bool parity) {
    auto t = safe_.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), d.path,
                                d.node, d.id, parity ? 1u : 0u);
    t.Wait();
    return t->GetReturnCode();
  }

  /** Unplug a disk: the array marks it faulty. @return RemoveBdev rc. */
  clio::run::u32 Unplug(const Disk &d) {
    auto t = safe_.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(), d.id, 1);
    t.Wait();
    return t->GetReturnCode();
  }

  /** Rebuild an unplugged disk's contents onto a replacement. @return rc. */
  clio::run::u32 Recover(const Disk &old_disk, const Disk &fresh) {
    auto t = safe_.AsyncRecoverBdev(clio::run::PoolQuery::Dynamic(),
                                    old_disk.id, fresh.path, fresh.node,
                                    fresh.id);
    t.Wait();
    return t->GetReturnCode();
  }

  /** Bring every stripe's parity up to date. @return rc. */
  clio::run::u32 BuildParity() {
    auto t = safe_.AsyncBuildParity(clio::run::PoolQuery::Dynamic(), 0);
    t.Wait();
    return t->GetReturnCode();
  }

  /**
   * Allocate, write and record one logical block.
   * @param seed pattern seed
   * @return true if the array took it
   */
  bool WriteOne(clio::run::u32 seed);

  /**
   * Overwrite every recorded block in place with a new pattern (as the CTE
   * rewrites a page), recording the new seed.
   * @param seed_base first new seed
   * @return blocks the array refused
   */
  size_t OverwriteAll(clio::run::u32 seed_base);

  /**
   * Read back every recorded block.
   * @param what label for the log
   * @return blocks that failed to read or mismatched
   */
  size_t VerifyAll(const char *what);

  /** @return blocks recorded so far. */
  size_t Written() {
    std::lock_guard<std::mutex> g(mu_);
    return stored_.size();
  }

  sb::Client &safe() { return safe_; }

 private:
  /** One recorded logical write. */
  struct Stored {
    std::vector<bd::Block> blocks;
    clio::run::u32 seed;
  };
  /** @return true if the blocks read back as the seed's pattern. */
  bool ReadMatches(const Stored &s);

  std::string tag_;
  clio::run::u32 base_;
  clio::run::u32 next_disk_ = 0;
  std::string log_;
  Disk first_;
  clio::run::u32 max_failures_ = 1;
  bool distributed_ = false;
  sb::Client safe_;
  std::mutex mu_;
  std::vector<Stored> stored_;
};

inline bool Array::WriteOne(clio::run::u32 seed) {
  auto a = safe_.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), kIoLen);
  a.Wait();
  if (a->GetReturnCode() != 0 || a->blocks_.empty()) return false;
  Stored s;
  s.seed = seed;
  clio::run::priv::vector<bd::Block> wb(CTP_MALLOC);
  for (size_t i = 0; i < a->blocks_.size(); ++i) {
    s.blocks.push_back(a->blocks_[i]);
    wb.push_back(a->blocks_[i]);
  }
  const auto pat = Pattern(seed);
  auto buf = CLIO_IPC->AllocateBuffer(kIoLen);
  if (buf.IsNull()) return false;
  memcpy(buf.ptr_, pat.data(), kIoLen);
  auto w = safe_.AsyncWrite(clio::run::PoolQuery::Dynamic(), wb,
                            buf.shm_.template Cast<void>(), kIoLen);
  w.Wait();
  const bool ok = w->GetReturnCode() == 0 && w->bytes_written_ == kIoLen;
  CLIO_IPC->FreeBuffer(buf);
  if (!ok) return false;
  std::lock_guard<std::mutex> g(mu_);
  stored_.push_back(std::move(s));
  return true;
}

inline size_t Array::OverwriteAll(clio::run::u32 seed_base) {
  std::lock_guard<std::mutex> g(mu_);
  size_t refused = 0;
  for (size_t i = 0; i < stored_.size(); ++i) {
    Stored &st = stored_[i];
    const clio::run::u32 seed = seed_base + static_cast<clio::run::u32>(i);
    clio::run::priv::vector<bd::Block> wb(CTP_MALLOC);
    for (const auto &b : st.blocks) wb.push_back(b);
    const auto pat = Pattern(seed);
    auto buf = CLIO_IPC->AllocateBuffer(kIoLen);
    if (buf.IsNull()) return stored_.size();
    memcpy(buf.ptr_, pat.data(), kIoLen);
    auto w = safe_.AsyncWrite(clio::run::PoolQuery::Dynamic(), wb,
                              buf.shm_.template Cast<void>(), kIoLen);
    w.Wait();
    if (w->GetReturnCode() == 0 && w->bytes_written_ == kIoLen) {
      st.seed = seed;
    } else {
      ++refused;
    }
    CLIO_IPC->FreeBuffer(buf);
  }
  return refused;
}

inline bool Array::ReadMatches(const Stored &s) {
  clio::run::priv::vector<bd::Block> rb(CTP_MALLOC);
  for (const auto &b : s.blocks) rb.push_back(b);
  auto buf = CLIO_IPC->AllocateBuffer(kIoLen);
  if (buf.IsNull()) return false;
  memset(buf.ptr_, 0, kIoLen);
  auto r = safe_.AsyncRead(clio::run::PoolQuery::Dynamic(), rb,
                           buf.shm_.template Cast<void>(), kIoLen);
  r.Wait();
  bool ok = r->GetReturnCode() == 0 && r->bytes_read_ == kIoLen;
  if (!ok) {
    HLOG(kWarning, "safe_bdev stress: seed {} read failed rc={} bytes={}",
         s.seed, r->GetReturnCode(), r->bytes_read_);
  } else {
    // A read that "succeeds" with the wrong bytes is the worst outcome:
    // name the first mismatching chunk.
    const auto pat = Pattern(s.seed);
    for (size_t c = 0; c < kIoLen / 65536 && ok; ++c) {
      if (memcmp(buf.ptr_ + c * 65536, pat.data() + c * 65536, 65536) != 0) {
        HLOG(kWarning, "safe_bdev stress: seed {} WRONG BYTES in chunk {} "
             "(block offset {})", s.seed, c,
             c < s.blocks.size() ? s.blocks[c].offset_ : 0);
        ok = false;
      }
    }
  }
  CLIO_IPC->FreeBuffer(buf);
  return ok;
}

inline size_t Array::VerifyAll(const char *what) {
  std::vector<Stored> snap;
  {
    std::lock_guard<std::mutex> g(mu_);
    snap = stored_;
  }
  size_t bad = 0;
  for (const auto &s : snap) {
    if (!ReadMatches(s)) ++bad;
  }
  HLOG(kInfo, "safe_bdev stress [{}]: {} of {} blocks bad", what, bad,
       snap.size());
  return bad;
}

/** Write `n` blocks, requiring every one to land. */
inline void WriteMany(Array &a, clio::run::u32 *seed, int n) {
  for (int i = 0; i < n; ++i) REQUIRE(a.WriteOne((*seed)++));
}


}  // namespace sbstress

#endif  // CLIO_SAFE_BDEV_TEST_SAFE_BDEV_STRESS_UTIL_H_

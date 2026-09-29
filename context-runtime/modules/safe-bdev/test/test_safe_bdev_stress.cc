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
 * safe_bdev stress: an array that starts on ONE file bdev and is grown to
 * two and then four data members (file bdevs standing in for disks of one
 * type), with members unplugged -- singly and two at once, data and parity
 * -- and recovered onto fresh bdevs. Every byte ever written is verified
 * after every membership change. A second case keeps I/O running on several
 * threads while the membership changes underneath it.
 *
 * Backing files go under $CLIO_SAFE_STRESS_DIR (default: the temp dir).
 */

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>
#include <clio_runtime/admin/admin_client.h>

using namespace std::chrono_literals;

namespace {

namespace sb = clio::run::safe_bdev;
namespace bd = clio::run::bdev;

/** Member size: holds the superblock plus ~250 64 KiB chunk rows. */
constexpr clio::run::u64 kMemberSize = 16ull << 20;
/** One logical write: 4 chunks, spread round-robin over the data members. */
constexpr clio::run::u64 kIoLen = 4 * 65536;
/** Blocks the live-churn case keeps (4 data disks x ~250 slots / 4 chunks,
 *  well under full so allocation failures mean a bug, not a full array). */
constexpr size_t kLiveCap = 150;

bool g_initialized = false;

/** Start the embedded runtime once per process. */
void EnsureInit() {
  if (g_initialized) return;
  g_initialized = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (g_initialized) {
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
    std::this_thread::sleep_for(500ms);
  }
}

/** @return directory for member backing files. */
std::filesystem::path StressDir() {
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
std::vector<ctp::u8> Pattern(clio::run::u32 seed) {
  std::vector<ctp::u8> v(kIoLen);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<ctp::u8>((seed * 131u + i * 7u + (i >> 9)) & 0xFF);
  }
  return v;
}

/** A disk: a file bdev pool with its backing path. */
struct Disk {
  std::string path;
  clio::run::PoolId id;
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
    d->path = (StressDir() / (tag_ + "_" + std::to_string(getpid()) + "_d" +
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
  clio::run::u32 Create(const Disk &first, clio::run::u32 max_failures) {
    first_ = first;
    max_failures_ = max_failures;
    std::vector<sb::MemberBdevDesc> m;
    m.emplace_back(first.path, 0, first.id);
    log_ = (StressDir() / (tag_ + "_" + std::to_string(getpid()) + ".alog"))
               .string();
    std::error_code ec;
    std::filesystem::remove(log_, ec);
    std::filesystem::remove(log_ + ".members", ec);
    auto t = safe_.AsyncCreate(clio::run::PoolQuery::Dynamic(), tag_,
                               safe_.pool_id_, max_failures, m, log_);
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
    m.emplace_back(first_.path, 0, first_.id);
    auto t = safe_.AsyncCreate(clio::run::PoolQuery::Dynamic(), tag_,
                               safe_.pool_id_, max_failures_, m, log_);
    t.Wait();
    return t->GetReturnCode();
  }

  /** @return AddBdev rc. */
  clio::run::u32 Add(const Disk &d, bool parity) {
    auto t = safe_.AsyncAddBdev(clio::run::PoolQuery::Dynamic(), d.path, 0,
                                d.id, parity ? 1u : 0u);
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
                                    old_disk.id, fresh.path, 0, fresh.id);
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
  sb::Client safe_;
  std::mutex mu_;
  std::vector<Stored> stored_;
};

bool Array::WriteOne(clio::run::u32 seed) {
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

size_t Array::OverwriteAll(clio::run::u32 seed_base) {
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

bool Array::ReadMatches(const Stored &s) {
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

size_t Array::VerifyAll(const char *what) {
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
void WriteMany(Array &a, clio::run::u32 *seed, int n) {
  for (int i = 0; i < n; ++i) REQUIRE(a.WriteOne((*seed)++));
}

}  // namespace

TEST_CASE("safe_bdev_stress_grow_unplug_recover", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_grow", 90000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d0, d1, d2, d3, p0, p1;

  // One disk, then a mirror (parity) of it.
  REQUIRE(a.NewDisk(&d0));
  REQUIRE(a.Create(d0, /*max_failures=*/2) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("1 data + 1 parity") == 0);

  // Upgrade to two data disks.
  REQUIRE(a.NewDisk(&d1));
  REQUIRE(a.Add(d1, false) == 0);
  WriteMany(a, &seed, 30);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("2 data") == 0);

  // Unplug the original disk; everything must still read (degraded), then
  // rebuild it onto a new disk.
  REQUIRE(a.Unplug(d0) == 0);
  REQUIRE(a.VerifyAll("2 data, d0 unplugged") == 0);
  Disk d0b;
  REQUIRE(a.NewDisk(&d0b));
  REQUIRE(a.Recover(d0, d0b) == 0);
  REQUIRE(a.VerifyAll("d0 recovered") == 0);

  // Upgrade to four data disks and a second parity disk.
  REQUIRE(a.NewDisk(&d2));
  REQUIRE(a.Add(d2, false) == 0);
  REQUIRE(a.NewDisk(&d3));
  REQUIRE(a.Add(d3, false) == 0);
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("4 data + 2 parity") == 0);

  // Two disks unplugged at once (the array's full failure budget).
  REQUIRE(a.Unplug(d1) == 0);
  REQUIRE(a.Unplug(d3) == 0);
  REQUIRE(a.VerifyAll("d1 + d3 unplugged") == 0);
  Disk d1b, d3b;
  REQUIRE(a.NewDisk(&d1b));
  REQUIRE(a.NewDisk(&d3b));
  REQUIRE(a.Recover(d1, d1b) == 0);
  REQUIRE(a.Recover(d3, d3b) == 0);
  REQUIRE(a.VerifyAll("d1 + d3 recovered") == 0);

  // A parity disk fails, and a data disk with it.
  REQUIRE(a.Unplug(p0) == 0);
  REQUIRE(a.Unplug(d2) == 0);
  REQUIRE(a.VerifyAll("p0 + d2 unplugged") == 0);
  Disk p0b, d2b;
  REQUIRE(a.NewDisk(&p0b));
  REQUIRE(a.NewDisk(&d2b));
  REQUIRE(a.Recover(d2, d2b) == 0);
  REQUIRE(a.Recover(p0, p0b) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("final") == 0);
}

/**
 * Lose one data disk and the FIRST parity disk, so every degraded read
 * depends on the second parity alone.
 * @param a array with >= 2 data disks and 2 parity disks, parity current
 * @param data a data disk to unplug
 * @param p0 the first parity disk
 * @return blocks that read wrong or failed
 */
size_t VerifyOnSecondParity(Array &a, const Disk &data, const Disk &p0) {
  REQUIRE(a.Unplug(data) == 0);
  REQUIRE(a.Unplug(p0) == 0);
  return a.VerifyAll("data + p0 unplugged (second parity only)");
}

TEST_CASE("safe_bdev_stress_second_parity_static", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_p2s",
          92000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(VerifyOnSecondParity(a, d[1], p0) == 0);
}

TEST_CASE("safe_bdev_stress_second_parity_grown", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_p2g",
          93000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(VerifyOnSecondParity(a, d[1], p0) == 0);
}

TEST_CASE("safe_bdev_stress_two_data_lost", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_2d",
          94000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.Unplug(d[1]) == 0);
  REQUIRE(a.Unplug(d[3]) == 0);
  REQUIRE(a.VerifyAll("two data disks lost") == 0);
}

TEST_CASE("safe_bdev_stress_restart_after_growth", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_rst",
          96000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0, p1;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 2) == 0);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  WriteMany(a, &seed, 20);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p1));
  REQUIRE(a.Add(p1, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  // One disk replaced, another left unplugged when the array stops.
  REQUIRE(a.Unplug(d[1]) == 0);
  Disk d1b;
  REQUIRE(a.NewDisk(&d1b));
  REQUIRE(a.Recover(d[1], d1b) == 0);
  REQUIRE(a.Unplug(d[2]) == 0);
  REQUIRE(a.VerifyAll("before restart") == 0);

  REQUIRE(a.Restart() == 0);
  REQUIRE(a.VerifyAll("after restart (d2 still out)") == 0);
  Disk d2b;
  REQUIRE(a.NewDisk(&d2b));
  REQUIRE(a.Recover(d[2], d2b) == 0);
  WriteMany(a, &seed, 20);
  REQUIRE(a.BuildParity() == 0);
  REQUIRE(a.VerifyAll("after restart + recovery") == 0);
}

TEST_CASE("safe_bdev_stress_overwrite_while_degraded", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_ovw",
          97000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  clio::run::u32 seed = 1;
  Disk d[4], p0;
  REQUIRE(a.NewDisk(&d[0]));
  REQUIRE(a.Create(d[0], 1) == 0);
  for (int i = 1; i < 4; ++i) {
    REQUIRE(a.NewDisk(&d[i]));
    REQUIRE(a.Add(d[i], false) == 0);
  }
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);
  WriteMany(a, &seed, 40);
  REQUIRE(a.BuildParity() == 0);
  // A disk dies; the pages on it are rewritten while it is gone. The new
  // bytes must live in the parity (nothing else holds them).
  REQUIRE(a.Unplug(d[1]) == 0);
  REQUIRE(a.OverwriteAll(5000) == 0);
  REQUIRE(a.VerifyAll("overwritten while d1 down") == 0);
  Disk d1b;
  REQUIRE(a.NewDisk(&d1b));
  REQUIRE(a.Recover(d[1], d1b) == 0);
  REQUIRE(a.VerifyAll("d1 recovered after overwrites") == 0);
}

TEST_CASE("safe_bdev_stress_io_during_membership", "[safe_bdev][stress]") {
  EnsureInit();
  REQUIRE(g_initialized);
  Array a("sbs_live",
          91000u + static_cast<clio::run::u32>(getpid() & 0xFFF) * 64);
  Disk d0, p0;
  REQUIRE(a.NewDisk(&d0));
  REQUIRE(a.Create(d0, 1) == 0);
  REQUIRE(a.NewDisk(&p0));
  REQUIRE(a.Add(p0, true) == 0);

  // Writers keep adding blocks; readers keep checking what is recorded.
  std::atomic<bool> stop{false};
  std::atomic<clio::run::u32> seed{1000};
  std::atomic<size_t> write_fail{0};
  std::vector<std::thread> io;
  for (int t = 0; t < 4; ++t) {
    io.emplace_back([&] {
      while (!stop.load()) {
        if (a.Written() >= kLiveCap) {
          std::this_thread::sleep_for(5ms);
          continue;
        }
        if (!a.WriteOne(seed.fetch_add(1))) {
          ++write_fail;  // full until the next disk arrives: back off
          std::this_thread::sleep_for(20ms);
        }
      }
    });
  }
  // Membership churn under load: grow 1 -> 2 -> 4 data disks, with parity
  // brought current before each unplug (the array's recovery precondition).
  std::vector<Disk> data{d0};
  for (int k = 1; k < 4; ++k) {
    std::this_thread::sleep_for(300ms);
    Disk d;
    REQUIRE(a.NewDisk(&d));
    REQUIRE(a.Add(d, false) == 0);
    data.push_back(d);
  }
  for (int round = 0; round < 3; ++round) {
    std::this_thread::sleep_for(300ms);
    stop.store(true);  // quiesce writers so every slot gets parity
    for (auto &th : io) th.join();
    io.clear();
    REQUIRE(a.BuildParity() == 0);
    const Disk victim = data[(round + 1) % data.size()];
    REQUIRE(a.Unplug(victim) == 0);
    stop.store(false);
    for (int t = 0; t < 4; ++t) {
      io.emplace_back([&] {
        while (!stop.load()) {
          if (a.Written() < kLiveCap && !a.WriteOne(seed.fetch_add(1))) {
            ++write_fail;
          }
          std::this_thread::sleep_for(2ms);
        }
      });
    }
    Disk fresh;
    REQUIRE(a.NewDisk(&fresh));
    REQUIRE(a.Recover(victim, fresh) == 0);
    data[(round + 1) % data.size()] = fresh;
  }
  stop.store(true);
  for (auto &th : io) th.join();
  REQUIRE(a.BuildParity() == 0);
  HLOG(kInfo, "safe_bdev stress live: {} writes recorded, {} refused",
       a.Written(), write_fail.load());
  REQUIRE(a.VerifyAll("after live churn") == 0);
}

SIMPLE_TEST_MAIN()

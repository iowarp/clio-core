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

#include <clio_runtime/safe_bdev/safe_bdev_runtime.h>

#include <clio_ctp/serialize/msgpack_wrapper.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // ::fsync, ::fileno (member-log durability)
#endif

namespace clio::run::safe_bdev {

//===========================================================================
// Helpers
//===========================================================================

clio::run::TaskStat Runtime::GetTaskStats(const clio::run::Task *task) const {
  if (task == nullptr) {
    return clio::run::TaskStat();
  }
  switch (task->method_) {
    case Method::kWrite: {
      const auto *wt = static_cast<const WriteTask *>(task);
      clio::run::TaskStat stat;
      stat.io_size_ = wt->length_;
      const size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0F;
      return stat;
    }
    case Method::kRead: {
      const auto *rt = static_cast<const ReadTask *>(task);
      clio::run::TaskStat stat;
      stat.io_size_ = rt->length_;
      const size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0F;
      return stat;
    }
    default:
      return clio::run::TaskStat();
  }
}

clio::run::priv::vector<clio::run::bdev::Block> Runtime::MemberBlocks(
    clio::run::u64 offset, clio::run::u64 len) const {
  clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
  blocks.push_back(clio::run::bdev::Block(offset, len, 0));
  return blocks;
}

//===========================================================================
// Segment I/O helpers (run inside task fibers; co_await member bdev I/O)
//===========================================================================

clio::run::TaskResume Runtime::WriteDataSegment(size_t d, clio::run::u64 offset,
                                          const uint8_t *src, clio::run::u64 len,
                                          bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
  if (buf.IsNull()) {
    CLIO_CO_RETURN;
  }
  std::memcpy(buf.ptr_, src, len);
  auto fut = data_clients_[d].AsyncWrite(
      MemberQuery(), MemberBlocks(offset, len),
      buf.shm_.template Cast<void>(), len);
  CLIO_CO_AWAIT(fut);
  ok = (fut->return_code_ == 0) && (fut->bytes_written_ == len);
  MaybeFaultData(d, fut->io_error_);
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadDataSegment(size_t d, clio::run::u64 offset,
                                         uint8_t *dst, clio::run::u64 len,
                                         bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
  if (buf.IsNull()) {
    CLIO_CO_RETURN;
  }
  auto fut = data_clients_[d].AsyncRead(
      MemberQuery(), MemberBlocks(offset, len),
      buf.shm_.template Cast<void>(), len);
  CLIO_CO_AWAIT(fut);
  if (fut->return_code_ == 0 && fut->bytes_read_ == len) {
    std::memcpy(dst, buf.ptr_, len);
    ok = true;
  }
  MaybeFaultData(d, fut->io_error_);
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

//===========================================================================
// Member superblock helpers
//===========================================================================

clio::run::TaskResume Runtime::WriteSuperblock(bool is_parity, size_t idx,
                                         bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kSuperblockSize);
  if (buf.IsNull()) {
    CLIO_CO_RETURN;
  }
  // Serialize the array identity into a zero-padded superblock buffer.
  std::memset(buf.ptr_, 0, kSuperblockSize);
  const MemberSlot &m =
      is_parity ? parity_members_[idx] : data_members_[idx];
  MemberSuperblock sb;
  sb.magic = kMemberSuperblockMagic;
  sb.format_version = kMemberSuperblockVersion;
  sb.flags = 0;
  sb.array_major = static_cast<uint64_t>(pool_id_.major_);
  sb.array_minor = static_cast<uint64_t>(pool_id_.minor_);
  sb.member_slot = static_cast<uint32_t>(idx);
  sb.role = static_cast<uint32_t>(m.role_);
  sb.index = static_cast<uint32_t>(m.index_);
  sb.max_failures = max_failures_;
  sb.shard_len = kChunkLen;
  sb.epoch = 0;
  sb.checksum = sb.ComputeChecksum();
  std::memcpy(buf.ptr_, &sb, sizeof(sb));

  // The superblock lives at absolute offset 0 on the member.
  auto &client = is_parity ? parity_clients_[idx] : data_clients_[idx];
  auto fut = client.AsyncWrite(MemberQuery(),
                               MemberBlocks(0, kSuperblockSize),
                               buf.shm_.template Cast<void>(), kSuperblockSize);
  CLIO_CO_AWAIT(fut);
  ok = (fut->return_code_ == 0) && (fut->bytes_written_ == kSuperblockSize);
  if (is_parity) {
    MaybeFaultParity(idx, fut->io_error_);
  } else {
    MaybeFaultData(idx, fut->io_error_);
  }
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadSuperblock(bool is_parity, size_t idx,
                                        MemberSuperblock &sb, bool &present,
                                        bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  present = false;
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kSuperblockSize);
  if (buf.IsNull()) {
    CLIO_CO_RETURN;
  }
  std::memset(buf.ptr_, 0, kSuperblockSize);
  auto &client = is_parity ? parity_clients_[idx] : data_clients_[idx];
  auto fut = client.AsyncRead(MemberQuery(), MemberBlocks(0, kSuperblockSize),
                              buf.shm_.template Cast<void>(), kSuperblockSize);
  CLIO_CO_AWAIT(fut);
  if (fut->return_code_ == 0 && fut->bytes_read_ == kSuperblockSize) {
    ok = true;
    std::memcpy(&sb, buf.ptr_, sizeof(sb));
    present = sb.Validate();
  }
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

//===========================================================================
// Stripe reconstruction (decode a stripe's k_s data chunks from survivors)
//===========================================================================

clio::run::TaskResume Runtime::ReconstructStripe(
    clio::run::u64 s, const std::vector<int> &stripe, int exclude_member,
    std::vector<std::vector<uint8_t>> &out, bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  const int k_s = static_cast<int>(stripe.size());
  if (k_s <= 0) {
    CLIO_CO_RETURN;
  }
  const clio::run::u64 off = SlotPhysOffset(s);

  // Gather up to k_s active survivors among the stripe's k_s data + m parity
  // members' chunks at this slot, excluding `exclude_member` and any non-active
  // member. A data survivor's RS shard index is its POSITION in `stripe`; a
  // parity survivor's is k_s + j.
  std::vector<int> survivor_index;
  std::vector<std::vector<uint8_t>> survivor_buf;

  for (int pos = 0; pos < k_s; ++pos) {
    const int d = stripe[static_cast<size_t>(pos)];
    if (d == exclude_member) {
      continue;
    }
    if (data_members_[static_cast<size_t>(d)].state_ != ec::EcState::kActive) {
      continue;
    }
    std::vector<uint8_t> buf(kChunkLen, 0);
    bool rd_ok = false;
    CLIO_CO_AWAIT(ReadDataSegment(static_cast<size_t>(d), off, buf.data(),
                                  kChunkLen, rd_ok));
    if (!rd_ok) {
      continue;
    }
    survivor_index.push_back(pos);
    survivor_buf.push_back(std::move(buf));
    if (static_cast<int>(survivor_index.size()) == k_s) {
      break;
    }
  }

  for (int j = 0; j < static_cast<int>(parity_level_) &&
                  static_cast<int>(survivor_index.size()) < k_s;
       ++j) {
    if (parity_members_[static_cast<size_t>(j)].state_ !=
        ec::EcState::kActive) {
      continue;
    }
    std::vector<uint8_t> buf(kChunkLen, 0);
    bool rd_ok = false;
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kChunkLen);
    if (rbuf.IsNull()) {
      CLIO_CO_RETURN;
    }
    auto fut = parity_clients_[static_cast<size_t>(j)].AsyncRead(
        MemberQuery(), MemberBlocks(off, kChunkLen),
        rbuf.shm_.template Cast<void>(), kChunkLen);
    CLIO_CO_AWAIT(fut);
    rd_ok = (fut->return_code_ == 0) && (fut->bytes_read_ == kChunkLen);
    if (rd_ok) {
      std::memcpy(buf.data(), rbuf.ptr_, kChunkLen);
    }
    MaybeFaultParity(static_cast<size_t>(j), fut->io_error_);
    ipc->FreeBuffer(rbuf);
    if (!rd_ok) {
      continue;
    }
    survivor_index.push_back(k_s + j);
    survivor_buf.push_back(std::move(buf));
  }

  if (static_cast<int>(survivor_index.size()) < k_s) {
    CLIO_CO_RETURN;  // Too many failures to reconstruct.
  }
  std::vector<const uint8_t *> ptrs(survivor_buf.size());
  for (size_t i = 0; i < survivor_buf.size(); ++i) {
    ptrs[i] = survivor_buf[i].data();
  }
  ec::ReedSolomon *codec = GetCodec(k_s);
  ok = codec->DecodeData(survivor_index, ptrs, kChunkLen, &out);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

//===========================================================================
// Member manifest WAL (membership + recovery state persistence)
//===========================================================================

// The member manifest is a small WRITE-AHEAD LOG (mirrors the bdev
// AllocatorLog). It carries NO msgpack -- a hand-rolled, append-friendly binary
// stream of variable-length member records (native-endian u32; same-host):
//   [role][index][pool_major][pool_minor][node_id][state][recovering]
//   [name_len][name bytes...]  (repeated)
// A membership change APPENDS a snapshot of every current member (cheap
// fopen("ab") -- no rename); replay keeps the LAST record per (role,index)
// slot; and CompactMemberManifest() periodically rewrites the log to one record
// per member (temp file + atomic std::filesystem::rename, which -- unlike C's
// std::rename -- replaces an existing file on every platform, including
// Windows). A crash mid-append leaves a partial trailing record that replay
// simply stops at.

void Runtime::WriteMemberRecord(std::FILE *f, const MemberSlot &m,
                                clio::run::u32 role) const {
  const auto put_u32 = [&](clio::run::u32 v) {
    std::fwrite(&v, sizeof(v), 1, f);
  };
  put_u32(role);
  put_u32(static_cast<clio::run::u32>(m.index_));
  put_u32(m.pool_id_.major_);
  put_u32(m.pool_id_.minor_);
  put_u32(m.node_id_);
  put_u32(static_cast<clio::run::u32>(m.state_));
  put_u32(m.recovering_ ? 1u : 0u);
  put_u32(static_cast<clio::run::u32>(m.pool_name_.size()));
  if (!m.pool_name_.empty()) {
    std::fwrite(m.pool_name_.data(), 1, m.pool_name_.size(), f);
  }
}

void Runtime::PersistMemberManifest() {
  if (members_manifest_path_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lk(member_log_mu_);
  std::FILE *f = std::fopen(members_manifest_path_.c_str(), "ab");
  if (f == nullptr) {
    HLOG(kWarning, "safe_bdev: cannot open member log '{}' for append",
         members_manifest_path_);
    return;
  }
  for (const auto &m : data_members_) WriteMemberRecord(f, m, 0);
  for (const auto &m : parity_members_) WriteMemberRecord(f, m, 1);
  std::fflush(f);
#ifndef _WIN32
  int fd = ::fileno(f);
  if (fd >= 0) {
    ::fsync(fd);
  }
#endif
  std::fclose(f);
  member_log_records_ += data_members_.size() + parity_members_.size();
}

void Runtime::CompactMemberManifest() {
  if (members_manifest_path_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lk(member_log_mu_);
  const std::string tmp = members_manifest_path_ + ".compact.tmp";
  std::FILE *f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) {
    return;
  }
  clio::run::u64 written = 0;
  for (const auto &m : data_members_) {
    WriteMemberRecord(f, m, 0);
    ++written;
  }
  for (const auto &m : parity_members_) {
    WriteMemberRecord(f, m, 1);
    ++written;
  }
  std::fflush(f);
#ifndef _WIN32
  int fd = ::fileno(f);
  if (fd >= 0) {
    ::fsync(fd);
  }
#endif
  std::fclose(f);
  std::error_code ec;
  std::filesystem::rename(tmp, members_manifest_path_, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    HLOG(kWarning, "safe_bdev: member log compaction rename failed: {}",
         ec.message());
    return;
  }
  member_log_records_ = written;
}

bool Runtime::MemberManifestNeedsCompaction() const {
  clio::run::u64 recs = 0;
  {
    std::lock_guard<std::mutex> lk(member_log_mu_);
    recs = member_log_records_;
  }
  const clio::run::u64 live = data_members_.size() + parity_members_.size();
  return recs > std::max<clio::run::u64>(64, live * 8);
}

bool Runtime::LoadMemberManifest(std::vector<MemberManifestEntry> &out) const {
  out.clear();
  if (members_manifest_path_.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lk(member_log_mu_);

  // Roll forward a compaction that crashed mid-replace (log absent, temp
  // present -> the temp is complete; finish the rename). A rename onto an absent
  // target is a plain create-rename, atomic on every platform.
  namespace fs = std::filesystem;
  const std::string tmp = members_manifest_path_ + ".compact.tmp";
  std::error_code fec;
  if (!fs::exists(members_manifest_path_, fec) && fs::exists(tmp, fec)) {
    fs::rename(tmp, members_manifest_path_, fec);
  }

  std::ifstream ifs(members_manifest_path_, std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }
  const auto get_u32 = [&](clio::run::u32 &v) -> bool {
    return static_cast<bool>(ifs.read(reinterpret_cast<char *>(&v), sizeof(v)));
  };
  std::map<clio::run::u64, MemberManifestEntry> latest;
  clio::run::u64 total = 0;
  while (true) {
    MemberManifestEntry e;
    clio::run::u32 name_len = 0;
    if (!get_u32(e.role_)) {
      break;  // clean EOF
    }
    if (e.role_ > 1) {
      break;  // corrupt / foreign -> stop at the valid prefix
    }
    if (!get_u32(e.index_) || !get_u32(e.pool_major_) ||
        !get_u32(e.pool_minor_) || !get_u32(e.node_id_) || !get_u32(e.state_) ||
        !get_u32(e.recovering_) || !get_u32(name_len)) {
      break;  // partial trailing record (crash mid-append) -> stop
    }
    if (name_len > (1u << 20)) {  // sanity bound
      break;
    }
    e.pool_name_.resize(name_len);
    if (name_len > 0 &&
        !ifs.read(&e.pool_name_[0], static_cast<std::streamsize>(name_len))) {
      break;
    }
    const clio::run::u64 key =
        (static_cast<clio::run::u64>(e.role_) << 32) | e.index_;
    latest[key] = std::move(e);
    ++total;
  }
  member_log_records_ = total;
  for (auto &kv : latest) {
    out.push_back(std::move(kv.second));
  }
  return !out.empty();
}

//===========================================================================
// Method handlers
//===========================================================================

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN

  CreateParams params = task->GetParams();

  max_failures_ = params.max_failures_;
  parity_level_ = 0;
  rr_cursor_ = 0;
  data_members_.clear();
  parity_members_.clear();
  data_clients_.clear();
  parity_clients_.clear();
  data_alloc_.clear();
  rs_cache_.clear();
  reattached_members_ = 0;
  {
    std::lock_guard<std::mutex> g(slot_mu_);
    dirty_slots_.clear();
    written_slots_.clear();
  }

  // DATA members come from the config member list and are re-attached by
  // SUPERBLOCK identity (pool ids may change across restart). The durable
  // manifest augments this with the runtime-added PARITY members and per-member
  // RECOVERY state so an interrupted recovery resumes. Fresh create has none.
  members_manifest_path_ =
      params.alloc_log_path_.empty() ? std::string()
                                     : (params.alloc_log_path_ + ".members");
  std::vector<MemberManifestEntry> manifest;
  const bool have_manifest = LoadMemberManifest(manifest);

  std::vector<MemberManifestEntry> eff_parity;  // parity members from manifest
  std::vector<bool> data_recovering(params.members_.size(), false);
  if (have_manifest) {
    for (const auto &e : manifest) {
      if (e.role_ == 1) {
        eff_parity.push_back(e);
      } else if (e.recovering_ != 0 && e.index_ < data_recovering.size()) {
        data_recovering[e.index_] = true;
      }
    }
    std::sort(eff_parity.begin(), eff_parity.end(),
              [](const MemberManifestEntry &a, const MemberManifestEntry &b) {
                return a.index_ < b.index_;
              });
    HLOG(kInfo,
         "safe_bdev Create: manifest '{}' augments membership with {} parity "
         "member(s); {} data column(s) mid-recovery",
         members_manifest_path_, eff_parity.size(),
         std::count(data_recovering.begin(), data_recovering.end(), true));
  }

  const int k0 = static_cast<int>(params.members_.size());
  if (k0 <= 0) {
    HLOG(kError, "safe_bdev Create: no member bdevs supplied");
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  // Open the persistent allocator-state log (WAL). Empty path => disabled. On
  // recover, its single group (kAllocGroup) exists and live(kAllocGroup) yields
  // every live banded offset to rebuild the per-member slot allocators.
  if (!alloc_log_.Open(params.alloc_log_path_, /*recover=*/true)) {
    HLOG(kWarning,
         "safe_bdev Create: failed to open alloc log at '{}', logging disabled",
         params.alloc_log_path_);
  }
  const bool recovered =
      alloc_log_.enabled() && !alloc_log_.groups().empty();

  // Seat each data member as a DATA column, sizing its slot allocator from the
  // member's own capacity (per-member -> full aggregate capacity). Classify the
  // superblock: blank -> FRESH (stamp identity); present+ours -> re-attach;
  // present+foreign -> REFUSE.
  int col = 0;
  for (const auto &desc : params.members_) {
    data_clients_.emplace_back(desc.pool_id_);
    auto stats = data_clients_.back().AsyncGetStats();
    CLIO_CO_AWAIT(stats);
    if (stats->return_code_ != 0) {
      HLOG(kError, "safe_bdev Create: GetStats failed for member '{}'",
           desc.pool_name_);
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    const clio::run::u64 remaining = stats->remaining_size_;
    const clio::run::u64 avail =
        (remaining > kSuperblockSize) ? (remaining - kSuperblockSize) : 0;
    const clio::run::u64 cap_slots = avail / kChunkLen;

    MemberSlot slot;
    slot.pool_id_ = desc.pool_id_;
    slot.pool_name_ = desc.pool_name_;
    slot.node_id_ = desc.node_id_;
    slot.role_ = ec::EcRole::kData;
    slot.state_ = ec::EcState::kActive;
    slot.index_ = col;
    data_members_.push_back(slot);
    MemberAlloc a;
    a.cap_slots_ = cap_slots;
    data_alloc_.push_back(std::move(a));

    MemberSuperblock sb;
    bool present = false;
    bool sb_ok = false;
    CLIO_CO_AWAIT(ReadSuperblock(/*is_parity=*/false, static_cast<size_t>(col),
                                 sb, present, sb_ok));
    if (!sb_ok) {
      HLOG(kError, "safe_bdev Create: superblock read failed for member '{}'",
           desc.pool_name_);
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    if (!present) {
      bool wr_ok = false;
      CLIO_CO_AWAIT(WriteSuperblock(/*is_parity=*/false,
                                    static_cast<size_t>(col), wr_ok));
      if (!wr_ok) {
        HLOG(kError,
             "safe_bdev Create: superblock write failed for fresh member '{}'",
             desc.pool_name_);
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }
      HLOG(kInfo,
           "safe_bdev Create: initialized fresh member '{}' (slot {}, {} slots)",
           desc.pool_name_, col, cap_slots);
    } else if (sb.array_major == static_cast<uint64_t>(pool_id_.major_) &&
               sb.array_minor == static_cast<uint64_t>(pool_id_.minor_)) {
      ++reattached_members_;
      HLOG(kInfo,
           "safe_bdev Create: re-attached member '{}' (slot {}) already owned "
           "by this array ({},{})",
           desc.pool_name_, col, pool_id_.major_, pool_id_.minor_);
    } else {
      HLOG(kError,
           "safe_bdev Create: REFUSING member '{}' — already initialized by a "
           "FOREIGN array ({},{}); this array is ({},{})",
           desc.pool_name_, sb.array_major, sb.array_minor, pool_id_.major_,
           pool_id_.minor_);
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    ++col;
  }

  // A data member restored from the manifest as mid-recovery stays non-active
  // (kFaulty + recovering) so degraded reads serve its data until
  // ResumeRecoveries() finishes the rebuild.
  for (size_t i = 0; i < data_members_.size(); ++i) {
    if (i < data_recovering.size() && data_recovering[i]) {
      data_members_[i].state_ = ec::EcState::kFaulty;
      data_members_[i].recovering_ = true;
    }
  }

  if (recovered) {
    // Rebuild every data member's slot allocator from the recovered live set:
    // decode each live banded offset -> (member d, slot s) -> live_. Then set
    // high_water = max(live)+1 and free = [0,high_water)\live (reclaims freed
    // gaps). written_slots_ = union of all live slots (dirty stays empty: parity
    // was persisted on the parity members before restart).
    const std::vector<clio::run::bdev::LiveBlock> &live =
        alloc_log_.live(kAllocGroup);
    for (const auto &b : live) {
      clio::run::u32 d = 0;
      clio::run::u64 s = 0, within = 0;
      Unband(b.offset, d, s, within);
      if (d < data_alloc_.size()) {
        data_alloc_[d].live_.insert(s);
      }
    }
    clio::run::u64 total_live = 0;
    for (auto &a : data_alloc_) {
      clio::run::u64 maxs = 0;
      bool any = false;
      for (clio::run::u64 s : a.live_) {
        any = true;
        if (s > maxs) maxs = s;
      }
      a.high_water_ = any ? (maxs + 1) : 0;
      a.free_.clear();
      for (clio::run::u64 s = 0; s < a.high_water_; ++s) {
        if (a.live_.count(s) == 0) {
          a.free_.push_back(s);
        }
      }
      total_live += a.live_.size();
    }
    {
      std::lock_guard<std::mutex> g(slot_mu_);
      for (const auto &a : data_alloc_) {
        for (clio::run::u64 s : a.live_) {
          written_slots_.insert(s);
        }
      }
    }
    HLOG(kInfo,
         "safe_bdev Create: RECOVERED slot allocators from alloc log '{}' "
         "({} live chunks across {} data members)",
         params.alloc_log_path_, total_live, data_alloc_.size());
  } else {
    // FRESH: register the single allocator group so a later restart recovers.
    alloc_log_.LogGroupOpen(kAllocGroup, /*k=*/0, /*first_row=*/0,
                            /*num_rows=*/0, /*logical_base=*/0);
    HLOG(kInfo,
         "safe_bdev Create: pool='{}', k0={}, M={} (dynamic view-group: "
         "per-chunk round-robin + dedicated parity)",
         task->pool_name_.str(), k0, max_failures_);
  }

  // Restore PARITY members from the manifest (empty on fresh create). Each is
  // probed by superblock: mid-recovery -> adopt+recovering; present+ours ->
  // adopt; otherwise SKIP (stale entry, will be re-added).
  for (const auto &pe : eff_parity) {
    MemberSlot slot;
    slot.pool_id_ = clio::run::PoolId(pe.pool_major_, pe.pool_minor_);
    slot.pool_name_ = pe.pool_name_;
    slot.node_id_ = pe.node_id_;
    slot.role_ = ec::EcRole::kParity;
    slot.index_ = static_cast<int>(pe.index_);
    const bool par_recovering = (pe.recovering_ != 0);

    parity_clients_.emplace_back(slot.pool_id_);
    parity_members_.push_back(slot);
    const size_t pj = parity_members_.size() - 1;
    MemberSuperblock sb;
    bool present = false;
    bool sb_ok = false;
    CLIO_CO_AWAIT(ReadSuperblock(/*is_parity=*/true, pj, sb, present, sb_ok));
    const bool ours = sb_ok && present &&
                      sb.array_major == static_cast<uint64_t>(pool_id_.major_) &&
                      sb.array_minor == static_cast<uint64_t>(pool_id_.minor_);
    if (par_recovering) {
      parity_members_[pj].state_ = ec::EcState::kFaulty;
      parity_members_[pj].recovering_ = true;
    } else if (ours) {
      parity_members_[pj].state_ = ec::EcState::kActive;
      parity_members_[pj].recovering_ = false;
    } else {
      parity_members_.pop_back();
      parity_clients_.pop_back();
      HLOG(kInfo,
           "safe_bdev Create: skipping stale manifest parity member "
           "(pool {}.{} not reachable/ours)",
           pe.pool_major_, pe.pool_minor_);
    }
  }
  parity_level_ = static_cast<clio::run::u32>(parity_members_.size());

  // NOTE: dirty_slots_ stays EMPTY on restart. Parity is durably persisted on
  // the parity members (they are real bdevs that survive the reboot), and the
  // recovered stripe membership matches what that parity was built over, so the
  // persisted parity is current -- re-dirtying it would only open a window where
  // degraded reads see an (unnecessarily) unprotected stripe until the async
  // builder catches up. A write whose parity had not yet been built before an
  // unclean shutdown is the same bounded gap as the pre-existing design.

  if (alloc_log_.enabled()) {
    client_.AsyncFlushAllocLog(MemberQuery(), kFlushAllocLogPeriodUs);
  }
  client_.AsyncBuildParity(MemberQuery(), /*max_batch=*/0,
                           /*period_us=*/kBuildParityPeriodUs);

  CLIO_CO_AWAIT(ResumeRecoveries());
  CompactMemberManifest();

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AllocateBlocks(
    clio::run::shared_ptr<AllocateBlocksTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Per-chunk round-robin over the non-full, active data members. Each kChunkLen
  // chunk of the request takes one whole SLOT on the next member; the returned
  // block bands (member,slot) into a logical offset (§ addressing). A multi-
  // chunk request therefore spreads across members and comes back as a block
  // list (one block per chunk). Allocation makes the slot LIVE (stripe
  // membership changes) so we mark it dirty -> BuildParity refreshes parity for
  // the widened stripe before any degraded read can use stale parity.
  const clio::run::u64 size = task->size_;
  task->blocks_.clear();
  if (size == 0) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (data_members_.empty()) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  const size_t nmembers = data_members_.size();
  clio::run::u64 remaining = size;
  while (remaining > 0) {
    // Advance the round-robin cursor to the next member that is active and has
    // a free slot. Give up after a full sweep (array full).
    size_t scanned = 0;
    int chosen = -1;
    while (scanned < nmembers) {
      const size_t d = rr_cursor_ % nmembers;
      rr_cursor_ = static_cast<clio::run::u32>((d + 1) % nmembers);
      ++scanned;
      if (data_members_[d].state_ == ec::EcState::kActive &&
          !data_alloc_[d].Full()) {
        chosen = static_cast<int>(d);
        break;
      }
    }
    if (chosen < 0) {
      HLOG(kError,
           "safe_bdev AllocateBlocks: no data member has a free slot ({} bytes "
           "unsatisfied of {})",
           remaining, size);
      task->blocks_.clear();
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    const size_t d = static_cast<size_t>(chosen);
    const clio::run::u64 s = data_alloc_[d].Take();
    const clio::run::u64 seg = std::min<clio::run::u64>(kChunkLen, remaining);
    const clio::run::u64 off = BandOffset(static_cast<clio::run::u32>(d), s);

    clio::run::bdev::Block block(off, seg, 0);
    task->blocks_.push_back(block);
    alloc_log_.LogAlloc(kAllocGroup, off, seg, 0);
    // Do NOT dirty the slot here. Parity tracks COMMITTED (written) data, not
    // reservations: an allocated-but-unwritten slot holds no meaningful bytes,
    // and dirtying it lets the async BuildParity encode parity over UNWRITTEN
    // data. Worse, such a stale build (started between alloc and the write) can
    // land AFTER the write's parity flush and overwrite the correct parity with
    // garbage -- a timing-dependent corruption. Write (and stripe-narrowing
    // Free) are what dirty a slot; a widening add-drive is covered because the
    // new member's WRITE dirties the stripes it joins.
    remaining -= seg;
  }

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FreeBlocks(clio::run::shared_ptr<FreeBlocksTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Decode each block to (member d, slot s) and release the slot on that member.
  // The slot leaves the stripe -> if the stripe still has members, mark it dirty
  // so BuildParity re-derives the narrower parity; if it is now empty, forget it.
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const clio::run::bdev::Block &b = task->blocks_[i];
    clio::run::u32 d = 0;
    clio::run::u64 s = 0, within = 0;
    Unband(b.offset_, d, s, within);
    if (d >= data_alloc_.size() || data_alloc_[d].live_.count(s) == 0) {
      HLOG(kWarning,
           "safe_bdev FreeBlocks: block at logical offset {} not live "
           "(member {}, slot {}); skipping",
           b.offset_, d, s);
      continue;
    }
    data_alloc_[d].Release(s);
    alloc_log_.LogFree(kAllocGroup, b.offset_, b.size_, 0);
    if (StripeMembers(s).empty()) {
      ForgetSlotIfEmpty(s);
    } else {
      MarkSlotDirty(s);
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Write(clio::run::shared_ptr<WriteTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (data_members_.empty() || task->blocks_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> data =
      ipc->ToFullPtr(task->data_).template Cast<char>();

  // Each block decodes to (member d, slot s0, within); a block is contiguous on
  // ONE member (banding keeps a member's consecutive slots contiguous), so it
  // writes in a single AsyncWrite to member d at kSuperblockSize + s0*kChunkLen
  // + within. All blocks' writes are dispatched in parallel, then awaited. Every
  // slot the write touches is marked dirty (parity deferred to BuildParity).
  clio::run::u64 buf_pos = 0;
  clio::run::u64 bytes_written = 0;
  std::vector<clio::run::Future<WriteTask>> futs;
  std::vector<ctp::ipc::FullPtr<char>> bufs;
  std::set<clio::run::u64> touched;
  bool dispatch_ok = true;

  for (size_t bi = 0; bi < task->blocks_.size() && dispatch_ok; ++bi) {
    const clio::run::u64 off = task->blocks_[bi].offset_;
    const clio::run::u64 len = task->blocks_[bi].size_;
    if (len == 0) {
      continue;
    }
    clio::run::u32 d = 0;
    clio::run::u64 s0 = 0, within = 0;
    Unband(off, d, s0, within);
    if (d >= data_members_.size()) {
      HLOG(kError, "safe_bdev Write: block offset {} decodes to member {} out "
           "of range", off, d);
      dispatch_ok = false;
      break;
    }
    const clio::run::u64 phys = SlotPhysOffset(s0) + within;

    if (data_members_[d].state_ == ec::EcState::kActive) {
      ctp::ipc::FullPtr<char> seg = ipc->AllocateBuffer(len);
      if (seg.IsNull()) {
        dispatch_ok = false;
        break;
      }
      std::memcpy(seg.ptr_, data.ptr_ + buf_pos, len);
      bufs.push_back(seg);
      futs.push_back(data_clients_[d].AsyncWrite(
          MemberQuery(), MemberBlocks(phys, len),
          seg.shm_.template Cast<void>(), len));
    }
    // Mark every slot this block spans dirty (contiguous slots on member d).
    const clio::run::u64 chunk_first = off / kChunkLen;
    const clio::run::u64 chunk_last = (off + len - 1) / kChunkLen;
    for (clio::run::u64 c = chunk_first; c <= chunk_last; ++c) {
      touched.insert(c % kSlotsPerMember);
    }
    buf_pos += len;
    bytes_written += len;
  }

  bool ok = dispatch_ok;
  for (auto &f : futs) {
    CLIO_CO_AWAIT(f);
    if (f->return_code_ != 0) {
      ok = false;
    }
  }
  for (auto &b : bufs) {
    ipc->FreeBuffer(b);
  }
  if (!ok) {
    task->bytes_written_ = 0;
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  for (clio::run::u64 s : touched) {
    MarkSlotDirty(s);
  }
  task->bytes_written_ = bytes_written;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Read(clio::run::shared_ptr<ReadTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (data_members_.empty() || task->blocks_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> data =
      ipc->ToFullPtr(task->data_).template Cast<char>();

  clio::run::u64 buf_pos = 0;
  clio::run::u64 bytes_read = 0;

  for (size_t bi = 0; bi < task->blocks_.size(); ++bi) {
    const clio::run::u64 off = task->blocks_[bi].offset_;
    const clio::run::u64 len = task->blocks_[bi].size_;
    if (len == 0) {
      continue;
    }
    clio::run::u32 d = 0;
    clio::run::u64 s0 = 0, within0 = 0;
    Unband(off, d, s0, within0);
    if (d >= data_members_.size()) {
      HLOG(kError, "safe_bdev Read: block offset {} decodes to member {} out of "
           "range", off, d);
      task->bytes_read_ = bytes_read;
      task->length_ = bytes_read;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    if (data_members_[d].state_ == ec::EcState::kActive) {
      // Healthy: one contiguous read from member d.
      const clio::run::u64 phys = SlotPhysOffset(s0) + within0;
      bool seg_ok = false;
      CLIO_CO_AWAIT(ReadDataSegment(
          static_cast<size_t>(d), phys,
          reinterpret_cast<uint8_t *>(data.ptr_) + buf_pos, len, seg_ok));
      if (!seg_ok) {
        task->bytes_read_ = bytes_read;
        task->length_ = bytes_read;
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }
      buf_pos += len;
    } else {
      // Degraded: member d is down -> reconstruct each chunk of this block from
      // its stripe's survivors + parity, chunk by chunk.
      const clio::run::u64 hi = off + len;
      clio::run::u64 cur = off;
      while (cur < hi) {
        clio::run::u32 dd = 0;
        clio::run::u64 s = 0, within = 0;
        Unband(cur, dd, s, within);
        const clio::run::u64 chunk_base = (cur / kChunkLen) * kChunkLen;
        const clio::run::u64 seg_end =
            std::min<clio::run::u64>(hi, chunk_base + kChunkLen);
        const clio::run::u64 seg_len = seg_end - cur;
        if (IsSlotDirty(s)) {
          HLOG(kError,
               "safe_bdev Read: slot {} is dirty (parity not built) and data "
               "member {} is down — cannot reconstruct",
               s, dd);
          task->bytes_read_ = bytes_read;
          task->length_ = bytes_read;
          task->return_code_ = 1;
          CLIO_CO_RETURN;
        }
        const std::vector<int> stripe = StripeMembers(s);
        int pos = -1;
        for (size_t i = 0; i < stripe.size(); ++i) {
          if (stripe[i] == static_cast<int>(dd)) {
            pos = static_cast<int>(i);
            break;
          }
        }
        std::vector<std::vector<uint8_t>> chunks;
        bool rec_ok = false;
        if (pos >= 0) {
          CLIO_CO_AWAIT(ReconstructStripe(s, stripe, static_cast<int>(dd),
                                          chunks, rec_ok));
        }
        if (!rec_ok || pos < 0) {
          task->bytes_read_ = bytes_read;
          task->length_ = bytes_read;
          task->return_code_ = 1;
          CLIO_CO_RETURN;
        }
        std::memcpy(data.ptr_ + buf_pos,
                    chunks[static_cast<size_t>(pos)].data() + within, seg_len);
        buf_pos += seg_len;
        cur = seg_end;
      }
    }
    bytes_read += len;
  }

  task->bytes_read_ = bytes_read;
  task->length_ = bytes_read;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetStats(clio::run::shared_ptr<GetStatsTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Full aggregate capacity: every active data member contributes its own
  // remaining slots (bump headroom + reusable frees). No stranding.
  clio::run::u64 remaining = 0;
  for (size_t d = 0; d < data_alloc_.size(); ++d) {
    if (data_members_[d].state_ == ec::EcState::kActive) {
      remaining += data_alloc_[d].RemainingSlots() * kChunkLen;
    }
  }
  task->remaining_size_ = remaining;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AddBdev(clio::run::shared_ptr<AddBdevTask> &task) {
  CLIO_TASK_BODY_BEGIN

  const bool as_parity = (task->as_parity_ != 0);

  if (!as_parity) {
    clio::run::u32 lowest_ttl = 7;
    int failing_data_idx = -1;
    int failing_parity_idx = -1;

    for (size_t i = 0; i < data_clients_.size(); ++i) {
      if (data_members_[i].state_ != ec::EcState::kActive) continue;
      auto stats = data_clients_[i].AsyncGetStats();
      CLIO_CO_AWAIT(stats);
      if (stats->return_code_ == 0 && stats->predicted_ttl_days_ < lowest_ttl) {
        lowest_ttl = stats->predicted_ttl_days_;
        failing_data_idx = static_cast<int>(i);
        failing_parity_idx = -1;
      }
    }
    for (size_t i = 0; i < parity_clients_.size(); ++i) {
      if (parity_members_[i].state_ != ec::EcState::kActive) continue;
      auto stats = parity_clients_[i].AsyncGetStats();
      CLIO_CO_AWAIT(stats);
      if (stats->return_code_ == 0 && stats->predicted_ttl_days_ < lowest_ttl) {
        lowest_ttl = stats->predicted_ttl_days_;
        failing_parity_idx = static_cast<int>(i);
        failing_data_idx = -1;
      }
    }

    if (failing_data_idx >= 0 || failing_parity_idx >= 0) {
      bool is_data_fail = failing_data_idx >= 0;
      int failed_idx = is_data_fail ? failing_data_idx : failing_parity_idx;
      const clio::run::PoolId &failed_pool_id = is_data_fail
          ? data_members_[static_cast<size_t>(failed_idx)].pool_id_
          : parity_members_[static_cast<size_t>(failed_idx)].pool_id_;

      HLOG(kInfo,
           "safe_bdev AddBdev: degrading {} member {} has TTL={} days (<7); "
           "preemptively migrating its data onto new member '{}'",
           is_data_fail ? "data" : "parity", failed_idx, lowest_ttl,
           task->pool_name_.str());

      // Mark the degrading member faulty BEFORE recovery so ReconstructRow
      // skips it and reads from survivors + parity.
      if (is_data_fail) {
        data_members_[static_cast<size_t>(failed_idx)].state_ =
            ec::EcState::kFaulty;
      } else {
        parity_members_[static_cast<size_t>(failed_idx)].state_ =
            ec::EcState::kFaulty;
      }

      // Drive the recovery: reconstruct all of the failing member's chunks
      // onto the new pool, then bring the slot back online pointing at
      // the new pool.
      auto rec_fut = client_.AsyncRecoverBdev(
          MemberQuery(), failed_pool_id, task->pool_name_.str(),
          task->node_id_, task->member_pool_id_);
      CLIO_CO_AWAIT(rec_fut);

      if (rec_fut->return_code_ != 0) {
        HLOG(kError,
             "safe_bdev AddBdev: preemptive migration failed (rc={}); "
             "member '{}' not added",
             rec_fut->return_code_.load(), task->pool_name_.str());
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }

      HLOG(kInfo,
           "safe_bdev AddBdev: preemptive migration complete - {} member {} "
           "rebuilt onto '{}'",
           is_data_fail ? "data" : "parity", failed_idx,
           task->pool_name_.str());
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // Add a DATA member: append it empty (fresh slot allocator sized from its
    // own capacity). NO data movement -- the round-robin cursor now includes it,
    // so new writes land on it and dirty the slots they widen, which BuildParity
    // re-parities. Existing data on other members is untouched.
    data_clients_.emplace_back(task->member_pool_id_);
    auto stats = data_clients_.back().AsyncGetStats();
    CLIO_CO_AWAIT(stats);
    if (stats->return_code_ != 0) {
      HLOG(kError, "safe_bdev AddBdev: GetStats failed for data member '{}'",
           task->pool_name_.str());
      data_clients_.pop_back();
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    const clio::run::u64 remaining = stats->remaining_size_;
    const clio::run::u64 avail =
        (remaining > kSuperblockSize) ? (remaining - kSuperblockSize) : 0;
    const clio::run::u64 cap_slots = avail / kChunkLen;

    const int new_col = static_cast<int>(data_members_.size());
    MemberSlot slot;
    slot.pool_id_ = task->member_pool_id_;
    slot.pool_name_ = task->pool_name_.str();
    slot.node_id_ = task->node_id_;
    slot.role_ = ec::EcRole::kData;
    slot.state_ = ec::EcState::kActive;
    slot.index_ = new_col;
    data_members_.push_back(slot);
    MemberAlloc a;
    a.cap_slots_ = cap_slots;
    data_alloc_.push_back(std::move(a));

    MemberSuperblock sb;
    bool present = false;
    bool sb_ok = false;
    CLIO_CO_AWAIT(ReadSuperblock(/*is_parity=*/false,
                                 static_cast<size_t>(new_col), sb,
                                 present, sb_ok));
    if (!sb_ok) {
      HLOG(kError, "safe_bdev AddBdev: superblock read failed for member '{}'",
           task->pool_name_.str());
      data_members_.pop_back();
      data_alloc_.pop_back();
      data_clients_.pop_back();
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    if (present && (sb.array_major != static_cast<uint64_t>(pool_id_.major_) ||
                    sb.array_minor != static_cast<uint64_t>(pool_id_.minor_))) {
      HLOG(kError,
           "safe_bdev AddBdev: REFUSING data member '{}' — owned by FOREIGN "
           "array ({},{})",
           task->pool_name_.str(), sb.array_major, sb.array_minor);
      data_members_.pop_back();
      data_alloc_.pop_back();
      data_clients_.pop_back();
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    if (!present) {
      bool wr_ok = false;
      CLIO_CO_AWAIT(WriteSuperblock(/*is_parity=*/false,
                                    static_cast<size_t>(new_col), wr_ok));
      if (!wr_ok) {
        HLOG(kWarning,
             "safe_bdev AddBdev: superblock write failed for new data member "
             "'{}' (seated; will be re-stamped on next attach)",
             task->pool_name_.str());
      }
    }

    HLOG(kInfo,
         "safe_bdev AddBdev(as_data): added data member {} '{}' ({} slots); no "
         "data movement (round-robin now includes it)",
         new_col, task->pool_name_.str(), cap_slots);
    PersistMemberManifest();
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  if (parity_level_ >= max_failures_) {
    HLOG(kWarning,
         "safe_bdev AddBdev: parity already at max_failures={} (member '{}' "
         "not added)",
         max_failures_, task->pool_name_.str());
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  // Append a PARITY member. Its parity row j == parity_level_; each stripe s
  // carries that member's RS shard j under the stripe's width code. Re-dirty
  // every written slot so BuildParity computes the new parity column off the
  // critical path.
  parity_clients_.emplace_back(task->member_pool_id_);
  MemberSlot slot;
  slot.pool_id_ = task->member_pool_id_;
  slot.pool_name_ = task->pool_name_.str();
  slot.node_id_ = task->node_id_;
  slot.role_ = ec::EcRole::kParity;
  slot.state_ = ec::EcState::kActive;
  slot.index_ = static_cast<int>(parity_level_);
  const size_t new_j = parity_members_.size();
  parity_members_.push_back(slot);
  ++parity_level_;

  {
    std::lock_guard<std::mutex> g(slot_mu_);
    for (clio::run::u64 s : written_slots_) {
      dirty_slots_.insert(s);
    }
  }
  HLOG(kInfo,
       "safe_bdev AddBdev: parity drive added (parity_level={}, all written "
       "slots re-dirtied; parity build deferred to BuildParity)",
       parity_level_);

  {
    bool wr_ok = false;
    CLIO_CO_AWAIT(WriteSuperblock(/*is_parity=*/true, new_j, wr_ok));
    if (!wr_ok) {
      HLOG(kWarning,
           "safe_bdev AddBdev: superblock write failed for new parity member "
           "'{}' (member seated; will be re-stamped on next attach)",
           task->pool_name_.str());
    }
  }

  PersistMemberManifest();
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RemoveBdev(clio::run::shared_ptr<RemoveBdevTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const auto apply = [&](MemberSlot &m, const char *kind, size_t idx) {
    if (task->was_faulty_ != 0) {
      m.state_ = ec::EcState::kFaulty;
      HLOG(kInfo, "safe_bdev RemoveBdev: {} member {} marked faulty", kind, idx);
    } else {
      m.state_ = ec::EcState::kRemoved;
      HLOG(kInfo, "safe_bdev RemoveBdev: {} member {} unlinked (marked removed)",
           kind, idx);
    }
  };
  bool found = false;
  for (size_t i = 0; i < data_members_.size() && !found; ++i) {
    if (data_members_[i].pool_id_ == task->target_pool_id_) {
      apply(data_members_[i], "data", i);
      found = true;
    }
  }
  for (size_t i = 0; i < parity_members_.size() && !found; ++i) {
    if (parity_members_[i].pool_id_ == task->target_pool_id_) {
      apply(parity_members_[i], "parity", i);
      found = true;
    }
  }
  if (found) {
    PersistMemberManifest();
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RebuildMember(bool is_data, int idx, bool &ok,
                                             bool &completed) {
  CLIO_TASK_BODY_BEGIN
  ok = true;
  completed = true;
  auto &client = is_data ? data_clients_[static_cast<size_t>(idx)]
                         : parity_clients_[static_cast<size_t>(idx)];
  auto *ipc = CLIO_IPC;

  // Test hook: stop after this many slots to simulate a recovery interrupted by
  // a crash (0 / unset => rebuild everything). The manifest keeps the member
  // recovering, so the next Create() resumes it.
  clio::run::u64 max_rows = 0;
  if (const char *env = std::getenv("CLIO_SAFE_BDEV_RECOVER_MAX_ROWS")) {
    max_rows = std::strtoull(env, nullptr, 10);
  }

  // The set of slots to rebuild: a DATA member rebuilds its own live slots; a
  // PARITY member rebuilds every written slot (all stripes).
  std::vector<clio::run::u64> slots;
  if (is_data) {
    for (clio::run::u64 s : data_alloc_[static_cast<size_t>(idx)].live_) {
      slots.push_back(s);
    }
  } else {
    std::lock_guard<std::mutex> g(slot_mu_);
    for (clio::run::u64 s : written_slots_) {
      slots.push_back(s);
    }
  }
  std::sort(slots.begin(), slots.end());

  recovery_ops_total_.store(slots.size(), std::memory_order_relaxed);
  recovery_ops_completed_.store(0, std::memory_order_relaxed);
  recovery_ops_in_flight_.store(0, std::memory_order_relaxed);
  recovering_is_parity_.store(is_data ? 0u : 1u, std::memory_order_relaxed);
  recovering_index_.store(idx, std::memory_order_relaxed);
  recovery_active_.store(1, std::memory_order_release);

  clio::run::u64 done = 0;
  for (clio::run::u64 s : slots) {
    if (max_rows != 0 && done >= max_rows) {
      completed = false;  // interrupted: leave the member recovering
      recovery_active_.store(0, std::memory_order_release);
      CLIO_CO_RETURN;
    }
    if (is_data && IsSlotDirty(s)) {
      // Reconstructing a DATA chunk needs current parity; a parity rebuild
      // recomputes from the (active) data members directly, so it does not.
      HLOG(kError,
           "safe_bdev RebuildMember: slot {} dirty (parity not built); cannot "
           "reconstruct data member",
           s);
      ok = false;
      recovery_active_.store(0, std::memory_order_release);
      CLIO_CO_RETURN;
    }
    const std::vector<int> stripe = StripeMembers(s);
    const int k_s = static_cast<int>(stripe.size());
    if (k_s <= 0) {
      ++done;
      continue;
    }

    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kChunkLen);
    if (buf.IsNull()) {
      ok = false;
      recovery_active_.store(0, std::memory_order_release);
      CLIO_CO_RETURN;
    }

    if (is_data) {
      // Reconstruct this member's chunk from the stripe's survivors.
      int pos = -1;
      for (size_t i = 0; i < stripe.size(); ++i) {
        if (stripe[i] == idx) {
          pos = static_cast<int>(i);
          break;
        }
      }
      std::vector<std::vector<uint8_t>> chunks;
      bool rec_ok = false;
      if (pos >= 0) {
        CLIO_CO_AWAIT(ReconstructStripe(s, stripe, idx, chunks, rec_ok));
      }
      if (!rec_ok || pos < 0) {
        ipc->FreeBuffer(buf);
        ok = false;
        recovery_active_.store(0, std::memory_order_release);
        CLIO_CO_RETURN;
      }
      std::memcpy(buf.ptr_, chunks[static_cast<size_t>(pos)].data(), kChunkLen);
    } else {
      // Recompute this parity member's shard from the stripe's data chunks
      // (all data members must be active).
      std::vector<std::vector<uint8_t>> dchunks(
          static_cast<size_t>(k_s), std::vector<uint8_t>(kChunkLen, 0));
      bool rd_ok = true;
      for (int pos = 0; pos < k_s; ++pos) {
        const int d = stripe[static_cast<size_t>(pos)];
        if (data_members_[static_cast<size_t>(d)].state_ !=
            ec::EcState::kActive) {
          rd_ok = false;
          break;
        }
        bool one = false;
        CLIO_CO_AWAIT(ReadDataSegment(static_cast<size_t>(d), SlotPhysOffset(s),
                                      dchunks[static_cast<size_t>(pos)].data(),
                                      kChunkLen, one));
        rd_ok = one;
        if (!rd_ok) break;
      }
      if (!rd_ok) {
        ipc->FreeBuffer(buf);
        ok = false;
        recovery_active_.store(0, std::memory_order_release);
        CLIO_CO_RETURN;
      }
      std::vector<const uint8_t *> ptrs(static_cast<size_t>(k_s));
      for (int pos = 0; pos < k_s; ++pos) {
        ptrs[static_cast<size_t>(pos)] = dchunks[static_cast<size_t>(pos)].data();
      }
      GetCodec(k_s)->EncodeParityShard(idx, ptrs, kChunkLen,
                                       reinterpret_cast<uint8_t *>(buf.ptr_));
    }

    recovery_ops_in_flight_.store(1, std::memory_order_relaxed);
    auto fut = client.AsyncWrite(MemberQuery(),
                                 MemberBlocks(SlotPhysOffset(s), kChunkLen),
                                 buf.shm_.template Cast<void>(), kChunkLen);
    CLIO_CO_AWAIT(fut);
    const bool wok =
        (fut->return_code_ == 0) && (fut->bytes_written_ == kChunkLen);
    ipc->FreeBuffer(buf);
    recovery_ops_in_flight_.store(0, std::memory_order_relaxed);
    if (!wok) {
      ok = false;
      recovery_active_.store(0, std::memory_order_release);
      CLIO_CO_RETURN;
    }
    recovery_ops_completed_.fetch_add(1, std::memory_order_relaxed);
    ++done;
  }
  recovery_active_.store(0, std::memory_order_release);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RecoverBdev(clio::run::shared_ptr<RecoverBdevTask> &task) {
  CLIO_TASK_BODY_BEGIN

  int failed_col = -1;
  int failed_par = -1;
  for (size_t i = 0; i < data_members_.size(); ++i) {
    if (data_members_[i].pool_id_ == task->old_bdev_id_) {
      failed_col = static_cast<int>(i);
      break;
    }
  }
  if (failed_col < 0) {
    for (size_t i = 0; i < parity_members_.size(); ++i) {
      if (parity_members_[i].pool_id_ == task->old_bdev_id_) {
        failed_par = static_cast<int>(i);
        break;
      }
    }
  }
  if (failed_col < 0 && failed_par < 0) {
    HLOG(kError, "safe_bdev RecoverBdev: old member {} not found",
         task->old_bdev_id_.ToU64());
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  const bool is_data = (failed_col >= 0);
  const int idx = is_data ? failed_col : failed_par;

  auto &client = is_data ? data_clients_[static_cast<size_t>(idx)]
                         : parity_clients_[static_cast<size_t>(idx)];
  client = clio::run::bdev::Client(task->new_pool_id_);
  MemberSlot &m = is_data ? data_members_[static_cast<size_t>(idx)]
                          : parity_members_[static_cast<size_t>(idx)];
  m.pool_id_ = task->new_pool_id_;
  m.pool_name_ = task->pool_name_.str();
  m.node_id_ = task->node_id_;
  m.state_ = ec::EcState::kFaulty;  // non-active until rebuild completes
  m.recovering_ = true;
  PersistMemberManifest();

  HLOG(kInfo, "safe_bdev RecoverBdev: rebuilding {} member {} onto '{}'",
       is_data ? "data" : "parity", idx, task->pool_name_.str());

  bool ok = false;
  bool completed = false;
  CLIO_CO_AWAIT(RebuildMember(is_data, idx, ok, completed));
  if (!ok) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  if (!completed) {
    HLOG(kWarning,
         "safe_bdev RecoverBdev: rebuild interrupted (test hook) -- recovery "
         "persisted; the next Create() resumes it");
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  m.state_ = ec::EcState::kActive;
  m.recovering_ = false;
  {
    bool wr_ok = false;
    CLIO_CO_AWAIT(WriteSuperblock(!is_data, static_cast<size_t>(idx), wr_ok));
    if (!wr_ok) {
      HLOG(kWarning,
           "safe_bdev RecoverBdev: superblock write failed for recovered "
           "member (data reconstructed; will be re-stamped on next attach)");
    }
  }
  PersistMemberManifest();
  HLOG(kInfo, "safe_bdev RecoverBdev: {} member {} recovered onto '{}'",
       is_data ? "data" : "parity", idx, task->pool_name_.str());
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ResumeRecoveries() {
  CLIO_TASK_BODY_BEGIN
  for (size_t i = 0; i < data_members_.size(); ++i) {
    if (!data_members_[i].recovering_) continue;
    HLOG(kInfo,
         "safe_bdev: resuming interrupted recovery of data member {} onto '{}'",
         i, data_members_[i].pool_name_);
    bool ok = false;
    bool completed = false;
    CLIO_CO_AWAIT(RebuildMember(true, static_cast<int>(i), ok, completed));
    if (ok && completed) {
      data_members_[i].state_ = ec::EcState::kActive;
      data_members_[i].recovering_ = false;
      bool wr = false;
      CLIO_CO_AWAIT(WriteSuperblock(false, i, wr));
      PersistMemberManifest();
      HLOG(kInfo, "safe_bdev: recovery of data member {} completed on restart",
           i);
    } else {
      HLOG(kWarning,
           "safe_bdev: resume of data member {} did not complete "
           "(ok={}, completed={})",
           i, ok, completed);
    }
  }
  for (size_t j = 0; j < parity_members_.size(); ++j) {
    if (!parity_members_[j].recovering_) continue;
    bool ok = false;
    bool completed = false;
    CLIO_CO_AWAIT(RebuildMember(false, static_cast<int>(j), ok, completed));
    if (ok && completed) {
      parity_members_[j].state_ = ec::EcState::kActive;
      parity_members_[j].recovering_ = false;
      bool wr = false;
      CLIO_CO_AWAIT(WriteSuperblock(true, j, wr));
      PersistMemberManifest();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::BuildParity(clio::run::shared_ptr<BuildParityTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (parity_level_ == 0) {
    std::lock_guard<std::mutex> g(slot_mu_);
    dirty_slots_.clear();
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  // Snapshot a batch of dirty slots (max_batch_==0 drains all). A slot is erased
  // only AFTER its parity is durably written, so an observer that sees an empty
  // dirty set knows every stripe is protected. A slot that cannot be built (a
  // stripe member is down) is left dirty for a later pass.
  std::vector<clio::run::u64> batch;
  {
    std::lock_guard<std::mutex> g(slot_mu_);
    for (clio::run::u64 s : dirty_slots_) {
      batch.push_back(s);
      if (task->max_batch_ != 0 &&
          batch.size() >= static_cast<size_t>(task->max_batch_)) {
        break;
      }
    }
  }

  auto *ipc = CLIO_IPC;
  clio::run::u32 built = 0;
  for (clio::run::u64 s : batch) {
    const std::vector<int> stripe = StripeMembers(s);
    const int k_s = static_cast<int>(stripe.size());
    if (k_s <= 0) {
      // Stale dirty mark for an empty stripe; drop it.
      std::lock_guard<std::mutex> g(slot_mu_);
      dirty_slots_.erase(s);
      continue;
    }

    // Read the k_s data chunks of this stripe (all stripe members must be
    // active to build parity over the full chunks).
    std::vector<std::vector<uint8_t>> dchunks(
        static_cast<size_t>(k_s), std::vector<uint8_t>(kChunkLen, 0));
    bool rd_ok = true;
    for (int pos = 0; pos < k_s; ++pos) {
      const int d = stripe[static_cast<size_t>(pos)];
      if (data_members_[static_cast<size_t>(d)].state_ !=
          ec::EcState::kActive) {
        rd_ok = false;
        break;
      }
      bool one = false;
      CLIO_CO_AWAIT(ReadDataSegment(static_cast<size_t>(d), SlotPhysOffset(s),
                                    dchunks[static_cast<size_t>(pos)].data(),
                                    kChunkLen, one));
      rd_ok = one;
      if (!rd_ok) break;
    }
    if (!rd_ok) {
      continue;  // leave dirty; retry next pass
    }

    std::vector<const uint8_t *> ptrs(static_cast<size_t>(k_s));
    for (int pos = 0; pos < k_s; ++pos) {
      ptrs[static_cast<size_t>(pos)] = dchunks[static_cast<size_t>(pos)].data();
    }
    ec::ReedSolomon *codec = GetCodec(k_s);

    bool wr_ok = true;
    for (int j = 0; j < static_cast<int>(parity_level_); ++j) {
      if (parity_members_[static_cast<size_t>(j)].state_ !=
          ec::EcState::kActive) {
        continue;
      }
      ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kChunkLen);
      if (buf.IsNull()) {
        wr_ok = false;
        break;
      }
      codec->EncodeParityShard(j, ptrs, kChunkLen,
                               reinterpret_cast<uint8_t *>(buf.ptr_));
      auto fut = parity_clients_[static_cast<size_t>(j)].AsyncWrite(
          MemberQuery(), MemberBlocks(SlotPhysOffset(s), kChunkLen),
          buf.shm_.template Cast<void>(), kChunkLen);
      CLIO_CO_AWAIT(fut);
      wr_ok = (fut->return_code_ == 0) && (fut->bytes_written_ == kChunkLen);
      MaybeFaultParity(static_cast<size_t>(j), fut->io_error_);
      ipc->FreeBuffer(buf);
      if (!wr_ok) break;
    }
    if (!wr_ok) {
      continue;  // leave dirty; retry next pass
    }
    {
      std::lock_guard<std::mutex> g2(slot_mu_);
      dirty_slots_.erase(s);
    }
    ++built;
  }
  if (built > 0) {
    HLOG(kDebug, "safe_bdev BuildParity: built parity for {} stripe(s)", built);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushAllocLog(
    clio::run::shared_ptr<FlushAllocLogTask> &task) {
  CLIO_TASK_BODY_BEGIN
  alloc_log_.Flush();
  const clio::run::u64 live = alloc_log_.live_block_count();
  const clio::run::u64 on_disk = alloc_log_.records_on_disk();
  const clio::run::u64 threshold =
      std::max<clio::run::u64>(kMinCompactRecords, live * kCompactGrowthFactor);
  if (on_disk > threshold) {
    alloc_log_.Compact();
  }
  if (MemberManifestNeedsCompaction()) {
    CompactMemberManifest();
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->query_ == "stats") {
    clio::run::u32 dirty = 0;
    clio::run::u32 written = 0;
    {
      std::lock_guard<std::mutex> g(slot_mu_);
      dirty = static_cast<clio::run::u32>(dirty_slots_.size());
      written = static_cast<clio::run::u32>(written_slots_.size());
    }
    clio::run::u64 total_slots = 0;
    for (const auto &a : data_alloc_) {
      total_slots += a.cap_slots_;
    }

    const clio::run::u64 rec_total =
        recovery_ops_total_.load(std::memory_order_relaxed);
    const clio::run::u64 rec_done =
        recovery_ops_completed_.load(std::memory_order_relaxed);
    const clio::run::u64 rec_inflight =
        recovery_ops_in_flight_.load(std::memory_order_relaxed);
    const clio::run::u64 rec_remaining =
        (rec_total > rec_done) ? (rec_total - rec_done) : 0;
    const clio::run::u32 rec_active =
        recovery_active_.load(std::memory_order_acquire);
    const clio::run::u32 rec_is_par =
        recovering_is_parity_.load(std::memory_order_relaxed);
    const int rec_idx = recovering_index_.load(std::memory_order_relaxed);

    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(14);
    pk.pack("pool_name");     pk.pack(pool_name_);
    pk.pack("max_failures");  pk.pack(max_failures_);
    pk.pack("data_count");
    pk.pack(static_cast<clio::run::u32>(data_members_.size()));
    pk.pack("parity_level");  pk.pack(parity_level_);
    pk.pack("total_slots");   pk.pack(total_slots);
    pk.pack("written_slots"); pk.pack(written);
    pk.pack("dirty_slots");   pk.pack(dirty);
    pk.pack("reattached_members"); pk.pack(reattached_members_);
    pk.pack("alloc_log_records");
    pk.pack(static_cast<clio::run::u64>(alloc_log_.records_on_disk()));
    pk.pack("recovery_active");        pk.pack(rec_active);
    pk.pack("recovery_ops_total");     pk.pack(rec_total);
    pk.pack("recovery_ops_completed"); pk.pack(rec_done);
    pk.pack("recovery_ops_in_flight"); pk.pack(rec_inflight);
    pk.pack("recovery_ops_remaining"); pk.pack(rec_remaining);
    const auto pack_members = [&](const std::vector<MemberSlot> &vec,
                                  bool is_parity) {
      for (const auto &m : vec) {
        const char *st = (m.state_ == ec::EcState::kFaulty)  ? "faulty"
                         : (m.state_ == ec::EcState::kRemoved) ? "removed"
                                                               : "active";
        const bool recovering = (rec_active != 0) &&
                                ((is_parity ? 1u : 0u) == rec_is_par) &&
                                (m.index_ == rec_idx);
        pk.pack_map(6);
        pk.pack("role");
        pk.pack(std::string(is_parity ? "parity" : "data"));
        pk.pack("index");     pk.pack(static_cast<clio::run::u32>(m.index_));
        pk.pack("pool_name"); pk.pack(m.pool_name_);
        pk.pack("pool_id");   pk.pack(m.pool_id_.ToU64());
        pk.pack("state");
        pk.pack(std::string(recovering ? "recovering" : st));
        pk.pack("recovering");
        pk.pack(static_cast<clio::run::u32>(recovering ? 1 : 0));
      }
    };
    pk.pack("members");
    pk.pack_array(static_cast<uint32_t>(data_members_.size() +
                                        parity_members_.size()));
    pack_members(data_members_, false);
    pack_members(parity_members_, true);
    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  alloc_log_.Flush();
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

}  // namespace clio::run::safe_bdev

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::safe_bdev::Runtime)

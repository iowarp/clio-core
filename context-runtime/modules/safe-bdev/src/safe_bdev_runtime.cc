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
#include <chrono>
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
      // compute_ feeds InferCpuTime; a safe-bdev write additionally stages a
      // per-block copy before forwarding to each member, so it carries roughly
      // twice the CPU per byte of a plain bdev write.
      stat.compute_ = static_cast<size_t>(aligned / 5000.0F) + 3;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0F;
      return stat;
    }
    case Method::kRead: {
      const auto *rt = static_cast<const ReadTask *>(task);
      clio::run::TaskStat stat;
      stat.io_size_ = rt->length_;
      const size_t aligned = ((stat.io_size_ + 4095) / 4096) * 4096;
      stat.compute_ = static_cast<size_t>(aligned / 5000.0F) + 3;
      stat.wall_time_ = static_cast<float>(aligned) / 500.0F;
      return stat;
    }
    case Method::kAllocateBlocks:
    case Method::kFreeBlocks: {
      // Slot bookkeeping under alloc_mu_ — small and bounded.
      clio::run::TaskStat stat;
      stat.compute_ = 6;
      stat.wall_time_ = 10.0F;
      return stat;
    }
    case Method::kBuildParity: {
      // Reed-Solomon encode over a dirty stripe: the CPU-heaviest verb here.
      clio::run::TaskStat stat;
      stat.compute_ = 500;
      stat.wall_time_ = 600.0F;
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
      DataQuery(d), MemberBlocks(offset, len),
      buf.shm_.template Cast<void>(), len);
  CLIO_CO_AWAIT(fut);
  ok = (fut->return_code_ == 0) && (fut->bytes_written_ == len);
  FaultOnIoError(/*is_parity=*/false, d, !ok, fut->io_error_);
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
      DataQuery(d), MemberBlocks(offset, len),
      buf.shm_.template Cast<void>(), len);
  CLIO_CO_AWAIT(fut);
  if (fut->return_code_ == 0 && fut->bytes_read_ == len) {
    std::memcpy(dst, buf.ptr_, len);
    ok = true;
  }
  FaultOnIoError(/*is_parity=*/false, d, !ok, fut->io_error_);
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

//===========================================================================
// Runtime member fault detection
//===========================================================================

bool Runtime::FaultMember(bool is_parity, size_t idx, const char *why) {
  std::vector<MemberSlot> &vec = is_parity ? parity_members_ : data_members_;
  if (idx >= vec.size()) return false;
  MemberSlot &m = vec[idx];
  // A recovery target is already non-active; its rebuild reports its own
  // failures.
  if (m.recovering_) return false;
  if (!m.state_.Transition(ec::EcState::kActive, ec::EcState::kFaulty)) {
    return false;  // already down: someone else logged and persisted it
  }
  HLOG(kError,
       "safe_bdev {}: {} member {} ('{}') FAILED at runtime ({}); marked "
       "faulty -- reads reconstruct it, writes go degraded; {} of {} members "
       "down, max_failures={}",
       pool_name_, is_parity ? "parity" : "data", idx, m.pool_name_, why,
       CountDownMembers(), data_members_.size() + parity_members_.size(),
       max_failures_);
  PersistMemberManifest();
  return true;
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
  auto fut = client.AsyncWrite(is_parity ? ParityQuery(idx) : DataQuery(idx),
                               MemberBlocks(0, kSuperblockSize),
                               buf.shm_.template Cast<void>(), kSuperblockSize);
  CLIO_CO_AWAIT(fut);
  // No auto-fault here: every caller handles a failed stamp itself (refuses
  // or unwinds the member), and a member it pops must not be persisted.
  ok = (fut->return_code_ == 0) && (fut->bytes_written_ == kSuperblockSize);
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
  auto fut = client.AsyncRead(is_parity ? ParityQuery(idx) : DataQuery(idx), MemberBlocks(0, kSuperblockSize),
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

clio::run::TaskResume Runtime::GatherSurvivors(
    clio::run::u64 s, const std::vector<int> &code,
    const std::vector<int> &exclude, std::vector<int> &idx,
    std::vector<std::vector<uint8_t>> &bufs) {
  CLIO_TASK_BODY_BEGIN
  const int k = static_cast<int>(code.size());
  const clio::run::u64 off = SlotPhysOffset(s);
  // A data survivor's RS shard index is its POSITION in `code`; a parity
  // survivor's is k + j.
  for (int pos = 0; pos < k && static_cast<int>(idx.size()) < k; ++pos) {
    const int d = code[static_cast<size_t>(pos)];
    if (std::find(exclude.begin(), exclude.end(), d) != exclude.end() ||
        static_cast<size_t>(d) >= data_members_.size() ||
        !DataActive(static_cast<size_t>(d))) {
      continue;
    }
    std::vector<uint8_t> buf(kChunkLen, 0);
    bool rd_ok = false;
    CLIO_CO_AWAIT(ReadDataSegment(static_cast<size_t>(d), off, buf.data(),
                                  kChunkLen, rd_ok));
    if (!rd_ok) continue;  // faulted (or transient): try another shard
    idx.push_back(pos);
    bufs.push_back(std::move(buf));
  }
  auto *ipc = CLIO_IPC;
  for (int j = 0; j < static_cast<int>(parity_level_) &&
                  static_cast<int>(idx.size()) < k;
       ++j) {
    if (parity_members_[static_cast<size_t>(j)].state_ !=
        ec::EcState::kActive) {
      continue;
    }
    ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kChunkLen);
    if (rbuf.IsNull()) CLIO_CO_RETURN;
    auto fut = parity_clients_[static_cast<size_t>(j)].AsyncRead(
        ParityQuery(static_cast<size_t>(j)), MemberBlocks(off, kChunkLen),
        rbuf.shm_.template Cast<void>(), kChunkLen);
    CLIO_CO_AWAIT(fut);
    const bool rd_ok =
        (fut->return_code_ == 0) && (fut->bytes_read_ == kChunkLen);
    if (rd_ok) {
      idx.push_back(k + j);
      bufs.emplace_back(reinterpret_cast<uint8_t *>(rbuf.ptr_),
                        reinterpret_cast<uint8_t *>(rbuf.ptr_) + kChunkLen);
    }
    FaultOnIoError(/*is_parity=*/true, static_cast<size_t>(j), !rd_ok,
                   fut->io_error_);
    ipc->FreeBuffer(rbuf);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReconstructStripe(
    clio::run::u64 s, const std::vector<int> &stripe,
    const std::vector<int> &exclude, std::vector<std::vector<uint8_t>> &out,
    bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  if (stripe.empty()) CLIO_CO_RETURN;
  // Fixed-width parity (#1126): the code spans every data column.
  const std::vector<int> code = CodeColumns();
  const int k = static_cast<int>(code.size());
  std::vector<int> idx;
  std::vector<std::vector<uint8_t>> bufs;
  CLIO_CO_AWAIT(GatherSurvivors(s, code, exclude, idx, bufs));
  if (static_cast<int>(idx.size()) < k) {
    CLIO_CO_RETURN;  // Too many failures to reconstruct.
  }
  std::vector<const uint8_t *> ptrs(bufs.size());
  for (size_t i = 0; i < bufs.size(); ++i) ptrs[i] = bufs[i].data();
  std::vector<std::vector<uint8_t>> decoded;
  if (!GetCodec(k)->DecodeData(idx, ptrs, kChunkLen, &decoded)) {
    CLIO_CO_RETURN;
  }
  out.assign(stripe.size(), std::vector<uint8_t>());
  for (size_t pos = 0; pos < stripe.size(); ++pos) {
    out[pos] = std::move(decoded[static_cast<size_t>(stripe[pos])]);
  }
  ok = true;
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
  put_u32(static_cast<clio::run::u32>(m.state_.Load()));
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

clio::run::TaskResume Runtime::QueryMemberSlots(clio::run::bdev::Client client,
                                                clio::run::PoolQuery q,
                                                clio::run::u64 &slots,
                                                bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  slots = 0;
  auto st = client.AsyncGetStats(q);
  CLIO_CO_AWAIT(st);
  if (st->return_code_ != 0) CLIO_CO_RETURN;
  // The device's own allocatable size (safe_bdev never allocates through the
  // member's allocator, so remaining == total for a member).
  const clio::run::u64 bytes =
      st->total_size_ != 0 ? st->total_size_ : st->remaining_size_;
  slots = bytes > kSuperblockSize ? (bytes - kSuperblockSize) / kChunkLen : 0;
  ok = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::vector<Runtime::DataSeatSpec> Runtime::BuildDataMemberPlan(
    const CreateParams &params,
    const std::vector<MemberManifestEntry> &manifest) const {
  std::vector<DataSeatSpec> plan;
  for (const auto &d : params.members_) {
    if (d.parity_) continue;  // seated by SeatParityMembers
    DataSeatSpec spec;
    spec.desc = d;
    plan.push_back(spec);
  }
  std::vector<const MemberManifestEntry *> data;
  for (const auto &e : manifest) {
    if (e.role_ == 0) data.push_back(&e);
  }
  std::sort(data.begin(), data.end(),
            [](const MemberManifestEntry *a, const MemberManifestEntry *b) {
              return a->index_ < b->index_;
            });
  for (const MemberManifestEntry *e : data) {
    if (e->index_ > plan.size()) {
      HLOG(kError, "safe_bdev Create: manifest data column {} leaves a gap "
           "after column {}; ignoring it", e->index_, plan.size());
      continue;
    }
    if (e->index_ == plan.size()) plan.emplace_back();  // added at runtime
    DataSeatSpec &spec = plan[e->index_];
    spec.desc = MemberBdevDesc(e->pool_name_, e->node_id_,
                               clio::run::PoolId(e->pool_major_,
                                                 e->pool_minor_));
    spec.state = e->state_;
    spec.recovering = e->recovering_ != 0;
  }
  return plan;
}

clio::run::TaskResume Runtime::SeatDataMember(DataSeatSpec spec, int col,
                                              clio::run::u32 &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  const MemberBdevDesc &desc = spec.desc;
  data_clients_.emplace_back(desc.pool_id_);
  MemberSlot slot;
  slot.pool_id_ = desc.pool_id_;
  slot.pool_name_ = desc.pool_name_;
  slot.node_id_ = desc.node_id_;
  slot.role_ = ec::EcRole::kData;
  slot.state_ = ec::EcState::kActive;
  slot.index_ = col;
  const bool down = !spec.recovering &&
                    spec.state != static_cast<clio::run::u32>(
                                      ec::EcState::kActive);
  if (down) {
    // Faulty/removed when the array stopped: its device may be gone, so do
    // not touch it. Reads reconstruct its chunks; it takes no allocations.
    slot.state_ = static_cast<ec::EcState>(spec.state);
    data_members_.push_back(slot);
    data_alloc_.emplace_back();
    HLOG(kWarning, "safe_bdev Create: data column {} ('{}') restored as down",
         col, desc.pool_name_);
    CLIO_CO_RETURN;
  }
  clio::run::u64 cap_slots = 0;
  bool qok = false;
  CLIO_CO_AWAIT(QueryMemberSlots(data_clients_.back(), MemberQuery(slot),
                                 cap_slots, qok));
  if (!qok) {
    HLOG(kError, "safe_bdev Create: GetStats failed for member '{}'",
         desc.pool_name_);
    rc = 1;
    CLIO_CO_RETURN;
  }
  const clio::run::u64 avail = cap_slots * kChunkLen;
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
    rc = 1;
    CLIO_CO_RETURN;
  }
  if (!present) {
    bool wr_ok = false;
    CLIO_CO_AWAIT(WriteSuperblock(/*is_parity=*/false,
                                  static_cast<size_t>(col), wr_ok));
    if (!wr_ok) {
      HLOG(kError, "safe_bdev Create: superblock write failed for fresh "
           "member '{}'", desc.pool_name_);
      rc = 1;
      CLIO_CO_RETURN;
    }
    HLOG(kInfo, "safe_bdev Create: initialized fresh member '{}' (slot {}, "
         "{} slots)", desc.pool_name_, col, avail / kChunkLen);
  } else if (sb.array_major == static_cast<uint64_t>(pool_id_.major_) &&
             sb.array_minor == static_cast<uint64_t>(pool_id_.minor_)) {
    ++reattached_members_;
    HLOG(kInfo, "safe_bdev Create: re-attached member '{}' (slot {})",
         desc.pool_name_, col);
  } else {
    HLOG(kError, "safe_bdev Create: REFUSING member '{}' -- initialized by a "
         "FOREIGN array ({},{}); this array is ({},{})", desc.pool_name_,
         sb.array_major, sb.array_minor, pool_id_.major_, pool_id_.minor_);
    rc = 2;
    CLIO_CO_RETURN;
  }
  if (spec.recovering) {
    // Mid-recovery onto this member: stays non-active (degraded reads serve
    // it) until ResumeRecoveries() finishes the rebuild.
    data_members_.back().state_ = ec::EcState::kFaulty;
    data_members_.back().recovering_ = true;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SeatConfigParity(MemberBdevDesc desc,
                                                clio::run::u32 &rc,
                                                bool &fresh) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  fresh = false;
  MemberSlot slot;
  slot.pool_id_ = desc.pool_id_;
  slot.pool_name_ = desc.pool_name_;
  slot.node_id_ = desc.node_id_;
  slot.role_ = ec::EcRole::kParity;
  slot.state_ = ec::EcState::kActive;
  slot.index_ = static_cast<int>(parity_members_.size());
  parity_clients_.emplace_back(desc.pool_id_);
  parity_members_.push_back(slot);
  const size_t pj = parity_members_.size() - 1;
  MemberSuperblock sb;
  bool present = false;
  bool sb_ok = false;
  CLIO_CO_AWAIT(ReadSuperblock(/*is_parity=*/true, pj, sb, present, sb_ok));
  if (!sb_ok) {
    HLOG(kError, "safe_bdev Create: superblock read failed for parity member "
         "'{}'", desc.pool_name_);
    rc = 1;
    CLIO_CO_RETURN;
  }
  if (!present) {
    bool wr_ok = false;
    CLIO_CO_AWAIT(WriteSuperblock(/*is_parity=*/true, pj, wr_ok));
    if (!wr_ok) {
      HLOG(kError, "safe_bdev Create: superblock write failed for fresh "
           "parity member '{}'", desc.pool_name_);
      rc = 1;
      CLIO_CO_RETURN;
    }
    fresh = true;
    HLOG(kInfo, "safe_bdev Create: initialized fresh parity member '{}' "
         "(row {})", desc.pool_name_, pj);
  } else if (sb.array_major == static_cast<uint64_t>(pool_id_.major_) &&
             sb.array_minor == static_cast<uint64_t>(pool_id_.minor_)) {
    ++reattached_members_;
    HLOG(kInfo, "safe_bdev Create: re-attached parity member '{}' (row {})",
         desc.pool_name_, pj);
  } else {
    HLOG(kError, "safe_bdev Create: REFUSING parity member '{}' -- "
         "initialized by a FOREIGN array ({},{})", desc.pool_name_,
         sb.array_major, sb.array_minor);
    rc = 2;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SeatParityMembers(
    const CreateParams &params,
    const std::vector<MemberManifestEntry> &manifest_parity,
    clio::run::u32 &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  for (const auto &pe : manifest_parity) {
    MemberSlot slot;
    slot.pool_id_ = clio::run::PoolId(pe.pool_major_, pe.pool_minor_);
    slot.pool_name_ = pe.pool_name_;
    slot.node_id_ = pe.node_id_;
    slot.role_ = ec::EcRole::kParity;
    slot.index_ = static_cast<int>(pe.index_);
    parity_clients_.emplace_back(slot.pool_id_);
    parity_members_.push_back(slot);
    const size_t pj = parity_members_.size() - 1;
    if (pe.recovering_ != 0) {
      parity_members_[pj].state_ = ec::EcState::kFaulty;
      parity_members_[pj].recovering_ = true;
      continue;
    }
    if (pe.state_ != static_cast<clio::run::u32>(ec::EcState::kActive)) {
      // Faulty/removed when the array stopped: its device may be gone, so do
      // not touch it -- and keep its row, or every later row's RS shard
      // index would shift.
      parity_members_[pj].state_ = static_cast<ec::EcState>(pe.state_);
      HLOG(kWarning, "safe_bdev Create: parity row {} ('{}') restored as down",
           pj, pe.pool_name_);
      continue;
    }
    MemberSuperblock sb;
    bool present = false;
    bool sb_ok = false;
    CLIO_CO_AWAIT(ReadSuperblock(/*is_parity=*/true, pj, sb, present, sb_ok));
    const bool ours = sb_ok && present &&
                      sb.array_major == static_cast<uint64_t>(pool_id_.major_) &&
                      sb.array_minor == static_cast<uint64_t>(pool_id_.minor_);
    if (!ours) {
      parity_members_.pop_back();
      parity_clients_.pop_back();
      HLOG(kInfo, "safe_bdev Create: skipping stale manifest parity member "
           "(pool {}.{} not reachable/ours)", pe.pool_major_, pe.pool_minor_);
    }
  }
  if (!manifest_parity.empty()) CLIO_CO_RETURN;

  std::vector<MemberBdevDesc> cfg;
  for (const auto &m : params.members_) {
    if (m.parity_) cfg.push_back(m);
  }
  if (cfg.size() > max_failures_) {
    HLOG(kError, "safe_bdev Create: {} members are marked parity but "
         "max_failures is {}", cfg.size(), max_failures_);
    rc = 3;
    CLIO_CO_RETURN;
  }
  bool any_fresh = false;
  for (size_t i = 0; i < cfg.size(); ++i) {
    bool fresh = false;
    CLIO_CO_AWAIT(SeatConfigParity(cfg[i], rc, fresh));
    if (rc != 0) CLIO_CO_RETURN;
    any_fresh = any_fresh || fresh;
  }
  if (any_fresh) {
    // A parity disk stamped just now holds no parity for data the array
    // already has (restart with parity newly configured): rebuild it.
    std::lock_guard<std::mutex> g(slot_mu_);
    for (clio::run::u64 s : written_slots_) {
      dirty_slots_.insert(s);
      ++slot_gen_[s];
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::FaultMembersOnDeadNodes() {
  if (!distributed_) return;
  auto *ipc = CLIO_IPC;
  bool changed = false;
  auto check = [&](MemberSlot &m, const char *kind, size_t i) {
    if (m.state_ != ec::EcState::kActive || ipc->IsAlive(m.node_id_)) return;
    m.state_ = ec::EcState::kFaulty;
    changed = true;
    HLOG(kWarning, "safe_bdev: {} member {} ('{}') is on dead node {}; "
         "marked faulty", kind, i, m.pool_name_, m.node_id_);
  };
  for (size_t i = 0; i < data_members_.size(); ++i) {
    check(data_members_[i], "data", i);
  }
  for (size_t i = 0; i < parity_members_.size(); ++i) {
    check(parity_members_[i], "parity", i);
  }
  if (changed) PersistMemberManifest();
}

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN

  CreateParams params = task->GetParams();

  // A distributed array runs on its first member's node; the pool's other
  // containers stay passive (their nodes' disks are reached from the home).
  distributed_ = params.distributed_;
  is_home_ = true;
  if (distributed_ && !params.members_.empty()) {
    auto *ipc = CLIO_IPC;
    is_home_ = ipc->GetNodeId() == params.members_[0].node_id_;
    if (!is_home_) {
      HLOG(kInfo, "safe_bdev Create: distributed array '{}' runs on node {}; "
           "this container is passive", task->pool_name_.str(),
           params.members_[0].node_id_);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  max_failures_ = params.max_failures_;
  parity_level_ = 0;
  rr_cursor_ = 0;
  data_members_.clear();
  parity_members_.clear();
  data_clients_.clear();
  parity_clients_.clear();
  data_alloc_.clear();
  data_members_.reserve(kMaxMembers);
  data_clients_.reserve(kMaxMembers);
  data_alloc_.reserve(kMaxMembers);
  parity_members_.reserve(kMaxMembers);
  parity_clients_.reserve(kMaxMembers);
  rs_cache_.clear();
  reattached_members_ = 0;
  {
    std::lock_guard<std::mutex> g(slot_mu_);
    dirty_slots_.clear();
    written_slots_.clear();
  }
  array_high_water_ = 0;

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
  if (have_manifest) {
    for (const auto &e : manifest) {
      if (e.role_ == 1) eff_parity.push_back(e);
    }
    std::sort(eff_parity.begin(), eff_parity.end(),
              [](const MemberManifestEntry &a, const MemberManifestEntry &b) {
                return a.index_ < b.index_;
              });
    HLOG(kInfo, "safe_bdev Create: manifest '{}' restores {} parity member(s)",
         members_manifest_path_, eff_parity.size());
  }
  const std::vector<DataSeatSpec> plan =
      BuildDataMemberPlan(params, have_manifest ? manifest
                                                : std::vector<MemberManifestEntry>());

  const int k0 = static_cast<int>(plan.size());
  if (k0 <= 0) {
    HLOG(kError, "safe_bdev Create: no member bdevs supplied");
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  if (static_cast<clio::run::u64>(k0) + max_failures_ > 256) {
    // The fixed-width Cauchy code needs N + m distinct GF(2^8) points.
    HLOG(kError, "safe_bdev Create: {} data members + max_failures {} > 256",
         k0, max_failures_);
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
  intent_sync_ = params.intent_sync_;
  if (alloc_log_.enabled()) StartIntentSync();
  const bool recovered =
      alloc_log_.enabled() && !alloc_log_.groups().empty();

  // Seat each data column (see BuildDataMemberPlan / SeatDataMember).
  for (size_t col = 0; col < plan.size(); ++col) {
    clio::run::u32 seat_rc = 0;
    CLIO_CO_AWAIT(SeatDataMember(plan[col], static_cast<int>(col), seat_rc));
    if (seat_rc != 0) {
      task->return_code_ = seat_rc;
      CLIO_CO_RETURN;
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
        if (b.size < kChunkLen) data_alloc_[d].part_len_[s] = b.size;
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
    RestoreParityCoverage();
    HLOG(kInfo,
         "safe_bdev Create: RECOVERED slot allocators from alloc log '{}' "
         "({} live chunks across {} data members)",
         params.alloc_log_path_, total_live, data_alloc_.size());
  } else {
    // FRESH: register the single allocator group so a later restart recovers.
    alloc_log_.LogGroupOpen(kAllocGroup, /*k=*/0, /*first_row=*/0,
                            /*num_rows=*/0, /*logical_base=*/0);
    alloc_log_.LogGroupOpen(kFormatGroup, kParityFormat, 0, 0, 0);
    HLOG(kInfo,
         "safe_bdev Create: pool='{}', k0={}, M={} (dynamic view-group: "
         "per-chunk round-robin + dedicated parity)",
         task->pool_name_.str(), k0, max_failures_);
  }

  // Seat the PARITY columns: the manifest's (runtime-added, recovered,
  // faulty), else the configured `parity: true` members.
  {
    clio::run::u32 par_rc = 0;
    CLIO_CO_AWAIT(SeatParityMembers(params, eff_parity, par_rc));
    if (par_rc != 0) {
      task->return_code_ = par_rc;
      CLIO_CO_RETURN;
    }
  }
  parity_level_ = static_cast<clio::run::u32>(parity_members_.size());
  for (size_t j = 0; j < parity_members_.size(); ++j) {
    clio::run::u64 slots = 0;
    bool qok = false;
    CLIO_CO_AWAIT(QueryMemberSlots(parity_clients_[j], ParityQuery(j), slots,
                                   qok));
    if (qok) parity_members_[j].cap_slots_ = slots;
  }
  ClampDataCaps();

  // Stripes whose parity was being (re)built when the array went down are
  // still in the intent log (#1121): their data may have reached a member
  // without the matching parity, so they are dirty -- degraded reads refuse
  // them rather than decode from stale parity -- until the builder below
  // re-encodes them. Every other stripe's persisted parity is current.
  const size_t replayed = ReplayStripeIntents();
  if (replayed != 0) {
    HLOG(kWarning, "safe_bdev Create: {} stripe(s) were mid-write at the last "
         "shutdown; re-encoding their parity", replayed);
  }

  if (alloc_log_.enabled()) {
    client_.AsyncFlushAllocLog(SelfQuery(), kFlushAllocLogPeriodUs);
  }
  client_.AsyncBuildParity(SelfQuery(), /*max_batch=*/0,
                           /*period_us=*/kBuildParityPeriodUs);

  CLIO_CO_AWAIT(ResumeRecoveries());
  CompactMemberManifest();

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::RestoreParityCoverage() {
  clio::run::u32 format = 0;
  for (const auto &g : alloc_log_.groups()) {
    if (g.group_id == kHighWaterGroup) {
      array_high_water_ = std::max(array_high_water_, g.first_row);
    } else if (g.group_id == kFormatGroup) {
      format = g.k;
    }
  }
  for (const auto &a : data_alloc_) {
    array_high_water_ = std::max(array_high_water_, a.high_water_);
  }
  std::lock_guard<std::mutex> g(slot_mu_);
  for (clio::run::u64 s = 0; s < array_high_water_; ++s) {
    written_slots_.insert(s);
  }
  if (format < kParityFormat) {
    // Parity written under the variable-width code: none of it is valid
    // for the fixed-width one. Re-encode every slot before trusting any.
    for (clio::run::u64 s : written_slots_) {
      dirty_slots_.insert(s);
      ++slot_gen_[s];
    }
    alloc_log_.LogGroupOpen(kFormatGroup, kParityFormat, 0, 0, 0);
    alloc_log_.LogGroupOpen(kHighWaterGroup, 0, array_high_water_, 0, 0);
    HLOG(kWarning, "safe_bdev Create: parity predates the fixed-width "
         "format; re-encoding {} slot(s)", written_slots_.size());
  }
}

clio::run::TaskResume Runtime::AllocateBlocks(
    clio::run::shared_ptr<AllocateBlocksTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "AllocateBlocks");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
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

  // Guards data_alloc_/rr_cursor_/alloc_log_ against the other workers that can
  // be executing this container concurrently. No co_await below, so a plain
  // mutex is safe here.
  std::lock_guard<std::mutex> alloc_g(alloc_mu_);
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
    if (s + 1 > array_high_water_) {
      array_high_water_ = s + 1;
      alloc_log_.LogGroupOpen(kHighWaterGroup, 0, array_high_water_, 0, 0);
    }
    const clio::run::u64 seg = std::min<clio::run::u64>(kChunkLen, remaining);
    if (seg < kChunkLen) data_alloc_[d].part_len_[s] = seg;
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
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "FreeBlocks");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
  // Decode each block to (member d, slot s) and release the slot on that
  // member. Parity is untouched (#1126): it covers the column's physical
  // bytes, allocated or not, and freeing changes none of them -- the next
  // write to the slot updates parity from whatever the chunk then holds.
  // Same allocator lock as AllocateBlocks (see alloc_mu_).
  std::lock_guard<std::mutex> alloc_g(alloc_mu_);
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
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LoadDegradedStripe(clio::run::u64 s,
                                                  DegradedStripe &out,
                                                  bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  if (IsSlotDirty(s)) {
    HLOG(kError, "safe_bdev Write: slot {} has a down member and stale "
         "parity; refusing the write (it could not be recovered)", s);
    CLIO_CO_RETURN;
  }
  out.members = CodeColumns();
  const std::vector<int> none;
  CLIO_CO_AWAIT(ReconstructStripe(s, out.members, none, out.chunks, ok));
  if (!ok) {
    HLOG(kError, "safe_bdev Write: cannot reconstruct slot {} for a write to "
         "a down member", s);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StoreDegradedParity(clio::run::u64 s,
                                                   const DegradedStripe &st,
                                                   bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  std::vector<const uint8_t *> ptrs(st.chunks.size());
  for (size_t i = 0; i < st.chunks.size(); ++i) ptrs[i] = st.chunks[i].data();
  ec::ReedSolomon *codec = GetCodec(static_cast<int>(st.members.size()));
  auto *ipc = CLIO_IPC;
  clio::run::u32 written = 0;
  for (int j = 0; j < static_cast<int>(parity_level_); ++j) {
    if (parity_members_[static_cast<size_t>(j)].state_ !=
        ec::EcState::kActive) {
      continue;
    }
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kChunkLen);
    if (buf.IsNull()) CLIO_CO_RETURN;
    codec->EncodeParityShard(j, ptrs, kChunkLen,
                             reinterpret_cast<uint8_t *>(buf.ptr_));
    auto fut = parity_clients_[static_cast<size_t>(j)].AsyncWrite(
        ParityQuery(static_cast<size_t>(j)), MemberBlocks(SlotPhysOffset(s), kChunkLen),
        buf.shm_.template Cast<void>(), kChunkLen);
    CLIO_CO_AWAIT(fut);
    const bool wok =
        fut->return_code_ == 0 && fut->bytes_written_ == kChunkLen;
    FaultOnIoError(/*is_parity=*/true, static_cast<size_t>(j), !wok,
                   fut->io_error_);
    ipc->FreeBuffer(buf);
    if (!wok) {
      // A parity disk that just failed is out of the stripe; one that is
      // still active holds a stale shard, which the stripe cannot carry.
      if (parity_members_[static_cast<size_t>(j)].state_ ==
          ec::EcState::kActive) {
        CLIO_CO_RETURN;
      }
      continue;
    }
    ++written;
  }
  // The stripe's down data chunks exist only in the parity: it needs at
  // least one written shard per down member to give them back.
  ok = written > 0 && written >= DownInStripe(st);
  if (!ok) {
    HLOG(kError, "safe_bdev Write: slot {} has {} down data member(s) but "
         "only {} parity shard(s) could be written; refusing the write",
         s, DownInStripe(st), written);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::vector<Runtime::WritePiece> Runtime::SplitWrite(
    const WriteTask &task) const {
  std::vector<WritePiece> pieces;
  clio::run::u64 buf_pos = 0;
  for (size_t bi = 0; bi < task.blocks_.size(); ++bi) {
    const clio::run::u64 off = task.blocks_[bi].offset_;
    const clio::run::u64 len = task.blocks_[bi].size_;
    for (clio::run::u64 cur = off; cur < off + len;) {
      WritePiece p;
      clio::run::u32 d = 0;
      Unband(cur, d, p.slot, p.within);
      p.member = d;
      p.len = std::min<clio::run::u64>(off + len - cur, kChunkLen - p.within);
      p.buf_off = buf_pos + (cur - off);
      pieces.push_back(p);
      cur += p.len;
    }
    buf_pos += len;
  }
  return pieces;
}

void Runtime::OverlayPieces(clio::run::u64 s,
                            const std::vector<WritePiece> &pieces,
                            const char *data, DegradedStripe &st) {
  for (const WritePiece &p : pieces) {
    if (p.slot != s) continue;
    for (size_t pos = 0; pos < st.members.size(); ++pos) {
      if (st.members[pos] == static_cast<int>(p.member)) {
        std::memcpy(st.chunks[pos].data() + p.within, data + p.buf_off, p.len);
      }
    }
  }
}

bool Runtime::DownBytesClobbered(clio::run::u64 s, const DegradedStripe &st,
                                 const std::vector<clio::run::u64> &need_len,
                                 const std::vector<WritePiece> &pieces) const {
  std::vector<uint8_t> landed(kChunkLen, 0);
  for (const WritePiece &p : pieces) {
    if (p.slot == s && DataActive(p.member)) {
      std::fill_n(landed.begin() + p.within, p.len, 1);
    }
  }
  for (size_t pos = 0; pos < st.members.size(); ++pos) {
    const int d = st.members[pos];
    if (DataActive(static_cast<size_t>(d))) continue;
    std::vector<uint8_t> need(kChunkLen, 0);
    std::fill_n(need.begin(), need_len[pos], 1);
    for (const WritePiece &p : pieces) {
      if (p.slot == s && p.member == static_cast<clio::run::u32>(d)) {
        std::fill_n(need.begin() + p.within, p.len, 0);
      }
    }
    for (clio::run::u64 x = 0; x < kChunkLen; ++x) {
      if (need[x] != 0 && landed[x] != 0) {
        HLOG(kError, "safe_bdev Write: slot {}: data member {} failed after "
             "this write changed bytes its parity decode needs", s, d);
        return true;
      }
    }
  }
  return false;
}

clio::run::TaskResume Runtime::RetryStripeDegraded(
    clio::run::u64 s, const std::vector<WritePiece> &pieces, const char *data,
    bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  if (IsSlotDirty(s)) {
    HLOG(kError, "safe_bdev Write: slot {} lost a member mid-write while its "
         "parity was stale; refusing the write", s);
    CLIO_CO_RETURN;
  }
  DegradedStripe st;
  std::vector<clio::run::u64> need_len;  // per stripe position
  {
    std::lock_guard<std::mutex> g(alloc_mu_);
    st.members = CodeColumns();
    for (int d : st.members) {
      // An unallocated chunk holds nothing anyone will read back.
      const auto &a = data_alloc_[static_cast<size_t>(d)];
      need_len.push_back(a.live_.count(s) != 0 ? a.LiveLen(s) : 0);
    }
  }
  // Members this write already changed hold NEW bytes, but the parity still
  // encodes their OLD ones. Treat them as erasures too: when the parity
  // covers them and the down members, the old stripe decodes exactly.
  std::vector<int> landed;
  for (const WritePiece &p : pieces) {
    const int d = static_cast<int>(p.member);
    if (p.slot == s && DataActive(p.member) &&
        std::find(landed.begin(), landed.end(), d) == landed.end()) {
      landed.push_back(d);
    }
  }
  bool rok = false;
  CLIO_CO_AWAIT(ReconstructStripe(s, st.members, landed, st.chunks, rok));
  if (!rok) {
    // Too many erasures for that. Decode from the survivors as they are now:
    // exact wherever this write left the survivors untouched, which is all
    // that matters unless a down member needs a byte the write overwrote.
    const std::vector<int> none;
    CLIO_CO_AWAIT(ReconstructStripe(s, st.members, none, st.chunks, rok));
    if (!rok) {
      HLOG(kError, "safe_bdev Write: slot {} lost a member mid-write and "
           "cannot be reconstructed (too many members down)", s);
      CLIO_CO_RETURN;
    }
    if (DownBytesClobbered(s, st, need_len, pieces)) CLIO_CO_RETURN;
  }
  OverlayPieces(s, pieces, data, st);
  CLIO_CO_AWAIT(StoreDegradedParity(s, st, ok));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::BlocksInRange(const WriteTask &task) const {
  for (size_t bi = 0; bi < task.blocks_.size(); ++bi) {
    clio::run::u32 d = 0;
    clio::run::u64 s0 = 0, within = 0;
    Unband(task.blocks_[bi].offset_, d, s0, within);
    if (task.blocks_[bi].size_ != 0 && d >= data_members_.size()) {
      HLOG(kError, "safe_bdev Write: block offset {} decodes to member {} out "
           "of range", task.blocks_[bi].offset_, d);
      return false;
    }
  }
  return true;
}

bool Runtime::DispatchMemberWrites(const WriteTask &task, const char *data,
                                   MemberWrites &mw) {
  auto *ipc = CLIO_IPC;
  clio::run::u64 buf_pos = 0;
  for (size_t bi = 0; bi < task.blocks_.size(); ++bi) {
    const clio::run::u64 off = task.blocks_[bi].offset_;
    const clio::run::u64 len = task.blocks_[bi].size_;
    if (len == 0) continue;
    clio::run::u32 d = 0;
    clio::run::u64 s0 = 0, within = 0;
    Unband(off, d, s0, within);
    for (clio::run::u64 c = off / kChunkLen; c <= (off + len - 1) / kChunkLen;
         ++c) {
      mw.touched.insert(c % kSlotsPerMember);
    }
    if (DataActive(d)) {
      ctp::ipc::FullPtr<char> seg = ipc->AllocateBuffer(len);
      if (seg.IsNull()) return false;
      std::memcpy(seg.ptr_, data + buf_pos, len);
      mw.bufs.push_back(seg);
      mw.members.push_back(d);
      mw.futs.push_back(data_clients_[d].AsyncWrite(
          DataQuery(d), MemberBlocks(SlotPhysOffset(s0) + within, len),
          seg.shm_.template Cast<void>(), len));
    }
    buf_pos += len;
    mw.bytes = buf_pos;
  }
  return true;
}

clio::run::TaskResume Runtime::AwaitMemberWrites(MemberWrites &mw,
                                                 bool &any_failed, bool &ok) {
  CLIO_TASK_BODY_BEGIN
  // A member write is done only when all its bytes are. A member that fails
  // is faulted (unless the error was transient); the caller redoes the
  // stripes it shares with the write degraded.
  for (size_t i = 0; i < mw.futs.size(); ++i) {
    auto &f = mw.futs[i];
    CLIO_CO_AWAIT(f);
    if (f->return_code_ == 0 && f->bytes_written_ == f->length_) continue;
    const size_t d = mw.members[i];
    FaultOnIoError(/*is_parity=*/false, d, true, f->io_error_);
    if (DataActive(d)) {
      HLOG(kWarning, "safe_bdev Write: member {} write returned rc={} with "
           "{} of {} bytes (transient)", d, f->return_code_,
           f->bytes_written_, f->length_);
      ok = false;
    } else {
      any_failed = true;
    }
  }
  auto *ipc = CLIO_IPC;
  for (auto &b : mw.bufs) ipc->FreeBuffer(b);
  mw.bufs.clear();
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Write(clio::run::shared_ptr<WriteTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "Write");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
  task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kOk);
  if (data_members_.empty() || task->blocks_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  if (!BlocksInRange(*task)) {
    task->bytes_written_ = 0;
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> data =
      ipc->ToFullPtr(task->data_).template Cast<char>();
  const std::vector<WritePiece> pieces = SplitWrite(*task);
  std::set<clio::run::u64> slots;
  for (const WritePiece &p : pieces) slots.insert(p.slot);

  // Synchronous parity (#1121): hold every stripe this write touches from
  // before its data lands until its parity does, and log them dirty (fsynced)
  // first. The write is acked only once each stripe's parity is current, so
  // a disk dying at any later moment can always rebuild it; a crash before
  // that leaves the stripes in the intent log for Create to re-encode.
  // The intent goes first and outside the stripe locks, so concurrent
  // writers -- which share stripes, allocation being round-robin -- can
  // share its fsync (AwaitIntentDurable); it is still durable before any
  // data of this write reaches a member.
  std::vector<IntentKey> intents;
  const clio::run::u64 intent_seq = LogDirtyIntents(slots, intents);
  CLIO_CO_AWAIT(AwaitIntentDurable(intent_seq));
  CLIO_CO_AWAIT(LockStripes(slots));
  const clio::run::u64 watermark = IntentWatermark();
  bool ok = false;
  std::set<clio::run::u64> clean;
  CLIO_CO_AWAIT(WriteStripes(task, pieces, data.ptr_, ok, clean));
  UnlockStripes(slots);
  LogCleanIntents(intents, clean);
  LogStripesEncoded(clean, watermark);  // older stale intents it settled
  if (!ok) {
    task->bytes_written_ = 0;
    task->return_code_ = 1;
    task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kDeviceFault);
    CLIO_CO_RETURN;
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WriteStripes(
    clio::run::shared_ptr<WriteTask> &task,
    const std::vector<WritePiece> &pieces, const char *data, bool &ok,
    std::set<clio::run::u64> &clean) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  clean.clear();
  // Stripes with a DOWN data member, reconstructed before any member write
  // lands (its parity is still consistent then) and overlaid with every byte
  // this write puts in them (see DegradedStripe). A stripe counts when ANY of
  // its data members is down, not only when this write lands on the down one.
  std::map<clio::run::u64, DegradedStripe> degraded;
  std::set<clio::run::u64> down_slots;
  {
    std::lock_guard<std::mutex> g(alloc_mu_);
    for (const WritePiece &p : pieces) {
      if (StripeHasDownMember(p.slot)) down_slots.insert(p.slot);
    }
  }
  for (clio::run::u64 ds : down_slots) {
    bool lok = false;
    CLIO_CO_AWAIT(LoadDegradedStripe(ds, degraded[ds], lok));
    if (!lok) CLIO_CO_RETURN;
  }
  for (auto &kv : degraded) OverlayPieces(kv.first, pieces, data, kv.second);
  // Healthy stripes whose parity encodes exactly their current members get
  // their parity updated from the change alone (DeltaEncodeStripe): keep the
  // bytes this write replaces, read before it lands.
  std::set<clio::run::u64> delta_slots;
  std::vector<std::vector<uint8_t>> old_bytes;
  if (degraded.empty()) {
    CLIO_CO_AWAIT(ReadReplacedBytes(pieces, delta_slots, old_bytes));
  }

  // One AsyncWrite per block to a live member, all dispatched, then awaited.
  // A down member's bytes live in the parity.
  MemberWrites mw;
  bool wok = DispatchMemberWrites(*task, data, mw);
  bool any_failed = false;
  CLIO_CO_AWAIT(AwaitMemberWrites(mw, any_failed, wok));
  const std::set<clio::run::u64> &touched = mw.touched;

  // Stripes that gained a down member under this write are redone degraded.
  std::set<clio::run::u64> retried;
  if (any_failed || CountDownMembers() != 0) {
    std::lock_guard<std::mutex> g(alloc_mu_);
    for (clio::run::u64 s : touched) {
      if (degraded.count(s) == 0 && StripeHasDownMember(s)) retried.insert(s);
    }
  }
  for (clio::run::u64 s : retried) {
    if (!wok) break;
    CLIO_CO_AWAIT(RetryStripeDegraded(s, pieces, data, wok));
    if (wok) clean.insert(s);
  }
  for (auto &kv : degraded) {
    if (!wok) break;
    CLIO_CO_AWAIT(StoreDegradedParity(kv.first, kv.second, wok));
    if (wok) clean.insert(kv.first);
  }
  if (!wok) {
    // Whatever landed made these stripes' parity stale: dirty them all, so a
    // down member's chunk there is refused rather than decoded wrong.
    for (clio::run::u64 s : touched) MarkSlotDirty(s);
    for (auto &kv : degraded) MarkSlotDirty(kv.first);
    clean.clear();
    CLIO_CO_RETURN;
  }
  // Healthy stripes: encode their parity now, before the write is acked.
  // (Degraded and retried stripes came out with current parity above.)
  if (FaultSkipParity()) {
    // Test fault: behave as if the process died right here -- data landed,
    // parity did not, and the intent log still says so.
    for (clio::run::u64 s : touched) MarkSlotDirty(s);
    task->bytes_written_ = mw.bytes;
    ok = true;
    CLIO_CO_RETURN;
  }
  for (clio::run::u64 s : touched) {
    if (degraded.count(s) != 0 || retried.count(s) != 0) continue;
    NoteSlotWritten(s);
    bool eok = false;
    if (delta_slots.count(s) != 0) {
      CLIO_CO_AWAIT(DeltaEncodeStripe(s, pieces, data, old_bytes, eok));
    }
    if (!eok) CLIO_CO_AWAIT(EncodeStripe(s, eok));
    if (!eok && StripeHasDownMemberLocked(s)) {
      // A member went down between the data landing and this encode: redo
      // the stripe degraded, as for a member write that failed outright.
      CLIO_CO_AWAIT(RetryStripeDegraded(s, pieces, data, eok));
      if (!eok) {
        MarkSlotDirty(s);  // its lost chunk is refused, never decoded wrong
        CLIO_CO_RETURN;    // the write fails: not all of it is protected
      }
    } else if (!eok) {
      // Members all up but the encode failed (e.g. no SHM buffer): the data
      // is intact on every member; the background builder retries.
      MarkSlotDirty(s);
      continue;
    }
    clean.insert(s);
  }
  task->bytes_written_ = mw.bytes;
  ok = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadBlockDegraded(clio::run::u64 off,
                                                 clio::run::u64 len, char *dst,
                                                 bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  const clio::run::u64 hi = off + len;
  clio::run::u64 cur = off;
  while (cur < hi) {
    clio::run::u32 dd = 0;
    clio::run::u64 s = 0, within = 0;
    Unband(cur, dd, s, within);
    const clio::run::u64 chunk_base = (cur / kChunkLen) * kChunkLen;
    const clio::run::u64 seg_end =
        std::min<clio::run::u64>(hi, chunk_base + kChunkLen);
    if (IsSlotDirty(s)) {
      HLOG(kError,
           "safe_bdev Read: slot {} is dirty (parity not built) and data "
           "member {} is down — cannot reconstruct",
           s, dd);
      CLIO_CO_RETURN;
    }
    const std::vector<int> stripe = CodeColumns();
    const int pos = static_cast<int>(dd) < static_cast<int>(stripe.size())
                        ? static_cast<int>(dd)
                        : -1;
    std::vector<std::vector<uint8_t>> chunks;
    bool rec_ok = false;
    if (pos >= 0) {
      const std::vector<int> excl{static_cast<int>(dd)};
      CLIO_CO_AWAIT(ReconstructStripe(s, stripe, excl, chunks, rec_ok));
    }
    if (!rec_ok || pos < 0) {
      HLOG(kError, "safe_bdev Read: cannot reconstruct slot {} of down data "
           "member {} (too many members down)", s, dd);
      CLIO_CO_RETURN;
    }
    std::memcpy(dst + (cur - off),
                chunks[static_cast<size_t>(pos)].data() + within,
                seg_end - cur);
    cur = seg_end;
  }
  ok = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Read(clio::run::shared_ptr<ReadTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "Read");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
  task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kOk);
  if (data_members_.empty() || task->blocks_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> data =
      ipc->ToFullPtr(task->data_).template Cast<char>();

  clio::run::u64 buf_pos = 0;
  for (size_t bi = 0; bi < task->blocks_.size(); ++bi) {
    const clio::run::u64 off = task->blocks_[bi].offset_;
    const clio::run::u64 len = task->blocks_[bi].size_;
    if (len == 0) {
      continue;
    }
    clio::run::u32 d = 0;
    clio::run::u64 s0 = 0, within0 = 0;
    Unband(off, d, s0, within0);
    bool seg_ok = false;
    if (d >= data_members_.size()) {
      HLOG(kError, "safe_bdev Read: block offset {} decodes to member {} out of "
           "range", off, d);
    } else {
      if (DataActive(d)) {
        // Healthy: one contiguous read from member d.
        CLIO_CO_AWAIT(ReadDataSegment(
            static_cast<size_t>(d), SlotPhysOffset(s0) + within0,
            reinterpret_cast<uint8_t *>(data.ptr_) + buf_pos, len, seg_ok));
      }
      // Member d is down -- or just failed this read and was faulted:
      // reconstruct the block from its stripes' survivors + parity.
      if (!seg_ok && !DataActive(d)) {
        CLIO_CO_AWAIT(ReadBlockDegraded(off, len, data.ptr_ + buf_pos, seg_ok));
        if (!seg_ok) {
          task->io_error_ =
              static_cast<clio::run::u32>(ctp::IoError::kDeviceFault);
        }
      }
    }
    if (!seg_ok) {
      task->bytes_read_ = buf_pos;
      task->length_ = buf_pos;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    buf_pos += len;
  }

  task->bytes_read_ = buf_pos;
  task->length_ = buf_pos;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::ArrayCapacity(clio::run::u64 *total,
                            clio::run::u64 *remaining) {
  *total = 0;
  *remaining = 0;
  // data_alloc_ is mutated by concurrent AllocateBlocks/FreeBlocks.
  std::lock_guard<std::mutex> alloc_g(alloc_mu_);
  for (size_t d = 0; d < data_alloc_.size(); ++d) {
    if (data_members_[d].state_ == ec::EcState::kActive) {
      *remaining += data_alloc_[d].RemainingSlots() * kChunkLen;
      *total += data_alloc_[d].cap_slots_ * kChunkLen;
    } else {
      // Its data is still stored (reconstructable): it stays "used", and
      // only its free space leaves the array.
      *total += data_alloc_[d].live_.size() * kChunkLen;
    }
  }
}

clio::run::TaskResume Runtime::GetStats(clio::run::shared_ptr<GetStatsTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // The array's usable capacity (after parity: data members only, each
  // clamped to what the parity members cover) and what is left of it. A
  // down member keeps only its live chunks in the total (they are still
  // stored, served degraded) and none of its free space: it takes no
  // allocations. A passive container of a distributed array holds
  // none of it and answers zeros (callers count the home's answer).
  clio::run::u64 remaining = 0;
  clio::run::u64 total = 0;
  if (IsHome()) {
    FaultMembersOnDeadNodes();
    ArrayCapacity(&total, &remaining);
  }
  task->remaining_size_ = remaining;
  task->total_size_ = total;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AddBdev(clio::run::shared_ptr<AddBdevTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "AddBdev");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();

  const bool as_parity = (task->as_parity_ != 0);
  if ((as_parity ? parity_members_.size() : data_members_.size()) >=
          kMaxMembers ||
      (!as_parity && data_members_.size() + 1 + max_failures_ > 256)) {
    HLOG(kError, "safe_bdev AddBdev: the array already has {} {} members",
         kMaxMembers, as_parity ? "parity" : "data");
    task->return_code_ = 4;
    CLIO_CO_RETURN;
  }

  if (!as_parity) {
    clio::run::u32 lowest_ttl = 7;
    int failing_data_idx = -1;
    int failing_parity_idx = -1;

    for (size_t i = 0; i < data_clients_.size(); ++i) {
      if (data_members_[i].state_ != ec::EcState::kActive) continue;
      auto stats = data_clients_[i].AsyncGetStats(DataQuery(i));
      CLIO_CO_AWAIT(stats);
      if (stats->return_code_ == 0 && stats->predicted_ttl_days_ < lowest_ttl) {
        lowest_ttl = stats->predicted_ttl_days_;
        failing_data_idx = static_cast<int>(i);
        failing_parity_idx = -1;
      }
    }
    for (size_t i = 0; i < parity_clients_.size(); ++i) {
      if (parity_members_[i].state_ != ec::EcState::kActive) continue;
      auto stats = parity_clients_[i].AsyncGetStats(ParityQuery(i));
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
          SelfQuery(), failed_pool_id, task->pool_name_.str(),
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
    MemberSlot probe;
    probe.node_id_ = task->node_id_;
    clio::run::u64 cap_slots = 0;
    bool qok = false;
    CLIO_CO_AWAIT(QueryMemberSlots(data_clients_.back(), MemberQuery(probe),
                                   cap_slots, qok));
    if (!qok) {
      HLOG(kError, "safe_bdev AddBdev: GetStats failed for data member '{}'",
           task->pool_name_.str());
      data_clients_.pop_back();
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    // A data slot needs its parity slot on every parity member.
    cap_slots = std::min(cap_slots, MinParityCap());

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
        // The member cannot be written: refuse it, as for the other
        // unusable-member cases above, instead of seating a column whose
        // every write would fail.
        HLOG(kError,
             "safe_bdev AddBdev: REFUSING data member '{}' -- its superblock "
             "cannot be written",
             task->pool_name_.str());
        data_members_.pop_back();
        data_alloc_.pop_back();
        data_clients_.pop_back();
        task->return_code_ = 3;
        CLIO_CO_RETURN;
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
  clio::run::u64 par_slots = 0;
  {
    MemberSlot probe;
    probe.node_id_ = task->node_id_;
    bool qok = false;
    CLIO_CO_AWAIT(QueryMemberSlots(parity_clients_.back(), MemberQuery(probe),
                                   par_slots, qok));
    if (!qok || par_slots < MaxDataHighWater()) {
      // Parity for slots already written would land past its end.
      HLOG(kError, "safe_bdev AddBdev: parity member '{}' holds {} slots, "
           "the data already uses {}; refused", task->pool_name_.str(),
           par_slots, MaxDataHighWater());
      parity_clients_.pop_back();
      task->return_code_ = 5;
      CLIO_CO_RETURN;
    }
  }
  MemberSlot slot;
  slot.pool_id_ = task->member_pool_id_;
  slot.pool_name_ = task->pool_name_.str();
  slot.node_id_ = task->node_id_;
  slot.role_ = ec::EcRole::kParity;
  slot.state_ = ec::EcState::kActive;
  slot.index_ = static_cast<int>(parity_level_);
  slot.cap_slots_ = par_slots;
  const size_t new_j = parity_members_.size();
  parity_members_.push_back(slot);
  ++parity_level_;
  {
    std::lock_guard<std::mutex> g(alloc_mu_);
    ClampDataCaps();  // no data slot may outgrow this parity member
  }

  {
    std::lock_guard<std::mutex> g(slot_mu_);
    for (clio::run::u64 s : written_slots_) {
      dirty_slots_.insert(s);
      ++slot_gen_[s];
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
      HLOG(kError,
           "safe_bdev AddBdev: superblock write failed for new parity member "
           "'{}'; seated faulty",
           task->pool_name_.str());
      parity_members_[new_j].state_ = ec::EcState::kFaulty;
      PersistMemberManifest();
      task->return_code_ = 3;
      CLIO_CO_RETURN;
    }
  }

  PersistMemberManifest();
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RemoveBdev(clio::run::shared_ptr<RemoveBdevTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "RemoveBdev");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
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
    // Every column, allocated or not (fixed-width parity, #1126).
    const std::vector<int> stripe = CodeColumns();
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
        const std::vector<int> excl{idx};
        CLIO_CO_AWAIT(ReconstructStripe(s, stripe, excl, chunks, rec_ok));
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
    const clio::run::PoolQuery q =
        is_data ? DataQuery(static_cast<size_t>(idx))
                : ParityQuery(static_cast<size_t>(idx));
    auto fut = client.AsyncWrite(q, MemberBlocks(SlotPhysOffset(s), kChunkLen),
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
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "RecoverBdev");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();

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

  // Size the replacement BEFORE seating it: it must hold every slot the
  // member already used (a data member's own, or -- for parity -- any data
  // member's), or rebuilt chunks would land past its end.
  clio::run::u64 new_slots = 0;
  {
    MemberSlot probe;
    probe.node_id_ = task->node_id_;
    bool qok = false;
    CLIO_CO_AWAIT(QueryMemberSlots(clio::run::bdev::Client(task->new_pool_id_),
                                   MemberQuery(probe), new_slots, qok));
    const clio::run::u64 need =
        is_data ? data_alloc_[static_cast<size_t>(idx)].high_water_
                : MaxDataHighWater();
    if (!qok || new_slots < need) {
      HLOG(kError, "safe_bdev RecoverBdev: replacement '{}' holds {} slots, "
           "the member needs {}; refused", task->pool_name_.str(), new_slots,
           need);
      task->return_code_ = 5;
      CLIO_CO_RETURN;
    }
  }

  auto &client = is_data ? data_clients_[static_cast<size_t>(idx)]
                         : parity_clients_[static_cast<size_t>(idx)];
  client = clio::run::bdev::Client(task->new_pool_id_);
  {
    // The member now has the replacement's capacity, not the old disk's.
    std::lock_guard<std::mutex> g(alloc_mu_);
    if (is_data) {
      data_alloc_[static_cast<size_t>(idx)].cap_slots_ = new_slots;
    } else {
      parity_members_[static_cast<size_t>(idx)].cap_slots_ = new_slots;
    }
    ClampDataCaps();
  }
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
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "BuildParity");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
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

  clio::run::u32 built = 0;
  if (FaultSkipParity()) batch.clear();  // test fault: the "crash" pending
  for (clio::run::u64 s : batch) {
    // Wait out whoever holds the stripe (a Write encodes its own; another
    // builder pass may be mid-encode) -- skipping it would let a drain
    // return while the stripe is still unprotected. Then encode only if it
    // is still dirty.
    const std::set<clio::run::u64> one{s};
    CLIO_CO_AWAIT(LockStripes(one));
    const clio::run::u64 wm = IntentWatermark();
    bool eok = !IsSlotDirty(s);
    if (!eok) CLIO_CO_AWAIT(EncodeStripe(s, eok));
    UnlockStripes(one);
    if (!eok) continue;  // a member is down or I/O failed: stays dirty
    LogStripesEncoded(one, wm);
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
    // 18 = the 17 scalar stats below plus the trailing "members" array. The
    // count was 14 while 15 pairs were packed, so a spec-conforming unpacker
    // (msgpack::unpack reads exactly the declared map) silently DROPPED the
    // members roster -- found by the dashboard's member-table test (#990).
    clio::run::u64 cap_total = 0, cap_remaining = 0;
    if (IsHome()) ArrayCapacity(&cap_total, &cap_remaining);
    pk.pack_map(18);
    pk.pack("pool_name");     pk.pack(pool_name_);
    pk.pack("max_failures");  pk.pack(max_failures_);
    pk.pack("data_count");
    pk.pack(static_cast<clio::run::u32>(data_members_.size()));
    pk.pack("parity_level");  pk.pack(parity_level_);
    pk.pack("total_slots");   pk.pack(total_slots);
    // Usable bytes, as GetStats (and so df) reports them.
    pk.pack("total_capacity");      pk.pack(cap_total);
    pk.pack("remaining_capacity");  pk.pack(cap_remaining);
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
    // Members out of service (faulty or removed), data + parity.
    pk.pack("faulty_members");         pk.pack(CountDownMembers());
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
  StopIntentSync();
  alloc_log_.Flush();
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

//=============================================================================
// Synchronous parity and the stripe intent log (#1121)
//=============================================================================

bool Runtime::TryLockStripes(const std::set<clio::run::u64> &slots) {
  std::lock_guard<std::mutex> g(slot_mu_);
  for (clio::run::u64 s : slots) {
    if (busy_stripes_.count(s) != 0) return false;
  }
  busy_stripes_.insert(slots.begin(), slots.end());
  return true;
}

clio::run::TaskResume Runtime::LockStripes(
    const std::set<clio::run::u64> &slots) {
  CLIO_TASK_BODY_BEGIN
  // All-or-nothing, re-checked every time the worker runs us again: no lock
  // order to get wrong and no wakeup to lose. A thread-blocking lock would
  // deadlock the worker the moment a holder suspends at a member I/O.
  while (!TryLockStripes(slots)) {
    CLIO_CO_AWAIT(clio::run::yield(kStripeLockPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::UnlockStripes(const std::set<clio::run::u64> &slots) {
  std::lock_guard<std::mutex> g(slot_mu_);
  for (clio::run::u64 s : slots) busy_stripes_.erase(s);
}

clio::run::u64 Runtime::LogDirtyIntents(const std::set<clio::run::u64> &slots,
                                        std::vector<IntentKey> &keys) {
  keys.clear();
  if (!alloc_log_.enabled() || slots.empty()) return 0;
  std::lock_guard<std::mutex> g(intent_mu_);
  for (clio::run::u64 s : slots) {
    // A key of its own per (write, stripe): two writes' intents on one
    // stripe coexist, so one write's clean record never erases another's
    // dirty one. The stripe rides in the record's size field.
    const clio::run::u64 key = ++intent_key_gen_;
    alloc_log_.LogAlloc(kIntentGroup, key, s, 0);
    keys.push_back(IntentKey{key, s});
  }
  // Into the kernel now (survives this process dying); AwaitIntentDurable
  // fsyncs it before the data may be written.
  alloc_log_.Append();
  return intent_seq_.fetch_add(1, std::memory_order_acq_rel) + 1;
}

void Runtime::LogCleanIntents(const std::vector<IntentKey> &keys,
                              const std::set<clio::run::u64> &clean) {
  if (!alloc_log_.enabled() || keys.empty()) return;
  std::lock_guard<std::mutex> g(intent_mu_);
  for (const IntentKey &k : keys) {
    if (clean.count(k.slot) != 0) {
      alloc_log_.LogFree(kIntentGroup, k.key, k.slot, 0);
    } else {
      // Not encoded (its write failed, or a member went down): the stripe
      // stays stale until some later encode settles it.
      intent_stale_[k.slot].insert(k.key);
    }
  }
  // Lazily flushed: a lost clean record costs one needless re-encode.
}

clio::run::u64 Runtime::IntentWatermark() {
  std::lock_guard<std::mutex> g(intent_mu_);
  return intent_key_gen_;
}

void Runtime::LogStripesEncoded(const std::set<clio::run::u64> &slots,
                                clio::run::u64 watermark) {
  if (!alloc_log_.enabled() || slots.empty()) return;
  std::lock_guard<std::mutex> g(intent_mu_);
  for (clio::run::u64 s : slots) {
    auto it = intent_stale_.find(s);
    if (it == intent_stale_.end()) continue;
    // Only stale intents logged before the encode began: a free that
    // narrowed the stripe meanwhile is not covered by it. In-flight writes'
    // intents are never here (their own write settles them).
    auto &keys = it->second;
    for (auto k = keys.begin(); k != keys.end() && *k <= watermark;) {
      alloc_log_.LogFree(kIntentGroup, *k, s, 0);
      k = keys.erase(k);
    }
    if (keys.empty()) intent_stale_.erase(it);
  }
}

clio::run::TaskResume Runtime::AwaitIntentDurable(clio::run::u64 seq) {
  CLIO_TASK_BODY_BEGIN
  // Group commit on a dedicated thread (IntentSyncMain): one fsync covers
  // every intent appended before it began, and it runs OFF the worker, so
  // other writes keep going while it is in flight. A per-write fsync on the
  // worker was ~7 of the 8 ms a 128 KiB write took.
  if (seq != 0 && intent_durable_.load(std::memory_order_acquire) < seq) {
    intent_cv_.notify_one();
    while (intent_durable_.load(std::memory_order_acquire) < seq) {
      CLIO_CO_AWAIT(clio::run::yield(kIntentSyncPollUs));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::StartIntentSync() {
  if (intent_thread_.joinable()) return;
  intent_stop_.store(false);
  intent_thread_ = std::thread(&Runtime::IntentSyncMain, this);
}

void Runtime::StopIntentSync() {
  if (!intent_thread_.joinable()) return;
  intent_stop_.store(true);
  intent_cv_.notify_one();
  intent_thread_.join();
}

void Runtime::IntentSyncMain() {
  while (true) {
    {
      std::unique_lock<std::mutex> lk(intent_cv_mu_);
      intent_cv_.wait_for(lk, std::chrono::milliseconds(5), [this] {
        return intent_stop_.load() ||
               intent_seq_.load(std::memory_order_acquire) >
                   intent_durable_.load(std::memory_order_acquire);
      });
    }
    const clio::run::u64 target = intent_seq_.load(std::memory_order_acquire);
    if (!intent_sync_ &&
        target > intent_durable_.load(std::memory_order_acquire)) {
      // intent_sync: false -- the records already reached the kernel when
      // they were appended (LogDirtyIntents); that survives a process crash.
      intent_durable_.store(target, std::memory_order_release);
      continue;
    }
    if (target > intent_durable_.load(std::memory_order_acquire)) {
      if (!alloc_log_.SyncAppended()) {
        HLOG(kError, "safe_bdev: fsync of the intent log failed; writes "
             "wait until it succeeds");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      intent_durable_.store(target, std::memory_order_release);
    }
    if (intent_stop_.load() &&
        intent_seq_.load(std::memory_order_acquire) <=
            intent_durable_.load(std::memory_order_acquire)) {
      return;
    }
  }
}

size_t Runtime::ReplayStripeIntents() {
  if (!alloc_log_.enabled()) return 0;
  const std::vector<clio::run::bdev::LiveBlock> &live =
      alloc_log_.live(kIntentGroup);
  std::lock_guard<std::mutex> g(slot_mu_);
  size_t n = 0;
  {
    std::lock_guard<std::mutex> ig(intent_mu_);
    for (const auto &b : live) {
      intent_key_gen_ = std::max(intent_key_gen_, b.offset);
      intent_stale_[b.size].insert(b.offset);
    }
  }
  for (const auto &b : live) {
    const clio::run::u64 s = b.size;  // keys are unique; the stripe is here
    if (written_slots_.count(s) == 0) continue;  // stripe since emptied
    // The parity on disk encodes some earlier version of this stripe: never
    // reconstruct from it until it has been re-encoded.
    dirty_slots_.insert(s);
    ++slot_gen_[s];
    ++n;
  }
  return n;
}

clio::run::TaskResume Runtime::EncodeStripe(clio::run::u64 s, bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  clio::run::u64 gen = 0;
  // Every data column's physical bytes, allocated or not (#1126).
  const std::vector<int> stripe = CodeColumns();
  {
    std::lock_guard<std::mutex> g(slot_mu_);
    gen = slot_gen_[s];
  }
  const int k_s = static_cast<int>(stripe.size());
  if (k_s <= 0) {
    std::lock_guard<std::mutex> g(slot_mu_);
    dirty_slots_.erase(s);  // nothing left to protect
    ok = true;
    CLIO_CO_RETURN;
  }
  std::vector<std::vector<uint8_t>> dchunks(
      static_cast<size_t>(k_s), std::vector<uint8_t>(kChunkLen, 0));
  for (int pos = 0; pos < k_s; ++pos) {
    const size_t d = static_cast<size_t>(stripe[static_cast<size_t>(pos)]);
    if (data_members_[d].state_ != ec::EcState::kActive) CLIO_CO_RETURN;
    bool one = false;
    CLIO_CO_AWAIT(ReadDataSegment(d, SlotPhysOffset(s),
                                  dchunks[static_cast<size_t>(pos)].data(),
                                  kChunkLen, one));
    if (!one) CLIO_CO_RETURN;
  }
  std::vector<const uint8_t *> ptrs(static_cast<size_t>(k_s));
  for (int pos = 0; pos < k_s; ++pos) {
    ptrs[static_cast<size_t>(pos)] = dchunks[static_cast<size_t>(pos)].data();
  }
  ec::ReedSolomon *codec = GetCodec(k_s);
  auto *ipc = CLIO_IPC;
  for (size_t j = 0; j < parity_level_; ++j) {
    if (parity_members_[j].state_ != ec::EcState::kActive) continue;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kChunkLen);
    if (buf.IsNull()) CLIO_CO_RETURN;
    codec->EncodeParityShard(static_cast<int>(j), ptrs, kChunkLen,
                             reinterpret_cast<uint8_t *>(buf.ptr_));
    auto fut = parity_clients_[j].AsyncWrite(
        ParityQuery(j), MemberBlocks(SlotPhysOffset(s), kChunkLen),
        buf.shm_.template Cast<void>(), kChunkLen);
    CLIO_CO_AWAIT(fut);
    const bool wr_ok =
        fut->return_code_ == 0 && fut->bytes_written_ == kChunkLen;
    // A parity member that fails is faulted and drops out of the array; the
    // shards the others hold are current, so the stripe stays protected by
    // what remains (and RecoverBdev rebuilds the lost column).
    FaultOnIoError(/*is_parity=*/true, j, !wr_ok, fut->io_error_);
    ipc->FreeBuffer(buf);
    if (!wr_ok && parity_members_[j].state_ == ec::EcState::kActive) {
      CLIO_CO_RETURN;  // failed but not faulted: parity state unknown
    }
  }
  {
    std::lock_guard<std::mutex> g(slot_mu_);
    if (slot_gen_[s] == gen) dirty_slots_.erase(s);
  }
  ok = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::DeltaEligible(clio::run::u64 s) {
  // Fixed-width parity (#1126): any write into a stripe whose parity is
  // current can update it from the change alone -- fresh chunks included,
  // whose "old" bytes are whatever the column holds.
  if (parity_level_ == 0 || IsSlotDirty(s)) return false;
  std::lock_guard<std::mutex> g(alloc_mu_);
  return !StripeHasDownMember(s);
}

clio::run::TaskResume Runtime::ReadReplacedBytes(
    const std::vector<WritePiece> &pieces, std::set<clio::run::u64> &slots,
    std::vector<std::vector<uint8_t>> &old_bytes) {
  CLIO_TASK_BODY_BEGIN
  slots.clear();
  old_bytes.assign(pieces.size(), std::vector<uint8_t>());
  std::set<clio::run::u64> seen;
  for (const WritePiece &p : pieces) {
    if (seen.insert(p.slot).second && DeltaEligible(p.slot)) {
      slots.insert(p.slot);
    }
  }
  for (size_t i = 0; i < pieces.size(); ++i) {
    const WritePiece &p = pieces[i];
    if (slots.count(p.slot) == 0) continue;
    old_bytes[i].assign(p.len, 0);
    bool ok = false;
    CLIO_CO_AWAIT(ReadDataSegment(p.member, SlotPhysOffset(p.slot) + p.within,
                                  old_bytes[i].data(), p.len, ok));
    if (!ok) slots.erase(p.slot);  // full encode for this stripe instead
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DeltaEncodeStripe(
    clio::run::u64 s, const std::vector<WritePiece> &pieces, const char *data,
    const std::vector<std::vector<uint8_t>> &old_bytes, bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  const std::vector<int> stripe = CodeColumns();
  if (stripe.empty()) CLIO_CO_RETURN;
  ec::ReedSolomon *codec = GetCodec(static_cast<int>(stripe.size()));
  // Linear code: parity_j' = parity_j + c(j, pos) * (new - old), per byte,
  // over GF(2^8) where + and - are XOR. Only the written range of each
  // parity shard is read and rewritten.
  for (size_t i = 0; i < pieces.size(); ++i) {
    const WritePiece &p = pieces[i];
    if (p.slot != s) continue;
    const auto it = std::find(stripe.begin(), stripe.end(),
                              static_cast<int>(p.member));
    if (it == stripe.end() || old_bytes[i].size() != p.len) CLIO_CO_RETURN;
    const int pos = static_cast<int>(it - stripe.begin());
    std::vector<uint8_t> delta(old_bytes[i]);
    const uint8_t *nw = reinterpret_cast<const uint8_t *>(data + p.buf_off);
    for (clio::run::u64 b = 0; b < p.len; ++b) delta[b] ^= nw[b];
    for (size_t j = 0; j < parity_level_; ++j) {
      if (parity_members_[j].state_ != ec::EcState::kActive) continue;
      bool pok = false;
      CLIO_CO_AWAIT(ParityRangeAdd(j, SlotPhysOffset(s) + p.within,
                                   delta.data(), p.len,
                                   codec->CauchyCoeff(static_cast<int>(j), pos),
                                   pok));
      // A parity shard half updated is stale: the caller re-encodes fully.
      if (!pok) CLIO_CO_RETURN;
    }
  }
  ok = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ParityRangeAdd(size_t j, clio::run::u64 offset,
                                              const uint8_t *delta,
                                              clio::run::u64 len,
                                              uint8_t coeff, bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = false;
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
  if (buf.IsNull()) CLIO_CO_RETURN;
  auto rd = parity_clients_[j].AsyncRead(ParityQuery(j), MemberBlocks(offset, len),
                                         buf.shm_.template Cast<void>(), len);
  CLIO_CO_AWAIT(rd);
  bool rok = rd->return_code_ == 0 && rd->bytes_read_ == len;
  FaultOnIoError(/*is_parity=*/true, j, !rok, rd->io_error_);
  if (rok) {
    ec::GfMulAddRegion(reinterpret_cast<uint8_t *>(buf.ptr_), delta, coeff,
                       len);
    auto wr = parity_clients_[j].AsyncWrite(
        ParityQuery(j), MemberBlocks(offset, len),
        buf.shm_.template Cast<void>(), len);
    CLIO_CO_AWAIT(wr);
    ok = wr->return_code_ == 0 && wr->bytes_written_ == len;
    FaultOnIoError(/*is_parity=*/true, j, !ok, wr->io_error_);
  }
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Sync(clio::run::shared_ptr<SyncTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!IsHome()) {
    HLOG(kError, "safe_bdev: {} sent to a passive container of a distributed "
         "array; route it to the array's home node", "Sync");
    task->return_code_ = kNotHomeRc;
    CLIO_CO_RETURN;
  }
  FaultMembersOnDeadNodes();
  task->return_code_ = 0;
  // 1. Every stripe still dirty gets its parity now (writes encode their own
  //    stripes before acking; frees and membership changes leave theirs to
  //    the background builder, and fsync must not return ahead of it).
  std::vector<clio::run::u64> dirty;
  {
    std::lock_guard<std::mutex> g(slot_mu_);
    dirty.assign(dirty_slots_.begin(), dirty_slots_.end());
  }
  for (clio::run::u64 s : dirty) {
    const std::set<clio::run::u64> one{s};
    CLIO_CO_AWAIT(LockStripes(one));
    const clio::run::u64 wm = IntentWatermark();
    bool eok = !IsSlotDirty(s);
    if (!eok) CLIO_CO_AWAIT(EncodeStripe(s, eok));
    UnlockStripes(one);
    if (eok) {
      LogStripesEncoded(one, wm);
    } else if (CountDownMembers() == 0) {
      task->return_code_ = 1;  // all members up, yet parity cannot be built
    }
  }
  // 2. Every active member's data (and its own allocator state) to media.
  // A member that cannot sync cannot persist anything: it is faulted like a
  // member whose I/O failed, and the array stays durable through the rest.
  // Sync fails only if that leaves the array past max_failures. (Failing it
  // outright turned one dying disk into an fsync EIO for every writer.)
  std::vector<clio::run::Future<clio::run::bdev::SyncTask>> futs;
  std::vector<std::pair<bool, size_t>> who;  // (is_parity, index)
  for (size_t d = 0; d < data_clients_.size(); ++d) {
    if (data_members_[d].state_ != ec::EcState::kActive) continue;
    futs.push_back(data_clients_[d].AsyncSync(DataQuery(d)));
    who.emplace_back(false, d);
  }
  for (size_t j = 0; j < parity_clients_.size(); ++j) {
    if (parity_members_[j].state_ != ec::EcState::kActive) continue;
    futs.push_back(parity_clients_[j].AsyncSync(ParityQuery(j)));
    who.emplace_back(true, j);
  }
  for (size_t i = 0; i < futs.size(); ++i) {
    CLIO_CO_AWAIT(futs[i]);
    if (futs[i]->GetReturnCode() != 0) {
      FaultOnIoError(who[i].first, who[i].second, true,
                     static_cast<clio::run::u32>(ctp::IoError::kDeviceFault));
    }
  }
  if (CountDownMembers() > max_failures_) task->return_code_ = 1;
  // 3. The allocation and intent log.
  alloc_log_.Flush();
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::run::safe_bdev

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::safe_bdev::Runtime)

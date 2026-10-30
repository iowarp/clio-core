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
 * CTE core: each tag's xattr section (TagXattrTask). A key/value map per
 * tag, kept in memory on the container a task reaches and made durable one
 * key at a time: every change appends a kSetTagXattr / kDelTagXattr record
 * to the WAL, and FlushMetadata writes the whole section (snapshot entry 7)
 * before it truncates the WAL.
 */

#include <clio_cte/core/core_runtime.h>

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace clio::cte::core {

namespace {
/** Pack a tag id into the key of tag_xattrs_. */
clio::run::u64 PackTag(const TagId &id) {
  return (static_cast<clio::run::u64>(id.major_) << 32) | id.minor_;
}
}  // namespace

clio::run::PoolQuery Runtime::TagXattrOwner(const TagId &tag_id) {
  return clio::run::PoolQuery::DirectHash(static_cast<clio::run::u32>(
      std::hash<clio::run::u64>{}(PackTag(tag_id))));
}

clio::run::TaskResume Runtime::TagXattr(
    clio::run::shared_ptr<TagXattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    task->return_code_ = RunTagXattr(*task);
  } catch (const std::exception &e) {
    HLOG(kError, "TagXattr failed: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u32 Runtime::RunTagXattr(TagXattrTask &t) {
  const std::string key = t.key_.str();
  const clio::run::u64 packed = PackTag(t.tag_id_);
  t.found_ = 0;
  if (t.op_ != kTagXattrList && key.size() > kTagXattrMaxKey) {
    return kTagXattrTooBigRc;
  }
  std::lock_guard<std::mutex> g(tag_xattr_mu_);
  auto it = tag_xattrs_.find(packed);
  switch (t.op_) {
    case kTagXattrSet: {
      if (key.empty()) return kTagXattrBadOpRc;
      const std::string value = t.value_.str();
      if (value.size() > kTagXattrMaxValue) return kTagXattrTooBigRc;
      const bool exists =
          it != tag_xattrs_.end() && it->second.count(key) != 0;
      if ((t.flags_ & kTagXattrCreate) && exists) return kTagXattrExistsRc;
      if ((t.flags_ & kTagXattrReplace) && !exists) return kTagXattrMissingRc;
      tag_xattrs_[packed][key] = value;
      LogTagXattr(TxnType::kSetTagXattr, t.tag_id_, key, value);
      return 0;
    }
    case kTagXattrGet: {
      if (it == tag_xattrs_.end()) return 0;
      auto kit = it->second.find(key);
      if (kit == it->second.end()) return 0;
      t.value_ = clio::run::priv::string(CLIO_PRIV_ALLOC, kit->second);
      t.found_ = 1;
      return 0;
    }
    case kTagXattrDel: {
      const bool exists =
          it != tag_xattrs_.end() &&
          (key.empty() ? !it->second.empty() : it->second.count(key) != 0);
      if (!exists) {
        return (t.flags_ & kTagXattrMustExist) ? kTagXattrMissingRc : 0;
      }
      if (key.empty()) {
        tag_xattrs_.erase(it);
      } else {
        it->second.erase(key);
        if (it->second.empty()) tag_xattrs_.erase(it);
      }
      LogTagXattr(TxnType::kDelTagXattr, t.tag_id_, key, std::string());
      return 0;
    }
    case kTagXattrList: {
      t.keys_.clear();
      if (it == tag_xattrs_.end()) return 0;
      auto kit = key.empty() ? it->second.begin() : it->second.upper_bound(key);
      for (; kit != it->second.end(); ++kit) {
        if (t.max_keys_ != 0 && t.keys_.size() >= t.max_keys_) {
          t.found_ = 1;  // more remain
          break;
        }
        t.keys_.push_back(kit->first);
      }
      return 0;
    }
    default:
      return kTagXattrBadOpRc;
  }
}

void Runtime::LogTagXattr(TxnType type, const TagId &tag_id,
                          const std::string &key, const std::string &value) {
  if (tag_txn_logs_.empty()) return;
  TxnTagXattr txn;
  txn.tag_major_ = tag_id.major_;
  txn.tag_minor_ = tag_id.minor_;
  txn.key_ = key;
  txn.value_ = value;
  auto *worker = CLIO_CUR_WORKER;
  const clio::run::u32 wid =
      worker != nullptr ? worker->GetWorkerStats().worker_id_ : 0;
  tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(type, txn);
}

void Runtime::EraseTagXattrs(const TagId &tag_id) {
  std::lock_guard<std::mutex> g(tag_xattr_mu_);
  tag_xattrs_.erase(PackTag(tag_id));
}

bool Runtime::HasTagXattrs(const TagId &tag_id) {
  std::lock_guard<std::mutex> g(tag_xattr_mu_);
  auto it = tag_xattrs_.find(PackTag(tag_id));
  return it != tag_xattrs_.end() && !it->second.empty();
}

void Runtime::ApplyWalTagXattr(const std::vector<char> &payload, bool del) {
  auto txn = TransactionLog::DeserializeTagXattr(payload);
  const clio::run::u64 packed = PackTag(TagId{txn.tag_major_, txn.tag_minor_});
  std::lock_guard<std::mutex> g(tag_xattr_mu_);
  if (!del) {
    tag_xattrs_[packed][txn.key_] = txn.value_;
    return;
  }
  auto it = tag_xattrs_.find(packed);
  if (it == tag_xattrs_.end()) return;
  if (txn.key_.empty()) {
    tag_xattrs_.erase(it);
    return;
  }
  it->second.erase(txn.key_);
  if (it->second.empty()) tag_xattrs_.erase(it);
}

size_t Runtime::WriteTagXattrSnapshot(std::ofstream &ofs) {
  // Copy under the lock, write outside it: a set holds the lock while it
  // logs, so every record before the caller's snapshot_seq is in the copy.
  std::vector<std::pair<clio::run::u64, std::map<std::string, std::string>>>
      sections;
  {
    std::lock_guard<std::mutex> g(tag_xattr_mu_);
    sections.assign(tag_xattrs_.begin(), tag_xattrs_.end());
  }
  auto put_str = [&ofs](const std::string &s) {
    const uint32_t len = static_cast<uint32_t>(s.size());
    ofs.write(reinterpret_cast<const char *>(&len), sizeof(len));
    ofs.write(s.data(), len);
  };
  for (const auto &sec : sections) {
    const uint8_t entry_type = kSnapTagXattrEntry;
    const TagId id{static_cast<clio::run::u32>(sec.first >> 32),
                   static_cast<clio::run::u32>(sec.first)};
    const uint32_t count = static_cast<uint32_t>(sec.second.size());
    ofs.write(reinterpret_cast<const char *>(&entry_type), sizeof(entry_type));
    ofs.write(reinterpret_cast<const char *>(&id), sizeof(id));
    ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto &kv : sec.second) {
      put_str(kv.first);
      put_str(kv.second);
    }
  }
  return sections.size();
}

void Runtime::RestoreTagXattrSection(
    const TagId &tag_id, std::vector<std::pair<std::string, std::string>> kvs) {
  std::lock_guard<std::mutex> g(tag_xattr_mu_);
  auto &sec = tag_xattrs_[PackTag(tag_id)];
  sec.clear();
  for (auto &kv : kvs) sec[std::move(kv.first)] = std::move(kv.second);
  if (sec.empty()) tag_xattrs_.erase(PackTag(tag_id));
}

}  // namespace clio::cte::core

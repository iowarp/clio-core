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

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/indexer/indexer_runtime.h>

namespace clio::cte::indexer {

namespace {

/**
 * Execute a freshly-built core-vocabulary task on the co-located core
 * container as a NESTED call on the current fiber — the same mechanism
 * ForwardToCore uses for interposed traffic, applied to the indexer's own
 * sub-ops (size probe / read-back / tag-name resolve).
 *
 * This exists because the first cut issued these through the CLIENT API
 * from inside the runtime: every re-index paid 2-3 full task
 * dispatch/queue/schedule/complete round-trips whose payload was a local
 * metadata lookup — the same cost profile CLIO_RUN_INLINE (#862) documents
 * for puts, and the dominant term in the indexer's measured put overhead.
 * Unlike CLIO_RUN_INLINE this is not env-gated: an interposer executing on
 * its next pool's local container is the sanctioned chain mechanism, not an
 * opt-in optimization.
 *
 * Same synthesized-completed-future shape as detail::RunInlineOrSend.
 */
template <typename TaskT>
clio::run::TaskResume RunOnCore(clio::run::ContainerHold core,
                                clio::run::u32 method,
                                clio::run::shared_ptr<TaskT> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    if (core == nullptr) {
      task->SetReturnCode(1);
      CLIO_CO_RETURN;
    }
    task->BeginRunContext();
    clio::run::Future<TaskT> fut(task->pool_id_, task->method_, task);
    fut.GetFutureShm()->origin_ = clio::run::ClientOrigin::kClientShm;
    task->RunFuture() = fut.template Cast<clio::run::Task>();
    clio::run::shared_ptr<clio::run::Task> base =
        task.template Cast<clio::run::Task>();
    CLIO_CO_AWAIT(core->Run(method, base));
    task->SetComplete();
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/** CLIO_INDEXER_PASSIVE=1: forward-only interposition — no index
 *  maintenance at all. Kill switch for production triage and the measured
 *  baseline separating interposition cost from indexing cost. */
bool IndexerPassive() {
  static const bool v = std::getenv("CLIO_INDEXER_PASSIVE") != nullptr;
  return v;
}

// Tokenize raw bytes as lowercase alphanumeric runs of length >= 2.
// Classification semantics MUST stay identical to the core's old in-core
// scan (C-locale isalnum/tolower over ASCII): the python/CAE semantic-search
// tests assert on its ranking behavior. Unlike the old scan this runs on
// EVERY put, so the per-byte cost is the put path's floor — the locale
// isalnum()/tolower() calls it used were ~3-10ns/byte (measured ~350MB/s put
// ceiling); the precomputed table below classifies+lowers in one L1 load.
// 0 = not a token byte; else the lowercased byte.
const unsigned char *SemSearchByteTable() {
  // C-locale alnum only: '0'-'9', 'A'-'Z' (lowered), 'a'-'z'. High bytes
  // stay 0 — matching C-locale isalnum on this runtime's daemons.
  static const std::array<unsigned char, 256> table = [] {
    std::array<unsigned char, 256> t{};
    for (int c = '0'; c <= '9'; ++c) t[c] = static_cast<unsigned char>(c);
    for (int c = 'a'; c <= 'z'; ++c) t[c] = static_cast<unsigned char>(c);
    for (int c = 'A'; c <= 'Z'; ++c)
      t[c] = static_cast<unsigned char>(c - 'A' + 'a');
    return t;
  }();
  return table.data();
}

std::vector<std::string> SemSearchTokenize(const char *data, size_t size) {
  const unsigned char *tbl = SemSearchByteTable();
  std::vector<std::string> tokens;
  std::string cur;
  cur.reserve(16);
  for (size_t i = 0; i < size; ++i) {
    unsigned char mapped = tbl[static_cast<unsigned char>(data[i])];
    if (mapped != 0) {
      cur.push_back(static_cast<char>(mapped));
    } else {
      if (cur.size() >= 2) tokens.push_back(std::move(cur));
      cur.clear();
    }
  }
  if (cur.size() >= 2) tokens.push_back(std::move(cur));
  return tokens;
}

}  // namespace

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    next_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }
  // Compile the module scope once. A bad regex disables scoping open
  // (index everything) rather than failing the pool — config is
  // best-effort, matching LoadConfig.
  scope_match_all_ = config_.tag_re_ == ".*" && config_.blob_re_ == ".*";
  try {
    scope_tag_re_ = std::regex(config_.tag_re_);
    scope_blob_re_ = std::regex(config_.blob_re_);
  } catch (const std::regex_error &e) {
    HLOG(kError, "Indexer: bad scope regex (tag='{}' blob='{}'): {} — "
         "indexing everything", config_.tag_re_, config_.blob_re_, e.what());
    scope_match_all_ = true;
  }
  // Persistence (issue #905): restore snapshot + WAL — a restart NEVER
  // rescans storage. Restore-then-compact leaves a fresh snapshot and an
  // empty WAL; the append stream opens after compaction.
  if (!config_.index_log_path_.empty()) {
    std::lock_guard<std::mutex> lock(index_mutex_);
    if (RestoreIndex()) {
      HLOG(kInfo, "Indexer: restored {} docs ({} pending) from {}",
           index_.size(), pending_.size(), config_.index_log_path_);
    }
    SnapshotIndex();  // compact: snapshot current state, truncate WAL
    wal_.open(config_.index_log_path_ + ".wal",
              std::ios::binary | std::ios::app);
    wal_bytes_ = 0;
    if (!wal_.is_open()) {
      HLOG(kError, "Indexer: cannot open WAL at {}.wal — persistence OFF",
           config_.index_log_path_);
    }
  }
  // Background progress for the asynchronous index drain (same periodic
  // pattern as replication's write-through sweep). Searches drain
  // synchronously regardless — the sweep only bounds how stale the index
  // can sit between searches.
  if (config_.index_sweep_period_ms_ > 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto sweep = ipc->NewTask<IndexSweepTask>(
        clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Local());
    sweep->SetPeriod(static_cast<double>(config_.index_sweep_period_ms_),
                     clio::run::kMilli);
    sweep->SetFlags(TASK_PERIODIC);  // SetPeriod alone does not mark it
    ipc->Send(sweep);
    HLOG(kInfo, "Indexer: async index sweep every {} ms",
         config_.index_sweep_period_ms_);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    // Clean shutdown: persist EVERYTHING including still-dirty keys, so the
    // next boot resumes exactly here (the restored pending set re-drains).
    SnapshotIndex();
    if (wal_.is_open()) wal_.close();
    index_.clear();
    tag_names_.clear();
    pending_.clear();
    pending_tag_backfills_.clear();
    tags_seen_.clear();
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::cte::core::Client *Runtime::GetNextClient() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

// ---------------------------------------------------------------------------
// Persistence: snapshot + append WAL (issue #905). Binary, versioned magic,
// same hand-rolled discipline as the core's FlushMetadata/TransactionLog.
// All writers assume index_mutex_ is HELD by the caller (they record a
// mutation made under that lock). Restart restores; it never rescans.
// ---------------------------------------------------------------------------

namespace {

constexpr char kIdxMagic[8] = {'C', 'L', 'I', 'O', 'I', 'D', 'X', '1'};
enum IdxWalType : unsigned char {
  kWalDoc = 1,
  kWalDelDoc = 2,
  kWalDelTag = 3,
  kWalRenameTag = 4,
  kWalTagSeen = 5,
};

template <typename T>
void WPod(std::ofstream &os, const T &v) {
  os.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
void WStr(std::ofstream &os, const std::string &s) {
  auto len = static_cast<clio::run::u32>(s.size());
  WPod(os, len);
  os.write(s.data(), len);
}
template <typename T>
bool RPod(std::ifstream &is, T *v) {
  is.read(reinterpret_cast<char *>(v), sizeof(T));
  return is.good();
}
bool RStr(std::ifstream &is, std::string *s) {
  clio::run::u32 len = 0;
  if (!RPod(is, &len)) return false;
  s->resize(len);
  is.read(s->data(), len);
  return is.good() || (len == 0 && !is.bad());
}

/** Anchor a literal tag name as a full-match regex. */
std::string EscapeRegex(const std::string &raw) {
  std::string out;
  out.reserve(raw.size() + 8);
  for (char c : raw) {
    if (std::strchr("\\^$.|?*+()[]{}", c) != nullptr) out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

}  // namespace

void Runtime::WalAppendDoc(const IndexedDoc &doc) {
  if (!wal_.is_open()) return;
  WPod(wal_, static_cast<unsigned char>(kWalDoc));
  WPod(wal_, doc.tag_id_.major_);
  WPod(wal_, doc.tag_id_.minor_);
  WStr(wal_, doc.tag_name_);
  WStr(wal_, doc.blob_name_);
  WPod(wal_, static_cast<clio::run::u64>(doc.length_));
  WPod(wal_, static_cast<clio::run::u32>(doc.tf_.size()));
  for (const auto &kv : doc.tf_) {
    WStr(wal_, kv.first);
    WPod(wal_, static_cast<clio::run::u32>(kv.second));
  }
  wal_.flush();
  wal_bytes_ += 32 + doc.tag_name_.size() + doc.blob_name_.size() +
                doc.tf_.size() * 12;
}

void Runtime::WalAppendDelDoc(const TagId &tag_id,
                              const std::string &blob_name) {
  if (!wal_.is_open()) return;
  WPod(wal_, static_cast<unsigned char>(kWalDelDoc));
  WPod(wal_, tag_id.major_);
  WPod(wal_, tag_id.minor_);
  WStr(wal_, blob_name);
  wal_.flush();
  wal_bytes_ += 16 + blob_name.size();
}

void Runtime::WalAppendDelTag(const TagId &tag_id) {
  if (!wal_.is_open()) return;
  WPod(wal_, static_cast<unsigned char>(kWalDelTag));
  WPod(wal_, tag_id.major_);
  WPod(wal_, tag_id.minor_);
  wal_.flush();
  wal_bytes_ += 9;
}

void Runtime::WalAppendRenameTag(const TagId &tag_id,
                                 const std::string &new_name) {
  if (!wal_.is_open()) return;
  WPod(wal_, static_cast<unsigned char>(kWalRenameTag));
  WPod(wal_, tag_id.major_);
  WPod(wal_, tag_id.minor_);
  WStr(wal_, new_name);
  wal_.flush();
  wal_bytes_ += 16 + new_name.size();
}

void Runtime::WalAppendTagSeen(const TagId &tag_id) {
  if (!wal_.is_open()) return;
  WPod(wal_, static_cast<unsigned char>(kWalTagSeen));
  WPod(wal_, tag_id.major_);
  WPod(wal_, tag_id.minor_);
  wal_.flush();
  wal_bytes_ += 9;
}

void Runtime::SnapshotIndex() {
  if (config_.index_log_path_.empty()) return;
  const bool reopen = wal_.is_open();
  if (reopen) wal_.close();
  {
    std::ofstream snap(config_.index_log_path_,
                       std::ios::binary | std::ios::trunc);
    if (!snap.is_open()) {
      HLOG(kError, "Indexer: cannot write snapshot at {}",
           config_.index_log_path_);
      return;
    }
    snap.write(kIdxMagic, sizeof(kIdxMagic));
    WPod(snap, static_cast<clio::run::u32>(1));  // version
    WPod(snap, static_cast<clio::run::u32>(tag_names_.size()));
    for (const auto &kv : tag_names_) {
      WPod(snap, kv.first);
      WStr(snap, kv.second);
    }
    WPod(snap, static_cast<clio::run::u32>(tags_seen_.size()));
    for (clio::run::u64 key : tags_seen_) {
      WPod(snap, key);
    }
    WPod(snap, static_cast<clio::run::u64>(index_.size()));
    for (const auto &kv : index_) {
      const IndexedDoc &doc = kv.second;
      WPod(snap, doc.tag_id_.major_);
      WPod(snap, doc.tag_id_.minor_);
      WStr(snap, doc.tag_name_);
      WStr(snap, doc.blob_name_);
      WPod(snap, static_cast<clio::run::u64>(doc.length_));
      WPod(snap, static_cast<clio::run::u32>(doc.tf_.size()));
      for (const auto &tf : doc.tf_) {
        WStr(snap, tf.first);
        WPod(snap, static_cast<clio::run::u32>(tf.second));
      }
    }
    // Dirty keys survive a clean shutdown: the restored pending set
    // re-drains, so acked-but-unindexed content is never silently lost.
    WPod(snap, static_cast<clio::run::u32>(
                   pending_.size() + pending_tag_backfills_.size()));
    for (const auto &kv : pending_) {
      WPod(snap, kv.second.tag_id_.major_);
      WPod(snap, kv.second.tag_id_.minor_);
      WStr(snap, kv.second.blob_name_);
    }
    // Un-run tag backfills persist as dirty markers with an empty blob
    // name; restore re-registers them.
    for (const auto &kv : pending_tag_backfills_) {
      WPod(snap, kv.second.major_);
      WPod(snap, kv.second.minor_);
      WStr(snap, std::string());
    }
  }
  // WAL entries are superseded by the snapshot: truncate.
  std::ofstream(config_.index_log_path_ + ".wal",
                std::ios::binary | std::ios::trunc);
  wal_bytes_ = 0;
  if (reopen) {
    wal_.open(config_.index_log_path_ + ".wal",
              std::ios::binary | std::ios::app);
  }
}

bool Runtime::RestoreIndex() {
  bool any = false;
  auto in_scope = [&](const std::string &tag_name,
                      const std::string &blob_name) {
    if (scope_match_all_) return true;
    return std::regex_match(tag_name, scope_tag_re_) &&
           std::regex_match(blob_name, scope_blob_re_);
  };
  // ---- Snapshot ----
  {
    std::ifstream snap(config_.index_log_path_, std::ios::binary);
    char magic[8];
    if (snap.is_open() && snap.read(magic, 8) &&
        std::memcmp(magic, kIdxMagic, 8) == 0) {
      clio::run::u32 version = 0;
      RPod(snap, &version);
      clio::run::u32 n_tags = 0;
      RPod(snap, &n_tags);
      for (clio::run::u32 i = 0; i < n_tags && snap.good(); ++i) {
        clio::run::u64 key = 0;
        std::string name;
        if (!RPod(snap, &key) || !RStr(snap, &name)) break;
        tag_names_[key] = name;
      }
      clio::run::u32 n_seen = 0;
      RPod(snap, &n_seen);
      for (clio::run::u32 i = 0; i < n_seen && snap.good(); ++i) {
        clio::run::u64 key = 0;
        if (!RPod(snap, &key)) break;
        tags_seen_.insert(key);
      }
      clio::run::u64 n_docs = 0;
      RPod(snap, &n_docs);
      for (clio::run::u64 i = 0; i < n_docs && snap.good(); ++i) {
        IndexedDoc doc;
        clio::run::u64 length = 0;
        clio::run::u32 ntf = 0;
        if (!RPod(snap, &doc.tag_id_.major_) ||
            !RPod(snap, &doc.tag_id_.minor_) ||
            !RStr(snap, &doc.tag_name_) || !RStr(snap, &doc.blob_name_) ||
            !RPod(snap, &length) || !RPod(snap, &ntf)) {
          break;
        }
        doc.length_ = static_cast<size_t>(length);
        doc.tf_.reserve(ntf);
        bool ok = true;
        for (clio::run::u32 t = 0; t < ntf; ++t) {
          std::string term;
          clio::run::u32 cnt = 0;
          if (!RStr(snap, &term) || !RPod(snap, &cnt)) {
            ok = false;
            break;
          }
          doc.tf_.emplace(std::move(term), static_cast<int>(cnt));
        }
        if (!ok) break;
        if (in_scope(doc.tag_name_, doc.blob_name_)) {
          std::string key = DocKey(doc.tag_id_, doc.blob_name_);
          index_[key] = std::move(doc);
          any = true;
        }
      }
      clio::run::u32 n_dirty = 0;
      RPod(snap, &n_dirty);
      for (clio::run::u32 i = 0; i < n_dirty && snap.good(); ++i) {
        TagId tag_id = TagId::GetNull();
        std::string blob_name;
        if (!RPod(snap, &tag_id.major_) || !RPod(snap, &tag_id.minor_) ||
            !RStr(snap, &blob_name)) {
          break;
        }
        if (blob_name.empty()) {
          pending_tag_backfills_[TagKey(tag_id)] = tag_id;
        } else {
          pending_[DocKey(tag_id, blob_name)] =
              PendingReindex{tag_id, blob_name};
        }
        any = true;
      }
    }
  }
  // ---- WAL replay (last-wins on top of the snapshot) ----
  {
    std::ifstream wal(config_.index_log_path_ + ".wal", std::ios::binary);
    while (wal.is_open() && wal.good()) {
      unsigned char type = 0;
      if (!RPod(wal, &type)) break;
      if (type == kWalDoc) {
        IndexedDoc doc;
        clio::run::u64 length = 0;
        clio::run::u32 ntf = 0;
        if (!RPod(wal, &doc.tag_id_.major_) ||
            !RPod(wal, &doc.tag_id_.minor_) || !RStr(wal, &doc.tag_name_) ||
            !RStr(wal, &doc.blob_name_) || !RPod(wal, &length) ||
            !RPod(wal, &ntf)) {
          break;
        }
        doc.length_ = static_cast<size_t>(length);
        bool ok = true;
        for (clio::run::u32 t = 0; t < ntf; ++t) {
          std::string term;
          clio::run::u32 cnt = 0;
          if (!RStr(wal, &term) || !RPod(wal, &cnt)) {
            ok = false;
            break;
          }
          doc.tf_.emplace(std::move(term), static_cast<int>(cnt));
        }
        if (!ok) break;
        tag_names_[TagKey(doc.tag_id_)] = doc.tag_name_;
        if (in_scope(doc.tag_name_, doc.blob_name_)) {
          std::string key = DocKey(doc.tag_id_, doc.blob_name_);
          index_[key] = std::move(doc);
          any = true;
        }
      } else if (type == kWalDelDoc) {
        TagId tag_id = TagId::GetNull();
        std::string blob_name;
        if (!RPod(wal, &tag_id.major_) || !RPod(wal, &tag_id.minor_) ||
            !RStr(wal, &blob_name)) {
          break;
        }
        std::string key = DocKey(tag_id, blob_name);
        index_.erase(key);
        pending_.erase(key);
      } else if (type == kWalDelTag) {
        TagId tag_id = TagId::GetNull();
        if (!RPod(wal, &tag_id.major_) || !RPod(wal, &tag_id.minor_)) break;
        for (auto it = index_.begin(); it != index_.end();) {
          if (it->second.tag_id_.major_ == tag_id.major_ &&
              it->second.tag_id_.minor_ == tag_id.minor_) {
            it = index_.erase(it);
          } else {
            ++it;
          }
        }
        tag_names_.erase(TagKey(tag_id));
        tags_seen_.erase(TagKey(tag_id));
        pending_tag_backfills_.erase(TagKey(tag_id));
      } else if (type == kWalRenameTag) {
        TagId tag_id = TagId::GetNull();
        std::string new_name;
        if (!RPod(wal, &tag_id.major_) || !RPod(wal, &tag_id.minor_) ||
            !RStr(wal, &new_name)) {
          break;
        }
        tag_names_[TagKey(tag_id)] = new_name;
        for (auto &kv : index_) {
          if (kv.second.tag_id_.major_ == tag_id.major_ &&
              kv.second.tag_id_.minor_ == tag_id.minor_) {
            kv.second.tag_name_ = new_name;
          }
        }
      } else if (type == kWalTagSeen) {
        TagId tag_id = TagId::GetNull();
        if (!RPod(wal, &tag_id.major_) || !RPod(wal, &tag_id.minor_)) break;
        tags_seen_.insert(TagKey(tag_id));
      } else {
        HLOG(kWarning, "Indexer: unknown WAL record type {} — stopping "
             "replay", static_cast<int>(type));
        break;
      }
    }
  }
  return any;
}

// ---------------------------------------------------------------------------
// Index maintenance
// ---------------------------------------------------------------------------

clio::run::TaskResume Runtime::ResolveTagName(TagId tag_id,
                                              std::string *name_out) {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto it = tag_names_.find(TagKey(tag_id));
    if (it != tag_names_.end()) {
      *name_out = it->second;
      CLIO_CO_RETURN;
    }
  }
  {
    // Nested inline run on the co-located core: tag metadata is a local map
    // lookup; a client round-trip here costs more than the lookup itself.
    // Broadcast is unnecessary — every chain container is one-per-node and
    // the tag either resolves locally or the name stays uncached until a
    // path that can see it (rebuild seeds the cache from BlobQuery names).
    auto task = CLIO_CPU_IPC->NewTask<clio::cte::core::GetTagNameTask>(
        clio::run::CreateTaskId(), CorePoolId(), clio::run::PoolQuery::Local(),
        tag_id);
    CLIO_CO_AWAIT(RunOnCore(CoreContainer(),
                            clio::cte::core::Method::kGetTagName, task));
    if (task->GetReturnCode() == 0 && task->found_ != 0) {
      *name_out = task->tag_name_.str();
      std::lock_guard<std::mutex> lock(index_mutex_);
      tag_names_[TagKey(tag_id)] = *name_out;
    } else {
      name_out->clear();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ProbeBlobSize(TagId tag_id,
                                             const std::string &blob_name,
                                             clio::run::u64 *total_out) {
  CLIO_TASK_BODY_BEGIN
  {
    auto task = CLIO_CPU_IPC->NewTask<clio::cte::core::GetBlobSizeTask>(
        clio::run::CreateTaskId(), CorePoolId(), clio::run::PoolQuery::Local(),
        tag_id, blob_name, 0);
    CLIO_CO_AWAIT(RunOnCore(CoreContainer(),
                            clio::cte::core::Method::kGetBlobSize, task));
    *total_out = task->GetReturnCode() == 0 ? task->size_ : 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::IndexDocBytes(const TagId &tag_id, const std::string &tag_name,
                            const std::string &blob_name, const char *data,
                            clio::run::u64 len) {
  auto tokens = SemSearchTokenize(data, len);
  IndexedDoc doc;
  doc.tag_id_ = tag_id;
  doc.tag_name_ = tag_name;
  doc.blob_name_ = blob_name;
  doc.length_ = tokens.size();
  for (auto &t : tokens) ++doc.tf_[t];
  std::lock_guard<std::mutex> lock(index_mutex_);
  WalAppendDoc(doc);
  index_[DocKey(tag_id, blob_name)] = std::move(doc);
}

clio::run::TaskResume Runtime::ReindexBlob(TagId tag_id,
                                           std::string blob_name) {
  CLIO_TASK_BODY_BEGIN
  {
    auto *ipc_manager = CLIO_CPU_IPC;

    // Current logical size decides index membership: the old in-core scan
    // skipped zero-length blobs, so an empty (or vanished) blob leaves the
    // index rather than lingering as an empty doc.
    clio::run::u64 total = 0;
    CLIO_CO_AWAIT(ProbeBlobSize(tag_id, blob_name, &total));
    if (total == 0) {
      HLOG(kDebug,
           "Indexer: blob '{}' (tag {}.{}) not indexable (size 0 or stat "
           "failed) — dropping from index",
           blob_name, tag_id.major_, tag_id.minor_);
      std::lock_guard<std::mutex> lock(index_mutex_);
      if (index_.erase(DocKey(tag_id, blob_name)) != 0) {
        WalAppendDelDoc(tag_id, blob_name);
      }
      CLIO_CO_RETURN;
    }

    std::string tag_name;
    CLIO_CO_AWAIT(ResolveTagName(tag_id, &tag_name));

    // Module scope, tag half (the blob half was pre-checked at enqueue):
    // out-of-scope content is never tokenized and leaves the index if a
    // scope change stranded an old entry there.
    if (!scope_match_all_ &&
        (tag_name.empty() || !std::regex_match(tag_name, scope_tag_re_) ||
         !std::regex_match(blob_name, scope_blob_re_))) {
      std::lock_guard<std::mutex> lock(index_mutex_);
      if (index_.erase(DocKey(tag_id, blob_name)) != 0) {
        WalAppendDelDoc(tag_id, blob_name);
      }
      CLIO_CO_RETURN;
    }

    // Re-read the blob's CURRENT bytes from the chain — the always-correct
    // fallback for partial/vectored writes and truncates (whole-blob puts
    // never come here; they tokenize straight from the task payload). The
    // read runs as a nested inline call on the core container: no client
    // round-trip, no queue hop.
    ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(total);
    if (buf.IsNull()) {
      HLOG(kWarning,
           "Indexer: AllocateBuffer({}) failed for blob '{}'; leaving stale "
           "index entry",
           total, blob_name);
      CLIO_CO_RETURN;
    }
    clio::run::u32 read_rc = 0;
    {
      auto get_task = ipc_manager->NewTask<clio::cte::core::GetBlobTask>(
          clio::run::CreateTaskId(), CorePoolId(),
          clio::run::PoolQuery::Local(), tag_id, blob_name.c_str(),
          static_cast<clio::run::u64>(0), total, static_cast<clio::run::u32>(0),
          ctp::ipc::ShmPtr<>(buf.shm_), clio::cte::core::Context());
      CLIO_CO_AWAIT(RunOnCore(CoreContainer(),
                              clio::cte::core::Method::kGetBlob, get_task));
      read_rc = get_task->GetReturnCode();
    }
    if (read_rc != 0) {
      ipc_manager->FreeBuffer(buf);
      HLOG(kWarning,
           "Indexer: GetBlob failed for blob '{}' (rc={}); dropping index "
           "entry",
           blob_name, read_rc);
      std::lock_guard<std::mutex> lock(index_mutex_);
      if (index_.erase(DocKey(tag_id, blob_name)) != 0) {
        WalAppendDelDoc(tag_id, blob_name);
      }
      CLIO_CO_RETURN;
    }

    IndexDocBytes(tag_id, tag_name, blob_name, buf.ptr_, total);
    ipc_manager->FreeBuffer(buf);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::EnqueuePending(const TagId &tag_id,
                             const std::string &blob_name) {
  // Scope precheck on the half we can evaluate without a resolve: an
  // out-of-scope blob name never even enters the queue. The tag_re half is
  // enforced at drain time, where the tag name is resolved anyway.
  if (!scope_match_all_ && !std::regex_match(blob_name, scope_blob_re_)) {
    return;
  }
  std::lock_guard<std::mutex> lock(index_mutex_);
  // tag_re precheck when the tag's name is already cached: a scoped-out
  // tag's writes then cost NOTHING past this lookup (first writes before
  // the name is cached fall through; the drain filters and caches).
  if (!scope_match_all_) {
    auto name_it = tag_names_.find(TagKey(tag_id));
    if (name_it != tag_names_.end() &&
        !std::regex_match(name_it->second, scope_tag_re_)) {
      return;
    }
  }
  pending_[DocKey(tag_id, blob_name)] = PendingReindex{tag_id, blob_name};
  // First-insertion backfill: the first write a tag receives after the
  // indexer composed schedules a one-time scan of that tag's PRE-EXISTING
  // blobs (drained asynchronously like everything else). tags_seen_ is
  // persisted, so restarts never re-trigger it.
  clio::run::u64 tkey = TagKey(tag_id);
  if (tags_seen_.insert(tkey).second) {
    WalAppendTagSeen(tag_id);
    pending_tag_backfills_[tkey] = tag_id;
  }
}

clio::run::TaskResume Runtime::DrainPendingIndex() {
  CLIO_TASK_BODY_BEGIN
  for (;;) {
    bool have = false;
    bool have_backfill = false;
    bool wait = false;
    TagId tag_id = TagId::GetNull();
    std::string blob_name;
    {
      std::lock_guard<std::mutex> lock(index_mutex_);
      if (!pending_tag_backfills_.empty()) {
        // Backfills FIRST: they feed pending_, so draining them early keeps
        // one pass sufficient.
        auto it = pending_tag_backfills_.begin();
        tag_id = it->second;
        pending_tag_backfills_.erase(it);
        ++draining_;
        have_backfill = true;
      } else if (!pending_.empty()) {
        auto it = pending_.begin();
        tag_id = it->second.tag_id_;
        blob_name = std::move(it->second.blob_name_);
        pending_.erase(it);
        ++draining_;
        have = true;
      } else if (draining_ != 0) {
        // Another drainer is mid-entry: the barrier must outlast it.
        wait = true;
      }
    }
    if (have_backfill) {
      std::string tag_name;
      CLIO_CO_AWAIT(ResolveTagName(tag_id, &tag_name));
      if (!tag_name.empty() &&
          (scope_match_all_ ||
           std::regex_match(tag_name, scope_tag_re_))) {
        clio::run::u64 enq = 0;
        CLIO_CO_AWAIT(ScanAndEnqueue("^" + EscapeRegex(tag_name) + "$",
                                     config_.blob_re_, &enq));
        HLOG(kDebug, "Indexer: first-insertion backfill of tag '{}' "
             "enqueued {} blobs", tag_name, enq);
      }
      std::lock_guard<std::mutex> lock(index_mutex_);
      --draining_;
      continue;
    }
    if (have) {
      CLIO_CO_AWAIT(ReindexBlob(tag_id, blob_name));
      std::lock_guard<std::mutex> lock(index_mutex_);
      --draining_;
      continue;
    }
    if (wait) {
      CLIO_CO_AWAIT(clio::run::yield(50));
      continue;
    }
    break;  // pending empty and nothing in flight
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::IndexSweep(
    clio::run::shared_ptr<IndexSweepTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(DrainPendingIndex());
  {
    // WAL compaction: past the threshold, fold the log into a fresh
    // snapshot so replay stays O(index), not O(history).
    std::lock_guard<std::mutex> lock(index_mutex_);
    if (wal_.is_open() && wal_bytes_ > config_.index_wal_compact_bytes_) {
      SnapshotIndex();
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ---------------------------------------------------------------------------
// Intercepted mutating verbs: forward FIRST, then mark the doc dirty.
// Indexing is ASYNCHRONOUS (issue #905 perf): the ack path's only cost is
// an O(1) coalesced enqueue — the measured synchronous alternatives were
// put-throughput disasters (client-hop read-backs ~350MB/s; even a
// payload-direct inline tokenize capped puts at the scanner's ~0.7GB/s).
// The drain (sweep task or a search's barrier) re-reads CURRENT bytes per
// dirty key, so N overwrites cost one scan and the index converges to the
// latest content.
// ---------------------------------------------------------------------------

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                              task.template Cast<clio::run::Task>()));
  // Replica-addressed writes duplicate primary content the index already
  // tracks (the primary put flowed through here too) — never re-index them.
  if (task->GetReturnCode() == 0 && task->context_.replica_ == 0 &&
      !IndexerPassive()) {
    // blob_name_ is INOUT (the gpu page suffix is composed handler-side),
    // so read it AFTER the forward.
    EnqueuePending(task->tag_id_, task->blob_name_.str());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() == 0 && task->context_.replica_ == 0 &&
      !IndexerPassive()) {
    std::vector<clio::cte::core::MultiPutDesc> descs =
        clio::cte::core::DecodeMultiPutDescs(task->descs_);
    for (size_t i = 0; i < descs.size(); ++i) {
      EnqueuePending(descs[i].tag_id_, descs[i].blob_name_);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TruncateBlob(
    clio::run::shared_ptr<clio::cte::core::TruncateBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kTruncateBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() == 0 && !IndexerPassive()) {
    EnqueuePending(task->tag_id_, task->blob_name_.str());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelBlob(
    clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kDelBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() == 0) {
    std::lock_guard<std::mutex> lock(index_mutex_);
    std::string blob_name = task->blob_name_.str();
    std::string key = DocKey(task->tag_id_, blob_name);
    bool had = index_.erase(key) != 0;
    had = pending_.erase(key) != 0 || had;
    if (had) {
      WalAppendDelDoc(task->tag_id_, blob_name);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelTag(
    clio::run::shared_ptr<clio::cte::core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kDelTag,
                              task.template Cast<clio::run::Task>()));
  // tag_id_ is INOUT (resolved from tag_name_ when given by name), so it is
  // authoritative after the forward. POSIX-unlink alias survival (the tag
  // living on under another name) reports a nonzero "kept alive" path via
  // rc==0 with the tag intact — the conservative move either way is to drop
  // docs only when the tag is actually gone, which the core signals with
  // rc==0 on a cascade delete. A dropped-then-still-alive tag would heal on
  // the next put; a kept-then-deleted tag would serve ghosts, so we only
  // prune on success.
  if (task->GetReturnCode() == 0 && !task->tag_id_.IsNull()) {
    std::lock_guard<std::mutex> lock(index_mutex_);
    for (auto it = index_.begin(); it != index_.end();) {
      if (it->second.tag_id_.major_ == task->tag_id_.major_ &&
          it->second.tag_id_.minor_ == task->tag_id_.minor_) {
        it = index_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (it->second.tag_id_.major_ == task->tag_id_.major_ &&
          it->second.tag_id_.minor_ == task->tag_id_.minor_) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
    tag_names_.erase(TagKey(task->tag_id_));
    // A recreated tag is a NEW first insertion: it must re-backfill.
    tags_seen_.erase(TagKey(task->tag_id_));
    pending_tag_backfills_.erase(TagKey(task->tag_id_));
    WalAppendDelTag(task->tag_id_);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RenameTag(
    clio::run::shared_ptr<clio::cte::core::RenameTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kRenameTag,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() == 0 && !task->tag_id_.IsNull()) {
    std::string new_name = task->new_name_.str();
    std::lock_guard<std::mutex> lock(index_mutex_);
    tag_names_[TagKey(task->tag_id_)] = new_name;
    for (auto &kv : index_) {
      if (kv.second.tag_id_.major_ == task->tag_id_.major_ &&
          kv.second.tag_id_.minor_ == task->tag_id_.minor_) {
        kv.second.tag_name_ = new_name;
      }
    }
    WalAppendRenameTag(task->tag_id_, new_name);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ---------------------------------------------------------------------------
// SemanticSearch — BM25 over the maintained index (no blob reads).
// ---------------------------------------------------------------------------

clio::run::TaskResume Runtime::SemanticSearch(
    clio::run::shared_ptr<clio::cte::core::SemanticSearchTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->results_.clear();
  task->return_code_ = 0;
  // Read-your-writes barrier: indexing is asynchronous, so bring the index
  // current with every acked mutation BEFORE evaluating the query.
  CLIO_CO_AWAIT(DrainPendingIndex());
  {
    std::string tag_regex_str = task->tag_regex_.str();
    std::string blob_regex_str = task->blob_regex_.str();
    std::string query_text = task->query_text_.str();
    clio::run::u32 k = task->k_;

    std::regex tag_pattern;
    std::regex blob_pattern;
    try {
      tag_pattern = std::regex(tag_regex_str);
      blob_pattern = std::regex(blob_regex_str);
    } catch (const std::regex_error &e) {
      HLOG(kError, "Indexer SemanticSearch: bad regex (tag='{}' blob='{}'): {}",
           tag_regex_str, blob_regex_str, e.what());
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // The matched working set: same regex_match (full-string) semantics as
    // the core's old scan, evaluated against the index instead of the
    // metadata maps. Tag-name matches are cached per tag within the pass.
    struct MatchedDoc {
      const IndexedDoc *doc;
    };
    std::vector<MatchedDoc> matched;
    std::unordered_map<clio::run::u64, bool> tag_match_memo;
    {
      std::lock_guard<std::mutex> lock(index_mutex_);
      for (const auto &kv : index_) {
        const IndexedDoc &doc = kv.second;
        clio::run::u64 tkey = TagKey(doc.tag_id_);
        auto memo = tag_match_memo.find(tkey);
        bool tag_ok;
        if (memo != tag_match_memo.end()) {
          tag_ok = memo->second;
        } else {
          tag_ok = std::regex_match(doc.tag_name_, tag_pattern);
          tag_match_memo.emplace(tkey, tag_ok);
        }
        if (!tag_ok) continue;
        if (!std::regex_match(doc.blob_name_, blob_pattern)) continue;
        matched.push_back({&doc});
      }

      // BM25 with corpus statistics over the matched slice only — "rank
      // within this regex" semantics, identical constants and math to the
      // core's old implementation (k1=1.5 / b=0.75 Okapi defaults). Scoring
      // runs under the index lock (pointers into index_ stay valid); it is
      // pure in-memory arithmetic, no awaits.
      if (!matched.empty()) {
        constexpr double kK1 = 1.5;
        constexpr double kB = 0.75;
        std::unordered_map<std::string, int> df;
        double total_len = 0.0;
        for (const auto &m : matched) {
          total_len += static_cast<double>(m.doc->length_);
          for (const auto &kv2 : m.doc->tf_) df[kv2.first]++;
        }
        double avgdl = total_len / static_cast<double>(matched.size());
        if (avgdl <= 0.0) avgdl = 1.0;
        const size_t N = matched.size();

        auto qtokens = SemSearchTokenize(query_text.data(), query_text.size());
        std::unordered_set<std::string> uniq_q(qtokens.begin(), qtokens.end());

        std::vector<clio::cte::core::SemanticSearchResult> scored;
        scored.reserve(matched.size());
        for (const auto &m : matched) {
          const IndexedDoc &d = *m.doc;
          double score = 0.0;
          for (const auto &q : uniq_q) {
            auto df_it = df.find(q);
            if (df_it == df.end()) continue;
            auto tf_it = d.tf_.find(q);
            if (tf_it == d.tf_.end()) continue;
            double df_q = static_cast<double>(df_it->second);
            double idf = std::log((static_cast<double>(N) - df_q + 0.5) /
                                      (df_q + 0.5) +
                                  1.0);
            double tf_q = static_cast<double>(tf_it->second);
            double norm =
                1.0 - kB + kB * (static_cast<double>(d.length_) / avgdl);
            score += idf * (tf_q * (kK1 + 1.0)) / (tf_q + kK1 * norm);
          }
          scored.emplace_back(d.tag_id_, d.tag_name_, d.blob_name_, score);
        }

        std::sort(scored.begin(), scored.end(),
                  [](const clio::cte::core::SemanticSearchResult &a,
                     const clio::cte::core::SemanticSearchResult &b) {
                    return a.score_ > b.score_;
                  });
        if (k > 0 && scored.size() > k) scored.resize(k);
        task->results_ = std::move(scored);
      }
    }
    HLOG(kDebug,
         "Indexer SemanticSearch: tag='{}' blob='{}' query='{}' -> {} results "
         "(index size {})",
         tag_regex_str, blob_regex_str, query_text, task->results_.size(),
         index_.size());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ---------------------------------------------------------------------------
// On-demand scan (kReindexScan verb + first-insertion backfill): enumerate
// and ENQUEUE — the asynchronous drain does the indexing. Restarts restore
// persisted state and never come here.
// ---------------------------------------------------------------------------

clio::run::TaskResume Runtime::ScanAndEnqueue(std::string tag_regex,
                                              std::string blob_regex,
                                              clio::run::u64 *enqueued_out) {
  CLIO_TASK_BODY_BEGIN
  {
    auto *next = GetNextClient();

    // Enumerate THIS node's (tag, blob) pairs: Local keeps each container's
    // scan to the slice its co-located core container owns, matching the
    // interposer's owner-delegated routing (a Broadcast here would enqueue
    // every blob on every node).
    std::vector<std::string> tag_names;
    std::vector<std::string> blob_names;
    {
      auto query_fut = next->AsyncBlobQuery(tag_regex, blob_regex, 0,
                                            clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(query_fut);
      if (query_fut->GetReturnCode() != 0) {
        HLOG(kWarning, "Indexer: scan BlobQuery(tag='{}' blob='{}') failed "
             "(rc={})", tag_regex, blob_regex, query_fut->GetReturnCode());
        CLIO_CO_RETURN;
      }
      tag_names = query_fut->tag_names_;
      blob_names = query_fut->blob_names_;
    }

    for (size_t i = 0; i < tag_names.size() && i < blob_names.size(); ++i) {
      // Resolve the tag's id (GetOrCreateTag on an existing tag is a pure
      // lookup) and seed the name cache for the drain.
      TagId tag_id = TagId::GetNull();
      {
        auto tag_fut = next->AsyncGetOrCreateTag(tag_names[i]);
        CLIO_CO_AWAIT(tag_fut);
        if (tag_fut->GetReturnCode() != 0) {
          HLOG(kWarning, "Indexer: scan tag resolve failed for '{}'",
               tag_names[i]);
          continue;
        }
        tag_id = tag_fut->tag_id_;
      }
      {
        std::lock_guard<std::mutex> lock(index_mutex_);
        tag_names_[TagKey(tag_id)] = tag_names[i];
        // A scanned tag counts as seen: the scan IS its backfill.
        if (tags_seen_.insert(TagKey(tag_id)).second) {
          WalAppendTagSeen(tag_id);
        }
        if (scope_match_all_ ||
            std::regex_match(blob_names[i], scope_blob_re_)) {
          pending_[DocKey(tag_id, blob_names[i])] =
              PendingReindex{tag_id, blob_names[i]};
          if (enqueued_out != nullptr) ++(*enqueued_out);
        }
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReindexScan(
    clio::run::shared_ptr<ReindexScanTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::run::u64 enqueued = 0;
    CLIO_CO_AWAIT(ScanAndEnqueue(task->tag_regex_.str(),
                                 task->blob_regex_.str(), &enqueued));
    task->blobs_enqueued_ = enqueued;
    HLOG(kInfo, "Indexer: ReindexScan(tag='{}' blob='{}') enqueued {} blobs",
         task->tag_regex_.str(), task->blob_regex_.str(), enqueued);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::indexer

CLIO_TASK_CC(clio::cte::indexer::Runtime)

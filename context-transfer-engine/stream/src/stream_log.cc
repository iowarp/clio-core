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
 * Stream runtime, persistence: each container logs the sizes of the
 * streams it homes, drop tombstones, merge plans (with the staged names
 * they merge) and plan completions. A restart replays the log, compacts
 * it, and finishes plans left open by the crash.
 */

#include <clio_cte/stream/stream_runtime.h>

#include <cstring>

namespace clio::cte::stream {

namespace {
/** Drop tombstones older than this are not kept across compactions (ns). */
constexpr clio::run::u64 kTombstoneTtlNs = 3600ULL * 1000 * 1000 * 1000;
/** Merged names kept across compactions: younger than this (ns). */
constexpr clio::run::u64 kNameTtlNs = 600ULL * 1000 * 1000 * 1000;
/** Merged names per snapshot record. */
constexpr size_t kNamesPerRecord = 4096;

/** Little-endian record encoder. */
class Enc {
 public:
  explicit Enc(std::string *out) : out_(out) {}
  /** Append a u32. */
  void U32(clio::run::u32 v) { out_->append(reinterpret_cast<char *>(&v), 4); }
  /** Append a u64. */
  void U64(clio::run::u64 v) { out_->append(reinterpret_cast<char *>(&v), 8); }
  /** Append a length-prefixed string. */
  void Str(const std::string &s) {
    U32(static_cast<clio::run::u32>(s.size()));
    out_->append(s);
  }
  /** Append a tag id. */
  void Tag(const clio::cte::core::TagId &t) {
    U32(t.major_);
    U32(t.minor_);
  }

 private:
  std::string *out_;
};

/** Bounds-checked record decoder. */
class Dec {
 public:
  explicit Dec(const std::string &in) : in_(in) {}
  /** Read a u32. */
  bool U32(clio::run::u32 *v) { return Raw(v, 4); }
  /** Read a u64. */
  bool U64(clio::run::u64 *v) { return Raw(v, 8); }
  /** Read a length-prefixed string. */
  bool Str(std::string *s) {
    clio::run::u32 n = 0;
    if (!U32(&n) || off_ + n > in_.size()) return false;
    s->assign(in_.data() + off_, n);
    off_ += n;
    return true;
  }
  /** Read a tag id. */
  bool Tag(clio::cte::core::TagId *t) {
    return U32(&t->major_) && U32(&t->minor_);
  }

 private:
  /** Copy `n` raw bytes into `dst`. */
  bool Raw(void *dst, size_t n) {
    if (off_ + n > in_.size()) return false;
    std::memcpy(dst, in_.data() + off_, n);
    off_ += n;
    return true;
  }
  const std::string &in_;
  size_t off_ = 0;
};

/** Encode a plan record body. */
std::string EncodePlan(const StreamPlan &plan) {
  std::string p;
  Enc e(&p);
  e.U64(plan.id_);
  e.Tag(plan.tag_);
  e.U64(plan.base_);
  e.U32(static_cast<clio::run::u32>(plan.entries_.size()));
  for (const auto &x : plan.entries_) {
    e.Str(x.staged_name_);
    e.U64(x.size_);
    e.U64(x.clock_);
    e.U32(x.origin_);
    e.U64(x.counter_);
  }
  return p;
}

/** Decode a plan record body. */
bool DecodePlan(const std::string &payload, StreamPlan *plan) {
  Dec d(payload);
  clio::run::u32 n = 0;
  if (!d.U64(&plan->id_) || !d.Tag(&plan->tag_) || !d.U64(&plan->base_) ||
      !d.U32(&n)) {
    return false;
  }
  plan->entries_.resize(n);
  for (auto &x : plan->entries_) {
    if (!d.Str(&x.staged_name_) || !d.U64(&x.size_) || !d.U64(&x.clock_) ||
        !d.U32(&x.origin_) || !d.U64(&x.counter_)) {
      return false;
    }
  }
  return true;
}
}  // namespace

void Runtime::OpenLog() {
  if (config_.log_path_.empty()) return;
  const std::string path =
      config_.log_path_ + "." + std::to_string(container_id_);
  if (!log_.Open(path)) {
    HLOG(kError, "stream: cannot open log {}; sizes will not survive a restart",
         path);
    return;
  }
  std::lock_guard<std::mutex> g(mu_);
  if (!is_restart_) {
    log_.Rewrite({});  // a fresh start discards the previous run's state
    return;
  }
  size_t n = log_.Replay([this](clio::run::u32 t, const std::string &p) {
    ReplayRecord(t, p);
  });
  CompactLocked();
  // Every restored size may be stale (its file was served elsewhere while
  // this node was down): hold them until the owner reconciles them.
  for (const auto &kv : streams_) unverified_.insert(kv.first);
  gate_deadline_ = std::chrono::steady_clock::now() +
                   std::chrono::seconds(kRestoreGateS);
  HLOG(kInfo, "stream: replayed {} log records: {} streams, {} open plans", n,
       streams_.size(), open_plans_.size());
}

void Runtime::ReplayRecord(clio::run::u32 type, const std::string &payload) {
  Dec d(payload);
  clio::cte::core::TagId tag;
  clio::run::u64 v = 0;
  switch (type) {
    case kRecSize:
      if (d.Tag(&tag) && d.U64(&v)) {
        streams_[tag].size_ = v;
        dropped_.erase(tag);
      }
      break;
    case kRecDrop:
      if (d.Tag(&tag) && d.U64(&v)) {
        streams_.erase(tag);
        dropped_[tag] = v;
      }
      break;
    case kRecPlan: {
      StreamPlan plan;
      if (!DecodePlan(payload, &plan)) break;
      for (const auto &x : plan.entries_) {
        merged_names_.emplace(x.staged_name_, x.clock_);
      }
      next_plan_id_ = std::max(next_plan_id_, plan.id_ + 1);
      open_plans_[plan.id_] = std::move(plan);
      break;
    }
    case kRecDone:
      if (d.U64(&v)) open_plans_.erase(v);
      break;
    case kRecNames: {
      clio::run::u32 n = 0;
      if (!d.U32(&n)) break;
      for (clio::run::u32 i = 0; i < n; ++i) {
        std::string name;
        if (!d.Str(&name) || !d.U64(&v)) break;
        merged_names_.emplace(std::move(name), v);
      }
      break;
    }
    default:
      break;
  }
}

void Runtime::CompactLocked() {
  if (!log_.IsOpen()) return;
  const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
  std::vector<std::pair<clio::run::u32, std::string>> recs;
  for (const auto &kv : streams_) {
    std::string p;
    Enc e(&p);
    e.Tag(kv.first);
    e.U64(kv.second.size_);
    recs.emplace_back(kRecSize, std::move(p));
  }
  for (const auto &kv : dropped_) {
    if (now - kv.second > kTombstoneTtlNs) continue;
    std::string p;
    Enc e(&p);
    e.Tag(kv.first);
    e.U64(kv.second);
    recs.emplace_back(kRecDrop, std::move(p));
  }
  for (const auto &kv : open_plans_) {
    recs.emplace_back(kRecPlan, EncodePlan(kv.second));
  }
  std::vector<std::pair<std::string, clio::run::u64>> names;
  for (const auto &kv : merged_names_) {
    if (now - kv.second <= kNameTtlNs) names.emplace_back(kv);
  }
  for (size_t i = 0; i < names.size(); i += kNamesPerRecord) {
    const size_t end = std::min(names.size(), i + kNamesPerRecord);
    std::string p;
    Enc e(&p);
    e.U32(static_cast<clio::run::u32>(end - i));
    for (size_t k = i; k < end; ++k) {
      e.Str(names[k].first);
      e.U64(names[k].second);
    }
    recs.emplace_back(kRecNames, std::move(p));
  }
  if (!log_.Rewrite(recs)) HLOG(kError, "stream: log compaction failed");
}

void Runtime::LogSizeLocked(const clio::cte::core::TagId &tag,
                            clio::run::u64 size) {
  if (!log_.IsOpen()) return;
  std::string p;
  Enc e(&p);
  e.Tag(tag);
  e.U64(size);
  log_.Append(kRecSize, p);
}

void Runtime::LogPlanLocked(const StreamPlan &plan) {
  if (log_.IsOpen()) log_.Append(kRecPlan, EncodePlan(plan));
}

void Runtime::LogDoneLocked(clio::run::u64 plan_id) {
  if (!log_.IsOpen()) return;
  std::string p;
  Enc(&p).U64(plan_id);
  log_.Append(kRecDone, p);
}

void Runtime::LogDropLocked(const clio::cte::core::TagId &tag) {
  if (!log_.IsOpen()) return;
  std::string p;
  Enc e(&p);
  e.Tag(tag);
  e.U64(clio::cte::core::GetWallTimeNs());
  log_.Append(kRecDrop, p);
}

}  // namespace clio::cte::stream

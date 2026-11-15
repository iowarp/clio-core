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
 * Tag names in the shm cache: published name records are routed to their
 * tag's home (hash of the tag id, the same rule as every key), which keeps
 * them; a node resolves a name's ancestors through the cache (fetching a
 * missing one from its home, registered so a rename invalidates it); and a
 * TagQuery arm returns the names its container is home for, resolved and
 * matched. See DESIGN.md "Tags and metadata".
 */

#include <map>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_runtime.h>

namespace clio::cte::shm_wtc {

namespace wtc = clio::cte::core::wtc;
using clio::cte::core::TagNameOp;
bool WtcTrace();  // core_shm_cache_writethru_flush.cc
using clio::cte::core::TagNameOpRec;

namespace {

/** Parse a stored "$tagid{M.m}/leaf" reference (false for "/" or flat). */
bool ParseRef(const std::string &stored, TagId *parent, std::string *leaf) {
  static const std::string kPrefix = "$tagid{";
  if (stored.compare(0, kPrefix.size(), kPrefix) != 0) return false;
  const size_t close = stored.find('}', kPrefix.size());
  if (close == std::string::npos) return false;
  const std::string id = stored.substr(kPrefix.size(), close - kPrefix.size());
  const size_t dot = id.find('.');
  if (dot == std::string::npos) return false;
  try {
    parent->major_ = static_cast<clio::run::u32>(std::stoul(id.substr(0, dot)));
    parent->minor_ = static_cast<clio::run::u32>(std::stoul(id.substr(dot + 1)));
  } catch (...) {
    return false;
  }
  std::string rest = stored.substr(close + 1);
  if (!rest.empty() && rest[0] == '/') rest.erase(0, 1);
  *leaf = rest;
  return true;
}

/** Packed tag id. */
clio::run::u64 Pack(const TagId &t) {
  return (static_cast<clio::run::u64>(t.major_) << 32) | t.minor_;
}

/** Remove one occurrence of `name` from `names`. */
void EraseName(std::vector<std::string> *names, const std::string &name) {
  for (auto it = names->begin(); it != names->end(); ++it) {
    if (*it == name) {
      names->erase(it);
      return;
    }
  }
}

}  // namespace

void Runtime::ApplyHomedOp(const TagNameOpRec &r) {
  if (WtcTrace()) {
    HLOG(kInfo, "[WTC-TRACE] homed-op at={} op={} tag={}.{} seq={} '{}' -> "
         "'{}'", container_id_, static_cast<int>(r.op_), r.tag_.major_,
         r.tag_.minor_, r.seq_, r.name_, r.name2_);
  }
  std::lock_guard<std::mutex> g(homed_mu_);
  auto stamp = [this](const std::string &n) {
    auto it = homed_seq_.find(n);
    return it == homed_seq_.end() ? 0ull : it->second;
  };
  std::vector<std::string> &names = homed_names_[Pack(r.tag_)];
  switch (r.op_) {
    case TagNameOp::kAddName:
      if (r.seq_ < stamp(r.name_)) break;  // a newer op owns the name
      homed_seq_[r.name_] = r.seq_;
      EraseName(&names, r.name_);
      names.push_back(r.name_);
      break;
    case TagNameOp::kRemoveName:
      if (r.seq_ < stamp(r.name_)) break;
      homed_seq_[r.name_] = r.seq_;
      EraseName(&names, r.name_);
      break;
    case TagNameOp::kRename: {
      const bool from_ok = r.seq_ >= stamp(r.name_);
      const bool to_ok = r.seq_ >= stamp(r.name2_);
      if (from_ok) homed_seq_[r.name_] = r.seq_;
      if (to_ok) homed_seq_[r.name2_] = r.seq_;
      bool replaced = false;
      if (from_ok && to_ok) {
        for (std::string &n : names) {  // keep its position (canonical)
          if (n == r.name_) {
            n = r.name2_;
            replaced = true;
            break;
          }
        }
      }
      if (!replaced) {
        if (from_ok) EraseName(&names, r.name_);
        if (to_ok) {
          EraseName(&names, r.name2_);
          names.push_back(r.name2_);
        }
      }
      break;
    }
    case TagNameOp::kSetRoot:
      names.assign(1, "/");
      break;
    case TagNameOp::kResetPublished:
      break;  // node-local (the publisher's core); never routed
  }
  if (names.empty()) homed_names_.erase(Pack(r.tag_));
}

clio::run::TaskResume Runtime::NameRecs(
    clio::run::shared_ptr<clio::cte::core::UpdateTagNamesTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::vector<TagNameOpRec> ops;
    const std::string buf = task->ops_.str();
    clio::cte::core::DecodeTagNameOps(buf.data(), buf.size(), &ops);
    for (const TagNameOpRec &r : ops) ApplyHomedOp(r);
    task->applied_ = static_cast<clio::run::u32>(ops.size());
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RouteNameRecords(
    const std::vector<TagNameOpRec> &ops) {
  CLIO_TASK_BODY_BEGIN
  {
    std::map<clio::run::u32, std::string> by_home;
    for (const TagNameOpRec &r : ops) {
      if (r.op_ == TagNameOp::kResetPublished || r.tag_.IsNull()) continue;
      clio::cte::core::EncodeTagNameOp(&by_home[TagHome(r.tag_)], r.op_, r.tag_,
                                       r.seq_, r.name_, r.name2_);
    }
    auto *ipc = CLIO_IPC;
    std::vector<clio::run::Future<clio::cte::core::UpdateTagNamesTask>> futs;
    for (auto &h : by_home) {
      if (h.first == container_id_) {
        std::vector<TagNameOpRec> mine;
        clio::cte::core::DecodeTagNameOps(h.second.data(), h.second.size(),
                                          &mine);
        for (const TagNameOpRec &r : mine) ApplyHomedOp(r);
        continue;
      }
      auto t = ipc->NewTask<clio::cte::core::UpdateTagNamesTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::DirectId(h.first), h.second);
      t->method_ = Method::kNameRecs;
      futs.push_back(ipc->Send(t));
    }
    for (auto &f : futs) CLIO_CO_AWAIT(f);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TagRec(
    clio::run::shared_ptr<clio::cte::core::GetTagNameTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> g(homed_mu_);
    auto it = homed_names_.find(Pack(task->tag_id_));
    if (it != homed_names_.end() && !it->second.empty()) {
      task->tag_name_ =
          clio::run::priv::string(CLIO_PRIV_ALLOC, it->second.front());
      task->found_ = 1;
    } else {
      task->found_ = 0;
    }
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StoredNameOf(const TagId &tag,
                                            std::string *stored, bool *found) {
  CLIO_TASK_BODY_BEGIN
  *found = false;
  {
    const clio::run::u32 home = TagHome(tag);
    if (home == container_id_) {
      std::lock_guard<std::mutex> g(homed_mu_);
      auto it = homed_names_.find(Pack(tag));
      if (it != homed_names_.end() && !it->second.empty()) {
        *stored = it->second.front();
        *found = true;
      }
      CLIO_CO_RETURN;
    }
    if (view_.Ready() && view_.LookupTagName(tag.major_, tag.minor_, stored)) {
      *found = true;
      CLIO_CO_RETURN;
    }
    // Miss: register with the home BEFORE fetching, so a rename landing
    // after this point invalidates the copy (or voids the fill).
    const clio::run::u64 seq = view_.Ready() ? view_.TagInvalSeq() : 0;
    CLIO_CO_AWAIT(RegisterSharer(
        tag, std::string(wtc::kTagMetaName, wtc::kTagMetaNameLen)));
    auto *ipc = CLIO_IPC;
    auto t = ipc->NewTask<clio::cte::core::GetTagNameTask>(
        clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::DirectId(home),
        tag);
    t->method_ = Method::kTagRec;
    auto f = ipc->Send(t);
    CLIO_CO_AWAIT(f);
    if (f->GetReturnCode() == 0 && f->found_ != 0) {
      *stored = f->tag_name_.str();
      *found = true;
      if (view_.Ready()) {
        const bool filled = view_.FillTagName(tag.major_, tag.minor_,
                                              stored->data(), stored->size(),
                                              seq);
        if (WtcTrace()) {
          HLOG(kInfo, "[WTC-TRACE] name-fill at={} tag={}.{} home={} '{}' "
               "filled={}", container_id_, tag.major_, tag.minor_, home,
               *stored, filled);
        }
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ResolveStored(const std::string &stored,
                                             std::string *abs, bool *ok) {
  CLIO_TASK_BODY_BEGIN
  *ok = false;
  {
    std::vector<std::string> leaves;
    std::string cur = stored;
    for (int depth = 0; depth < 256; ++depth) {
      TagId parent;
      std::string leaf;
      if (cur == "/" || !ParseRef(cur, &parent, &leaf)) {
        if (cur != "/" && !leaves.empty()) break;  // a flat name is no parent
        std::string out = cur == "/" ? std::string() : cur;
        for (auto it = leaves.rbegin(); it != leaves.rend(); ++it) {
          out += "/" + *it;
        }
        *abs = out.empty() ? "/" : out;
        *ok = true;
        break;
      }
      leaves.push_back(leaf);
      bool found = false;
      CLIO_CO_AWAIT(StoredNameOf(parent, &cur, &found));
      if (!found) break;  // an ancestor not published (yet): skip the name
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TagSearch(
    clio::run::shared_ptr<clio::cte::core::TagQueryTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string pattern = task->tag_regex_.str();
    const clio::run::u32 max = task->max_tags_;
    task->results_.clear();
    task->result_ids_.clear();
    // Flat names live in this node's core (hierarchical ones are served
    // from the homed records below, never twice).
    auto f = Next()->AsyncTagQuery(pattern, 0, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(f);
    for (size_t i = 0; i < f->results_.size(); ++i) {
      const std::string &n = f->results_[i];
      if (!n.empty() && n[0] == '/') continue;
      if (max != 0 && task->results_.size() >= max) break;
      task->results_.push_back(n);
      task->result_ids_.push_back(i < f->result_ids_.size() ? f->result_ids_[i]
                                                           : 0);
    }
    std::vector<std::pair<clio::run::u64, std::string>> recs;
    {
      std::lock_guard<std::mutex> g(homed_mu_);
      for (const auto &e : homed_names_) {
        for (const std::string &n : e.second) recs.emplace_back(e.first, n);
      }
    }
    std::regex re;
    bool re_ok = true;
    try {
      re = std::regex(pattern);
    } catch (...) {
      re_ok = false;
    }
    for (size_t i = 0; re_ok && i < recs.size(); ++i) {
      std::string abs;
      bool ok = false;
      CLIO_CO_AWAIT(ResolveStored(recs[i].second, &abs, &ok));
      if (!ok || !std::regex_match(abs, re)) continue;
      if (max != 0 && task->results_.size() >= max) break;
      task->results_.push_back(abs);
      task->result_ids_.push_back(recs[i].first);
    }
    task->total_tags_matched_ = task->results_.size();
    task->SetReturnCode(re_ok ? 0 : 1);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::shm_wtc

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
 * Replication handoff log: the changes a container made on behalf of a dead
 * owner (see replication_remote.cc) outlive a restart of that container.
 * Without it a successor that restarted before its owner returned forgot
 * what to hand back, and the owner came back serving its stale copies.
 *
 * Records (RecordLog framing): kHandoffNote / kHandoffDone, each
 * [u32 owner][u8 deleted][u32 tag major][u32 tag minor][blob name].
 */

#include <clio_cte/replication/replication_runtime.h>

#include <cstdint>
#include <cstring>

namespace clio::cte::replication {

namespace {

/**
 * Append a fixed-width integer to a record payload.
 * @param out payload being built
 * @param v value
 */
template <typename T>
void Put(std::string *out, T v) {
  out->append(reinterpret_cast<const char *>(&v), sizeof(v));
}

/**
 * Read a fixed-width integer from a record payload.
 * @param p payload
 * @param off read cursor, advanced on success
 * @param v receives the value
 * @return false if the payload is too short
 */
template <typename T>
bool Get(const std::string &p, size_t *off, T *v) {
  if (*off + sizeof(T) > p.size()) return false;
  std::memcpy(v, p.data() + *off, sizeof(T));
  *off += sizeof(T);
  return true;
}

/**
 * Encode one handoff record.
 * @param owner owner container
 * @param deleted delete (true) or write/truncate (false)
 * @param tag blob's tag
 * @param name blob name
 * @return payload
 */
std::string Encode(clio::run::u32 owner, bool deleted, const TagId &tag,
                   const std::string &name) {
  std::string p;
  Put<clio::run::u32>(&p, owner);
  Put<std::uint8_t>(&p, deleted ? 1 : 0);
  Put<clio::run::u32>(&p, tag.major_);
  Put<clio::run::u32>(&p, tag.minor_);
  p += name;
  return p;
}

/**
 * Handoff map key of a blob (same form NoteHandoff uses).
 * @param tag blob's tag
 * @param name blob name
 * @return "major.minor.name"
 */
std::string Key(const TagId &tag, const std::string &name) {
  return std::to_string(tag.major_) + "." + std::to_string(tag.minor_) + "." +
         name;
}

}  // namespace

void Runtime::OpenHandoffLog() {
  if (config_.handoff_log_path_.empty()) return;
  const std::string path =
      config_.handoff_log_path_ + "." + std::to_string(container_id_);
  if (!handoff_log_.Open(path)) {
    HLOG(kError, "replication: cannot open handoff log {}; failover changes "
         "will not survive a restart of this node", path);
    return;
  }
  std::lock_guard<std::mutex> g(handoff_mu_);
  if (!is_restart_) {
    handoff_log_.Rewrite({});  // a fresh start discards the previous run's
    return;
  }
  size_t n = handoff_log_.Replay(
      [this](clio::run::u32 type, const std::string &p) {
        size_t off = 0;
        clio::run::u32 owner = 0, major = 0, minor = 0;
        std::uint8_t deleted = 0;
        if (!Get(p, &off, &owner) || !Get(p, &off, &deleted) ||
            !Get(p, &off, &major) || !Get(p, &off, &minor)) {
          return;
        }
        HandoffEntry e{TagId(major, minor), p.substr(off), deleted != 0};
        const std::string key = Key(e.tag_, e.name_);
        if (type == kHandoffNote) {
          handoff_[owner][key] = e;
          return;
        }
        auto it = handoff_.find(owner);
        if (it == handoff_.end()) return;
        auto ent = it->second.find(key);
        if (ent != it->second.end() && ent->second.deleted_ == e.deleted_) {
          it->second.erase(ent);
        }
        if (it->second.empty()) handoff_.erase(it);
      });
  size_t pending = 0;
  for (const auto &kv : handoff_) pending += kv.second.size();
  HLOG(kInfo, "replication: replayed {} handoff records, {} change(s) still "
       "to hand back", n, pending);
  CompactHandoffLogLocked();
}

void Runtime::LogHandoff(clio::run::u32 type, clio::run::u32 owner,
                         const HandoffEntry &e) {
  if (!handoff_log_.IsOpen()) return;
  handoff_log_.Append(type, Encode(owner, e.deleted_, e.tag_, e.name_));
}

void Runtime::CompactHandoffLogLocked() {
  if (!handoff_log_.IsOpen()) return;
  std::vector<std::pair<clio::run::u32, std::string>> recs;
  for (const auto &kv : handoff_) {
    for (const auto &ent : kv.second) {
      recs.emplace_back(kHandoffNote, Encode(kv.first, ent.second.deleted_,
                                             ent.second.tag_,
                                             ent.second.name_));
    }
  }
  if (!handoff_log_.Rewrite(recs)) {
    HLOG(kError, "replication: handoff log compaction failed");
  }
}

}  // namespace clio::cte::replication

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

#ifndef CLIO_CTE_SHM_WRITETHRU_CACHE_TASKS_H_
#define CLIO_CTE_SHM_WRITETHRU_CACHE_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/config_parse.h>
#include <clio_ctp/util/msan.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/shm_writethru_cache/autogen/shm_writethru_cache_methods.h>

#include <cstring>
#include <string>
#include <vector>

namespace clio::cte::shm_wtc {

/** Well-known pool id of the default-compose shm write-through cache. */
inline const clio::run::PoolId kShmWtcPoolId(565, 0);  // == wtc::kDefaultPoolId

/**
 * Pool configuration (compose YAML keys in brackets). Every field
 * round-trips through serialize().
 */
struct WtcConfig {
  static constexpr const char *chimod_lib_name =
      "clio_cte_shm_writethru_cache";

  clio::run::PoolId next_pool_id_;    ///< [next_pool_id] pool below this one
  clio::run::u64 capacity_ = 256ull << 20;      ///< [capacity] segment bytes
  clio::run::u32 log_depth_ = 65536;            ///< [log_depth] ranges
  clio::run::u64 flush_batch_bytes_ = 16ull << 20;  ///< [flush_batch_bytes]
  clio::run::u32 max_tags_ = 4096;              ///< [max_tags]
  clio::run::u32 max_blobs_ = 65536;            ///< [max_blobs]
  clio::run::u64 max_fill_ = 16ull << 20;       ///< [max_fill]
  clio::run::u32 evict_sample_ = 16;            ///< [evict_sample]
  clio::run::u64 half_life_ms_ = 10000;         ///< [half_life_ms]
  double flush_period_us_ = 50.0;               ///< [flush_period_us]
  clio::run::u32 flush_retries_ = 3;            ///< [flush_retries]
  clio::run::u32 max_sharer_entries_ = 65536;   ///< [max_sharer_entries]

  WtcConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  WtcConfig(const clio::run::PoolId &pool_id, const WtcConfig &other) {
    (void)pool_id;
    *this = other;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_, capacity_, log_depth_, flush_batch_bytes_, max_tags_,
       max_blobs_, max_fill_, evict_sample_, half_life_ms_, flush_period_us_,
       flush_retries_, max_sharer_entries_);
  }

  /** Parse "major.minor". */
  static clio::run::PoolId ParsePoolId(const std::string &s) {
    auto dot = s.find('.');
    if (dot == std::string::npos) {
      return clio::run::PoolId::GetNull();
    }
    return clio::run::PoolId(
        static_cast<clio::run::u32>(std::stoul(s.substr(0, dot))),
        static_cast<clio::run::u32>(std::stoul(s.substr(dot + 1))));
  }

  /** Load configuration from compose YAML. */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) {
      return;
    }
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      ctp::MsanUnpoisonYaml(node);
      if (node["next_pool_id"]) {
        next_pool_id_ = ParsePoolId(node["next_pool_id"].as<std::string>());
      }
      auto size_of = [&node](const char *key, clio::run::u64 *out) {
        if (node[key]) {
          *out = ctp::ConfigParse::ParseSize(node[key].as<std::string>());
        }
      };
      size_of("capacity", &capacity_);
      size_of("flush_batch_bytes", &flush_batch_bytes_);
      size_of("max_fill", &max_fill_);
      if (node["log_depth"]) log_depth_ = node["log_depth"].as<clio::run::u32>();
      if (node["max_tags"]) max_tags_ = node["max_tags"].as<clio::run::u32>();
      if (node["max_blobs"]) max_blobs_ = node["max_blobs"].as<clio::run::u32>();
      if (node["evict_sample"]) {
        evict_sample_ = node["evict_sample"].as<clio::run::u32>();
      }
      if (node["half_life_ms"]) {
        half_life_ms_ = node["half_life_ms"].as<clio::run::u64>();
      }
      if (node["flush_period_us"]) {
        flush_period_us_ = node["flush_period_us"].as<double>();
      }
      if (node["flush_retries"]) {
        flush_retries_ = node["flush_retries"].as<clio::run::u32>();
      }
      if (node["max_sharer_entries"]) {
        max_sharer_entries_ = node["max_sharer_entries"].as<clio::run::u32>();
      }
    } catch (...) {
      // Best-effort, like every other chimod's config.
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<WtcConfig>;
using MonitorTask = clio::run::admin::MonitorTask;

/** Tear the container down (unmaps and unlinks the segment). */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}
  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other) {
    Task::AggregateOut(other);
  }
  void Copy(const ctp::ipc::FullPtr<DestroyTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Drain the dirty-range log (periodic; the log's only consumer). One call
 * pops at most flush_batch_bytes of ranges and sends them down as
 * MultiPutBlob batches.
 */
struct FlushLogTask : public clio::run::Task {
  OUT clio::run::u64 ranges_;  ///< ranges completed by this call
  OUT clio::run::u64 bytes_;   ///< payload bytes shipped by this call

  FlushLogTask() : clio::run::Task(), ranges_(0), bytes_(0) {}
  explicit FlushLogTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kFlushLog),
        ranges_(0),
        bytes_(0) {}
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<FlushLogTask>();
    ranges_ += other->ranges_;
    bytes_ += other->bytes_;
  }
  void Copy(const ctp::ipc::FullPtr<FlushLogTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    ranges_ = other->ranges_;
    bytes_ = other->bytes_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(ranges_, bytes_);
  }
};

/** One blob key packed into an Invalidate/RegisterSharer list. */
struct BlobKey {
  clio::run::u32 major_ = 0;
  clio::run::u32 minor_ = 0;
  std::string name_;
};

/** Pack keys: [u32 major][u32 minor][u32 len][name]... */
inline std::string PackKeys(const std::vector<BlobKey> &keys) {
  std::string out;
  for (const BlobKey &k : keys) {
    clio::run::u32 hdr[3] = {k.major_, k.minor_,
                             static_cast<clio::run::u32>(k.name_.size())};
    out.append(reinterpret_cast<const char *>(hdr), sizeof(hdr));
    out.append(k.name_);
  }
  return out;
}

/** Inverse of PackKeys; stops at the first malformed record. */
inline std::vector<BlobKey> UnpackKeys(const std::string &in) {
  std::vector<BlobKey> keys;
  size_t pos = 0;
  while (pos + 3 * sizeof(clio::run::u32) <= in.size()) {
    clio::run::u32 hdr[3];
    std::memcpy(hdr, in.data() + pos, sizeof(hdr));
    pos += sizeof(hdr);
    if (pos + hdr[2] > in.size()) {
      break;
    }
    keys.push_back(BlobKey{hdr[0], hdr[1], in.substr(pos, hdr[2])});
    pos += hdr[2];
  }
  return keys;
}

/** What an InvalidateTask asks of its target container. */
enum InvalidateMode : clio::run::u32 {
  /** Drop cached copies of these blobs (sent by a blob's home). */
  kInvalidateDrop = 0,
  /** Record the sender as caching these blobs (sent to their home). */
  kInvalidateRegister = 1,
  /** Home: drop every node's copy of these blobs (DelBlob). */
  kInvalidateFanout = 2,
  /** Drop every cached blob of these tags (key name empty; DelTag). */
  kInvalidateDropTag = 3,
};

/**
 * Coherence message between cache containers: a home dropping other nodes'
 * copies, or a reader registering itself with a home before it fills.
 */
struct InvalidateTask : public clio::run::Task {
  IN clio::run::u32 mode_;          ///< InvalidateMode
  IN clio::run::u32 sender_node_;   ///< node that sent the message
  IN clio::run::priv::string keys_; ///< PackKeys()

  InvalidateTask()
      : clio::run::Task(), mode_(0), sender_node_(0), keys_(CLIO_PRIV_ALLOC) {}
  explicit InvalidateTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          clio::run::u32 mode, clio::run::u32 sender_node,
                          const std::string &keys)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kInvalidate),
        mode_(mode),
        sender_node_(sender_node),
        keys_(CLIO_PRIV_ALLOC, keys) {}
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other) {
    Task::AggregateOut(other);
  }
  void Copy(const ctp::ipc::FullPtr<InvalidateTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    mode_ = other->mode_;
    sender_node_ = other->sender_node_;
    keys_ = other->keys_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(mode_, sender_node_, keys_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * A flush batch landing at its blobs' home cache: the core's
 * MultiPutBlobTask (it already ships its payload across nodes), addressed to
 * this pool under kHomeFlush. context_.origin_node_ names the writer node.
 */
using HomeFlushTask = clio::cte::core::MultiPutBlobTask;

}  // namespace clio::cte::shm_wtc

#endif  // CLIO_CTE_SHM_WRITETHRU_CACHE_TASKS_H_
